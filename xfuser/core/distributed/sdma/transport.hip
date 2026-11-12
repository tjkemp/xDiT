// SDMA transport for xDiT's Ulysses all-to-all (see transport.py), loaded with ctypes. Built
// against MoRI's headers and libmori_application, whose anvil layer owns the SDMA queues.
//
// Buffers that peers write into are allocated here with hipMalloc (flag words: uncached) and
// shared with HIP IPC handles.
//
// A send is one kernel on the sending stream: for each peer, one lane writes the peer's copy
// packets into that peer's anvil queue (one queue per peer, on the SDMA engine KFD recommends
// for the link) and rings its doorbell once. Behind the copies, a FENCE packet on the same queue
// writes 1 into this rank's flag word at the peer; a queue runs in order, so the flag lands
// after the data, and after every earlier send's copies to that peer. A copy is linear, or 2D
// (`height` rows of `width` bytes at two pitches, one LARGE_SUB_WINDOW packet, SDMA 4.4:
// gfx942 and gfx950).
//
// The receiving stream waits for flags with sx_wait_flags, a one-thread kernel whose wait is
// bounded: a flag that never lands is recorded (sx_peek_error / sx_take_error), and the stream
// either moves on or stays held for the host to abort the process. The kernel resets the flags
// it waited for, so the next use of the same words needs no separate clear.
//
// Functions return 0 on success and a negative code otherwise.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "mori/application/transport/sdma/anvil.hpp"

// A bounded flag wait that ran out of time, written by sx_wait_flags into host-visible memory.
struct SxError {
  unsigned long long count;    // waits that timed out since the last sx_take_error
  unsigned long long missing;  // first one: bit i set if flag word i never read 1
  int tag;                     // first one: caller's tag (which wait) and call id
  int call;
};

namespace {

struct Copy {
  uint64_t src_off;    // from the source base of the send
  uint64_t dst;        // address in this process (a peer's buffer opened with IPC)
  uint64_t bytes;      // linear copy when height == 0; otherwise unused
  uint32_t width;      // 2D: bytes per row
  uint32_t height;     // 2D: rows; 0 for a linear copy
  uint32_t src_pitch;  // 2D
  uint32_t dst_pitch;  // 2D
};

struct Peer {
  anvil::SdmaQueueDeviceHandle* queue;
  uint32_t first, count;  // its copies in the route's copy array
  uint64_t flag;          // flag word to set to 1 behind them; 0 for none
};

struct Route {
  Peer* peers = nullptr;   // device
  Copy* copies = nullptr;  // device
  int n = 0;
};

std::vector<Route> g_routes;
SxError* g_error = nullptr;      // host-mapped
uint32_t* g_scratch = nullptr;   // device word that a dropped flag write goes to (test hook)
unsigned long long g_ticks_per_s = 0;

// The largest linear copy one packet takes (30-bit count, minus one).
constexpr uint64_t kMaxLinear = 1ull << 30;

// Block i posts peer i's copies and flag from `src_base`; peer `drop`'s flag goes to `scratch`.
__global__ void post(const Peer* peers, const Copy* copies, uint64_t src_base, int drop, uint32_t* scratch) {
  if (threadIdx.x != 0) return;
  // Write back L2 (buffer_wbl2): the engines read the source from memory, and it was written by
  // earlier kernels on this stream.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  const Peer p = peers[blockIdx.x];
  anvil::SdmaQueueDeviceHandle q = *p.queue;
  // One reservation and one doorbell for the peer's packets.
  uint64_t bytes = p.flag ? sizeof(SDMA_PKT_FENCE) : 0;
  for (uint32_t i = 0; i < p.count; ++i)
    bytes += copies[p.first + i].height ? sizeof(SDMA_PKT_COPY_LARGE_SUB_WINDOW) : sizeof(SDMA_PKT_COPY_LINEAR);
  if (!bytes) return;
  uint64_t offset = 0;
  const uint64_t base = q.ReserveQueueSpace(bytes, offset);
  uint64_t pending = base;
  for (uint32_t i = 0; i < p.count; ++i) {
    const Copy c = copies[p.first + i];
    void* src = (void*)(src_base + c.src_off);
    if (c.height) {
      auto packet = anvil::CreateCopy2DPacket(src, c.src_pitch, (void*)c.dst, c.dst_pitch, c.width, c.height);
      q.placePacket(packet, pending, offset);
    } else {
      auto packet = anvil::CreateCopyPacket(src, (void*)c.dst, c.bytes);
      q.placePacket(packet, pending, offset);
    }
    offset = 0;  // the wrap padding goes before the first packet only
  }
  if (p.flag) {
    void* flag = (int)blockIdx.x == drop ? (void*)scratch : (void*)p.flag;
    auto fence = anvil::CreateFencePacket((HSAuint64*)flag, 1);
    q.placePacket(fence, pending, offset);
  }
  q.submitPacket(base, pending);
}

// Waits until all `count` words at `words` but word `skip` (the waiter's own, which no engine
// writes) read 1 (acquire, system scope), then sets them back to 0. Once `budget` wall-clock
// ticks have passed it records which words were missing; then it returns, or with `block` keeps
// waiting, holding the stream so no work after it runs on data that never came (the host's
// watchdog aborts the process).
__global__ void wait_flags(int* words, int count, int skip, unsigned long long budget, SxError* err, int tag,
                           int call, int block) {
  unsigned long long start = wall_clock64();
  bool recorded = false;
  while (true) {
    unsigned long long missing = 0;
    for (int i = 0; i < count; ++i)
      if (i != skip && __hip_atomic_load(words + i, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != 1)
        missing |= 1ull << i;
    if (!missing) break;
    if (!recorded && wall_clock64() - start > budget) {
      if (__hip_atomic_fetch_add(&err->count, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
        err->missing = missing;
        err->tag = tag;
        err->call = call;
      }
      __threadfence_system();
      if (!block) break;
      recorded = true;
    }
    __builtin_amdgcn_s_sleep(2);
  }
  for (int i = 0; i < count; ++i) __hip_atomic_store(words + i, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
}

int take_error(int* tag, int* call, uint64_t* missing, bool clear) {
  volatile SxError* e = g_error;
  if (e == nullptr || e->count == 0) return 0;
  unsigned long long count = e->count;
  *tag = e->tag;
  *call = e->call;
  *missing = e->missing;
  if (clear) e->count = 0;
  return (int)(count > 0x7fffffff ? 0x7fffffff : count);
}

}  // namespace

extern "C" {

// The KFD node id of HIP device `hip_dev`: host-global, so ranks can exchange it.
int sx_kfd_node(int hip_dev) { return anvil::AnvilLib::kfdNodeIdForHipDevice(hip_dev); }

int sx_init(int hip_dev) {
  if (g_error) return 0;
  if (hipSetDevice(hip_dev) != hipSuccess) return -1;
  // Fails here, not mid-call, if the library has no code object for this GPU.
  hipFuncAttributes attr;
  if (hipFuncGetAttributes(&attr, (const void*)post) != hipSuccess) return -11;
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, hip_dev) != hipSuccess || khz <= 0) return -2;
  g_ticks_per_s = (unsigned long long)khz * 1000ull;
  if (hipMalloc(&g_scratch, sizeof(uint32_t)) != hipSuccess) return -3;
  if (hipHostMalloc((void**)&g_error, sizeof(SxError), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess)
    return -4;
  *g_error = SxError{};
  anvil::anvil.init();
  return 0;
}

// The device handle of the queue from KFD node `src` to `dst` (one per pair, made on first use).
int sx_queue(int src, int dst, uint64_t* out) {
  if (!anvil::anvil.connect(src, dst, 1)) return -1;
  anvil::SdmaQueue* q = anvil::anvil.getSdmaQueue(src, dst, 0);
  if (!q || !q->deviceHandle()) return -2;
  *out = (uint64_t)q->deviceHandle();
  return 0;
}

// Device memory for peers to write into: hipMalloc, or `uncached` (flag words, which a kernel
// polls). Sizes are rounded up to 2 MiB: a smaller uncached allocation that is not a whole
// number of pages cannot be exported with hipIpcGetMemHandle.
int sx_malloc(uint64_t bytes, int uncached, uint64_t* out) {
  constexpr uint64_t kGranule = 2ull << 20;
  bytes = (bytes + kGranule - 1) / kGranule * kGranule;
  void* p = nullptr;
  hipError_t e = uncached ? hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached) : hipMalloc(&p, bytes);
  if (e != hipSuccess) return -1;
  if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) return -2;
  *out = (uint64_t)p;
  return 0;
}

int sx_free(uint64_t ptr) { return hipFree((void*)ptr) == hipSuccess ? 0 : -1; }

int sx_ipc_handle_size() { return (int)sizeof(hipIpcMemHandle_t); }

// The IPC handle of an sx_malloc buffer, written to `out` (sx_ipc_handle_size bytes).
int sx_ipc_get(uint64_t ptr, void* out) {
  hipIpcMemHandle_t h;
  if (hipIpcGetMemHandle(&h, (void*)ptr) != hipSuccess) return -1;
  memcpy(out, &h, sizeof(h));
  return 0;
}

// Maps a peer's buffer from its IPC handle for this GPU.
int sx_ipc_open(const void* handle, uint64_t* out) {
  hipIpcMemHandle_t h;
  memcpy(&h, handle, sizeof(h));
  void* p = nullptr;
  if (hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess) != hipSuccess) return -1;
  *out = (uint64_t)p;
  return 0;
}

int sx_ipc_close(uint64_t ptr) { return hipIpcCloseMemHandle((void*)ptr) == hipSuccess ? 0 : -1; }

// A route: n peers; peer i posts counts[i] copies on queues[i], then sets flags[i] (flags null:
// none). Copy j: src_offsets, dsts, bytes (linear), or widths/heights/src_pitches/dst_pitches
// (2D, when heights[j] > 0). Returns the route id (>= 0).
int sx_route_create(int n, const uint64_t* queues, const int* counts, const uint64_t* flags,
                    const uint64_t* src_offsets, const uint64_t* dsts, const uint64_t* bytes, const uint32_t* widths,
                    const uint32_t* heights, const uint32_t* src_pitches, const uint32_t* dst_pitches) {
  std::vector<Peer> peers(n);
  std::vector<Copy> copies;
  for (int i = 0; i < n; ++i) {
    peers[i] = {(anvil::SdmaQueueDeviceHandle*)queues[i], (uint32_t)copies.size(), (uint32_t)counts[i],
                flags ? flags[i] : 0};
    for (int k = 0; k < counts[i]; ++k) {
      size_t j = copies.size();
      if (heights[j] == 0 && (bytes[j] == 0 || bytes[j] > kMaxLinear)) return -3;
      if (heights[j] && (widths[j] == 0 || widths[j] > src_pitches[j] || widths[j] > dst_pitches[j])) return -4;
      copies.push_back({src_offsets[j], dsts[j], bytes[j], widths[j], heights[j], src_pitches[j], dst_pitches[j]});
    }
  }
  Route r;
  r.n = n;
  if (hipMalloc(&r.peers, sizeof(Peer) * (n ? n : 1)) != hipSuccess ||
      hipMalloc(&r.copies, sizeof(Copy) * (copies.empty() ? 1 : copies.size())) != hipSuccess)
    return -1;
  if (hipMemcpy(r.peers, peers.data(), sizeof(Peer) * n, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(r.copies, copies.data(), sizeof(Copy) * copies.size(), hipMemcpyHostToDevice) != hipSuccess)
    return -2;
  for (size_t id = 0; id < g_routes.size(); ++id)
    if (!g_routes[id].peers) {
      g_routes[id] = r;
      return (int)id;
    }
  g_routes.push_back(r);
  return (int)g_routes.size() - 1;
}

int sx_route_free(int id) {
  if (id < 0 || id >= (int)g_routes.size() || !g_routes[id].peers) return -1;
  // Its tables may still be read by a post kernel in flight.
  if (hipDeviceSynchronize() != hipSuccess) return -2;
  hipError_t peers = hipFree(g_routes[id].peers), copies = hipFree(g_routes[id].copies);
  g_routes[id] = Route();
  return peers == hipSuccess && copies == hipSuccess ? 0 : -3;
}

// Posts route `id`'s copies from `src_base` once `stream` gets here. `drop` (test hook): the peer
// index whose flag write goes to a scratch word instead, or -1.
int sx_send(hipStream_t stream, int id, uint64_t src_base, int drop) {
  if (id < 0 || id >= (int)g_routes.size() || !g_routes[id].peers) return -1;
  const Route& r = g_routes[id];
  if (!r.n) return 0;
  hipLaunchKernelGGL(post, dim3(r.n), dim3(64), 0, stream, r.peers, r.copies, src_base, drop, g_scratch);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// Enqueues on `stream` a wait until the `count` 32-bit flags at `words`, all but word `skip`,
// read 1, then resets them to 0. After `seconds` a wait that is still missing flags is recorded
// (sx_peek_error, sx_take_error); then the stream moves on, or with `block` stays held (for the
// host to abort the process).
int sx_wait_flags(hipStream_t stream, uint64_t words, int count, int skip, double seconds, int tag, int call,
                  int block) {
  if (count > 64) return -1;
  unsigned long long budget = (unsigned long long)(seconds * (double)g_ticks_per_s);
  hipLaunchKernelGGL(wait_flags, dim3(1), dim3(1), 0, stream, (int*)words, count, skip, budget, g_error, tag, call,
                     block);
  return hipGetLastError() == hipSuccess ? 0 : -2;
}

// How many bounded waits ran out (0 if none), and the first one's tag, call id and missing-flag
// mask. sx_take_error clears the record; sx_peek_error (for a watchdog that polls it) does not.
int sx_take_error(int* tag, int* call, uint64_t* missing) { return take_error(tag, call, missing, true); }

int sx_peek_error(int* tag, int* call, uint64_t* missing) { return take_error(tag, call, missing, false); }

}  // extern "C"
