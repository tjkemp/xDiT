"""Build the SDMA transport (transport.hip) into a shared library; needs hipcc and MoRI, not a GPU.

Standard library only, so an image build can run it without importing xfuser
(whose dependencies may look for a GPU on import):

    python xfuser/core/distributed/sdma/build.py --out DIR [--offload-arch gfx942,gfx950]

The library, DIR/libxdit_sdma.so, is what XDIT_SDMA_LIB points at. Built into
this directory, transport.py loads it while it matches the source. It links
MoRI's libmori_application (see `mori_paths`).
"""

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("transport.hip")
LIB_NAME = "libxdit_sdma.so"
# What `build` compiles for by default.
BUILD_ARCHS = ("gfx942", "gfx950")

MORI_MISSING = (
    "--ulysses_a2a sdma needs MoRI (https://github.com/ROCm/mori), and it was not found. Install it "
    "(`pip install git+https://github.com/ROCm/mori`), or point XDIT_MORI_INCLUDE (MoRI's include and "
    "3rdparty/spdlog/include, ':'-separated) and XDIT_MORI_LIB_DIR (the directory holding "
    "libmori_application.so) at a MoRI build. Alternatively, set XDIT_SDMA_LIB to a transport library "
    f"built where MoRI is (`python {Path(__file__)} --out DIR`)."
)


def source_hash() -> str:
    return hashlib.sha1(SOURCE.read_bytes()).hexdigest()


def mori_paths():
    """(include directories, library directory) of MoRI, or None if it is not found.

    XDIT_MORI_INCLUDE (os.pathsep-separated) and XDIT_MORI_LIB_DIR, as for a source build
    (MoRI's include/ and 3rdparty/spdlog/include; the directory holding
    libmori_application.so); else the installed `mori` package, found without importing it.
    """
    include, lib = os.environ.get("XDIT_MORI_INCLUDE", ""), os.environ.get("XDIT_MORI_LIB_DIR", "")
    if include and lib:
        return [Path(p) for p in include.split(os.pathsep) if p], Path(lib)
    try:
        spec = importlib.util.find_spec("mori")
    except (ImportError, ValueError):
        spec = None
    for root in (spec.submodule_search_locations or []) if spec else []:
        root = Path(root)
        sources = root / "_jit-sources"
        if (root / "libmori_application.so").is_file() and (sources / "include").is_dir():
            return [sources / "include", sources / "3rdparty" / "spdlog" / "include"], root
    return None


def build(out_dir, archs) -> Path:
    """Compile the transport for `archs` (e.g. ["gfx942", "gfx950"]) into out_dir/LIB_NAME."""
    paths = mori_paths()
    if paths is None:
        raise RuntimeError(MORI_MISSING)
    includes, lib = paths
    hipcc = shutil.which("hipcc") or str(Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin" / "hipcc")
    if not Path(hipcc).exists():
        raise RuntimeError(
            "the SDMA transport needs hipcc to build. Build it where hipcc is available with `python "
            f"{Path(__file__)} --out DIR` and set XDIT_SDMA_LIB=DIR/{LIB_NAME}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    so = out / LIB_NAME
    tmp = so.with_suffix(f".{os.getpid()}.tmp")
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    cmd = [hipcc, "-O2", "-std=c++17", "-shared", "-fPIC", *[f"--offload-arch={a}" for a in archs]]
    cmd += [f"-I{p}" for p in includes if p.is_dir()] + [f"-I{rocm / 'include'}"]
    cmd += ["-o", str(tmp), str(SOURCE), f"-L{lib}", "-lmori_application", f"-Wl,-rpath,{lib}"]
    subprocess.run(cmd, check=True)
    tmp.rename(so)
    (out / (LIB_NAME + ".source-sha1")).write_text(source_hash() + "\n")
    return so


def main(argv=None, prog=None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=f"compile the SDMA transport into DIR/{LIB_NAME}")
    parser.add_argument("--out", required=True, metavar="DIR")
    parser.add_argument("--offload-arch", default=",".join(BUILD_ARCHS), help="comma-separated GPU architectures")
    args = parser.parse_args(argv)
    print(build(args.out, [a for a in args.offload_arch.split(",") if a]))


if __name__ == "__main__":
    main()
