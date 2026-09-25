import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from xfuser.model_executor.cache import (
    DBCachePreset,
    CacheDitAdapterConfig,
    DBCacheSettings,
)
from xfuser.model_executor.models.runner_models.base_model import (
    register_model,
    xFuserModel,
    ModelCapabilities,
    DefaultInputValues,
    DiffusionOutput,
    ModelSettings,
    DIFFUSERS_FROM_SOURCE,
    _parse_attention_backend,
)
from xfuser import xFuserArgs
from xfuser.core.distributed import get_runtime_state
from xfuser.core.distributed.attention_backend import AttentionBackendType
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)

# Sparse and Sparge backends are already refused by the capability checks. FLEX_VSA_H3 is
# neither, but needs per-head gate inputs that only the MiniMax-H3 processors supply.
_QWEN_IMAGE_21_UNSUPPORTED_ATTENTION_BACKENDS = frozenset({AttentionBackendType.FLEX_VSA_H3})
# num_attention_heads of the Qwen/Qwen-Image-2.1 transformer; Ulysses splits heads across ranks.
_QWEN_IMAGE_21_NUM_HEADS = 32

@register_model("Qwen/Qwen-Image-Edit-2511")
@register_model("Qwen/Qwen-Image-Edit-2509")
@register_model("Qwen/Qwen-Image-Edit")
@register_model("Qwen-Image-Edit-2511")
@register_model("Qwen-Image-Edit-2509")
@register_model("Qwen-Image-Edit")
class xFuserQwenImageEditModel(xFuserModel):
    min_diffusers_version = "0.37.0"

    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        use_cfg_parallel=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_fp8_text_encoder=True,
        use_fp8_comms=True,
        use_parallel_vae=True,
        use_parallel_vae_encoder=True,
        enable_tiling=True,
        enable_slicing=True,
        supports_step_caching=True,
    )
    default_input_values = DefaultInputValues(
        num_inference_steps=50,
        guidance_scale=4.0,
        negative_prompt=" ",
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image-Edit",
        output_name="qwen_image_edit",
        model_output_type="image",
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        step_cache_config={
            "dbcache": DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("transformer_blocks", "Pattern_1"),),
                    enable_separate_cfg=True,
                ),
                preset=DBCachePreset(Fn_compute_blocks=6, residual_diff_threshold=0.12, scm_policy="ultra"),
            ),
        },
        fp8_text_encoder_module_list=["text_encoder.model.language_model.layers"],
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        if "2511" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2511"
            self.settings.output_name = "qwen_image_edit_2511"
        elif "2509" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-Edit-2509"
            self.settings.output_name = "qwen_image_edit_2509"

    def _load_model(self) -> DiffusionPipeline:
        from xfuser.model_executor.pipelines.pipeline_qwen_image_edit import (
            xFuserQwenImageEditPipeline,
        )
        from xfuser.model_executor.models.transformers.transformer_qwen import (
            xFuserQwenImageTransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImageTransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = xFuserQwenImageEditPipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "image": input_args["input_images"][0],
            "prompt": input_args["prompt"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": self._make_generator(input_args["seed"]),
        }
        if "height" in input_args: kwargs["height"] = input_args["height"]
        if "width" in input_args: kwargs["width"] = input_args["width"]

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)


    def _validate_args(self, input_args: dict) -> None:
        """ Validate input arguments """
        super()._validate_args(input_args)
        images = input_args.get("input_images", [])
        if len(images) != 1:
            raise ValueError("Exactly one input image is required for Qwen Image Edit model.")

@register_model("Qwen/Qwen-Image-2512")
@register_model("Qwen/Qwen-Image")
@register_model("Qwen-Image-2512")
@register_model("Qwen-Image")
class xFuserQwenImageModel(xFuserModel):
    min_diffusers_version = "0.37.0"

    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=True,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        supports_step_caching=True,
        use_fp8_text_encoder=True,
        use_fp8_comms=True,
        use_parallel_vae=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=928,
        width=1664,
        num_inference_steps=50,
        guidance_scale=0.0,
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image",
        output_name="qwen_image",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fp8_text_encoder_module_list=["text_encoder.model.language_model.layers"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
        step_cache_config={
            "dbcache": DBCacheSettings(
                adapter=CacheDitAdapterConfig(
                    blocks=(("transformer_blocks", "Pattern_1"),),
                    enable_separate_cfg=False,
                ),
                preset=DBCachePreset(Fn_compute_blocks=6, residual_diff_threshold=0.12, scm_policy="ultra"),
        )},
    )

    def _customize_settings(self, config: xFuserArgs) -> None:
        super()._customize_settings(config)
        if "2512" in config.model:
            self.settings.model_name = "Qwen/Qwen-Image-2512"
            self.settings.output_name = "qwen_image_2512"

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import QwenImagePipeline
        from xfuser.model_executor.models.transformers.transformer_qwen import (
            xFuserQwenImageTransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImageTransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        kwargs = {
            "prompt": input_args["prompt"],
            "height": input_args["height"],
            "width": input_args["width"],
            "negative_prompt": input_args["negative_prompt"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": input_args["guidance_scale"],
            "generator": self._make_generator(input_args["seed"]),
        }

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)


def _qwen_image21_output_size(images, height, width, resolution=1024):
    """Output (height, width): explicit when given, else the last condition image's aspect
    ratio at ``resolution**2`` area -- what QwenImage21Pipeline picks itself -- else square."""
    if height is not None and width is not None:
        return height, width
    if height is not None or width is not None:
        raise ValueError("Qwen-Image-2.1 needs both --height and --width, or neither.")
    if images:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        image_width, image_height = images[-1].size
        width, height, _ = calculate_dimensions(resolution * resolution, image_width / image_height)
        return height, width
    return resolution, resolution


@register_model("Qwen/Qwen-Image-2.1")
@register_model("Qwen-Image-2.1")
class xFuserQwenImage21Model(xFuserModel):
    """Qwen-Image 2.1 text-to-image and image-conditioned generation.

    ``--input_images`` are passed to the pipeline as condition images: the ``Qwen3-VL``
    encoder reads them as vision context and the VAE encodes them into prefix tokens.
    Without ``--height``/``--width`` the output follows the last image's aspect ratio.

    2.1 is a distinct architecture from the 1.x Qwen-Image models: a single-stream,
    block-causal transformer (``QwenImage21Transformer2DModel``) with interleaved
    text/image tokens and prefix KV caching, a ``Qwen3-VL`` text encoder and its own
    VAE. Those classes and the ``QwenImage21Pipeline`` ship only in a diffusers built
    from source, hence ``min_diffusers_version = DIFFUSERS_FROM_SOURCE``.

    Ulysses sequence parallelism shards the transformer's block loop over the sequence and
    gathers the whole sequence per head group for attention, so the block-causal prefill and
    the prefix KV cache work unchanged (see ``xFuserQwenImage21TransformerWrapper``). Ring is
    left off: it would split the block-causal mask across ranks.
    """

    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=True,
        ring_degree=False,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        enable_tiling=True,
        enable_slicing=True,
    )
    # No height/width: _preprocess_args_images resolves them, from the condition image when given.
    default_input_values = DefaultInputValues(
        num_inference_steps=40,
        # true_cfg_scale; 2.1 is meant to be sampled without guidance (cfg off at 1.0).
        # No default negative_prompt: with CFG off it would be ignored, and passing it
        # only makes diffusers warn that it has no effect.
        guidance_scale=1.0,
    )
    settings = ModelSettings(
        model_name="Qwen/Qwen-Image-2.1",
        output_name="qwen_image_21",
        model_output_type="image",
        fp8_gemm_module_list=["transformer.transformer_blocks"],
        fsdp_strategy={
            "transformer": {
                "wrap_attrs": ["transformer_blocks"],
            },
            "text_encoder": {
                "wrap_attrs": ["model.language_model.layers"],
            },
        },
    )

    def _load_model(self) -> DiffusionPipeline:
        from diffusers import QwenImage21Pipeline
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            xFuserQwenImage21TransformerWrapper,
        )

        transformer = self.loader.load_transformer(xFuserQwenImage21TransformerWrapper)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = QwenImage21Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _preprocess_args_images(self, input_args: dict) -> dict:
        input_args = super()._preprocess_args_images(input_args)
        explicit = input_args.get("height") is not None and input_args.get("width") is not None
        height, width = _qwen_image21_output_size(
            input_args["input_images"], input_args.get("height"), input_args.get("width")
        )
        input_args["height"], input_args["width"] = height, width
        if not explicit:
            source = "last input image's aspect ratio" if input_args["input_images"] else "model default"
            log(f"Qwen-Image-2.1: output size {width}x{height} (from {source}).")
        return input_args

    def _post_load_and_state_initialization(self, input_args: dict) -> None:
        super()._post_load_and_state_initialization(input_args)
        # After materialization: a processor set on the pre-materialized module is not the one that runs.
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            xFuserQwenImage21AttnProcessor,
        )

        blocks = self.pipe.transformer.transformer_blocks
        for block in blocks:
            block.attn.set_processor(xFuserQwenImage21AttnProcessor())
        log(f"Qwen-Image-2.1: decode attention uses {get_runtime_state().attention_backend.name}.")
        if not self.config.use_torch_compile:
            log("Qwen-Image-2.1: block-causal prefill uses per-segment SDPA.")

    def _validate_config(self, config: xFuserArgs) -> None:
        super()._validate_config(config)
        backend = _parse_attention_backend(config.attention_backend, "attention backend")
        if backend in _QWEN_IMAGE_21_UNSUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                f"Model {self.settings.model_name} does not support attention backend {backend.name}: "
                f"its decode attention is plain dense attention and cannot supply the backend's extra inputs."
            )
        ulysses_degree = config.ulysses_degree or 1
        if _QWEN_IMAGE_21_NUM_HEADS % ulysses_degree:
            raise ValueError(
                f"Model {self.settings.model_name} has {_QWEN_IMAGE_21_NUM_HEADS} attention heads; "
                f"--ulysses_degree must divide it, got {ulysses_degree}."
            )

    def _prefer_blockwise_compile(self) -> bool:
        # 2.1's forward syncs to host outside the block loop (prefix_len, token metadata, segment .tolist()).
        return True

    def _compile_model(self, input_args: dict) -> None:
        # flex_attention is only efficient compiled, so the one-call prefill is set up here, not at load.
        from xfuser.model_executor.models.transformers.transformer_qwenimage21 import (
            xFuserQwenImage21FlexAttnProcessor,
        )
        blocks = self.pipe.transformer.transformer_blocks
        try:
            for block in blocks:
                block.attn.set_processor(xFuserQwenImage21FlexAttnProcessor())
            log(
                f"Qwen-Image-2.1: block-causal prefill uses flex_attention on {len(blocks)} "
                f"blocks."
            )
        except ImportError as exc:
            # flex_attention needs torch>=2.5 with torch.nn.attention.flex_attention.
            log(
                f"Qwen-Image-2.1: flex_attention unavailable ({exc}); block-causal prefill "
                f"uses per-segment SDPA."
            )
        super()._compile_model(input_args)

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        true_cfg_scale = input_args["guidance_scale"]
        kwargs = {
            "prompt": input_args["prompt"],
            "image": input_args["input_images"] or None,
            "height": input_args["height"],
            "width": input_args["width"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": true_cfg_scale,
            "generator": self._make_generator(input_args["seed"]),
        }
        # With CFG off diffusers ignores a negative prompt and warns on every call.
        negative_prompt = input_args.get("negative_prompt")
        if true_cfg_scale > 1 and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)
