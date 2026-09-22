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
)
from xfuser import xFuserArgs
from xfuser.core.utils.runner_utils import log
from xfuser.model_executor.models.runner_models.loading.contracts import (
    LoadSupport,
    STANDARD_LOAD_ROUTES,
)

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


@register_model("Qwen/Qwen-Image-2.1")
@register_model("Qwen-Image-2.1")
class xFuserQwenImage21Model(xFuserModel):
    """Qwen-Image 2.1 text-to-image.

    2.1 is a distinct architecture from the 1.x Qwen-Image models: a single-stream,
    block-causal transformer (``QwenImage21Transformer2DModel``) with interleaved
    text/image tokens and prefix KV caching, a ``Qwen3-VL`` text encoder and its own
    VAE. Those classes and the ``QwenImage21Pipeline`` ship only in a diffusers built
    from source, hence ``min_diffusers_version = DIFFUSERS_FROM_SOURCE``.

    Sequence parallelism (Ulysses/Ring) is intentionally left off: 2.1's block-causal
    attention with KV caching does not fit the sequence-chunk USP wrapper the 1.x
    models use, and needs its own verified implementation. The supported scaling paths
    for now are single-GPU acceleration (FP8 GEMMs, VAE tiling/slicing, CPU offload),
    data parallelism and FSDP parameter sharding.
    """

    min_diffusers_version = DIFFUSERS_FROM_SOURCE

    load_support = LoadSupport(
        meta_transformers=('transformer',),
        meta_text_encoders=('text_encoder',),
        replicated_meta=True,
        routes=STANDARD_LOAD_ROUTES,
    )
    capabilities = ModelCapabilities(
        ulysses_degree=False,
        ring_degree=False,
        fully_shard_degree=True,
        use_fp8_gemms=True,
        use_parallel_vae=False,
        enable_tiling=True,
        enable_slicing=True,
    )
    default_input_values = DefaultInputValues(
        height=1024,
        width=1024,
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
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21Transformer2DModel,
        )

        transformer = self.loader.load_transformer(QwenImage21Transformer2DModel)
        te_kwargs, te_quant = self.loader.plan_text_encoders()
        pipe = QwenImage21Pipeline.from_pretrained(
            pretrained_model_name_or_path=self.settings.model_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            quantization_config=te_quant,
            **te_kwargs,
        )
        return pipe

    def _prefer_blockwise_compile(self) -> bool:
        # 2.1's forward does host-side scalar/index work outside the block loop --
        # prefix_len = int((~target_token_mask).sum()), build_token_metadata, and the
        # prefix-segment .tolist() -- which breaks a whole-model graph at each sync.
        # Compiling per block keeps that setup eager and traces only the blocks.
        return True

    def _compile_model(self, input_args: dict) -> None:
        # Swap to the flex attention processor here, after loading and materialization and
        # right before compile -- the latest safe point. Setting it in _load_model did not
        # take effect at runtime, because the transformer is materialized afterwards and the
        # processor on the pre-materialized module is not the one that ends up running.
        #
        # Flex expresses 2.1's block-causal prefill as one flex_attention call, which the
        # per-block compile below then traces; it is efficient only once compiled, so this
        # only runs on the compile path (_compile_model is only called under torch.compile).
        # Decode is unmasked full attention and stays on SDPA regardless.
        from diffusers.models.transformers.transformer_qwenimage21 import (
            QwenImage21FlexAttnProcessor,
        )
        blocks = self.pipe.transformer.transformer_blocks
        try:
            for block in blocks:
                block.attn.set_processor(QwenImage21FlexAttnProcessor())
            log(
                f"Qwen-Image-2.1: enabled QwenImage21FlexAttnProcessor on {len(blocks)} "
                f"blocks before compile."
            )
        except ImportError as exc:
            # flex_attention needs torch>=2.5 with torch.nn.attention.flex_attention.
            log(
                f"Qwen-Image-2.1: flex_attention unavailable ({exc}); compiling with the "
                f"default SDPA attention processor."
            )
        super()._compile_model(input_args)

    def _run_pipe(self, input_args: dict) -> DiffusionOutput:
        true_cfg_scale = input_args["guidance_scale"]
        kwargs = {
            "prompt": input_args["prompt"],
            "height": input_args["height"],
            "width": input_args["width"],
            "num_inference_steps": input_args["num_inference_steps"],
            "true_cfg_scale": true_cfg_scale,
            "generator": self._make_generator(input_args["seed"]),
        }
        # 2.1 samples without guidance by default (true_cfg_scale == 1.0). A negative
        # prompt only takes effect when CFG is on; passing it otherwise is ignored and
        # makes diffusers warn once per call. So only forward it when CFG is enabled.
        negative_prompt = input_args.get("negative_prompt")
        if true_cfg_scale > 1 and negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        output = self.pipe(**kwargs)
        return DiffusionOutput(images=output.images, pipe_args=input_args)
