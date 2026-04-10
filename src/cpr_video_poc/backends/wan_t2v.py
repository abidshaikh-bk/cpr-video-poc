from __future__ import annotations

import logging
from typing import Any

from cpr_video_poc.backends.base import BaseBackend, GenerationRequest, GenerationResult
from cpr_video_poc.backends.registry import register_backend
from cpr_video_poc.utils.seed import build_torch_generator

logger = logging.getLogger(__name__)


def _cuda_bf16_supported(torch_module: Any) -> bool:
    if not torch_module.cuda.is_available():
        return False
    is_bf16_supported = getattr(torch_module.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        return bool(is_bf16_supported())
    return False


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if not hasattr(torch_module, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    if dtype_name == "bfloat16" and torch_module.cuda.is_available() and not _cuda_bf16_supported(torch_module):
        logger.warning(
            "Requested bfloat16, but this CUDA device does not report BF16 support. Falling back to float16."
        )
        return torch_module.float16
    dtype = getattr(torch_module, dtype_name)
    if not torch_module.cuda.is_available() and dtype_name in {"float16", "bfloat16"}:
        return torch_module.float32
    return dtype


class WanT2VBackend(BaseBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.pipe = None
        self.model_id = config.get("model_id", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        load_cfg = self.config.get("load", {})
        vae_dtype = _resolve_torch_dtype(torch, str(load_cfg.get("vae_dtype", "float32")))
        pipe_dtype = _resolve_torch_dtype(
            torch,
            str(load_cfg.get("pipeline_dtype", "float16")),
        )
        revision = self.config.get("model_revision")
        variant = self.config.get("variant")
        low_cpu_mem_usage = bool(load_cfg.get("low_cpu_mem_usage", True))

        logger.info("Loading Wan backend: %s", self.model_id)
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=vae_dtype,
            revision=revision,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.pipe = WanPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=pipe_dtype,
            use_safetensors=bool(load_cfg.get("use_safetensors", True)),
            revision=revision,
            variant=variant,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        if bool(load_cfg.get("enable_vae_slicing", True)):
            self.pipe.vae.enable_slicing()
        if bool(load_cfg.get("enable_vae_tiling", True)):
            self.pipe.vae.enable_tiling()
        if bool(load_cfg.get("enable_attention_slicing", False)):
            self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(
            disable=bool(load_cfg.get("disable_progress_bar", False))
        )

        if bool(load_cfg.get("enable_cpu_offload", True)) and torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            device = load_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
            self.pipe.to(device)

        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded or self.pipe is None:
            self.load()

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = build_torch_generator(request.seed, device=device)

        logger.info(
            "Generating video with Wan | frames=%s size=%sx%s steps=%s guidance=%s seed=%s",
            request.num_frames,
            request.width,
            request.height,
            request.steps,
            request.guidance_scale,
            request.seed,
        )
        output = self.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.steps,
            generator=generator,
            output_type=request.extra.get("output_type", "np"),
        )
        frames = output.frames[0]
        metadata = {
            "backend": "wan_t2v_1_3b",
            "model_id": self.model_id,
            "model_revision": self.config.get("model_revision", "main"),
            "resolution": {"height": request.height, "width": request.width},
            "frames": request.num_frames,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "fps": request.fps,
            "dtype": str(self.pipe.transformer.dtype).replace("torch.", "")
            if getattr(self.pipe, "transformer", None) is not None
            else self.config.get("load", {}).get("pipeline_dtype", "float16"),
        }
        return GenerationResult(
            frames=frames,
            seed=request.seed,
            backend_name="wan_t2v_1_3b",
            model_id=self.model_id,
            metadata=metadata,
        )

    def unload(self) -> None:
        if self.pipe is None:
            return
        try:
            import gc
            import torch

            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self._loaded = False


register_backend("wan_t2v_1_3b", WanT2VBackend)
