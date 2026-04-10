from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import logging
from pathlib import Path
from typing import Any

from cpr_video_poc.backends import WanT2VBackend  # noqa: F401
from cpr_video_poc.backends.base import GenerationRequest
from cpr_video_poc.backends.registry import create_backend
from cpr_video_poc.backends.selector import resolve_backend_selection
from cpr_video_poc.logging_utils import configure_logging
from cpr_video_poc.pipeline.prompt_builder import build_prompts
from cpr_video_poc.pipeline.prompt_parser import parse_prompt
from cpr_video_poc.pipeline.save_run import save_run_artifacts
from cpr_video_poc.pipeline.sync import sync_run_if_enabled
from cpr_video_poc.settings import Settings, load_settings
from cpr_video_poc.utils.seed import resolve_seed, seed_everything

logger = logging.getLogger(__name__)


def _package_versions() -> dict[str, str]:
    package_names = ["accelerate", "diffusers", "torch", "transformers"]
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            continue
    return versions


def _build_metadata(
    *,
    prompt: str,
    parsed: dict[str, Any],
    prompt_bundle: dict[str, str],
    request: GenerationRequest,
    settings: Settings,
    backend_metadata: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        **backend_metadata,
        "project_name": settings.project.get("project_name"),
        "request": {
            "raw_prompt": prompt,
            "parsed_prompt": parsed,
            "final_prompt": prompt_bundle["prompt"],
            "negative_prompt": prompt_bundle["negative_prompt"],
        },
        "generation": {
            "backend": request.extra["backend_name"],
            "seed": request.seed,
            "resolution": {"height": request.height, "width": request.width},
            "frames": request.num_frames,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "fps": request.fps,
        },
        "config_snapshot": settings.as_dict(),
        "software": _package_versions(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def run_generation_with_settings(
    settings: Settings,
    *,
    prompt: str,
    seed: int | None = None,
    artifact_root_override: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_logging()
    generation_cfg = settings.generation
    storage_cfg = settings.storage

    parsed = parse_prompt(prompt)
    prompt_bundle = build_prompts(parsed)

    final_seed = resolve_seed(seed)
    seed_everything(final_seed)

    selection = resolve_backend_selection(settings)
    backend_name = selection.selected_backend
    backend = create_backend(backend_name, selection.backend_config)
    request = GenerationRequest(
        prompt=prompt_bundle["prompt"],
        negative_prompt=prompt_bundle["negative_prompt"],
        num_frames=int(generation_cfg["num_frames"]),
        height=int(generation_cfg["height"]),
        width=int(generation_cfg["width"]),
        steps=int(generation_cfg["steps"]),
        guidance_scale=float(generation_cfg["guidance_scale"]),
        seed=final_seed,
        fps=int(generation_cfg.get("fps", 8)),
        extra={
            "raw_prompt": prompt,
            "backend_name": backend_name,
            "output_type": selection.backend_config.get("load", {}).get("output_type", "np"),
        },
    )

    try:
        result = backend.generate(request)
        metadata = _build_metadata(
            prompt=prompt,
            parsed=parsed,
            prompt_bundle=prompt_bundle,
            request=request,
            settings=settings,
            backend_metadata=result.metadata,
            extra_metadata={
                **(extra_metadata or {}),
                "backend_selection": {
                    "requested_backend": selection.requested_backend,
                    "selected_backend": selection.selected_backend,
                    "fallback_used": selection.fallback_used,
                    "reason": selection.reason,
                    "backend_config_path": str(selection.backend_config_path),
                    "capabilities": selection.capabilities,
                },
            },
        )
        saved = save_run_artifacts(
            run_root=Path(artifact_root_override).resolve()
            if artifact_root_override is not None
            else settings.artifact_root,
            request_data={"prompt": prompt, "seed": final_seed},
            parsed_prompt=parsed,
            final_prompt=prompt_bundle["prompt"],
            negative_prompt=prompt_bundle["negative_prompt"],
            config=settings.as_dict(),
            metadata=metadata,
            frames=result.frames,
            fps=request.fps,
            output_basename=str(generation_cfg.get("output_basename", "output")),
            save_preview_gif=bool(storage_cfg.get("save_preview_gif", True)),
        )
        synced_to_hf = sync_run_if_enabled(saved["run_dir"], storage_cfg)
        logger.info("Run saved to %s", saved["run_dir"])
        return {
            "run_dir": str(saved["run_dir"]),
            "video_path": str(saved["video"]),
            "preview_gif_path": str(saved.get("preview_gif", "")),
            "seed": final_seed,
            "requested_backend": selection.requested_backend,
            "selected_backend": selection.selected_backend,
            "backend_fallback_used": selection.fallback_used,
            "parsed_prompt": parsed,
            "final_prompt": prompt_bundle["prompt"],
            "negative_prompt": prompt_bundle["negative_prompt"],
            "synced_to_hf": synced_to_hf,
        }
    finally:
        backend.unload()


def run_generation(
    prompt: str,
    seed: int | None = None,
    project_config_path: str | None = None,
    backend_config_path: str | None = None,
) -> dict[str, Any]:
    settings = load_settings(project_config_path, backend_config_path)
    return run_generation_with_settings(settings, prompt=prompt, seed=seed)
