from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpr_video_poc.paths import CONFIGS_DIR, REPO_ROOT
from cpr_video_poc.utils.yaml_io import load_yaml


REQUIRED_GENERATION_FIELDS = {
    "backend",
    "num_frames",
    "height",
    "width",
    "steps",
    "guidance_scale",
}


@dataclass(slots=True)
class Settings:
    project: dict[str, Any]
    backend: dict[str, Any]
    project_config_path: Path
    backend_config_path: Path

    @property
    def generation(self) -> dict[str, Any]:
        return self.project.get("generation", {})

    @property
    def storage(self) -> dict[str, Any]:
        return self.project.get("storage", {})

    @property
    def orchestration(self) -> dict[str, Any]:
        return self.project.get("orchestration", {})

    @property
    def artifact_root(self) -> Path:
        root = self.project.get("artifact_root", "./runs")
        return _resolve_path(root, base_dir=_artifact_base_dir(self.project_config_path))

    @property
    def evaluation_prompts(self) -> list[str]:
        prompts = self.project.get("evaluation_prompts", [])
        return [str(prompt) for prompt in prompts]

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "backend": self.backend,
            "project_config_path": str(self.project_config_path),
            "backend_config_path": str(self.backend_config_path),
        }


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _artifact_base_dir(project_config_path: Path) -> Path:
    try:
        project_config_path.resolve().relative_to(CONFIGS_DIR.resolve())
    except ValueError:
        return project_config_path.parent
    return REPO_ROOT


def _expect_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {context} to be a mapping, got {type(value)!r}")
    return value


def _validate_project_config(project: dict[str, Any]) -> dict[str, Any]:
    generation = _expect_mapping(project.get("generation", {}), "project.generation")
    missing_fields = sorted(REQUIRED_GENERATION_FIELDS - generation.keys())
    if missing_fields:
        raise ValueError(
            "Missing required generation settings: " + ", ".join(missing_fields)
        )

    generation["backend"] = str(generation["backend"])
    generation["num_frames"] = int(generation["num_frames"])
    generation["height"] = int(generation["height"])
    generation["width"] = int(generation["width"])
    generation["steps"] = int(generation["steps"])
    generation["guidance_scale"] = float(generation["guidance_scale"])
    generation["fps"] = int(generation.get("fps", 8))
    generation["output_basename"] = str(generation.get("output_basename", "output"))

    storage = _expect_mapping(project.get("storage", {}), "project.storage")
    orchestration = _expect_mapping(
        project.get("orchestration", {}),
        "project.orchestration",
    )
    project["generation"] = generation
    project["storage"] = storage
    project["orchestration"] = orchestration
    project["project_name"] = str(project.get("project_name", "cpr-video-poc"))
    return project


def _validate_backend_config(backend: dict[str, Any]) -> dict[str, Any]:
    if "model_id" not in backend:
        raise ValueError("Backend config must define 'model_id'")
    backend["model_id"] = str(backend["model_id"])
    backend["load"] = _expect_mapping(backend.get("load", {}), "backend.load")
    return backend


def load_settings(
    project_config_path: str | Path | None = None,
    backend_config_path: str | Path | None = None,
) -> Settings:
    project_path = Path(project_config_path or CONFIGS_DIR / "project.yaml").resolve()
    project = _validate_project_config(load_yaml(project_path))
    backend_name = project.get("generation", {}).get("backend", "wan_t2v_1_3b")
    backend_path = Path(
        backend_config_path or CONFIGS_DIR / "backends" / f"{backend_name}.yaml"
    ).resolve()
    backend = _validate_backend_config(load_yaml(backend_path))
    return Settings(
        project=project,
        backend=backend,
        project_config_path=project_path,
        backend_config_path=backend_path,
    )
