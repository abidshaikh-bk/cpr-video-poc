from pathlib import Path

import pytest

from cpr_video_poc.paths import REPO_ROOT
from cpr_video_poc.settings import load_settings



def test_load_settings_reads_default_configs():
    settings = load_settings()
    assert settings.project["project_name"] == "cpr-video-poc"
    assert settings.generation["backend"] == "wan_t2v_1_3b"
    assert settings.backend["model_id"] == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert settings.generation["num_frames"] == 21
    assert settings.artifact_root == (REPO_ROOT / "runs").resolve()


def test_artifact_root_is_resolved_relative_to_project_config(tmp_path: Path):
    config_dir = tmp_path / "configs"
    backend_dir = config_dir / "backends"
    backend_dir.mkdir(parents=True)

    project_path = config_dir / "project.yaml"
    project_path.write_text(
        """
project_name: test-project
artifact_root: ./custom-runs
generation:
  backend: wan_t2v_1_3b
  num_frames: 49
  height: 480
  width: 832
  steps: 30
  guidance_scale: 5.0
storage: {}
""".strip(),
        encoding="utf-8",
    )
    (backend_dir / "wan_t2v_1_3b.yaml").write_text(
        "model_id: Wan-AI/Wan2.1-T2V-1.3B-Diffusers\nload: {}\n",
        encoding="utf-8",
    )

    settings = load_settings(project_path)
    assert settings.artifact_root == (config_dir / "custom-runs").resolve()


def test_load_settings_rejects_missing_generation_fields(tmp_path: Path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    project_path = config_dir / "project.yaml"
    project_path.write_text(
        """
project_name: broken-project
generation:
  backend: wan_t2v_1_3b
storage: {}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required generation settings"):
        load_settings(project_path)
