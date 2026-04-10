from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cpr_video_poc.utils.video import save_gif, save_video_mp4
from cpr_video_poc.utils.yaml_io import dump_yaml


def create_run_dir(root: str | Path) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    path = root_path / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_text(path: str | Path, text: str) -> Path:
    p = Path(path)
    p.write_text(text, encoding="utf-8")
    return p


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    p = Path(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def save_run_artifacts(
    run_root: str | Path,
    request_data: dict[str, Any],
    parsed_prompt: dict[str, Any],
    final_prompt: str,
    negative_prompt: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    frames: list,
    fps: int,
    output_basename: str = "output",
    save_preview_gif: bool = True,
) -> dict[str, Path]:
    run_dir = create_run_dir(run_root)
    request_path = write_json(run_dir / "request.json", request_data)
    parsed_path = write_json(run_dir / "parsed_prompt.json", parsed_prompt)
    prompt_path = write_text(run_dir / "final_prompt.txt", final_prompt)
    negative_prompt_path = write_text(run_dir / "negative_prompt.txt", negative_prompt)
    config_path = run_dir / "config.yaml"
    dump_yaml(config_path, config)
    metadata_path = write_json(run_dir / "metadata.json", metadata)
    video_path = save_video_mp4(frames, run_dir / f"{output_basename}.mp4", fps=fps)
    result = {
        "run_dir": run_dir,
        "request": request_path,
        "parsed_prompt": parsed_path,
        "final_prompt": prompt_path,
        "negative_prompt": negative_prompt_path,
        "config": config_path,
        "metadata": metadata_path,
        "video": video_path,
    }
    if save_preview_gif:
        gif_path = save_gif(frames, run_dir / "preview.gif", fps=fps)
        result["preview_gif"] = gif_path
    return result
