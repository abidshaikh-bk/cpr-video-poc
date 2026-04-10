from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio
import numpy as np


def _frame_to_uint8_array(frame: object) -> np.ndarray:
    arr = np.asarray(frame)

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.max(arr)) if arr.size else 0.0
        min_value = float(np.min(arr)) if arr.size else 0.0

        if 0.0 <= min_value and max_value <= 1.0:
            arr = arr * 255.0

        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)

    raise TypeError(f"Unsupported frame dtype for video serialization: {arr.dtype!r}")


def pil_frames_to_ndarrays(frames: Iterable[object]) -> list[np.ndarray]:
    return [_frame_to_uint8_array(frame) for frame in frames]


def save_video_mp4(frames: Iterable[object], output_path: str | Path, fps: int = 8) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrs = pil_frames_to_ndarrays(frames)
    with imageio.get_writer(output, fps=fps, codec="libx264") as writer:
        for frame in arrs:
            writer.append_data(frame)
    return output


def save_gif(frames: Iterable[object], output_path: str | Path, fps: int = 8) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrs = pil_frames_to_ndarrays(frames)
    duration_seconds = 1 / max(fps, 1)
    imageio.mimsave(output, arrs, format="GIF", duration=duration_seconds, loop=0)
    return output
