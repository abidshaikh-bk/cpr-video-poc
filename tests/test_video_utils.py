from pathlib import Path

import numpy as np

from cpr_video_poc.utils.video import pil_frames_to_ndarrays, save_gif


def test_pil_frames_to_ndarrays_converts_float_frames_to_uint8():
    frames = [np.ones((4, 4, 3), dtype=np.float32) * 0.5]
    converted = pil_frames_to_ndarrays(frames)

    assert converted[0].dtype == np.uint8
    assert converted[0].shape == (4, 4, 3)
    assert int(converted[0][0, 0, 0]) == 127


def test_save_gif_accepts_float_frames(tmp_path: Path):
    frames = [np.ones((8, 8, 3), dtype=np.float32) * 0.25 for _ in range(2)]
    output_path = save_gif(frames, tmp_path / "preview.gif", fps=4)

    assert output_path.exists()
