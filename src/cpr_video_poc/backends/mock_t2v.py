from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cpr_video_poc.backends.base import BaseBackend, GenerationRequest, GenerationResult
from cpr_video_poc.backends.registry import register_backend


class MockT2VBackend(BaseBackend):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model_id = str(config.get("model_id", "mock/mock-t2v"))

    def load(self) -> None:
        self._loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded:
            self.load()

        seed_material = f"{request.seed}:{request.prompt}".encode("utf-8")
        seed = int.from_bytes(sha256(seed_material).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        frames: list[Image.Image] = []

        bg_color = tuple(int(value) for value in rng.integers(20, 180, size=3))
        accent_color = tuple(int(value) for value in rng.integers(120, 255, size=3))
        text_lines = [
            "Mock T2V Backend",
            request.extra.get("raw_prompt", request.prompt)[:72],
            f"seed={request.seed}",
        ]

        for frame_index in range(request.num_frames):
            image = Image.new("RGB", (request.width, request.height), color=bg_color)
            draw = ImageDraw.Draw(image)
            motion_offset = int((frame_index / max(request.num_frames - 1, 1)) * request.width * 0.4)

            draw.rounded_rectangle(
                [(40 + motion_offset, request.height // 2 - 80), (260 + motion_offset, request.height // 2 + 80)],
                radius=24,
                fill=accent_color,
            )
            draw.ellipse(
                [(request.width - 220 - motion_offset, 90), (request.width - 60 - motion_offset, 250)],
                fill=(255, 235, 180),
            )

            for idx, line in enumerate(text_lines):
                draw.text((40, 30 + idx * 24), line, fill=(255, 255, 255))

            draw.text(
                (40, request.height - 40),
                f"frame {frame_index + 1}/{request.num_frames}",
                fill=(230, 230, 230),
            )
            frames.append(image)

        metadata = {
            "backend": "mock_t2v",
            "model_id": self.model_id,
            "resolution": {"height": request.height, "width": request.width},
            "frames": request.num_frames,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "fps": request.fps,
            "mock": True,
        }
        return GenerationResult(
            frames=frames,
            seed=request.seed,
            backend_name="mock_t2v",
            model_id=self.model_id,
            metadata=metadata,
        )

    def unload(self) -> None:
        self._loaded = False


register_backend("mock_t2v", MockT2VBackend)
