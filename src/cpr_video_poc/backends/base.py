from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str
    num_frames: int
    height: int
    width: int
    steps: int
    guidance_scale: float
    seed: int
    fps: int = 8
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    frames: list[Any]
    seed: int
    backend_name: str
    model_id: str
    metadata: dict[str, Any]
    output_path: Path | None = None
    preview_gif_path: Path | None = None


class BaseBackend(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError
