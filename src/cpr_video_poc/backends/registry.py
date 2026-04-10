from __future__ import annotations

from typing import Type

from cpr_video_poc.backends.base import BaseBackend


_BACKENDS: dict[str, Type[BaseBackend]] = {}



def register_backend(name: str, backend_cls: Type[BaseBackend]) -> None:
    _BACKENDS[name] = backend_cls



def get_backend_class(name: str) -> Type[BaseBackend]:
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise KeyError(f"Backend '{name}' is not registered. Available: {available}")
    return _BACKENDS[name]



def create_backend(name: str, config: dict) -> BaseBackend:
    return get_backend_class(name)(config)



def registered_backends() -> list[str]:
    return sorted(_BACKENDS)
