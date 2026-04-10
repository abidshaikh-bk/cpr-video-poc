from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cpr_video_poc.backends.probe import detect_infra_capabilities, evaluate_backend_support
from cpr_video_poc.settings import Settings, load_backend_config


@dataclass(slots=True)
class BackendSelection:
    requested_backend: str
    selected_backend: str
    backend_config: dict[str, Any]
    backend_config_path: Path
    capabilities: dict[str, Any]
    reason: str
    fallback_used: bool


def _candidate_backends(settings: Settings) -> list[str]:
    generation = settings.generation
    requested = str(generation.get("backend", "wan_t2v_1_3b"))
    candidates = [requested]

    fallback_backend = generation.get("fallback_backend")
    if fallback_backend:
        candidates.append(str(fallback_backend))

    extra_candidates = generation.get("backend_candidates", [])
    if isinstance(extra_candidates, list):
        candidates.extend(str(name) for name in extra_candidates)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def resolve_backend_selection(settings: Settings) -> BackendSelection:
    capabilities = detect_infra_capabilities()
    candidates = _candidate_backends(settings)
    requested_backend = candidates[0]

    for index, backend_name in enumerate(candidates):
        if index == 0:
            backend_config = settings.backend
            backend_config_path = settings.backend_config_path
        else:
            backend_config, backend_config_path = load_backend_config(backend_name)
        supported, reason = evaluate_backend_support(backend_name, backend_config, capabilities)
        if supported:
            return BackendSelection(
                requested_backend=requested_backend,
                selected_backend=backend_name,
                backend_config=backend_config,
                backend_config_path=backend_config_path,
                capabilities=capabilities,
                reason=reason if index == 0 else f"fallback from {requested_backend}: {reason}",
                fallback_used=index != 0,
            )

    last_backend = candidates[-1]
    backend_config, backend_config_path = (
        (settings.backend, settings.backend_config_path)
        if last_backend == requested_backend
        else load_backend_config(last_backend)
    )
    return BackendSelection(
        requested_backend=requested_backend,
        selected_backend=last_backend,
        backend_config=backend_config,
        backend_config_path=backend_config_path,
        capabilities=capabilities,
        reason=f"No backend met infra requirements; defaulting to {last_backend}",
        fallback_used=last_backend != requested_backend,
    )
