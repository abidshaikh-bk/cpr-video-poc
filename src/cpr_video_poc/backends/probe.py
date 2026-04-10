from __future__ import annotations

import os
import platform
from typing import Any


def _system_ram_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * phys_pages) / (1024**3)
    except (AttributeError, OSError, ValueError):
        return None


def detect_infra_capabilities() -> dict[str, Any]:
    capabilities: dict[str, Any] = {
        "platform": platform.platform(),
        "system_ram_gb": _system_ram_gb(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "bf16_supported": False,
    }
    try:
        import torch
    except Exception:
        return capabilities

    cuda_available = bool(torch.cuda.is_available())
    capabilities["cuda_available"] = cuda_available
    if not cuda_available:
        return capabilities

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    capabilities["gpu_name"] = props.name
    capabilities["gpu_vram_gb"] = props.total_memory / (1024**3)
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    capabilities["bf16_supported"] = bool(is_bf16_supported()) if callable(is_bf16_supported) else False
    return capabilities


def evaluate_backend_support(
    backend_name: str,
    backend_config: dict[str, Any],
    capabilities: dict[str, Any],
) -> tuple[bool, str]:
    requirements = backend_config.get("requirements", {})

    min_system_ram_gb = requirements.get("min_system_ram_gb")
    if min_system_ram_gb is not None:
        system_ram_gb = capabilities.get("system_ram_gb")
        if system_ram_gb is None or system_ram_gb < float(min_system_ram_gb):
            return (
                False,
                f"{backend_name} requires at least {min_system_ram_gb} GB system RAM",
            )

    if bool(requirements.get("requires_cuda", False)) and not capabilities.get("cuda_available"):
        return False, f"{backend_name} requires CUDA"

    min_vram_gb = requirements.get("min_gpu_vram_gb")
    if min_vram_gb is not None:
        gpu_vram_gb = capabilities.get("gpu_vram_gb")
        if gpu_vram_gb is None or gpu_vram_gb < float(min_vram_gb):
            return (
                False,
                f"{backend_name} requires at least {min_vram_gb} GB GPU VRAM",
            )

    requires_bf16 = requirements.get("requires_bf16")
    if requires_bf16 and not capabilities.get("bf16_supported"):
        return False, f"{backend_name} requires BF16 support"

    return True, "supported"
