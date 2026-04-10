from pathlib import Path

import pytest

from cpr_video_poc.backends.selector import resolve_backend_selection
from cpr_video_poc.settings import load_settings


@pytest.fixture
def settings():
    return load_settings()


def test_selector_falls_back_to_mock_on_low_ram(monkeypatch, settings):
    monkeypatch.setattr(
        "cpr_video_poc.backends.selector.detect_infra_capabilities",
        lambda: {
            "platform": "test",
            "system_ram_gb": 8.0,
            "cuda_available": False,
            "gpu_name": None,
            "gpu_vram_gb": None,
            "bf16_supported": False,
        },
    )

    selection = resolve_backend_selection(settings)

    assert selection.requested_backend == "wan_t2v_1_3b"
    assert selection.selected_backend == "mock_t2v"
    assert selection.fallback_used is True


def test_selector_keeps_requested_backend_when_requirements_are_met(monkeypatch, settings):
    monkeypatch.setattr(
        "cpr_video_poc.backends.selector.detect_infra_capabilities",
        lambda: {
            "platform": "test",
            "system_ram_gb": 64.0,
            "cuda_available": True,
            "gpu_name": "Big GPU",
            "gpu_vram_gb": 24.0,
            "bf16_supported": True,
        },
    )

    selection = resolve_backend_selection(settings)

    assert selection.selected_backend == "wan_t2v_1_3b"
    assert selection.fallback_used is False
