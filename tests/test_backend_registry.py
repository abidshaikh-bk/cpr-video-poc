import pytest

from cpr_video_poc.backends.registry import create_backend, registered_backends
from cpr_video_poc.backends.wan_t2v import WanT2VBackend



def test_backend_registry_contains_wan_backend():
    assert "wan_t2v_1_3b" in registered_backends()
    backend = create_backend("wan_t2v_1_3b", {"model_id": "test-model"})
    assert isinstance(backend, WanT2VBackend)


def test_create_backend_raises_for_unknown_backend():
    with pytest.raises(KeyError, match="not registered"):
        create_backend("missing-backend", {})
