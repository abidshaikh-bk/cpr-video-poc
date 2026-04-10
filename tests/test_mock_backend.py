from cpr_video_poc.backends.mock_t2v import MockT2VBackend
from cpr_video_poc.backends.base import GenerationRequest


def test_mock_backend_generates_requested_number_of_frames():
    backend = MockT2VBackend({"model_id": "mock/mock-t2v"})
    request = GenerationRequest(
        prompt="test prompt",
        negative_prompt="",
        num_frames=5,
        height=120,
        width=160,
        steps=4,
        guidance_scale=1.0,
        seed=7,
        fps=4,
        extra={"raw_prompt": "Generate a mock video"},
    )

    result = backend.generate(request)

    assert result.backend_name == "mock_t2v"
    assert len(result.frames) == 5
