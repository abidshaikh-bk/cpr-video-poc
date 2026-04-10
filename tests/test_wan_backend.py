from cpr_video_poc.backends.wan_t2v import _resolve_torch_dtype


class _FakeCuda:
    def __init__(self, *, available: bool, bf16_supported: bool):
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self) -> bool:
        return self._bf16_supported


class _FakeTorch:
    float16 = "float16"
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self, *, cuda_available: bool, bf16_supported: bool):
        self.cuda = _FakeCuda(
            available=cuda_available,
            bf16_supported=bf16_supported,
        )


def test_resolve_torch_dtype_falls_back_from_bfloat16_when_cuda_device_lacks_support():
    fake_torch = _FakeTorch(cuda_available=True, bf16_supported=False)

    resolved = _resolve_torch_dtype(fake_torch, "bfloat16")

    assert resolved == "float16"


def test_resolve_torch_dtype_falls_back_to_float32_on_cpu_for_half_precision():
    fake_torch = _FakeTorch(cuda_available=False, bf16_supported=False)

    resolved = _resolve_torch_dtype(fake_torch, "float16")

    assert resolved == "float32"
