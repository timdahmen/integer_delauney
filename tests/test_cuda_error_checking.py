"""A CUDA allocation failure surfaces as a Python exception, not a crash.

The happy-path suites elsewhere never touch cuda_check.cuh's CUDA_CHECK path,
since every call there succeeds. This deliberately forces a real
cudaErrorMemoryAllocation through the public API (a canvas no real GPU has
enough VRAM for) and checks the failure is reported, not swallowed -- and that
it does not leave the process, or a subsequent normal Delaunay, broken.
"""
import pytest


def _cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="_delauney_cuda extension not built or no CUDA device",
)

if _cuda_available():
    from delauney import _delauney_cuda as _cu

# (W + 2P) * (H + 2P) * 2 * 4 bytes for d_grid_ alone is ~320 GB at P=0 --
# comfortably beyond any real GPU's VRAM, and independent of host RAM (the
# constructor's failing allocation is device-side; no host buffer this size
# is ever touched).
HUGE_SIDE = 200_000
MAX_SEEDS = 1_000


class TestConstructorAllocationFailure:
    def test_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            _cu.Delaunay(HUGE_SIDE, HUGE_SIDE, MAX_SEEDS, 0)

    def test_error_message_names_the_failure(self):
        with pytest.raises(RuntimeError) as excinfo:
            _cu.Delaunay(HUGE_SIDE, HUGE_SIDE, MAX_SEEDS, 0)
        message = str(excinfo.value)
        assert "cudaMalloc" in message
        assert "delaunay.cu" in message

    def test_process_and_subsequent_delaunay_still_work(self):
        # The failed construction above must not leak the buffers it did
        # allocate before the one that failed, nor leave the CUDA context
        # broken for anything constructed afterward.
        with pytest.raises(RuntimeError):
            _cu.Delaunay(HUGE_SIDE, HUGE_SIDE, MAX_SEEDS, 0)

        inc = _cu.Delaunay(64, 64, 256, 0)
        tri_map, tgrid = inc.insert([(10, 10), (50, 10), (30, 50)])
        assert len(tri_map) > 0
        assert tgrid.shape == (64, 64, 3)
