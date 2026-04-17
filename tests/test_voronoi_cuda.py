"""CUDA tests for RegularDelaunay — skipped automatically when no GPU is available."""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="_delauney_cuda extension not built or no CUDA device"
)


@pytest.fixture(scope="module")
def cuda_vd():
    from delauney._delauney_cuda import RegularDelaunay
    return RegularDelaunay()


@pytest.fixture(scope="module")
def ref_vd():
    from delauney.reference.voronoi import RegularDelaunay
    return RegularDelaunay()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cuda(cuda_vd, width, height, seeds):
    return cuda_vd.compute(width, height, seeds)


def compute_ref(ref_vd, width, height, seeds):
    return ref_vd.compute(width, height, seeds)


# ---------------------------------------------------------------------------
# CUDA vs reference comparison
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCudaMatchesReference:
    """CUDA output must be element-wise identical to the NumPy reference."""

    @pytest.mark.parametrize("seeds,W,H", [
        ([(2, 2)], 5, 5),
        ([(0, 0), (4, 0)], 5, 1),
        ([(0, 0), (0, 4)], 1, 5),
        ([(1, 1), (8, 1), (4, 8)], 10, 10),
        ([(1, 1), (8, 1), (1, 8), (8, 8)], 10, 10),
        ([(0, 0), (4, 0), (0, 4), (4, 4)], 5, 5),
    ])
    def test_matches_reference(self, cuda_vd, ref_vd, seeds, W, H):
        cuda_grid = compute_cuda(cuda_vd, W, H, seeds)
        ref_grid  = compute_ref(ref_vd, W, H, seeds)
        assert np.array_equal(cuda_grid, ref_grid), (
            f"CUDA and reference differ for seeds={seeds}, grid={W}x{H}\n"
            f"CUDA seed_ids:\n{cuda_grid[:,:,0]}\n"
            f"ref  seed_ids:\n{ref_grid[:,:,0]}"
        )

    def test_large_grid(self, cuda_vd, ref_vd):
        rng = np.random.default_rng(42)
        W, H = 128, 128
        coords = rng.integers(0, [W, H], size=(20, 2))
        seeds = list(map(tuple, coords.tolist()))
        # deduplicate
        seeds = list(dict.fromkeys(seeds))
        cuda_grid = compute_cuda(cuda_vd, W, H, seeds)
        ref_grid  = compute_ref(ref_vd, W, H, seeds)
        assert np.array_equal(cuda_grid, ref_grid)


# ---------------------------------------------------------------------------
# CUDA output contract (mirrors reference tests)
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCudaOutputContract:
    def test_shape(self, cuda_vd):
        g = compute_cuda(cuda_vd, 7, 5, [(3, 2)])
        assert g.shape == (5, 7, 2)

    def test_dtype(self, cuda_vd):
        g = compute_cuda(cuda_vd, 4, 4, [(1, 1)])
        assert g.dtype == np.int32

    def test_no_undefined(self, cuda_vd):
        g = compute_cuda(cuda_vd, 10, 10, [(1, 1), (8, 8)])
        assert np.all(g[:, :, 0] >= 0)

    def test_distances_non_negative(self, cuda_vd):
        g = compute_cuda(cuda_vd, 10, 10, [(1, 1), (8, 8)])
        assert np.all(g[:, :, 1] >= 0)

    def test_empty_seeds_raises(self, cuda_vd):
        with pytest.raises(Exception):
            cuda_vd.compute(4, 4, [])

    def test_seed_out_of_bounds_raises(self, cuda_vd):
        with pytest.raises(Exception):
            cuda_vd.compute(4, 4, [(10, 10)])
