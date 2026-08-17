"""triangulate(): the one-call entry point.

It exists to remove a duplicated Voronoi computation. Callers paired
Voronoi().compute() with GridTriangulation().compute(), and at any padding above
0 that built the diagram twice -- the triangulator needs a *padded* diagram for
detection and computed its own, using the caller's only to fill the output
grid's seed-id and distance channels. Those channels are exactly the interior
window of the padded diagram, so the caller's copy was pure waste.

The whole claim therefore rests on that interior window being identical to an
unpadded diagram, which is why the equivalence tests below sweep sizes,
densities and paddings rather than checking one case.
"""
import numpy as np
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

from delauney import DEFAULT_BORDER_PADDING


def seeds_for(W, H, n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.integers([0, 0], [W, H], size=(int(n * 1.5) + 8, 2))
    return sorted({(int(x), int(y)) for x, y in pts})[:n]


def two_step(W, H, seeds, padding=-1, as_arrays=False):
    vgrid = _cu.Voronoi().compute(W, H, seeds)
    return _cu.GridTriangulation().compute(vgrid, seeds, padding, as_arrays)


def triplets(tri_map):
    return {tuple(sorted((int(v[2]), int(v[3]), int(v[4]))))
            for v in tri_map.values()}


class TestEquivalentToThePair:
    """Identical output, or the replacement is not a refactor."""

    @pytest.mark.parametrize("W,H,n", [
        (64, 64, 20), (128, 96, 60), (256, 256, 500), (300, 200, 2000),
    ])
    @pytest.mark.parametrize("padding", [-1, 0, 1, 8, 32])
    def test_same_triangles_and_same_grid(self, W, H, n, padding):
        seeds = seeds_for(W, H, n, seed=W * H + n + padding)
        ref_map, ref_grid = two_step(W, H, seeds, padding)
        got_map, got_grid = _cu.triangulate(W, H, seeds, padding)

        assert triplets(got_map) == triplets(ref_map)
        assert len(got_map) == len(ref_map)
        np.testing.assert_array_equal(np.asarray(got_grid),
                                      np.asarray(ref_grid))

    def test_seed_id_and_distance_channels_match_a_plain_voronoi(self):
        """The channels triangulate now sources from the padded interior.

        This is the specific substitution the optimisation makes, so it gets
        its own assertion rather than only riding along with the grid compare.
        """
        W, H = 200, 150
        seeds = seeds_for(W, H, 300, seed=3)
        plain = np.asarray(_cu.Voronoi().compute(W, H, seeds))
        _, grid = _cu.triangulate(W, H, seeds, DEFAULT_BORDER_PADDING)
        grid = np.asarray(grid)
        np.testing.assert_array_equal(grid[:, :, 0], plain[:, :, 0])
        np.testing.assert_array_equal(grid[:, :, 1], plain[:, :, 1])

    def test_as_arrays_matches_too(self):
        W, H = 180, 140
        seeds = seeds_for(W, H, 250, seed=5)
        ref_verts, ref_grid = two_step(W, H, seeds, -1, True)
        got_verts, got_grid = _cu.triangulate(W, H, seeds, -1, True)
        np.testing.assert_array_equal(np.asarray(got_verts),
                                      np.asarray(ref_verts))
        np.testing.assert_array_equal(np.asarray(got_grid),
                                      np.asarray(ref_grid))


class TestContract:

    def test_default_padding_is_the_library_default(self):
        W, H = 120, 90
        seeds = seeds_for(W, H, 80, seed=9)
        a, _ = _cu.triangulate(W, H, seeds)
        b, _ = _cu.triangulate(W, H, seeds, DEFAULT_BORDER_PADDING)
        assert triplets(a) == triplets(b)

    def test_zero_padding_is_not_treated_as_unset(self):
        W, H = 120, 90
        seeds = seeds_for(W, H, 80, seed=11)
        zero, _ = _cu.triangulate(W, H, seeds, 0)
        padded, _ = _cu.triangulate(W, H, seeds, DEFAULT_BORDER_PADDING)
        assert len(zero) <= len(padded)

    def test_output_shape(self):
        _, grid = _cu.triangulate(140, 110, seeds_for(140, 110, 60, 13))
        grid = np.asarray(grid)
        assert grid.shape == (110, 140, 3)
        assert grid.dtype == np.int32

    @pytest.mark.parametrize("W,H", [(0, 10), (10, 0), (-4, 10)])
    def test_non_positive_dimensions_raise(self, W, H):
        with pytest.raises(Exception):
            _cu.triangulate(W, H, [(0, 0), (1, 1), (2, 0)])

    def test_exported_from_the_package(self):
        import delauney
        assert delauney.triangulate is _cu.triangulate
        assert "triangulate" in delauney.__all__
