"""Tests for DEFAULT_BORDER_PADDING and the border_padding=None default.

The default used to be a per-call density estimate, sqrt(area / n_seeds). It was
removed because it estimates the wrong quantity: circumcentre excursion depends
on triangle *shape*, and grows without limit as three points approach
collinearity, at any seed count. On a real frame it was out by 3.5x at the tail.

What replaces it is a constant plus a contract: a caller that samples the convex
hull boundary at spacing s is guaranteed no triangle is missed provided
border_padding >= s^2/8. See BORDER_PADDING_BOUND.md, and max_boundary_spacing()
for the inverse.

These tests therefore check that the default is honoured consistently across the
two paths and that 0 stays distinguishable from "unset" -- not that it takes any
particular numeric value beyond the one both paths must agree on.
"""
import numpy as np
import pytest

from delauney import DEFAULT_BORDER_PADDING, max_boundary_spacing
from delauney.reference.triangulation import GridTriangulation
from delauney.reference.voronoi import RegularDelaunay

_vd = RegularDelaunay()
_tri = GridTriangulation()


def _seeds(W, H, n, seed):
    rng = np.random.default_rng(seed)
    pts = set()
    while len(pts) < n:
        k = n - len(pts)
        for x, y in zip(rng.integers(1, W - 1, k), rng.integers(1, H - 1, k)):
            pts.add((int(x), int(y)))
    return sorted(pts)


class TestMaxBoundarySpacing:
    """The inverse of the bound, which is the number a caller actually needs."""

    def test_inverts_the_bound(self):
        for p in (1, 2, 8, 16, 32, 128):
            s = max_boundary_spacing(p)
            assert s * s / 8 <= p, "spacing must satisfy s^2/8 <= P"
            assert (s + 1) ** 2 / 8 > p, "and must be the largest such spacing"

    def test_default_admits_a_useful_spacing(self):
        # 16 px of padding buys 11 px boundary spacing; below about 4 the
        # boundary sampling cost stops being negligible.
        assert max_boundary_spacing(DEFAULT_BORDER_PADDING) >= 8

    def test_zero_padding_admits_no_guarantee(self):
        assert max_boundary_spacing(0) == 0
        assert max_boundary_spacing(-1) == 0


class TestPaddingDefault:
    W = H = 64

    def test_default_is_a_constant_not_a_function_of_seed_count(self):
        """The property the removed estimate did not have."""
        sparse = _seeds(self.W, self.H, 20, seed=1)
        dense = _seeds(self.W, self.H, 400, seed=2)
        for seeds in (sparse, dense):
            vg = _vd.compute(self.W, self.H, seeds)
            a, _ = _tri.compute(vg, seeds)
            b, _ = _tri.compute(vg, seeds, border_padding=DEFAULT_BORDER_PADDING)
            assert len(a) == len(b)

    def test_default_finds_at_least_as_many_triangles_as_no_padding(self):
        seeds = _seeds(self.W, self.H, 40, seed=5)
        vg = _vd.compute(self.W, self.H, seeds)
        unpadded, _ = _tri.compute(vg, seeds, border_padding=0)
        defaulted, _ = _tri.compute(vg, seeds)
        assert len(defaulted) >= len(unpadded)

    def test_default_leaves_no_more_pixels_unconfined(self):
        seeds = _seeds(self.W, self.H, 40, seed=7)
        vg = _vd.compute(self.W, self.H, seeds)
        _, tg0 = _tri.compute(vg, seeds, border_padding=0)
        _, tgd = _tri.compute(vg, seeds)
        assert int((tgd[:, :, 2] < 0).sum()) <= int((tg0[:, :, 2] < 0).sum())

    def test_explicit_zero_still_disables_padding(self):
        """0 must remain distinguishable from None, not be treated as unset."""
        seeds = _seeds(self.W, self.H, 40, seed=9)
        vg = _vd.compute(self.W, self.H, seeds)
        a, _ = _tri.compute(vg, seeds, border_padding=0)
        with_default, _ = _tri.compute(vg, seeds)
        assert len(with_default) >= len(a)

    def test_negative_padding_is_clamped(self):
        seeds = _seeds(self.W, self.H, 20, seed=11)
        vg = _vd.compute(self.W, self.H, seeds)
        a, _ = _tri.compute(vg, seeds, border_padding=-5)
        b, _ = _tri.compute(vg, seeds, border_padding=0)
        assert len(a) == len(b)

    def test_output_shape_unaffected_by_padding(self):
        seeds = _seeds(self.W, self.H, 30, seed=13)
        vg = _vd.compute(self.W, self.H, seeds)
        for p in (None, 0, 12):
            _, tg = _tri.compute(vg, seeds, border_padding=p)
            assert tg.shape == (self.H, self.W, 3)


def _cuda_available():
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="_delauney_cuda not built")
class TestCudaDefaultMatchesReference:
    """The CUDA path must default to the same padding as the reference.

    The reference spells "choose for me" as border_padding=None; the CUDA
    binding cannot take None through an int argument, so it uses a negative
    sentinel. Both must resolve to DEFAULT_BORDER_PADDING, or the two paths
    would silently disagree on how much of the image gets triangulated. That is
    now a constant shared between Python and C++ rather than a formula each has
    to reimplement identically -- which is one fewer way for them to drift.
    """
    W = H = 64

    def test_cuda_default_equals_explicit_constant(self):
        from delauney import _delauney_cuda as dc

        seeds = _seeds(self.W, self.H, 40, seed=17)
        vg = np.asarray(dc.RegularDelaunay().compute(self.W, self.H, seeds))

        default_map, default_grid = dc.GridTriangulation().compute(vg, seeds)
        explicit_map, explicit_grid = dc.GridTriangulation().compute(
            vg, seeds, DEFAULT_BORDER_PADDING)

        assert len(default_map) == len(explicit_map)
        np.testing.assert_array_equal(np.asarray(default_grid),
                                      np.asarray(explicit_grid))

    def test_cuda_and_reference_agree_on_the_default(self):
        """Both spellings of "unset" must land on the same canvas."""
        from delauney import _delauney_cuda as dc

        seeds = _seeds(self.W, self.H, 40, seed=18)
        vg = np.asarray(dc.RegularDelaunay().compute(self.W, self.H, seeds))
        cuda_map, _ = dc.GridTriangulation().compute(vg, seeds)
        cuda_explicit, _ = dc.GridTriangulation().compute(
            vg, seeds, DEFAULT_BORDER_PADDING)
        assert len(cuda_map) == len(cuda_explicit)

    def test_cuda_default_finds_at_least_as_many_as_zero(self):
        from delauney import _delauney_cuda as dc

        seeds = _seeds(self.W, self.H, 40, seed=19)
        vg = np.asarray(dc.RegularDelaunay().compute(self.W, self.H, seeds))
        default_map, _ = dc.GridTriangulation().compute(vg, seeds)
        zero_map, _ = dc.GridTriangulation().compute(vg, seeds, 0)
        assert len(default_map) >= len(zero_map)

    def test_explicit_zero_is_not_treated_as_unset(self):
        """0 must mean "no padding", not "please choose for me"."""
        from delauney import _delauney_cuda as dc

        seeds = _seeds(self.W, self.H, 25, seed=23)
        vg = np.asarray(dc.RegularDelaunay().compute(self.W, self.H, seeds))
        zero_map, _ = dc.GridTriangulation().compute(vg, seeds, 0)
        padded_map, _ = dc.GridTriangulation().compute(
            vg, seeds, DEFAULT_BORDER_PADDING)
        assert len(zero_map) <= len(padded_map)

    def test_incremental_default_matches_the_constant(self):
        from delauney import _delauney_cuda as dc

        inc = dc.IncrementalDelaunay(self.W, self.H, 512)
        assert inc.border_padding == DEFAULT_BORDER_PADDING
        assert dc.IncrementalDelaunay(self.W, self.H, 512, 5).border_padding == 5
        assert dc.IncrementalDelaunay(self.W, self.H, 512, 0).border_padding == 0
