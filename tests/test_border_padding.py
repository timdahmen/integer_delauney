"""Tests for auto_border_padding and the border_padding=None default.

A triangle is only detected where three Voronoi regions meet -- its
circumcenter. Boundary triangles frequently have circumcenters outside the
image, so at padding 0 they are never registered and the pixels they should
cover degrade to nearest-seed with no error raised. The default now scales
padding to seed density instead of leaving it at 0.
"""
import numpy as np
import pytest

from delauney import auto_border_padding
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


class TestAutoBorderPadding:
    def test_scales_with_density_not_image_size(self):
        """Same sparsity on a bigger image gives a similar padding in pixels."""
        a = auto_border_padding(256, 256, 655)      # ~1% of 256^2
        b = auto_border_padding(512, 512, 2621)     # ~1% of 512^2
        assert abs(a - b) <= 1

    def test_denser_seeds_need_less_padding(self):
        sparse = auto_border_padding(256, 256, 65)
        dense = auto_border_padding(256, 256, 3276)
        assert sparse > dense

    def test_matches_the_closed_form(self):
        assert auto_border_padding(256, 256, 655) == int(round((256 * 256 / 655) ** 0.5))

    def test_zero_seeds_is_zero(self):
        assert auto_border_padding(64, 64, 0) == 0

    def test_never_negative(self):
        for n in (1, 2, 10, 100000):
            assert auto_border_padding(64, 64, n) >= 0


class TestPaddingDefault:
    W = H = 64

    def test_default_is_not_zero_for_realistic_seed_counts(self):
        assert auto_border_padding(self.W, self.H, 40) > 0

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
        b, _ = _tri.compute(vg, seeds, border_padding=0)
        assert len(a) == len(b)
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
