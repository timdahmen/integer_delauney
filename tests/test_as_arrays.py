"""as_arrays returns the triangle vertex indices without building a dict."""
import numpy as np
import pytest

from delauney.reference.triangulation import GridTriangulation
from delauney.reference.voronoi import Voronoi

_vd = Voronoi()
_tri = GridTriangulation()

W = H = 48


def _seeds(n, seed):
    rng = np.random.default_rng(seed)
    pts = set()
    while len(pts) < n:
        k = n - len(pts)
        for x, y in zip(rng.integers(1, W - 1, k), rng.integers(1, H - 1, k)):
            pts.add((int(x), int(y)))
    return sorted(pts)


def _cuda_available():
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


class TestReferenceAsArrays:
    def test_shape_and_dtype(self):
        seeds = _seeds(30, 1)
        vg = _vd.compute(W, H, seeds)
        verts, tgrid = _tri.compute(vg, seeds, as_arrays=True)
        tri_map, _ = _tri.compute(vg, seeds)
        assert verts.shape == (len(tri_map), 3)
        assert verts.dtype == np.int32

    def test_rows_match_the_dict_by_triangle_id(self):
        """Row tid must be the same triangle the grid's tid channel refers to."""
        seeds = _seeds(30, 2)
        vg = _vd.compute(W, H, seeds)
        verts, _ = _tri.compute(vg, seeds, as_arrays=True)
        tri_map, _ = _tri.compute(vg, seeds)
        for tid, (_x, _y, a, b, c) in tri_map.items():
            assert tuple(verts[tid]) == (a, b, c)

    def test_grid_is_identical_either_way(self):
        seeds = _seeds(30, 3)
        vg = _vd.compute(W, H, seeds)
        _, g_arr = _tri.compute(vg, seeds, as_arrays=True)
        _, g_dict = _tri.compute(vg, seeds)
        np.testing.assert_array_equal(g_arr, g_dict)

    def test_default_still_returns_a_dict(self):
        seeds = _seeds(20, 4)
        vg = _vd.compute(W, H, seeds)
        tri_map, _ = _tri.compute(vg, seeds)
        assert isinstance(tri_map, dict)


@pytest.mark.skipif(not _cuda_available(), reason="_delauney_cuda not built")
class TestCudaAsArrays:
    def test_matches_its_own_dict(self):
        from delauney import _delauney_cuda as dc

        seeds = _seeds(30, 5)
        vg = np.asarray(dc.Voronoi().compute(W, H, seeds))
        verts, g_arr = dc.GridTriangulation().compute(vg, seeds, as_arrays=True)
        tri_map, g_dict = dc.GridTriangulation().compute(vg, seeds)

        verts = np.asarray(verts)
        assert verts.shape == (len(tri_map), 3)
        for tid, (_x, _y, a, b, c) in tri_map.items():
            assert tuple(verts[tid]) == (a, b, c)
        np.testing.assert_array_equal(np.asarray(g_arr), np.asarray(g_dict))

    def test_default_still_returns_a_dict(self):
        from delauney import _delauney_cuda as dc

        seeds = _seeds(20, 6)
        vg = np.asarray(dc.Voronoi().compute(W, H, seeds))
        tri_map, _ = dc.GridTriangulation().compute(vg, seeds)
        assert isinstance(tri_map, dict)
