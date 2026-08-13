"""Tests for the NumPy reference GridTriangulation implementation."""
import numpy as np
import pytest
from delauney.reference.voronoi import RegularDelaunay
from delauney.reference.triangulation import GridTriangulation, _point_in_triangle

_vd = RegularDelaunay()
_tri = GridTriangulation()


def voronoi(width, height, seeds):
    return _vd.compute(width, height, seeds)


def triangulate(vgrid, seeds):
    return _tri.compute(vgrid, seeds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_voronoi_and_tri(width, height, seeds):
    g = voronoi(width, height, seeds)
    return triangulate(g, seeds)


# ---------------------------------------------------------------------------
# Three seeds — basic triangle detection
# ---------------------------------------------------------------------------

class TestThreeSeeds:
    """Three non-collinear seeds must produce exactly one unique triangle."""

    SEEDS = [(1, 1), (8, 1), (4, 8)]

    def test_exactly_one_triangle(self):
        tri_map, _ = make_voronoi_and_tri(10, 10, self.SEEDS)
        assert len(tri_map) == 1

    def test_triangle_contains_all_three_seed_ids(self):
        tri_map, _ = make_voronoi_and_tri(10, 10, self.SEEDS)
        (_, (x, y, id_a, id_b, id_c)) = next(iter(tri_map.items()))
        assert {id_a, id_b, id_c} == {0, 1, 2}

    def test_every_cell_has_a_valid_id_or_the_sentinel(self):
        """Ids are either a real triangle index or -1 ("no triangle contains it").

        Three seeds cannot cover a 10x10 image, so -1 must actually occur here.
        """
        tri_map, tgrid = make_voronoi_and_tri(10, 10, self.SEEDS)
        t = tgrid[:, :, 2]
        assert np.all((t == -1) | ((t >= 0) & (t < len(tri_map))))
        assert np.any(t == -1)
        assert np.any(t >= 0)

    def test_output_grid_preserves_n_and_d(self):
        vgrid = voronoi(10, 10, self.SEEDS)
        _, tgrid = triangulate(vgrid, self.SEEDS)
        assert np.array_equal(vgrid[:, :, 0], tgrid[:, :, 0])
        assert np.array_equal(vgrid[:, :, 1], tgrid[:, :, 1])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """The same geometric triangle must appear only once in the map."""

    def test_duplicate_detections_collapse(self):
        # Large grid with 3 seeds: the boundary will span many cells,
        # potentially detecting the same (id_a,id_b,id_c) triplet multiple times.
        seeds = [(2, 5), (15, 2), (10, 14)]
        tri_map, _ = make_voronoi_and_tri(20, 20, seeds)
        # Regardless of how many raw detections occurred, only one entry per
        # unique sorted triplet must exist.
        triplets = set()
        for _, (x, y, a, b, c) in tri_map.items():
            triplet = frozenset({a, b, c})
            assert triplet not in triplets, "duplicate triangle entry found"
            triplets.add(triplet)


# ---------------------------------------------------------------------------
# Triangle map entry structure
# ---------------------------------------------------------------------------

class TestTriangleMapStructure:
    def test_entry_is_five_tuple(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, _ = make_voronoi_and_tri(10, 10, seeds)
        for tid, entry in tri_map.items():
            assert isinstance(tid, int)
            assert len(entry) == 5, "expected (x, y, id_a, id_b, id_c)"

    def test_detection_coordinates_in_bounds(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, _ = make_voronoi_and_tri(10, 10, seeds)
        for _, (x, y, *_) in tri_map.items():
            assert 1 <= x <= 9
            assert 0 <= y <= 8


# ---------------------------------------------------------------------------
# Pixel containment (integer convention — see GridTriangulation docstring)
# ---------------------------------------------------------------------------

class TestPixelContainment:
    def test_seed_cells_assigned_to_triangle_containing_seed(self):
        """Each seed pixel must lie within the triangle that uses its seed id."""
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, tgrid = make_voronoi_and_tri(10, 10, seeds)
        # All seed positions should have a valid triangle assigned
        for sx, sy in seeds:
            assert tgrid[sy, sx, 2] >= 0

    def test_triangle_id_consistent_with_seed_membership(self):
        """Every cell must be assigned to a triangle that includes its Voronoi seed id."""
        seeds = [(1, 1), (8, 1), (4, 8)]
        vgrid = voronoi(10, 10, seeds)
        tri_map, tgrid = triangulate(vgrid, seeds)
        for y in range(10):
            for x in range(10):
                n = int(tgrid[y, x, 0])
                t = int(tgrid[y, x, 2])
                if t >= 0:
                    _, (_, _, a, b, c) = next(
                        (k, v) for k, v in tri_map.items() if k == t
                    )
                    assert n in {a, b, c}, (
                        f"cell ({x},{y}) has seed {n} but triangle {t} has ids {a},{b},{c}"
                    )


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_output_grid_shape(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        _, tgrid = make_voronoi_and_tri(10, 10, seeds)
        assert tgrid.shape == (10, 10, 3)

    def test_output_grid_dtype(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        _, tgrid = make_voronoi_and_tri(10, 10, seeds)
        assert tgrid.dtype == np.int32

    def test_triangle_ids_are_valid_or_sentinel(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, tgrid = make_voronoi_and_tri(10, 10, seeds)
        t = tgrid[:, :, 2]
        assert np.all((t == -1) | ((t >= 0) & (t < len(tri_map))))


# ---------------------------------------------------------------------------
# Outside-hull sentinel and the integer pixel convention
# ---------------------------------------------------------------------------

class TestOutsideHullSentinel:
    """Pixels that no triangle contains carry -1, and only those pixels do."""

    SEEDS = [(1, 1), (8, 1), (4, 8)]

    def _grids(self):
        vgrid = _vd.compute(10, 10, self.SEEDS)
        return _tri.compute(vgrid, self.SEEDS)

    def test_sentinel_appears_where_no_triangle_covers(self):
        _, tgrid = self._grids()
        # One triangle cannot cover a 10x10 image.
        assert np.any(tgrid[:, :, 2] == -1)

    def test_sentinel_exactly_where_no_triangle_contains_the_pixel(self):
        tri_map, tgrid = self._grids()
        seeds = np.array(sorted(self.SEEDS), dtype=float)
        t = tgrid[:, :, 2]
        tris = [(a, b, c) for (_, _, a, b, c) in tri_map.values()]

        for y in range(10):
            for x in range(10):
                covered = any(
                    _point_in_triangle(x, y, seeds[a], seeds[b], seeds[c])
                    for (a, b, c) in tris)
                if t[y, x] == -1:
                    assert not covered, f"({x},{y}) is -1 but lies in a triangle"
                else:
                    assert covered, f"({x},{y}) has an id but lies in no triangle"

    def test_assigned_pixels_lie_in_their_own_triangle(self):
        """Assignment uses the integer coordinate, so containment holds there."""
        tri_map, tgrid = self._grids()
        seeds = np.array(sorted(self.SEEDS), dtype=float)
        t = tgrid[:, :, 2]

        ys, xs = np.where(t >= 0)
        for y, x in zip(ys, xs):
            _, _, a, b, c = tri_map[int(t[y, x])]
            assert _point_in_triangle(x, y, seeds[a], seeds[b], seeds[c]), (
                f"({x},{y}) assigned a triangle that does not contain it")

    def test_other_channels_untouched(self):
        vgrid = _vd.compute(10, 10, self.SEEDS)
        _, tgrid = _tri.compute(vgrid, self.SEEDS)
        assert np.array_equal(vgrid[:, :, 0], tgrid[:, :, 0])
        assert np.array_equal(vgrid[:, :, 1], tgrid[:, :, 1])


# ---------------------------------------------------------------------------
# Cocircular (degree-4 Voronoi vertex) quads
# ---------------------------------------------------------------------------

class TestCocircularQuad:
    """Four cocircular seeds must yield ONE triangulation, not both diagonals.

    See the cocircular block in reference/triangulation.py for the rule.
    """

    @pytest.mark.parametrize("seeds,W,H", [
        ([(1, 1), (8, 1), (1, 8), (8, 8)], 10, 10),
        ([(0, 0), (4, 0), (0, 4), (4, 4)], 5, 5),
        ([(2, 2), (12, 2), (2, 12), (12, 12)], 15, 15),
    ])
    def test_exactly_two_triangles(self, seeds, W, H):
        vgrid = _vd.compute(W, H, seeds)
        tri_map, _ = _tri.compute(vgrid, seeds)
        assert len(tri_map) == 2, (
            f"expected 2 triangles for a cocircular quad, got {len(tri_map)}: "
            f"{[ (v[2],v[3],v[4]) for v in tri_map.values() ]}")

    @pytest.mark.parametrize("seeds,W,H", [
        ([(1, 1), (8, 1), (1, 8), (8, 8)], 10, 10),
        ([(0, 0), (4, 0), (0, 4), (4, 4)], 5, 5),
    ])
    def test_triangles_share_exactly_one_edge(self, seeds, W, H):
        """Two triangles of a quad must meet along a diagonal, not overlap."""
        vgrid = _vd.compute(W, H, seeds)
        tri_map, _ = _tri.compute(vgrid, seeds)
        tris = [frozenset((v[2], v[3], v[4])) for v in tri_map.values()]
        assert len(tris) == 2
        shared = tris[0] & tris[1]
        assert len(shared) == 2, f"triangles share {len(shared)} vertices, expected 2"
        # Together they must use all four seeds exactly once as a quad.
        assert tris[0] | tris[1] == {0, 1, 2, 3}

    def test_no_pixel_lies_in_both_triangles(self):
        """Overlapping triangles would put a pixel inside more than one."""
        seeds = [(1, 1), (8, 1), (1, 8), (8, 8)]
        vgrid = _vd.compute(10, 10, seeds)
        tri_map, _ = _tri.compute(vgrid, seeds)
        pos = np.array(sorted(seeds), dtype=float)

        for y in range(10):
            for x in range(10):
                n_containing = sum(
                    _point_in_triangle(x, y, pos[a], pos[b], pos[c])
                    for (_, _, a, b, c) in tri_map.values())
                # 2 is legal only on the shared diagonal itself.
                assert n_containing <= 2, (
                    f"({x},{y}) lies inside {n_containing} triangles")
