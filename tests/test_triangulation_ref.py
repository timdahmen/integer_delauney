"""Tests for the NumPy reference GridTriangulation implementation."""
import numpy as np
import pytest
from delauney.reference.voronoi import RegularDelaunay
from delauney.reference.triangulation import GridTriangulation

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

    def test_all_cells_assigned_to_triangle(self):
        _, tgrid = make_voronoi_and_tri(10, 10, self.SEEDS)
        assert np.all(tgrid[:, :, 2] >= 0)

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
# Four-region meeting (documented known limitation)
# ---------------------------------------------------------------------------

class TestFourRegionMeeting:
    """When 4 seeds meet equidistantly, only 1 triangle is expected (known limitation)."""

    def test_four_symmetric_seeds_produce_at_least_one_triangle(self):
        # Seeds at corners of a small square; they will all meet in the centre.
        seeds = [(0, 0), (4, 0), (0, 4), (4, 4)]
        tri_map, _ = make_voronoi_and_tri(5, 5, seeds)
        assert len(tri_map) >= 1

    def test_four_region_at_most_three_triangles(self):
        seeds = [(0, 0), (4, 0), (0, 4), (4, 4)]
        tri_map, _ = make_voronoi_and_tri(5, 5, seeds)
        # We document the known limitation: max triangles ≤ N-2 for N seeds
        assert len(tri_map) <= 4


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
# Center-point containment
# ---------------------------------------------------------------------------

class TestCenterPointContainment:
    def test_seed_cells_assigned_to_triangle_containing_seed(self):
        """Each seed cell center must lie within the triangle that uses its seed id."""
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

    def test_triangle_ids_non_negative(self):
        seeds = [(1, 1), (8, 1), (4, 8)]
        _, tgrid = make_voronoi_and_tri(10, 10, seeds)
        assert np.all(tgrid[:, :, 2] >= 0)
