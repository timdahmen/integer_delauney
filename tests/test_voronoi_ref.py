"""Tests for the NumPy reference RegularDelaunay implementation."""
import numpy as np
import pytest
from delauney.reference.voronoi import RegularDelaunay

vd = RegularDelaunay()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute(width, height, seeds):
    return vd.compute(width, height, seeds)


def grid_n(g):
    return g[:, :, 0]


def grid_d(g):
    return g[:, :, 1]


# ---------------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------------

class TestSingleSeed:
    def test_all_cells_get_seed_id(self):
        g = compute(5, 5, [(2, 2)])
        assert np.all(grid_n(g) == 0)

    def test_distances_are_squared_l2(self):
        """The distance channel stores squared Euclidean distance, not Manhattan."""
        g = compute(5, 5, [(2, 2)])
        for y in range(5):
            for x in range(5):
                expected = (x - 2) ** 2 + (y - 2) ** 2
                assert grid_d(g)[y, x] == expected, f"cell ({x},{y})"

    def test_seed_cell_has_distance_zero(self):
        g = compute(7, 7, [(3, 4)])
        assert grid_d(g)[4, 3] == 0

    def test_single_seed_corner(self):
        g = compute(4, 4, [(0, 0)])
        assert grid_d(g)[0, 0] == 0
        assert grid_d(g)[3, 3] == 18  # squared L2 from (0,0) to (3,3): 9 + 9


# ---------------------------------------------------------------------------
# Seed ID assignment ordering
# ---------------------------------------------------------------------------

class TestSeedIdOrdering:
    def test_ascending_x_order(self):
        # Seeds at x=3 and x=1: seed at x=1 gets id 0, x=3 gets id 1
        g = compute(5, 1, [(3, 0), (1, 0)])
        assert grid_n(g)[0, 1] == 0   # seed at x=1 → id 0
        assert grid_n(g)[0, 3] == 1   # seed at x=3 → id 1

    def test_tiebreaker_ascending_y(self):
        # Two seeds at same x=2, at y=0 and y=4: y=0 gets lower id
        g = compute(5, 5, [(2, 4), (2, 0)])
        assert grid_n(g)[0, 2] == 0   # seed at (2,0) → id 0
        assert grid_n(g)[4, 2] == 1   # seed at (2,4) → id 1


# ---------------------------------------------------------------------------
# Two seeds — boundary and distances
# ---------------------------------------------------------------------------

class TestTwoSeeds:
    def test_horizontal_split(self):
        # Seeds at (0,0) and (4,0) on a 1-row grid
        # Equidistant cell at x=2 → higher id (id=1) wins
        g = compute(5, 1, [(0, 0), (4, 0)])
        n = grid_n(g)[0]
        # Left half belongs to id 0 (seed at x=0)
        assert n[0] == 0
        assert n[1] == 0
        # Right half belongs to id 1 (seed at x=4)
        assert n[3] == 1
        assert n[4] == 1
        # Tie at x=2: higher id wins
        assert n[2] == 1

    def test_distances_from_respective_seeds(self):
        """Distances are squared L2; on a single row that is dx**2."""
        g = compute(5, 1, [(0, 0), (4, 0)])
        d = grid_d(g)[0]
        assert d[0] == 0   # seed 0
        assert d[1] == 1
        assert d[2] == 4   # equidistant: dist to winning seed (id=1 at x=4) is 2**2
        assert d[3] == 1
        assert d[4] == 0   # seed 1

    def test_vertical_split(self):
        g = compute(1, 5, [(0, 0), (0, 4)])
        n = grid_n(g)[:, 0]
        assert n[0] == 0
        assert n[4] == 1
        assert n[2] == 1  # tie → higher id wins

    def test_two_seeds_same_position_raises(self):
        with pytest.raises(ValueError):
            compute(5, 5, [(2, 2), (2, 2)])


# ---------------------------------------------------------------------------
# Tie-breaking: higher seed id wins
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def test_equidistant_higher_id_wins(self):
        # 3 seeds arranged symmetrically; centre cell equidistant from 0 and 2
        # id=0 at (0,2), id=1 at (2,0), id=2 at (4,2)
        # Centre (2,2): distance to id=0 is 2, to id=2 is 2 → id=2 wins
        g = compute(5, 5, [(0, 2), (2, 0), (4, 2)])
        # Cell (2,2) is equidistant from seeds 0 (dist 2) and 2 (dist 2)
        # seed 1 at (2,0) has dist 2 too → all three equidistant → highest id=2 wins
        assert grid_n(g)[2, 2] == 2


# ---------------------------------------------------------------------------
# Completeness: all cells defined after compute
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_no_undefined_cells(self):
        g = compute(10, 10, [(1, 1), (8, 1), (1, 8), (8, 8)])
        assert np.all(grid_n(g) >= 0)

    def test_distances_non_negative(self):
        g = compute(10, 10, [(1, 1), (8, 1), (1, 8), (8, 8)])
        assert np.all(grid_d(g) >= 0)


# ---------------------------------------------------------------------------
# Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_shape(self):
        g = compute(7, 5, [(3, 2)])
        assert g.shape == (5, 7, 2)

    def test_dtype_int32(self):
        g = compute(4, 4, [(1, 1)])
        assert g.dtype == np.int32

    def test_empty_seeds_raises(self):
        with pytest.raises(ValueError):
            compute(4, 4, [])

    def test_seed_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            compute(4, 4, [(10, 10)])
