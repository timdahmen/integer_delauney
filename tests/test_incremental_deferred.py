"""Tests for the deferred insert / finalise path of Delaunay.

Correctness strategy
--------------------
The deferred path is not a new algorithm; it is the same one with the pixel
assignment postponed. So the bar is equivalence, and it is checked in two
directions:

  * against insert(), which is defined as insert_deferred() + finalise() and so
    must agree pixel for pixel, and
  * against the stateless batch pipeline, which is what test_incremental_cuda
    already validates insert() itself against.

The interesting cases are the ones the new code paths only see under load: a
large scattered batch, which is what drives the reassign mask to saturate and
the cost switch to choose a full assignment, and several deferred rounds in a
row, which is the pattern the mode exists for.
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

W, H = 64, 64
MAX_SEEDS = 4096

# Padding pinned to 0 throughout, matching batch_triangulate: the incremental
# constructor now defaults to auto padding like the batch API, so leaving it
# unset would compare a padded triangulation against an unpadded one and call
# the difference a bug. Padding equivalence has its own tests.
PADDING = 0


def tri_set(tri_map):
    """Canonical triplet set, so insertion-order ids do not matter."""
    if isinstance(tri_map, np.ndarray):
        return frozenset(tuple(sorted(row)) for row in tri_map.tolist())
    return frozenset(tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values())


def batch_triangulate(seeds):
    """Stateless batch pipeline. border_padding pinned to 0, as incremental
    has no padding support and would otherwise be a different feature set."""
    vgrid = _cu.Voronoi().compute(W, H, seeds)
    tri_map, tgrid = _cu.GridTriangulation().compute(vgrid, seeds, 0)
    return tri_set(tri_map), tgrid


def scattered_seeds(n, rng, w=W, h=H):
    """Distinct integer seeds spread over the whole grid.

    Spread rather than clustered on purpose: a scattered batch is what makes
    the dilated reassign region cover the image, which is the case the cost
    switch exists to handle.
    """
    pts = set()
    while len(pts) < n:
        pts.add((int(rng.integers(0, w)), int(rng.integers(0, h))))
    return [list(p) for p in sorted(pts)]


# ---------------------------------------------------------------------------
# Equivalence with insert()
# ---------------------------------------------------------------------------

class TestDeferredMatchesImmediate:

    def _run_immediate(self, batches):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map = tgrid = None
        for b in batches:
            tri_map, tgrid = inc.insert(b)
        return tri_set(tri_map), tgrid

    def _run_deferred(self, batches):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        for b in batches:
            inc.insert_deferred(b)
        tri_map, tgrid = inc.finalise()
        return tri_set(tri_map), tgrid

    def test_single_batch(self):
        rng = np.random.default_rng(0)
        seeds = scattered_seeds(40, rng)
        imm_set, imm_grid = self._run_immediate([seeds])
        def_set, def_grid = self._run_deferred([seeds])
        assert def_set == imm_set
        np.testing.assert_array_equal(def_grid, imm_grid)

    def test_several_batches(self):
        rng = np.random.default_rng(1)
        seeds = scattered_seeds(90, rng)
        batches = [seeds[:30], seeds[30:60], seeds[60:]]
        imm_set, imm_grid = self._run_immediate(batches)
        def_set, def_grid = self._run_deferred(batches)
        assert def_set == imm_set
        np.testing.assert_array_equal(def_grid, imm_grid)

    def test_five_rounds_matches_immediate(self):
        """The refinement access pattern: repeated scattered insertions."""
        rng = np.random.default_rng(2)
        seeds = scattered_seeds(250, rng)
        batches = [seeds[i::5] for i in range(5)]
        imm_set, imm_grid = self._run_immediate(batches)
        def_set, def_grid = self._run_deferred(batches)
        assert def_set == imm_set
        np.testing.assert_array_equal(def_grid, imm_grid)


# ---------------------------------------------------------------------------
# Equivalence with the batch pipeline
# ---------------------------------------------------------------------------

class TestDeferredMatchesBatch:

    def test_triangles_match_batch(self):
        rng = np.random.default_rng(3)
        seeds = scattered_seeds(120, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        for b in (seeds[:50], seeds[50:90], seeds[90:]):
            inc.insert_deferred(b)
        tri_map, _ = inc.finalise()
        batch_set, _ = batch_triangulate(sorted(seeds))
        assert tri_set(tri_map) == batch_set

    def test_dense_batch_forces_full_assign(self):
        """Enough scattered seeds that the dirty region saturates.

        This is the branch where the cost switch abandons the mask; the result
        must be identical either way, which is the only thing worth asserting
        about a performance heuristic.
        """
        rng = np.random.default_rng(4)
        seeds = scattered_seeds(600, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(seeds[:100])
        inc.insert_deferred(seeds[100:])
        tri_map, _ = inc.finalise()
        batch_set, _ = batch_triangulate(sorted(seeds))
        assert tri_set(tri_map) == batch_set


# ---------------------------------------------------------------------------
# API behaviour
# ---------------------------------------------------------------------------

class TestDeferredApi:

    def test_has_pending_tracks_state(self):
        rng = np.random.default_rng(5)
        seeds = scattered_seeds(30, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        assert not inc.has_pending
        inc.insert_deferred(seeds)
        assert inc.has_pending
        inc.finalise()
        assert not inc.has_pending

    def test_finalise_with_nothing_pending_is_idempotent(self):
        rng = np.random.default_rng(6)
        seeds = scattered_seeds(40, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(seeds)
        map1, grid1 = inc.finalise()
        map2, grid2 = inc.finalise()
        assert tri_set(map1) == tri_set(map2)
        np.testing.assert_array_equal(grid1, grid2)

    def test_get_triangles_matches_finalise(self):
        """Topology read between rounds must agree with what finalise reports.

        get_triangles uses insertion-order ids and finalise uses sorted ones, so
        they are compared through sorted_rank rather than directly -- which also
        checks that the translation the header promises actually holds.
        """
        rng = np.random.default_rng(7)
        seeds = scattered_seeds(80, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(seeds[:40])
        inc.insert_deferred(seeds[40:])

        internal = inc.get_triangles()
        rank = inc.sorted_rank
        translated = frozenset(
            tuple(sorted(rank[list(row)].tolist())) for row in internal
        )

        tri_map, _ = inc.finalise()
        assert translated == tri_set(tri_map)

    def test_get_triangles_needs_no_finalise(self):
        rng = np.random.default_rng(8)
        seeds = scattered_seeds(60, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(seeds)
        tris = inc.get_triangles()
        assert tris.ndim == 2 and tris.shape[1] == 3
        assert len(tris) > 0
        assert inc.has_pending           # unchanged by reading topology

    def test_as_arrays_matches_dict(self):
        rng = np.random.default_rng(9)
        seeds = scattered_seeds(70, rng)

        inc_a = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc_a.insert_deferred(seeds)
        map_dict, grid_dict = inc_a.finalise(as_arrays=False)

        inc_b = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc_b.insert_deferred(seeds)
        map_arr, grid_arr = inc_b.finalise(as_arrays=True)

        assert isinstance(map_arr, np.ndarray)
        assert tri_set(map_arr) == tri_set(map_dict)
        np.testing.assert_array_equal(grid_arr, grid_dict)

    def test_numpy_seed_array_accepted(self):
        """The fast path in _parse_seeds must agree with the generic one."""
        rng = np.random.default_rng(10)
        seeds = scattered_seeds(50, rng)

        inc_a = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc_a.insert_deferred(np.asarray(seeds, dtype=np.int32))
        map_a, grid_a = inc_a.finalise()

        inc_b = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc_b.insert_deferred(seeds)
        map_b, grid_b = inc_b.finalise()

        assert tri_set(map_a) == tri_set(map_b)
        np.testing.assert_array_equal(grid_a, grid_b)

    def test_duplicate_within_batch_still_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(ValueError):
            inc.insert_deferred([[1, 1], [2, 2], [1, 1]])

    def test_duplicate_across_deferred_batches_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred([[1, 1], [2, 2], [3, 5]])
        with pytest.raises(ValueError):
            inc.insert_deferred([[4, 4], [2, 2]])

    def test_timings_reported(self):
        rng = np.random.default_rng(11)
        seeds = scattered_seeds(40, rng)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        t_ins = inc.insert_deferred_timed(seeds)
        # Assignment has not run yet; that is the whole point of the mode.
        assert t_ins["assign_ms"] == 0.0
        _, _, t_fin = inc.finalise_timed()
        assert t_fin["assign_ms"] > 0.0
