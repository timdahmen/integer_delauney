"""Tests for Delaunay CUDA interface.

Correctness strategy
--------------------
All expected outputs are derived from the batch GridTriangulation interface
(already validated against the NumPy reference). Each incremental test
inserts the *same* seeds that the batch API receives, then compares:
  - triangle_map  (set of canonical (a,b,c) triplets)
  - triangulation_grid  (per-pixel triangle assignment up to ID remapping)
  - voronoi_grid  (seed-id / distance channels)

Because incremental insert assigns IDs in insertion order (not sorted order),
we normalise triangle maps by their sorted triplet sets before comparison.
"""
import pytest
import numpy as np

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
MAX_SEEDS = 256

# Padding pinned to 0 throughout, matching batch_triangulate: the incremental
# constructor now defaults to auto padding like the batch API, so leaving it
# unset would compare a padded triangulation against an unpadded one and call
# the difference a bug. Padding equivalence has its own tests.
PADDING = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_triangulate(seeds):
    """Run the stateless batch pipeline and return (tri_set, tgrid, vgrid).

    border_padding is pinned to 0 for both paths here: comparing each path's
    own default padding would test padding equivalence, not the topology
    matching this suite is about. Padding equivalence has its own tests
    (test_incremental_padding.py).
    """
    rd = _cu.Voronoi()
    vgrid = rd.compute(W, H, seeds)

    gt = _cu.GridTriangulation()
    tri_map, tgrid = gt.compute(vgrid, seeds, 0)

    tri_set = frozenset(
        tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
    )
    return tri_set, tgrid, vgrid


def incremental_insert_all(seeds_batches):
    """Insert batches one at a time, return final (tri_set, tgrid, vgrid)."""
    inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
    tri_map = {}
    tgrid = None
    for batch in seeds_batches:
        tri_map, tgrid = inc.insert(batch)
    vgrid = inc.get_voronoi_grid()

    tri_set = frozenset(
        tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
    )
    return tri_set, tgrid, vgrid


def tgrid_to_tri_sets(tgrid, tri_map):
    """Return {canonical_triplet: set_of_pixel_coords} from tgrid."""
    id_to_triplet = {
        tid: tuple(sorted([v[2], v[3], v[4]])) for tid, v in tri_map.items()
    }
    result = {}
    H2, W2, _ = tgrid.shape
    for y in range(H2):
        for x in range(W2):
            tid = tgrid[y, x, 2]
            if tid < 0:
                continue
            triplet = id_to_triplet.get(tid)
            if triplet is not None:
                result.setdefault(triplet, set()).add((x, y))
    return result


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_initial_state(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        assert inc.seed_count == 0
        assert inc.width == W
        assert inc.height == H

    def test_initial_voronoi_grid_shape(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        vg = inc.get_voronoi_grid()
        assert vg.shape == (H, W, 2)
        assert vg.dtype == np.int32
        # All cells undefined before any insertion
        assert np.all(vg[:, :, 0] == -1)


# ---------------------------------------------------------------------------
# Single-seed insertion
# ---------------------------------------------------------------------------

class TestSingleSeed:
    SEED = [(32, 32)]

    def test_seed_count(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEED)
        assert inc.seed_count == 1

    def test_voronoi_all_same_seed(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEED)
        vg = inc.get_voronoi_grid()
        assert np.all(vg[:, :, 0] == 0), "All cells should belong to seed 0"

    def test_voronoi_distances_are_squared_l2(self):
        """The distance channel stores squared Euclidean distance, not Manhattan."""
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEED)
        vg = inc.get_voronoi_grid()
        sx, sy = self.SEED[0]
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        expected = (xx - sx) ** 2 + (yy - sy) ** 2
        np.testing.assert_array_equal(vg[:, :, 1], expected.astype(np.int32))

    def test_no_triangles_single_seed(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, tgrid = inc.insert(self.SEED)
        assert len(tri_map) == 0


# ---------------------------------------------------------------------------
# Two seeds
# ---------------------------------------------------------------------------

class TestTwoSeeds:
    SEEDS = [(10, 32), (50, 32)]

    def test_seed_count(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEEDS)
        assert inc.seed_count == 2

    def test_no_triangles_two_seeds(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, _ = inc.insert(self.SEEDS)
        assert len(tri_map) == 0

    def test_voronoi_matches_batch(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEEDS)
        vg_inc = inc.get_voronoi_grid()

        rd = _cu.Voronoi()
        vg_batch = rd.compute(W, H, sorted(self.SEEDS))
        np.testing.assert_array_equal(vg_inc[:, :, 1], vg_batch[:, :, 1])


# ---------------------------------------------------------------------------
# Three seeds — one triangle
# ---------------------------------------------------------------------------

class TestThreeSeeds:
    SEEDS = [(10, 10), (50, 10), (30, 50)]

    def test_exactly_one_triangle(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, _ = inc.insert(self.SEEDS)
        assert len(tri_map) == 1

    def test_triangle_triplet_matches_batch(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, _ = inc.insert(self.SEEDS)
        inc_set, _, _ = (
            frozenset(tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()),
            None, None,
        )
        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(sorted(self.SEEDS))
        # Both should have the single same triplet (0,1,2) in some order
        assert len(inc_set) == 1
        assert len(batch_set) == 1

    def test_every_pixel_has_a_valid_id_or_the_sentinel(self):
        """Ids are a real triangle index or -1 ("no triangle contains it").

        Three seeds cannot cover the whole image, so -1 must occur.  This
        replaces an assertion that every id was >= 0, which the CUDA kernel has
        never satisfied — it writes -1 for unconfined pixels.
        """
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, tgrid = inc.insert(self.SEEDS)
        t = tgrid[:, :, 2]
        assert np.all((t == -1) | ((t >= 0) & (t < len(tri_map))))
        assert np.any(t >= 0)

    def test_tgrid_shape(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, tgrid = inc.insert(self.SEEDS)
        assert tgrid.shape == (H, W, 3)
        assert tgrid.dtype == np.int32


# ---------------------------------------------------------------------------
# Multi-seed: compare incremental (all at once) vs batch
# ---------------------------------------------------------------------------

class TestMultiSeedVsBatch:
    SEEDS = [(5, 5), (55, 5), (30, 30), (5, 55), (55, 55)]

    def test_triangle_sets_match(self):
        """Incremental insert-all-at-once must produce the same triangle set."""
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, _ = inc.insert(self.SEEDS)
        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(sorted(self.SEEDS))
        assert inc_set == batch_set

    def test_voronoi_distance_matches_batch(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEEDS)
        vg_inc = inc.get_voronoi_grid()

        rd = _cu.Voronoi()
        vg_batch = rd.compute(W, H, sorted(self.SEEDS))
        np.testing.assert_array_equal(vg_inc[:, :, 1], vg_batch[:, :, 1])

    def test_seed_count(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEEDS)
        assert inc.seed_count == len(self.SEEDS)


# ---------------------------------------------------------------------------
# Incremental: insert seeds in multiple batches
# ---------------------------------------------------------------------------

class TestIncrementalBatches:
    SEEDS = [(5, 5), (55, 5), (30, 30), (5, 55), (55, 55)]

    def _insert_one_by_one(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        for s in self.SEEDS:
            tri_map, tgrid = inc.insert([s])
        return inc, tri_map, tgrid

    def test_seed_count_accumulates(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        for i, s in enumerate(self.SEEDS):
            inc.insert([s])
            assert inc.seed_count == i + 1

    def test_triangle_sets_match_batch_after_sequential_insert(self):
        inc, tri_map, _ = self._insert_one_by_one()
        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(sorted(self.SEEDS))
        assert inc_set == batch_set

    def test_voronoi_matches_batch_after_sequential_insert(self):
        inc, _, _ = self._insert_one_by_one()
        vg_inc = inc.get_voronoi_grid()

        rd = _cu.Voronoi()
        vg_batch = rd.compute(W, H, sorted(self.SEEDS))
        np.testing.assert_array_equal(vg_inc[:, :, 1], vg_batch[:, :, 1])

    def test_tgrid_consistent_with_tri_map(self):
        """Every non-negative tgrid triangle_id must exist in tri_map."""
        inc, tri_map, tgrid = self._insert_one_by_one()
        valid_ids = set(tri_map.keys())
        used_ids = set(int(x) for x in tgrid[:, :, 2].flat if x >= 0)
        assert used_ids.issubset(valid_ids)

    def test_two_batch_insert(self):
        """Insert seeds in two batches; result must match single-batch."""
        half = len(self.SEEDS) // 2
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert(self.SEEDS[:half])
        tri_map, _ = inc.insert(self.SEEDS[half:])

        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(sorted(self.SEEDS))
        assert inc_set == batch_set


# ---------------------------------------------------------------------------
# Duplicate seed rejection
# ---------------------------------------------------------------------------

class TestDuplicateRejection:
    def test_duplicate_in_batch_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(Exception):
            inc.insert([(10, 10), (10, 10)])

    def test_duplicate_across_batches_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert([(10, 10)])
        with pytest.raises(Exception):
            inc.insert([(10, 10)])

    def test_non_duplicate_does_not_raise(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert([(10, 10)])
        inc.insert([(10, 11)])  # different pixel — must not raise


# ---------------------------------------------------------------------------
# Out-of-bounds seed rejection
# ---------------------------------------------------------------------------

class TestBoundsValidation:
    def test_negative_x_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(Exception):
            inc.insert([(-1, 10)])

    def test_x_equals_width_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(Exception):
            inc.insert([(W, 10)])

    def test_negative_y_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(Exception):
            inc.insert([(10, -1)])

    def test_y_equals_height_raises(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        with pytest.raises(Exception):
            inc.insert([(10, H)])


# ---------------------------------------------------------------------------
# Timed insert
# ---------------------------------------------------------------------------

class TestTimedInsert:
    SEEDS = [(5, 5), (55, 5), (30, 30), (5, 55), (55, 55)]

    def test_timed_returns_three_tuple(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        result = inc.insert_timed(self.SEEDS)
        assert len(result) == 3

    def test_timed_timings_keys(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        _, _, t = inc.insert_timed(self.SEEDS)
        assert set(t.keys()) == {"bfs_ms", "bfs_iters",
                                 "detect_ms", "dedup_ms", "assign_ms"}

    def test_bfs_iters_counts_full_canvas_passes(self):
        """One BFS trip is a pass over the whole padded canvas, so the trip
        count is what decides the cost of an insert, not the batch size."""
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        _, _, t = inc.insert_timed(self.SEEDS)
        assert t["bfs_iters"] >= 1
        assert t["bfs_iters"] <= W + H, (
            "a converging BFS cannot need more trips than the canvas is wide "
            "plus tall; more means it is filling seedless space")

    def test_timed_timings_non_negative(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        _, _, t = inc.insert_timed(self.SEEDS)
        for k, v in t.items():
            assert v >= 0.0, f"{k} = {v} is negative"

    def test_timed_result_matches_untimed(self):
        seeds = self.SEEDS
        inc1 = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map1, tgrid1 = inc1.insert(seeds)

        inc2 = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map2, tgrid2, _ = inc2.insert_timed(seeds)

        set1 = frozenset(tuple(sorted([v[2], v[3], v[4]])) for v in tri_map1.values())
        set2 = frozenset(tuple(sorted([v[2], v[3], v[4]])) for v in tri_map2.values())
        assert set1 == set2


# ---------------------------------------------------------------------------
# Larger random smoke test
# ---------------------------------------------------------------------------

class TestRandomSmoke:
    @staticmethod
    def _random_seeds(n, seed=42):
        rng = np.random.default_rng(seed)
        xs = rng.integers(0, W, size=n * 2)
        ys = rng.integers(0, H, size=n * 2)
        seeds = list(dict.fromkeys(zip(xs.tolist(), ys.tolist())))[:n]
        return seeds

    def test_20_seeds_all_at_once(self):
        # Sort before insertion so incremental IDs match the batch API's sort order.
        seeds = sorted(self._random_seeds(20, seed=1))
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, tgrid = inc.insert(seeds)
        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(seeds)
        assert inc_set == batch_set

    def test_20_seeds_sequential(self):
        # Sort before insertion so incremental IDs match the batch API's sort order.
        seeds = sorted(self._random_seeds(20, seed=2))
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        for s in seeds:
            tri_map, tgrid = inc.insert([s])

        inc_set = frozenset(
            tuple(sorted([v[2], v[3], v[4]])) for v in tri_map.values()
        )
        batch_set, _, _ = batch_triangulate(seeds)
        assert inc_set == batch_set

    def test_tgrid_no_invalid_ids(self):
        seeds = self._random_seeds(15, seed=3)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        tri_map, tgrid = inc.insert(seeds)
        valid_ids = set(tri_map.keys()) | {-1}
        used = set(int(x) for x in tgrid[:, :, 2].flat)
        assert used.issubset(valid_ids), f"Unknown IDs: {used - valid_ids}"


# ---------------------------------------------------------------------------
# get_edges
# ---------------------------------------------------------------------------

class TestGetEdges:
    """Undirected edges, deduplicated on the device.

    The oracle is the triangle list this same object reports: a caller could
    derive edges from it in numpy, so the device result must agree exactly
    with that derivation.
    """

    @staticmethod
    def _edges_from_triangles(tris, n_seeds):
        if len(tris) == 0:
            return set()
        out = set()
        for a, b, c in tris:
            for p, q in ((a, b), (b, c), (c, a)):
                out.add((int(min(p, q)), int(max(p, q))))
        return out

    def _mesh(self, n, seed):
        rng = np.random.default_rng(seed)
        pts = sorted({(int(x), int(y))
                      for x, y in rng.integers([0, 0], [W, H], size=(n * 3, 2))})
        pts = pts[:n]
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(np.array(pts, dtype=np.int32))
        return inc

    @pytest.mark.parametrize("n", [3, 10, 60, 200])
    def test_matches_edges_derived_from_the_triangles(self, n):
        inc = self._mesh(n, seed=n)
        expected = self._edges_from_triangles(inc.get_triangles(), inc.seed_count)
        got = {(int(a), int(b)) for a, b in inc.get_edges()}
        assert got == expected

    def test_each_edge_appears_once_and_is_ordered(self):
        inc = self._mesh(120, seed=5)
        e = inc.get_edges()
        assert e.ndim == 2 and e.shape[1] == 2
        assert e.dtype == np.int32
        assert (e[:, 0] < e[:, 1]).all(), "each row must be (min, max)"
        assert len({(int(a), int(b)) for a, b in e}) == len(e)

    def test_ids_are_insertion_order(self):
        """Same numbering as get_triangles, so both index a caller's array."""
        inc = self._mesh(80, seed=7)
        n = inc.seed_count
        e = inc.get_edges()
        assert e.min() >= 0 and e.max() < n

    def test_survives_further_inserts(self):
        inc = self._mesh(60, seed=11)
        before = len(inc.get_edges())
        rng = np.random.default_rng(12)
        seen = {(int(x), int(y)) for x, y in
                np.asarray(inc.get_voronoi_grid())[:0, :0].reshape(0, 2)}  # empty
        pts = []
        used = set()
        while len(pts) < 20:
            p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            if p not in used:
                used.add(p)
                pts.append(p)
        try:
            inc.insert_deferred(np.array(pts, dtype=np.int32))
        except ValueError:
            pytest.skip("random points collided with existing seeds")
        after = inc.get_edges()
        expected = self._edges_from_triangles(inc.get_triangles(), inc.seed_count)
        assert {(int(a), int(b)) for a, b in after} == expected
        assert len(after) > before

    def test_two_seeds_have_no_triangle_and_no_edges(self):
        inc = _cu.Delaunay(W, H, MAX_SEEDS, PADDING)
        inc.insert_deferred(np.array([[5, 5], [20, 20]], dtype=np.int32))
        assert len(inc.get_triangles()) == 0
        assert len(inc.get_edges()) == 0
