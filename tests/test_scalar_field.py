"""The scalar field on the vertices, and edge selection over it.

A caller refining a mesh carries one number per vertex and subdivides the edges
along which it changes most. Holding that field beside the coordinates and the
edge list means scoring an edge moves nothing; the alternative is shipping the
edge list out, scoring on the host, and shipping a list of chosen indices back.

Selection stays with the caller, expressed as a threshold rather than a list:
`select_midpoints` takes every edge beating (min_score, tie_index) under
"higher score first, lower edge index on a tie". Passing the k-th largest such
key takes exactly k edges, so what crosses the boundary is two numbers.

The oracle throughout is the numpy the device code replaced.
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

W, H, P = 240, 180, 16
MAX_SEEDS = 4000
MIN_LEN = 2.0


def make_mesh(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cand = np.array(sorted({(int(x), int(y)) for x, y in
                            rng.integers([0, 0], [W, H], size=(n * 4, 2))}),
                    dtype=np.int32)
    pts = cand[np.sort(rng.choice(len(cand), size=min(n, len(cand)),
                                  replace=False))]
    img = rng.random((H, W)).astype(np.float32)
    vals = img[pts[:, 1], pts[:, 0]]
    mesh = _cu.Delaunay(W, H, MAX_SEEDS, P)
    mesh.insert_deferred(pts, vals)
    return mesh, pts, vals


def host_scores(pts, vals, ea, eb, min_length):
    d = pts[eb].astype(np.float64) - pts[ea].astype(np.float64)
    length = np.hypot(d[:, 0], d[:, 1])
    contrast = np.abs(vals[eb].astype(np.float64) - vals[ea].astype(np.float64))
    s = contrast * length
    s[length < min_length] = 0.0
    return s


class TestEdgeScores:

    @pytest.mark.parametrize("n", [20, 120, 400])
    def test_matches_the_host_formula(self, n):
        mesh, pts, vals = make_mesh(n, seed=n)
        e = mesh.get_edges()
        got = mesh.edge_scores(MIN_LEN)
        want = host_scores(pts, vals, e[:, 0].astype(np.int64),
                           e[:, 1].astype(np.int64), MIN_LEN)
        assert len(got) == len(e)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_short_edges_score_zero(self):
        """Their midpoint rounds onto an endpoint, so they cannot be split."""
        mesh, pts, _ = make_mesh(400, seed=5)
        e = mesh.get_edges()
        d = pts[e[:, 1]].astype(np.float64) - pts[e[:, 0]].astype(np.float64)
        length = np.hypot(d[:, 0], d[:, 1])
        s = mesh.edge_scores(8.0)
        assert (s[length < 8.0] == 0).all()
        assert (s[length >= 8.0] >= 0).all()

    def test_order_matches_get_edges(self):
        """Scores index the edge list, so the two must not drift."""
        mesh, pts, vals = make_mesh(200, seed=7)
        e1 = mesh.get_edges()
        s = mesh.edge_scores(MIN_LEN)
        e2 = mesh.get_edges()
        np.testing.assert_array_equal(e1, e2)
        assert len(s) == len(e1)

    def test_without_values_it_refuses(self):
        mesh = _cu.Delaunay(W, H, MAX_SEEDS, P)
        mesh.insert_deferred(np.array([[10, 10], [60, 12], [30, 70]],
                                      dtype=np.int32))
        with pytest.raises(Exception):
            mesh.edge_scores(MIN_LEN)

    def test_values_must_match_the_batch(self):
        mesh = _cu.Delaunay(W, H, MAX_SEEDS, P)
        with pytest.raises(Exception):
            mesh.insert_deferred(np.array([[10, 10], [60, 12], [30, 70]],
                                          dtype=np.int32),
                                 np.array([1.0, 2.0], dtype=np.float32))


def kth_key(scores, take):
    """The k-th largest (score, -index) key, as the host computes it."""
    cand = np.flatnonzero(scores > 0)
    if len(cand) <= take:
        weakest = scores[cand].min()
        return float(weakest), int(cand[scores[cand] == weakest].max())
    s = scores[cand]
    ks = float(s[np.argpartition(-s, take - 1)[take - 1]])
    n_gt = int((s > ks).sum())
    tied = cand[s == ks]
    return ks, int(tied[take - n_gt - 1])


class TestSelectMidpoints:

    @pytest.mark.parametrize("take", [1, 10, 75, 300])
    def test_matches_the_host_selection(self, take):
        mesh, pts, vals = make_mesh(400, seed=11)
        e = mesh.get_edges()
        ea, eb = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
        scores = mesh.edge_scores(MIN_LEN)
        ms, ti = kth_key(scores, take)

        idx = np.arange(len(scores))
        picked = np.flatnonzero((scores > 0)
                                & ((scores > ms) | ((scores == ms) & (idx <= ti))))
        mid = np.rint((pts[ea[picked]].astype(np.float64)
                       + pts[eb[picked]].astype(np.float64)) / 2.0).astype(np.int64)
        np.clip(mid[:, 0], 0, W - 1, out=mid[:, 0])
        np.clip(mid[:, 1], 0, H - 1, out=mid[:, 1])
        taken = {(int(x), int(y)) for x, y in pts}
        seen, want = set(), []
        for mx, my in mid:
            t = (int(mx), int(my))
            if t in taken or t in seen:
                continue
            seen.add(t)
            want.append(t)

        got = {(int(x), int(y)) for x, y in mesh.select_midpoints(MIN_LEN, ms, ti)}
        assert got == set(want)

    def test_the_key_takes_exactly_k_edges(self):
        """The tie-break by index is what makes the count exact."""
        mesh, pts, vals = make_mesh(400, seed=13)
        scores = mesh.edge_scores(MIN_LEN)
        idx = np.arange(len(scores))
        for take in (5, 50, 200):
            ms, ti = kth_key(scores, take)
            n = int(((scores > 0)
                     & ((scores > ms) | ((scores == ms) & (idx <= ti)))).sum())
            assert n == take, f"key selected {n} edges, wanted {take}"

    def test_never_returns_a_pixel_that_is_already_a_seed(self):
        """The Voronoi diagram is the record of what has been sampled: a seed
        is a cell at distance zero, so no separate list is needed."""
        mesh, pts, vals = make_mesh(400, seed=17)
        scores = mesh.edge_scores(MIN_LEN)
        ms, ti = kth_key(scores, 300)
        mid = mesh.select_midpoints(MIN_LEN, ms, ti)
        seeds = {(int(x), int(y)) for x, y in pts}
        assert not ({(int(x), int(y)) for x, y in mid} & seeds)

    def test_midpoints_are_unique_and_in_bounds(self):
        mesh, pts, vals = make_mesh(400, seed=19)
        scores = mesh.edge_scores(MIN_LEN)
        ms, ti = kth_key(scores, 250)
        mid = mesh.select_midpoints(MIN_LEN, ms, ti)
        assert len(np.unique(mid, axis=0)) == len(mid)
        assert (mid >= 0).all()
        assert (mid[:, 0] < W).all() and (mid[:, 1] < H).all()

    def test_is_deterministic(self):
        """Emission races on an atomic counter, so the collision pass sorts by
        (pixel, edge index) to make the survivor a fixed choice."""
        mesh, pts, vals = make_mesh(400, seed=23)
        scores = mesh.edge_scores(MIN_LEN)
        ms, ti = kth_key(scores, 200)
        first = mesh.select_midpoints(MIN_LEN, ms, ti)
        for _ in range(4):
            np.testing.assert_array_equal(
                mesh.select_midpoints(MIN_LEN, ms, ti), first)

    def test_inserting_them_back_is_accepted(self):
        """Whatever comes out must be legal to insert: no duplicates, in
        bounds. delauney rejects both, so a round trip proves it."""
        mesh, pts, vals = make_mesh(400, seed=29)
        scores = mesh.edge_scores(MIN_LEN)
        ms, ti = kth_key(scores, 150)
        mid = mesh.select_midpoints(MIN_LEN, ms, ti)
        before = mesh.seed_count
        mesh.insert_deferred(mid.astype(np.int32),
                             np.zeros(len(mid), dtype=np.float32))
        assert mesh.seed_count == before + len(mid)
