"""The scalar field on the vertices, and edge selection over it.

A caller refining a mesh carries one number per vertex and subdivides the edges
along which it changes most. Holding that field beside the coordinates and the
edge list means scoring an edge moves nothing; the alternative is shipping the
edge list out, scoring on the host, and shipping a list of chosen indices back.

Selection happens there too, driven by a count: the edges are ordered by score
descending and edge index ascending, which is a total order, so taking the front
`count` of them is exact and needs no threshold with ties to settle. The caller
still decides how many, but that is all that crosses.

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


def host_selection(pts, ea, eb, scores, take, threshold=0.0):
    """The selection the device performs, done in numpy.

    Same total order -- score descending, edge index ascending -- so `take` is
    exact and there are no ties left to settle. Collisions keep the first
    survivor in that order, which is the better-scoring edge.
    """
    eligible = np.flatnonzero(scores > threshold)
    if len(eligible) == 0:
        return []
    order = eligible[np.lexsort((eligible, -scores[eligible]))][:take]
    mid = np.rint((pts[ea[order]].astype(np.float64)
                   + pts[eb[order]].astype(np.float64)) / 2.0).astype(np.int64)
    np.clip(mid[:, 0], 0, W - 1, out=mid[:, 0])
    np.clip(mid[:, 1], 0, H - 1, out=mid[:, 1])
    taken = {(int(x), int(y)) for x, y in pts}
    seen, out = set(), []
    for mx, my in mid:
        t = (int(mx), int(my))
        if t in taken or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


class TestSelectMidpoints:

    @pytest.mark.parametrize("take", [1, 10, 75, 300])
    def test_matches_the_host_selection(self, take):
        mesh, pts, vals = make_mesh(400, seed=11)
        e = mesh.get_edges()
        ea, eb = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
        scores = mesh.edge_scores(MIN_LEN)
        want = host_selection(pts, ea, eb, scores, take)
        got = {(int(x), int(y)) for x, y in mesh.select_midpoints(MIN_LEN, take)}
        assert got == set(want)

    @pytest.mark.parametrize("take", [5, 50, 200])
    def test_takes_exactly_the_count_asked_for(self, take):
        """The order is total, so the count is exact -- no ties to settle.

        Fewer can come back only because a midpoint collided, never because
        the score boundary was ambiguous.
        """
        mesh, pts, vals = make_mesh(400, seed=13)
        e = mesh.get_edges()
        ea, eb = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
        scores = mesh.edge_scores(MIN_LEN)
        assert int((scores > 0).sum()) >= take
        want = host_selection(pts, ea, eb, scores, take)
        got = mesh.select_midpoints(MIN_LEN, take)
        assert len(got) == len(want) <= take

    def test_asking_for_more_than_available_returns_what_there_is(self):
        mesh, pts, vals = make_mesh(120, seed=15)
        scores = mesh.edge_scores(MIN_LEN)
        n_eligible = int((scores > 0).sum())
        got = mesh.select_midpoints(MIN_LEN, n_eligible * 10)
        assert 0 < len(got) <= n_eligible

    def test_threshold_excludes_weak_edges(self):
        mesh, pts, vals = make_mesh(400, seed=16)
        scores = mesh.edge_scores(MIN_LEN)
        cut = float(np.median(scores[scores > 0]))
        got = mesh.select_midpoints(MIN_LEN, 10_000, cut)
        assert len(got) <= int((scores > cut).sum())
        assert len(mesh.select_midpoints(MIN_LEN, 10_000, float(scores.max()))) == 0

    def test_never_returns_a_pixel_that_is_already_a_seed(self):
        """The Voronoi diagram is the record of what has been sampled: a seed
        is a cell at distance zero, so no separate list is needed."""
        mesh, pts, vals = make_mesh(400, seed=17)
        mid = mesh.select_midpoints(MIN_LEN, 300)
        seeds = {(int(x), int(y)) for x, y in pts}
        assert not ({(int(x), int(y)) for x, y in mid} & seeds)

    def test_midpoints_are_unique_and_in_bounds(self):
        mesh, pts, vals = make_mesh(400, seed=19)
        mid = mesh.select_midpoints(MIN_LEN, 250)
        assert len(np.unique(mid, axis=0)) == len(mid)
        assert (mid >= 0).all()
        assert (mid[:, 0] < W).all() and (mid[:, 1] < H).all()

    def test_is_deterministic(self):
        """Emission races on an atomic counter, so the collision pass sorts by
        (pixel, rank) to make the survivor a fixed choice."""
        mesh, pts, vals = make_mesh(400, seed=23)
        first = mesh.select_midpoints(MIN_LEN, 200)
        for _ in range(4):
            np.testing.assert_array_equal(
                mesh.select_midpoints(MIN_LEN, 200), first)

    def test_inserting_them_back_is_accepted(self):
        """Whatever comes out must be legal to insert: no duplicates, in
        bounds. delauney rejects both, so a round trip proves it."""
        mesh, pts, vals = make_mesh(400, seed=29)
        mid = mesh.select_midpoints(MIN_LEN, 150)
        before = mesh.seed_count
        mesh.insert_deferred(mid.astype(np.int32),
                             np.zeros(len(mid), dtype=np.float32))
        assert mesh.seed_count == before + len(mid)


class TestSeedsAndValues:
    """The mesh is the single record of the point set and the field."""

    def test_seeds_are_insertion_order(self):
        mesh, pts, vals = make_mesh(200, seed=31)
        np.testing.assert_array_equal(mesh.get_seeds(), pts)

    def test_values_are_insertion_order(self):
        mesh, pts, vals = make_mesh(200, seed=33)
        np.testing.assert_allclose(mesh.get_values(), vals, rtol=0, atol=0)

    def test_both_grow_with_further_inserts(self):
        mesh, pts, vals = make_mesh(200, seed=35)
        mid = mesh.select_midpoints(MIN_LEN, 40)
        new = np.arange(len(mid), dtype=np.float32)
        mesh.insert_deferred(mid.astype(np.int32), new)
        seeds, values = mesh.get_seeds(), mesh.get_values()
        assert len(seeds) == len(pts) + len(mid) == len(values)
        np.testing.assert_array_equal(seeds[:len(pts)], pts)
        np.testing.assert_array_equal(seeds[len(pts):], mid)
        np.testing.assert_allclose(values[len(pts):], new)


class TestLocate:
    """Containment for a list of points, without rasterising the canvas.

    The oracle is the finalised grid, which answers the same question for every
    pixel -- so locate() reading differently at any point would be a bug, not a
    tolerance.
    """

    def _mesh_and_queries(self, n, nq, seed):
        mesh, pts, vals = make_mesh(n, seed)
        rng = np.random.default_rng(seed + 1)
        q = np.stack([rng.integers(0, W, nq), rng.integers(0, H, nq)],
                     axis=1).astype(np.int32)
        return mesh, pts, q

    @pytest.mark.parametrize("n,nq", [(50, 200), (400, 2000), (400, 9000)])
    def test_matches_the_finalised_grid(self, n, nq):
        mesh, pts, q = self._mesh_and_queries(n, nq, seed=n + nq)
        got = mesh.locate(q)
        _, grid = mesh.finalise(as_arrays=True)
        want = np.asarray(grid)[q[:, 1], q[:, 0], 2]
        np.testing.assert_array_equal(got, want)

    def test_valid_before_any_finalise(self):
        """The caller queries a mesh refinement left unfinalised, so the answer
        must not depend on an assignment pass having run."""
        mesh, pts, q = self._mesh_and_queries(400, 3000, seed=71)
        before = mesh.locate(q)
        _, grid = mesh.finalise(as_arrays=True)
        np.testing.assert_array_equal(before, np.asarray(grid)[q[:, 1], q[:, 0], 2])

    def test_ids_index_get_triangles(self):
        """locate's ids and get_triangles must agree, or a caller looking up the
        vertices of a located triangle gets a different triangle."""
        mesh, pts, q = self._mesh_and_queries(400, 2000, seed=73)
        tids = mesh.locate(q)
        tris = mesh.get_triangles()
        hit = tids >= 0
        assert hit.any()
        assert tids[hit].max() < len(tris)
        # Each located point really is inside the triangle it names.
        for i in np.flatnonzero(hit)[:200]:
            a, b, c = (pts[j] for j in tris[tids[i]])
            p = q[i].astype(np.float64)
            def z(u, v):                      # 2-D cross, scalar
                return float(u[0] * v[1] - u[1] * v[0])
            d1, d2, d3 = z(b - a, p - a), z(c - b, p - b), z(a - c, p - c)
            assert not ((min(d1, d2, d3) < 0) and (max(d1, d2, d3) > 0)), \
                f"point {p} is not inside triangle {tids[i]}"

    def test_points_outside_the_hull_report_no_triangle(self):
        mesh, pts, _ = self._mesh_and_queries(60, 10, seed=77)
        far = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]],
                       dtype=np.int32)
        _, grid = mesh.finalise(as_arrays=True)
        np.testing.assert_array_equal(
            mesh.locate(far), np.asarray(grid)[far[:, 1], far[:, 0], 2])

    def test_empty_query(self):
        mesh, _, _ = self._mesh_and_queries(100, 10, seed=79)
        assert len(mesh.locate(np.zeros((0, 2), dtype=np.int32))) == 0
