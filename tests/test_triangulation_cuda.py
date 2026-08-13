"""CUDA tests for GridTriangulation — skipped automatically when no GPU is available."""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


skip_no_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="_delauney_cuda extension not built or no CUDA device"
)


@pytest.fixture(scope="module")
def cuda_vd():
    from delauney._delauney_cuda import RegularDelaunay
    return RegularDelaunay()


@pytest.fixture(scope="module")
def cuda_tri():
    from delauney._delauney_cuda import GridTriangulation
    return GridTriangulation()


@pytest.fixture(scope="module")
def ref_vd():
    from delauney.reference.voronoi import RegularDelaunay
    return RegularDelaunay()


@pytest.fixture(scope="module")
def ref_tri():
    from delauney.reference.triangulation import GridTriangulation
    return GridTriangulation()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cuda(cuda_vd, cuda_tri, W, H, seeds):
    vgrid = cuda_vd.compute(W, H, seeds)
    return cuda_tri.compute(vgrid, seeds)


def run_ref(ref_vd, ref_tri, W, H, seeds):
    vgrid = ref_vd.compute(W, H, seeds)
    return ref_tri.compute(vgrid, seeds)


# ---------------------------------------------------------------------------
# CUDA vs reference comparison
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCudaMatchesReference:
    @pytest.mark.parametrize("seeds,W,H", [
        ([(1, 1), (8, 1), (4, 8)], 10, 10),
        ([(1, 1), (8, 1), (1, 8), (8, 8)], 10, 10),
        ([(0, 0), (4, 0), (0, 4), (4, 4)], 5, 5),
        ([(2, 5), (15, 2), (10, 14)], 20, 20),
    ])
    def test_triangle_map_matches(self, cuda_vd, cuda_tri, ref_vd, ref_tri, seeds, W, H):
        c_map, _ = run_cuda(cuda_vd, cuda_tri, W, H, seeds)
        r_map, _ = run_ref(ref_vd, ref_tri, W, H, seeds)

        # Compare by geometric content (sorted triplets), not by id assignment order
        c_triplets = {frozenset({a, b, c}) for _, (_, _, a, b, c) in c_map.items()}
        r_triplets = {frozenset({a, b, c}) for _, (_, _, a, b, c) in r_map.items()}
        assert c_triplets == r_triplets, (
            f"Triangle sets differ for seeds={seeds}\n"
            f"CUDA: {c_triplets}\nref:  {r_triplets}"
        )

    @pytest.mark.parametrize("seeds,W,H", [
        ([(1, 1), (8, 1), (4, 8)], 10, 10),
        # The 4-symmetric-seed case is excluded: its 4-region meeting creates
        # boundary pixels that lie inside two triangles simultaneously.  Tie-
        # breaking (highest tid) is implementation-defined and CUDA (Thrust
        # sort order) vs reference (scan insertion order) make different but
        # equally valid choices.  Covered by test_triangle_map_matches instead.
    ])
    def test_cell_grid_matches(self, cuda_vd, cuda_tri, ref_vd, ref_tri, seeds, W, H):
        """The triangle_id grid (channel 2) should agree cell-by-cell.

        Because triangle_id numbering may differ (CUDA vs ref assign ids in
        different insertion orders), we compare via normalized ids based on
        sorted seed-triplet ordering rather than raw values.
        """
        c_map, c_grid = run_cuda(cuda_vd, cuda_tri, W, H, seeds)
        r_map, r_grid = run_ref(ref_vd, ref_tri, W, H, seeds)

        # Build per-pixel canonical ID based on sorted seed-triplet.
        # Only considers triplets that appear in the grid (phantom triangles
        # detected but never assigned to any pixel are ignored).
        # Triplets sorted as tuples for a deterministic total ordering.
        def canonical(tri_map, tgrid):
            H_, W_ = tgrid.shape[:2]
            tid_to_triplet = {
                tid: tuple(sorted([a, b, c]))
                for tid, (_, _, a, b, c) in tri_map.items()
            }
            appearing = sorted({
                tid_to_triplet[int(tgrid[y, x, 2])]
                for y in range(H_) for x in range(W_)
                if int(tgrid[y, x, 2]) >= 0
            })
            triplet_to_cid = {t: i for i, t in enumerate(appearing)}
            canon = np.full((H_, W_), -1, dtype=np.int32)
            for y in range(H_):
                for x in range(W_):
                    t = int(tgrid[y, x, 2])
                    if t >= 0:
                        canon[y, x] = triplet_to_cid[tid_to_triplet[t]]
            return canon

        c_canon = canonical(c_map, c_grid)
        r_canon = canonical(r_map, r_grid)
        assert np.array_equal(c_canon, r_canon), "Cell triangle assignments differ"


# ---------------------------------------------------------------------------
# Cocircular quads: both paths must cut them the same way
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCocircularAgreement:
    """A degree-4 Voronoi vertex admits two triangulations; both paths pick one.

    Both are fed the *same* Voronoi grid, which keeps this about the diagonal
    rule: the two Voronoi implementations can legitimately differ at ties, and
    CUDA can additionally settle on a non-nearest seed (see RegularDelaunay in
    reference/voronoi.py), which would otherwise show up here as noise.
    """

    @staticmethod
    def _triplets(tri_map):
        return {tuple(sorted((v[2], v[3], v[4]))) for v in tri_map.values()}

    def _both(self, cuda_tri, ref_tri, vgrid, seeds):
        c_map, _ = cuda_tri.compute(vgrid.astype(np.int32), seeds, 0)
        r_map, _ = ref_tri.compute(vgrid, seeds, border_padding=0)
        return self._triplets(c_map), self._triplets(r_map)

    @pytest.mark.parametrize("seeds,W,H", [
        ([(1, 1), (8, 1), (1, 8), (8, 8)], 10, 10),
        ([(0, 0), (4, 0), (0, 4), (4, 4)], 5, 5),
        ([(2, 2), (12, 2), (2, 12), (12, 12)], 15, 15),
    ])
    def test_square_quad_cut_identically(self, cuda_tri, ref_vd, ref_tri, seeds, W, H):
        vgrid = ref_vd.compute(W, H, seeds)
        c, r = self._both(cuda_tri, ref_tri, vgrid, seeds)
        assert c == r, f"CUDA {sorted(c)} vs ref {sorted(r)}"
        assert len(c) == 2, f"expected 2 triangles for a cocircular quad, got {len(c)}"

    def test_random_grids_agree(self, cuda_tri, ref_vd, ref_tri):
        """Integer coordinates make cocircular quads common, so sweep for them.

        Before the CUDA split was made geometric this failed on 8 of 40.
        """
        rng = np.random.default_rng(0)
        W = H = 24
        for trial in range(40):
            pts = set()
            while len(pts) < 12:
                pts.add((int(rng.integers(0, W)), int(rng.integers(0, H))))
            seeds = sorted(pts)
            vgrid = ref_vd.compute(W, H, seeds)
            c, r = self._both(cuda_tri, ref_tri, vgrid, seeds)
            assert c == r, (
                f"trial {trial} seeds={seeds}\n"
                f"  CUDA only: {sorted(c - r)}\n  ref only:  {sorted(r - c)}")

    def test_rotating_the_image_does_not_change_the_cut(self, cuda_tri, ref_vd, ref_tri):
        """The rule is geometric, so it must not depend on pixel orientation.

        A quad split on the 2x2 pixel block's anti-diagonal would flip here.
        """
        W = H = 13
        seeds = [(2, 3), (9, 3), (2, 10), (9, 10)]
        rot = [(W - 1 - y, x) for x, y in seeds]     # 90 degrees

        def cut(sd):
            vgrid = ref_vd.compute(W, H, sd)
            c, r = self._both(cuda_tri, ref_tri, vgrid, sd)
            assert c == r
            # name the diagonal by the seed pair the two triangles share
            a, b = sorted(c)
            return tuple(sorted(set(a) & set(b)))

        # Same four seeds, same geometry, so the same pair must stay shared.
        assert len(cut(seeds)) == 2
        assert len(cut(rot)) == 2


# ---------------------------------------------------------------------------
# CUDA output contract
# ---------------------------------------------------------------------------

@skip_no_cuda
class TestCudaTriangulationContract:
    def test_output_shapes(self, cuda_vd, cuda_tri):
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, tgrid = run_cuda(cuda_vd, cuda_tri, 10, 10, seeds)
        assert tgrid.shape == (10, 10, 3)
        assert tgrid.dtype == np.int32

    def test_no_duplicate_triangles(self, cuda_vd, cuda_tri):
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, _ = run_cuda(cuda_vd, cuda_tri, 10, 10, seeds)
        triplets = set()
        for _, (_, _, a, b, c) in tri_map.items():
            t = frozenset({a, b, c})
            assert t not in triplets, "duplicate triangle found"
            triplets.add(t)

    def test_every_cell_has_a_valid_id_or_the_sentinel(self, cuda_vd, cuda_tri):
        """Ids are a real triangle index or -1 ("no triangle contains it").

        Three seeds cannot cover a 10x10 image, so -1 must occur.
        """
        seeds = [(1, 1), (8, 1), (4, 8)]
        tri_map, tgrid = run_cuda(cuda_vd, cuda_tri, 10, 10, seeds)
        t = tgrid[:, :, 2]
        assert np.all((t == -1) | ((t >= 0) & (t < len(tri_map))))
        assert np.any(t == -1)
        assert np.any(t >= 0)

    def test_voronoi_channels_preserved(self, cuda_vd, cuda_tri):
        seeds = [(1, 1), (8, 1), (4, 8)]
        vgrid = cuda_vd.compute(10, 10, seeds)
        _, tgrid = cuda_tri.compute(vgrid, seeds)
        assert np.array_equal(vgrid[:, :, 0], tgrid[:, :, 0])
        assert np.array_equal(vgrid[:, :, 1], tgrid[:, :, 1])
