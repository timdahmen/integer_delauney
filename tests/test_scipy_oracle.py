"""Both CUDA paths against scipy's exact Delaunay triangulation.

Why this file exists
--------------------
Every other correctness test here compares one raster implementation against
another: CUDA against the NumPy reference, or incremental against batch. Both
comparisons share the same idea of what a triangle is -- three Voronoi regions
meeting in a 2x2 pixel block -- so a misapplication of that idea present in
both sides is invisible. That is not hypothetical: the cocircular tie-break bug
was fixed in triangulation.cu while delaunay.cu kept a stale copy, and the
incremental-vs-batch tests could not see it because they only ever compared the
two against each other.

Unifying the kernel (which is the point of the next step) makes the
incremental-vs-batch comparison tautological -- the same code on both sides --
so an oracle that is not a raster implementation is needed first.

scipy.spatial.Delaunay is that oracle. It is exact computational geometry with
no raster anywhere, so it agrees with delauney only where delauney is right.

What can honestly be asserted
-----------------------------
delauney is a raster method and does not reproduce scipy everywhere. Measured
over 60 random 40-seed sets on 256x256, every single disagreement falls into
one of exactly two documented limitations, with nothing left over:

  * The circumcentre lies outside the padded canvas. A triangle is registered
    only where its circumcentre is witnessed as a grid cell, so these are
    invisible by construction. See BORDER_PADDING_BOUND.md.

  * Near-cocircularity. Let

        margin = dist(circumcentre, nearest rival seed) - circumradius

    For a quad that is exactly cocircular, margin == 0 and both diagonals are
    Delaunay, so either answer is correct. Just off cocircular, the two true
    Voronoi vertices are a sub-pixel distance apart and no 2x2 block can tell
    them apart. Measured: triangles delauney finds have a median margin of
    5.3 px; the ones it misses, 0.15 px, with 85% inside +-1 px. Extra
    triangles it emits sit at a median margin of -0.17 px -- barely, and
    unresolvably, non-Delaunay.

So the contract has three clauses:

  1. A triangle that is both resolvable and on-canvas must be found.
  2. No triangle may be emitted that is more than marginally non-Delaunay.
  3. The emitted triangles must not overlap.

MARGIN_PX = 2.0 sets "resolvable" with a safety factor over the 1 px where the
measured boundary actually sits. The residue was checked rather than assumed:
of 33 missed interior triangles, the 5 with margin > 1 px all had off-canvas
circumcentres. Unexplained: 0.

Clause 3 is not redundant, and scipy cannot supply it. The historical bug
emitted *all four* triangles of a cocircular quad -- each individually
Delaunay, since the fourth point lies exactly on the circumcircle, so clauses 1
and 2 both pass. Only the global structure is wrong. See
TestTheOverlapInvariantItself, which checks that clause against a triangle set
built to fail in exactly that way, because it is the one assertion in this file
with no external oracle behind it.
"""
import numpy as np
import pytest

from scipy.spatial import Delaunay as ScipyDelaunay


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

W = H = 256
PADDING = 64
MAX_SEEDS = 512

#: Margin, in pixels, above which a triangle is considered resolvable on the
#: raster. The measured boundary is at about 1 px; 2.0 leaves a safety factor.
MARGIN_PX = 2.0

#: How far inside the padded canvas the circumcentre must sit. Detection reads
#: 2x2 blocks, so a circumcentre exactly on the canvas edge has no full block.
EDGE_SLACK = 2.0


# ---------------------------------------------------------------------------
# Geometry, in plain numpy -- deliberately independent of anything in delauney
# ---------------------------------------------------------------------------

def circumcircle(p):
    ax, ay = p[0]
    bx, by = p[1]
    cx, cy = p[2]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
          + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
          + (cx**2 + cy**2) * (bx - ax)) / d
    centre = np.array([ux, uy])
    return centre, float(np.hypot(ax - ux, ay - uy))


def margin_of(seed_arr, triplet):
    """dist(circumcentre, nearest rival seed) - circumradius, in pixels.

    Positive means the circumcircle is empty with room to spare, which is what
    lets the raster resolve the triangle. Negative means some seed is inside,
    so the triangle is not Delaunay.
    """
    centre, radius = circumcircle(seed_arr[list(triplet)])
    d = np.linalg.norm(seed_arr - centre, axis=1)
    d[list(triplet)] = np.inf
    return float(d.min() - radius), centre


def on_canvas(centre, padding):
    return (-padding + EDGE_SLACK <= centre[0] < W + padding - EDGE_SLACK
            and -padding + EDGE_SLACK <= centre[1] < H + padding - EDGE_SLACK)


def random_seeds(rng, n, margin=24):
    pts = rng.integers([margin, margin], [W - margin, H - margin], size=(n, 2))
    return sorted({(int(x), int(y)) for x, y in pts})


def scipy_triangles(seeds):
    sd = ScipyDelaunay(np.asarray(seeds, dtype=np.float64))
    triplets = {tuple(sorted(int(v) for v in s)) for s in sd.simplices}
    hull = {int(v) for s in sd.convex_hull for v in s}
    return triplets, hull


def triplets_of(tri_map):
    return {tuple(sorted((int(v[2]), int(v[3]), int(v[4]))))
            for v in tri_map.values()}


def overlapping_pair(triplets, arr):
    """First pair of triangles that overlap, or None if the set is disjoint.

    With every triangle wound counter-clockwise, each directed edge belongs to
    at most one triangle of a valid triangulation: an interior edge is
    traversed once in each direction by the two triangles that share it. Two
    triangles that reuse a directed edge lie on the same side of it and so
    overlap.
    """
    seen = {}
    for t in triplets:
        a, b, c = (arr[i] for i in t)
        ia, ib, ic = t
        if ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0])) < 0:
            ib, ic = ic, ib
        for e in ((ia, ib), (ib, ic), (ic, ia)):
            if e in seen:
                return seen[e], t, e
            seen[e] = t
    return None


# ---------------------------------------------------------------------------
# The two CUDA paths, producing the same kind of answer
# ---------------------------------------------------------------------------

def batch_triplets(seeds, padding=PADDING):
    vgrid = _cu.Voronoi().compute(W, H, seeds)
    tri_map, _ = _cu.GridTriangulation().compute(vgrid, seeds, padding)
    return triplets_of(tri_map)


def incremental_triplets(seeds, padding=PADDING, batches=1):
    inc = _cu.Delaunay(W, H, MAX_SEEDS, padding)
    tri_map = {}
    chunks = np.array_split(np.asarray(seeds), batches)
    for chunk in chunks:
        if len(chunk):
            tri_map, _ = inc.insert([tuple(int(v) for v in p) for p in chunk])
    return triplets_of(tri_map)


PATHS = [("batch", batch_triplets), ("incremental", incremental_triplets)]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_name,path", PATHS)
class TestAgainstScipy:
    """The properties that hold for both paths, stated against exact geometry."""

    N_TRIALS = 12
    N_SEEDS = 40

    def _trials(self, seed=7):
        rng = np.random.default_rng(seed)
        for _ in range(self.N_TRIALS):
            seeds = random_seeds(rng, self.N_SEEDS)
            if len(seeds) < self.N_SEEDS - 5:
                continue
            yield seeds, np.asarray(seeds, dtype=np.float64)

    def test_every_resolvable_triangle_is_found(self, path_name, path):
        """The main claim: nothing that *can* be seen is missed.

        Resolvable means margin >= MARGIN_PX (not near-cocircular) and a
        circumcentre inside the padded canvas. Neither excuse applies, so a
        miss here is a defect.
        """
        checked = 0
        for seeds, arr in self._trials():
            ref, _ = scipy_triangles(seeds)
            got = path(seeds)
            for t in ref:
                m, centre = margin_of(arr, t)
                if m < MARGIN_PX or not on_canvas(centre, PADDING):
                    continue
                checked += 1
                assert t in got, (
                    f"{path_name} missed a resolvable triangle {t} "
                    f"(margin {m:.2f}px, circumcentre "
                    f"({centre[0]:.1f}, {centre[1]:.1f}))\n"
                    f"seeds={[seeds[i] for i in t]}")
        assert checked > 500, (
            f"only {checked} triangles exercised -- the test has gone vacuous")

    def test_no_triangle_is_deeply_non_delaunay(self, path_name, path):
        """Nothing emitted may have a seed comfortably inside its circumcircle.

        Marginal violations are the unresolvable near-cocircular case and are
        allowed; a violation past MARGIN_PX is a real one.
        """
        for seeds, arr in self._trials():
            for t in path(seeds):
                m, _ = margin_of(arr, t)
                assert m > -MARGIN_PX, (
                    f"{path_name} emitted {t} with a rival seed {-m:.2f}px "
                    f"inside its circumcircle\nseeds={[seeds[i] for i in t]}")

    def test_no_duplicate_or_degenerate_triangles(self, path_name, path):
        for seeds, arr in self._trials():
            got = path(seeds)
            for t in got:
                assert len(set(t)) == 3, f"{path_name} emitted degenerate {t}"
                a, b, c = arr[list(t)]
                area = abs((b[0] - a[0]) * (c[1] - a[1])
                           - (b[1] - a[1]) * (c[0] - a[0])) / 2
                assert area > 0, f"{path_name} emitted collinear {t}"

    def test_triangles_do_not_overlap(self, path_name, path):
        """The output must be a triangulation, not just a bag of triangles.

        scipy cannot police this on its own. An exactly cocircular quad admits
        two triangulations, and the historical bug emitted *all four* of its
        triangles -- but each one is individually Delaunay, with the fourth
        point exactly on the circumcircle, so every per-triangle test passes.
        What fails is the global structure: the four overlap pairwise.

        Orientation catches it exactly. With every triangle wound
        counter-clockwise, each directed edge belongs to at most one triangle
        in a valid triangulation, because an interior edge is traversed once in
        each direction by the two triangles sharing it. The four triangles of a
        doubly-cut quad reuse all four of its boundary edges in the same
        direction, so the invariant fails immediately.
        """
        for seeds, arr in self._trials():
            bad = overlapping_pair(path(seeds), arr)
            assert bad is None, (
                f"{path_name} emitted overlapping triangles {bad[0]} and "
                f"{bad[1]}, which share directed edge {bad[2]}")

    def test_disagreement_is_always_explained(self, path_name, path):
        """No disagreement may fall outside the two documented limitations.

        This is the characterisation the module docstring rests on, kept
        executable so that a regression cannot quietly widen it.
        """
        unexplained = []
        for seeds, arr in self._trials():
            ref, hull = scipy_triangles(seeds)
            got = path(seeds)
            for t in (ref - got):
                m, centre = margin_of(arr, t)
                if m < MARGIN_PX or not on_canvas(centre, PADDING):
                    continue
                unexplained.append(("missing", t, m, [seeds[i] for i in t]))
            for t in (got - ref):
                m, _ = margin_of(arr, t)
                if m > -MARGIN_PX:
                    continue
                unexplained.append(("extra", t, m, [seeds[i] for i in t]))
        assert not unexplained, (
            f"{path_name}: {len(unexplained)} disagreement(s) with scipy that "
            f"neither padding nor near-cocircularity explains:\n"
            + "\n".join(f"  {k} {t} margin={m:.2f}px pts={p}"
                        for k, t, m, p in unexplained[:10]))


# ---------------------------------------------------------------------------
# Cases with a single right answer, where the oracle is unambiguous
# ---------------------------------------------------------------------------

class TestExactSmallCases:
    """Small configurations whose triangulation is not in doubt.

    Deliberately chosen away from cocircularity, so scipy and delauney must
    agree exactly and any disagreement is a defect rather than a tie.
    """

    @pytest.mark.parametrize("path_name,path", PATHS)
    @pytest.mark.parametrize("seeds", [
        [(60, 60), (190, 70), (120, 180)],
        [(50, 50), (200, 60), (180, 200), (70, 190)],
        [(40, 40), (200, 45), (120, 120), (60, 200), (210, 190)],
        [(80, 40), (170, 60), (60, 140), (190, 150), (120, 210), (128, 100)],
    ])
    def test_matches_scipy_exactly(self, path_name, path, seeds):
        # delauney's seed ids index its own (x asc, y asc) sorted ordering,
        # while scipy's index the array as passed. Sorting here makes the two
        # numberings the same one, so the sets can be compared directly.
        seeds = sorted(seeds)
        arr = np.asarray(seeds, dtype=np.float64)
        ref, _ = scipy_triangles(seeds)

        # The premise of this test: no near-cocircular triangle, so the
        # answer is unique and delauney has no excuse.
        for t in ref:
            m, _ = margin_of(arr, t)
            assert abs(m) > MARGIN_PX, (
                f"fixture {seeds} is near-cocircular at {t} (margin {m:.2f}px) "
                f"and cannot serve as an exact oracle case")

        assert path(seeds) == ref, (
            f"{path_name} disagrees with scipy on an unambiguous case\n"
            f"  missing: {sorted(ref - path(seeds))}\n"
            f"  extra:   {sorted(path(seeds) - ref)}")

    def test_the_two_paths_agree_with_each_other(self):
        """Kept as a cross-check, but no longer the primary evidence.

        Once the detection kernel is shared this is tautological; it stays
        because the surrounding bookkeeping still differs between the paths.
        """
        rng = np.random.default_rng(3)
        for _ in range(6):
            seeds = random_seeds(rng, 30)
            assert batch_triplets(seeds) == incremental_triplets(seeds)


class TestTheOverlapInvariantItself:
    """The overlap check is the one assertion here that scipy cannot back up.

    Every other test is anchored to an external oracle, so a mistake in it
    shows up as a failure against scipy. This one is self-contained, which
    means it could silently accept everything and no other test would notice.
    So it is checked against a triangle set built to be wrong in precisely the
    historical way, and against valid sets that it must not reject.
    """

    SQUARE = np.array([(0, 0), (10, 0), (0, 10), (10, 10)], dtype=np.float64)

    def test_rejects_a_doubly_cut_cocircular_quad(self):
        """The exact shape of the old bug: both diagonals registered.

        Four cocircular seeds, all four triangles emitted. Each one is
        individually Delaunay -- the fourth point lies exactly on its
        circumcircle -- so no per-triangle test can object. Only the overlap
        invariant can.
        """
        both_diagonals = {(0, 1, 2), (1, 2, 3), (0, 1, 3), (0, 2, 3)}
        assert overlapping_pair(both_diagonals, self.SQUARE) is not None

    def test_rejects_either_pair_plus_one_extra(self):
        for extra in ((0, 1, 3), (0, 2, 3)):
            triples = {(0, 1, 2), (1, 2, 3), extra}
            assert overlapping_pair(triples, self.SQUARE) is not None, (
                f"adding {extra} to a complete quad must be caught")

    @pytest.mark.parametrize("triples", [
        {(0, 1, 2), (1, 2, 3)},        # one diagonal
        {(0, 1, 3), (0, 2, 3)},        # the other diagonal
        {(0, 1, 2)},                   # a lone triangle
        set(),                         # nothing at all
    ])
    def test_accepts_valid_triangulations(self, triples):
        assert overlapping_pair(triples, self.SQUARE) is None

    def test_accepts_a_real_triangulation(self):
        """A genuine scipy triangulation must pass, whatever its winding."""
        rng = np.random.default_rng(5)
        for _ in range(5):
            seeds = random_seeds(rng, 30)
            arr = np.asarray(seeds, dtype=np.float64)
            ref, _ = scipy_triangles(seeds)
            assert overlapping_pair(ref, arr) is None


class TestCocircularQuads:
    """Exactly cocircular quads: two triangles, whichever diagonal is chosen.

    This is the configuration the historical tie-break bug got wrong, and the
    one integer seed coordinates produce constantly. scipy is not the oracle
    here -- both cuts are correct -- so the assertion is on the count and on
    non-overlap, not on which pair comes back.
    """

    QUADS = [
        [(60, 60), (180, 60), (60, 180), (180, 180)],      # square
        [(60, 80), (200, 80), (200, 170), (60, 170)],      # rectangle
        [(80, 128), (128, 80), (176, 128), (128, 176)],    # rotated square
    ]

    @pytest.mark.parametrize("path_name,path", PATHS)
    @pytest.mark.parametrize("quad", QUADS)
    def test_exactly_two_triangles(self, path_name, path, quad):
        quad = sorted(quad)
        arr = np.asarray(quad, dtype=np.float64)

        # Confirm the premise: all four points really are on one circle.
        centre, radius = circumcircle(arr[[0, 1, 2]])
        assert abs(np.linalg.norm(arr[3] - centre) - radius) < 1e-9, (
            f"fixture {quad} is not cocircular, so it tests nothing here")

        got = path(quad)
        assert len(got) == 2, (
            f"{path_name} cut the cocircular quad {quad} into {len(got)} "
            f"triangles instead of 2: {sorted(got)}")

        # And the two must share a diagonal rather than overlap.
        a, b = sorted(got)
        assert len(set(a) & set(b)) == 2, (
            f"{path_name} returned {sorted(got)}, which do not share an edge")


class TestIncrementalInsertionOrder:
    """Splitting the insert into batches must not change the answer.

    This is where an incremental-only defect would show, and scipy rather than
    the batch path decides who is wrong.
    """

    @pytest.mark.parametrize("batches", [1, 2, 5])
    def test_batching_does_not_change_the_triangulation(self, batches):
        rng = np.random.default_rng(11)
        for _ in range(5):
            seeds = random_seeds(rng, 40)
            arr = np.asarray(seeds, dtype=np.float64)
            got = incremental_triplets(seeds, batches=batches)
            ref, _ = scipy_triangles(seeds)
            for t in ref:
                m, centre = margin_of(arr, t)
                if m < MARGIN_PX or not on_canvas(centre, PADDING):
                    continue
                assert t in got, (
                    f"inserting in {batches} batches lost a resolvable "
                    f"triangle {t} (margin {m:.2f}px)")
