"""Border padding in Delaunay.

A triangle is registered where three Voronoi regions meet -- its circumcentre --
and boundary triangles frequently have circumcentres outside the image, so at
padding 0 they are never detected and the pixels they cover degrade to
nearest-seed with no error raised. The batch pipeline handles this by re-running
detection on a padded canvas; this class holds the padded canvas as its
persistent state instead.

The bar is the same as everywhere else in this suite: at a given padding, the
incremental result must equal the batch result. That is checked across padding
values rather than at one, because padding is the one parameter where the two
implementations arrive at the padded canvas by different routes -- the batch
path builds it fresh per call, the incremental path keeps it.
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
    from delauney import DEFAULT_BORDER_PADDING

W, H = 64, 64
MAX_SEEDS = 2048


def tri_set(m):
    if isinstance(m, np.ndarray):
        return frozenset(tuple(sorted(r)) for r in m.tolist())
    return frozenset(tuple(sorted([v[2], v[3], v[4]])) for v in m.values())


def scattered(n, seed, w=W, h=H):
    rng = np.random.default_rng(seed)
    pts = set()
    while len(pts) < n:
        pts.add((int(rng.integers(0, w)), int(rng.integers(0, h))))
    return [list(p) for p in sorted(pts)]


def batch(seeds, padding):
    vg = _cu.Voronoi().compute(W, H, seeds)
    tm, tg = _cu.GridTriangulation().compute(vg, seeds, padding)
    return tri_set(tm), tg


def incremental(seeds, padding, batches=1):
    inc = _cu.Delaunay(W, H, MAX_SEEDS, padding)
    step = max(1, len(seeds) // batches)
    for i in range(0, len(seeds), step):
        inc.insert_deferred(seeds[i:i + step])
    tm, tg = inc.finalise()
    return tri_set(tm), tg


def pixel_triplets(tgrid, tri_map):
    """Per-pixel canonical vertex triple, or None where no triangle covers it.

    Raw triangle ids are not comparable across the two implementations: each
    numbers triangles in its own registry order, so an identical triangulation
    still yields different id rasters. Translating through the triangle map
    compares what the raster means rather than how it is labelled.
    """
    lut = {tid: tuple(sorted([v[2], v[3], v[4]])) for tid, v in tri_map.items()}
    tg = np.asarray(tgrid)
    return np.array(
        [[lut.get(int(tg[y, x, 2])) if tg[y, x, 2] >= 0 else None
          for x in range(tg.shape[1])] for y in range(tg.shape[0])],
        dtype=object)


class TestPaddingMatchesBatch:

    @pytest.mark.parametrize("padding", [0, 1, 3, 8, 16])
    def test_triangles_match_batch_at_each_padding(self, padding):
        seeds = scattered(120, 0)
        i_set, _ = incremental(seeds, padding)
        b_set, _ = batch(seeds, padding)
        assert i_set == b_set

    @pytest.mark.parametrize("n", [20, 60, 200, 400])
    def test_matches_batch_across_densities(self, n):
        seeds = scattered(n, n)
        pad = DEFAULT_BORDER_PADDING
        i_set, _ = incremental(seeds, pad)
        b_set, _ = batch(seeds, pad)
        assert i_set == b_set

    def test_matches_batch_after_several_deferred_inserts(self):
        """Padding must survive incremental insertion, not just a first build."""
        seeds = scattered(250, 7)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, 6)
        for i in range(0, len(seeds), 50):
            inc.insert_deferred(seeds[i:i + 50])
        i_map, i_grid = inc.finalise()

        vg = _cu.Voronoi().compute(W, H, seeds)
        b_map, b_grid = _cu.GridTriangulation().compute(vg, seeds, 6)

        assert tri_set(i_map) == tri_set(b_map)

        # Pixel assignment can differ on shared boundaries, and does so
        # independently of padding: assign_triangles_kernel resolves a pixel
        # covered by several triangles by taking the highest triangle id, and
        # the two implementations number triangles in their own registry order.
        # point_in_triangle is non-strict, so a pixel lying exactly on a shared
        # edge belongs to both and the tie is broken differently. Measured at
        # 0 / 2 / 18 mismatches for 40 / 120 / 250 seeds, the same at padding 0
        # and 6. Harmless -- the two triangles agree along the edge they share,
        # so the interpolated value is the same either way -- but it means the
        # assertion has to be that every disagreement is such a tie.
        pts = np.asarray(seeds)
        a = pixel_triplets(i_grid, i_map)
        b = pixel_triplets(b_grid, b_map)
        ys, xs = np.nonzero(a != b)
        for y, x in zip(ys, xs):
            ta, tb = a[y, x], b[y, x]
            assert ta is not None and tb is not None, \
                f"pixel ({x},{y}) covered by one implementation only"
            assert len(set(ta) & set(tb)) == 2, \
                f"pixel ({x},{y}) assigned to non-adjacent triangles {ta} vs {tb}"
            def cross(u, v):
                return u[0] * v[1] - u[1] * v[0]

            for tri in (ta, tb):
                p, q, r = pts[list(tri)].astype(np.float64)
                w = np.array([x, y], dtype=np.float64)
                d = [cross(q - p, w - p), cross(r - q, w - q), cross(p - r, w - r)]
                assert not (min(d) < 0 < max(d)), \
                    f"pixel ({x},{y}) is not actually inside {tri}"


class TestPaddingRecoversTriangles:

    def test_padding_finds_border_triangles_that_zero_misses(self):
        seeds = scattered(200, 1)
        n0, _ = incremental(seeds, 0)
        n8, _ = incremental(seeds, 8)
        assert len(n8) > len(n0), "padding recovered no triangles"
        # It only ever adds: an unpadded run cannot see a triangle a padded one
        # misses, since the padded canvas contains the unpadded one.
        assert n0 <= n8

    def test_padded_pixels_are_better_covered(self):
        """Fewer pixels left without a containing triangle."""
        seeds = scattered(200, 2)
        _, g0 = incremental(seeds, 0)
        _, g8 = incremental(seeds, 8)
        unassigned0 = int((np.asarray(g0)[:, :, 2] < 0).sum())
        unassigned8 = int((np.asarray(g8)[:, :, 2] < 0).sum())
        assert unassigned8 < unassigned0


class TestPaddingApi:

    def test_output_shapes_are_image_sized(self):
        """The padded canvas is internal; outputs stay (H, W, *)."""
        seeds = scattered(80, 3)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, 9)
        inc.insert_deferred(seeds)
        _, tgrid = inc.finalise()
        assert np.asarray(tgrid).shape == (H, W, 3)
        assert np.asarray(inc.get_voronoi_grid()).shape == (H, W, 2)

    def test_default_padding_is_the_library_constant(self):
        """Not a function of max_seeds any more.

        The old default was a density estimate resolved against max_seeds, so a
        mesh holding far fewer seeds than its capacity was under-padded relative
        to what the batch path would pick for it. A constant removes that
        disagreement by construction.
        """
        inc = _cu.Delaunay(W, H, MAX_SEEDS)
        assert inc.border_padding == DEFAULT_BORDER_PADDING
        assert _cu.Delaunay(W, H, MAX_SEEDS * 8).border_padding             == DEFAULT_BORDER_PADDING

    def test_explicit_padding_is_reported(self):
        assert _cu.Delaunay(W, H, MAX_SEEDS, 5).border_padding == 5
        assert _cu.Delaunay(W, H, MAX_SEEDS, 0).border_padding == 0

    def test_seed_bounds_are_image_coordinates_not_padded(self):
        """Padding must not quietly widen the accepted coordinate range."""
        inc = _cu.Delaunay(W, H, MAX_SEEDS, 8)
        with pytest.raises(ValueError):
            inc.insert_deferred([[W, 0]])
        with pytest.raises(ValueError):
            inc.insert_deferred([[0, H]])
        with pytest.raises(ValueError):
            inc.insert_deferred([[-1, 0]])

    def test_voronoi_matches_batch_under_padding(self):
        """Padding changes detection, not the Voronoi diagram of the image."""
        seeds = scattered(100, 4)
        inc = _cu.Delaunay(W, H, MAX_SEEDS, 7)
        inc.insert_deferred(seeds)
        inc.finalise()
        vg_inc = np.asarray(inc.get_voronoi_grid())
        vg_batch = np.asarray(_cu.Voronoi().compute(W, H, seeds))
        np.testing.assert_array_equal(vg_inc[:, :, 1], vg_batch[:, :, 1])
