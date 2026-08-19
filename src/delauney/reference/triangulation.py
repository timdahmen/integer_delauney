"""NumPy reference implementation of GridTriangulation."""
import numpy as np
from delauney.reference.voronoi import Voronoi as _RefVoronoi


class GridTriangulation:
    """Extracts an approximate Delaunay triangulation from a Voronoi grid.

    Scans the grid for L-shapes of three differing Voronoi regions in all four
    orientations, deduplicates by sorted seed-ID triplet, then assigns each
    cell to its containing triangle by testing the pixel's integer coordinate.

    Pixel convention: pixel (x, y) is the point (x, y), not its centre
    (x+0.5, y+0.5).  This matches the CUDA kernel and where seeds live, so
    evaluating at a seed's coordinate reproduces the measured value exactly.

    Pixels that no triangle contains keep the sentinel -1 in the triangle_id
    channel, so callers can tell "no containing triangle" from a real triangle.

    Known limitation: a triangle is only detected where three Voronoi regions
    meet, i.e. at its circumcenter, so a triangle whose circumcenter falls
    outside the (optionally padded) canvas is missed and its pixels stay -1.
    See ``border_padding``.
    """

    def compute(
        self,
        voronoi_grid: np.ndarray,
        seed_positions: list[tuple[int, int]],
        border_padding: int | None = None,
        as_arrays: bool = False,
    ) -> tuple[dict, np.ndarray]:
        """Return (triangle_map, triangulation_grid).

        Parameters
        ----------
        voronoi_grid:
            int32 array of shape (H, W, 2) produced by Voronoi.
        seed_positions:
            Seed coordinates in the same order as used to produce voronoi_grid
            (i.e. index == seed_id after the x-then-y sort applied internally
            by Voronoi).  Pass the *original* seed list; this method
            re-applies the same sort so that index == seed_id.
        border_padding:
            Pixels of Voronoi canvas to add on each side before triangle
            detection, exposing border triangles whose circumcenter lies
            outside the image.  None (the default) uses the fixed
            ``delauney.DEFAULT_BORDER_PADDING`` -- see its docstring for why a
            density-based estimate would be the wrong quantity; 0 disables
            padding.  The output grid is always the original (H, W, 3)
            resolution.
        as_arrays:
            Return the vertex indices as an (N_tri, 3) int32 array instead of
            a {tid: (x, y, id_a, id_b, id_c)} dict, skipping the per-triangle
            Python objects.  The (x, y) detection pixel is not included; ask
            for the dict if you need it.

        Returns
        -------
        triangle_map:
            dict mapping triangle_id (int) → (x, y, id_a, id_b, id_c), or an
            (N_tri, 3) int32 array of vertex indices when as_arrays is set.
        triangulation_grid:
            int32 array of shape (H, W, 3): channels (seed_id, distance, triangle_id).
        """
        seeds_arr = np.array(seed_positions, dtype=np.int32)
        order = np.lexsort((seeds_arr[:, 1], seeds_arr[:, 0]))
        sorted_seeds = seeds_arr[order].astype(np.float64)  # for geometric ops

        H, W = voronoi_grid.shape[:2]

        # Detection grid.  With padding we rerun the Voronoi on a larger canvas
        # so that vertices outside the image become visible.  Shifting seeds by
        # (P, P) preserves lexicographic order, so IDs match sorted_seeds.
        if border_padding is None:
            from delauney import DEFAULT_BORDER_PADDING
            border_padding = DEFAULT_BORDER_PADDING
        P = max(0, int(border_padding))
        if P > 0:
            shifted = [(int(x) + P, int(y) + P) for x, y in sorted_seeds]
            pad_vgrid = _RefVoronoi().compute(W + 2 * P, H + 2 * P, shifted)
            n_grid = pad_vgrid[:, :, 0]
        else:
            n_grid = voronoi_grid[:, :, 0]

        # Scan all 4 L-shape orientations so that every triple-region meeting
        # is found regardless of which diagonal it sits on.  Deduplication by
        # sorted seed-ID triplet keeps exactly one entry per geometric triangle.
        seen_triplets: dict[frozenset, int] = {}

        triangle_map: dict[int, tuple] = {}
        next_id = 0

        # Cocircular (4-region) meetings.
        #
        # A 2x2 block of four *different* Voronoi regions is a degree-4 Voronoi
        # vertex: four cocircular seeds, with two equally valid triangulations
        # (one per diagonal).  The four L-shape scans below would each report a
        # different triple and register all four, which overlaps — a pixel then
        # lands in several triangles at once.  Pick one diagonal and blacklist
        # the other's two triples.
        #
        # Rule: the shorter diagonal wins; on an exact tie — the usual case,
        # since the points are cocircular — take the diagonal that does *not*
        # contain the lowest seed id, which is always well defined.
        #
        # triangulation.cu implements the same rule, so given the same Voronoi
        # grid both paths cut a quad identically; keep them in step.
        blocked: set[frozenset] = set()
        _tl, _tr = n_grid[:-1, :-1], n_grid[:-1, 1:]
        _bl, _br = n_grid[1:, :-1], n_grid[1:, 1:]
        _quad = ((_tl != _tr) & (_tl != _bl) & (_tl != _br)
                 & (_tr != _bl) & (_tr != _br) & (_bl != _br))

        for ry, rx in zip(*np.where(_quad)):
            ids = [int(_tl[ry, rx]), int(_tr[ry, rx]),
                   int(_bl[ry, rx]), int(_br[ry, rx])]
            pts = np.array([sorted_seeds[i] for i in ids], dtype=np.float64)

            # Cyclic order around the centroid so that "diagonal" means
            # opposite corners rather than adjacent ones.
            centre = pts.mean(axis=0)
            order = sorted(range(4), key=lambda k: np.arctan2(pts[k, 1] - centre[1],
                                                             pts[k, 0] - centre[0]))
            d0 = (order[0], order[2])
            d1 = (order[1], order[3])

            def _len2(pair):
                p, q = pts[pair[0]], pts[pair[1]]
                return float((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

            if _len2(d0) < _len2(d1):
                chosen = d0
            elif _len2(d1) < _len2(d0):
                chosen = d1
            else:
                lowest = min(range(4), key=lambda k: ids[k])
                chosen = d1 if lowest in d0 else d0

            # The rejected diagonal (r, s) spans the two triangles {r,s,p} and
            # {r,s,q}, where (p, q) is the chosen diagonal — block exactly those.
            rejected = d1 if chosen is d0 else d0
            for k in chosen:
                blocked.add(frozenset({ids[rejected[0]], ids[rejected[1]], ids[k]}))

        def _register(gx: int, gy: int, a: int, b: int, c: int) -> None:
            nonlocal next_id
            triplet = frozenset({a, b, c})
            if len(triplet) < 3:
                return
            if triplet in blocked:
                return
            if triplet not in seen_triplets:
                seen_triplets[triplet] = next_id
                # Shift detection pixel back to original coordinate space.
                # Border triangles will have gx or gy outside [0,W-1]/[0,H-1].
                triangle_map[next_id] = (gx - P, gy - P, a, b, c)
                next_id += 1

        def _scan_lshape(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                        gx_offset: int, gy_offset: int) -> None:
            """Register a triangle at every position where a, b, c differ pairwise.

            The four call sites below differ only in which three corners of the
            2x2 block they compare and where the detection pixel lands -- the
            comparison and registration are the same at each.
            """
            mask = (a != b) & (a != c) & (b != c)
            for ry, rx in zip(*np.where(mask)):
                _register(rx + gx_offset, ry + gy_offset,
                          int(a[ry, rx]), int(b[ry, rx]), int(c[ry, rx]))

        # left+down: cell=(x,y), left=(x-1,y), down=(x,y+1)
        _scan_lshape(n_grid[:-1, :-1], n_grid[:-1, 1:], n_grid[1:, 1:], 1, 0)

        # right+down: cell=(x,y), right=(x+1,y), down=(x,y+1)
        _scan_lshape(n_grid[:-1, :-1], n_grid[:-1, 1:], n_grid[1:, :-1], 0, 0)

        # right+up: cell=(x,y), right=(x+1,y), up=(x,y-1)
        _scan_lshape(n_grid[1:, :-1], n_grid[1:, 1:], n_grid[:-1, :-1], 0, 1)

        # left+up: cell=(x,y), left=(x-1,y), up=(x,y-1)
        _scan_lshape(n_grid[1:, :-1], n_grid[1:, 1:], n_grid[:-1, 1:], 1, 1)

        # Assign each cell by testing its integer coordinate (see class
        # docstring).  Every triangle is tested, not just those sharing the
        # pixel's Voronoi seed_id: a pixel in region A can lie inside a
        # triangle (B,C,D) that A is not a vertex of.
        #
        # Ties go to the highest id; uncontained pixels keep -1 rather than
        # folding into some nearest triangle, so callers can distinguish them.

        tri_id_grid = np.full((H, W), -1, dtype=np.int32)

        for y in range(H):
            for x in range(W):
                best_tid = -1
                for tid, (_, _, a, b, c) in triangle_map.items():
                    if _point_in_triangle(x, y, sorted_seeds[a],
                                          sorted_seeds[b], sorted_seeds[c]):
                        if best_tid == -1 or tid > best_tid:
                            best_tid = tid

                tri_id_grid[y, x] = best_tid

        tgrid = np.empty((H, W, 3), dtype=np.int32)
        tgrid[:, :, 0] = voronoi_grid[:, :, 0]
        tgrid[:, :, 1] = voronoi_grid[:, :, 1]
        tgrid[:, :, 2] = tri_id_grid

        if as_arrays:
            # Keys are a dense 0..N-1 range by construction, so row index == tid
            # == the value in the grid's triangle_id channel.
            verts = np.empty((len(triangle_map), 3), dtype=np.int32)
            for tid, (_x, _y, a, b, c) in triangle_map.items():
                verts[tid] = (a, b, c)
            return verts, tgrid

        return triangle_map, tgrid


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cross2d(ox, oy, ax, ay, bx, by) -> float:
    """Signed area × 2 of triangle OAB (positive = counter-clockwise)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _point_in_triangle(
    px: float,
    py: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    """Return True if (px, py) is inside or on the boundary of triangle ABC.

    Uses the cross-product sign test; works for both CW and CCW vertex order.
    Seed coordinates are (x, y) with positive Y downward (screen space).
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    d1 = _cross2d(px, py, ax, ay, bx, by)
    d2 = _cross2d(px, py, bx, by, cx, cy)
    d3 = _cross2d(px, py, cx, cy, ax, ay)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)
