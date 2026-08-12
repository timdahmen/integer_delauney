"""NumPy reference implementation of GridTriangulation."""
import numpy as np
from delauney.reference.voronoi import RegularDelaunay as _RefVoronoi


class GridTriangulation:
    """Extracts an approximate Delaunay triangulation from a Voronoi grid.

    Scans the grid for left+down L-shapes where all three Voronoi regions
    differ, deduplicates by sorted seed-ID triplet, then assigns each cell
    to its containing triangle by testing the pixel's integer coordinate.

    Pixel convention: a pixel (x, y) is represented by the point (x, y), NOT by
    its centre (x+0.5, y+0.5).  This matches the CUDA implementation
    (``triangulation.cu``, ``assign_triangles_kernel``) and, critically, matches
    where seeds live: a seed at (5, 7) means pixel (5, 7) was sampled, so
    evaluating there reproduces the measured value exactly.  Under a centre
    convention an interpolator evaluating at integer coordinates disagrees with
    the measurement at the very pixels that were sampled.

    Pixels that no triangle contains keep the sentinel -1 in the triangle_id
    channel, so callers can tell "no containing triangle" from a real triangle.

    Known limitations, both of which leave pixels with no containing triangle:
      * 4-region meetings produce only 1 triangle (the one detected by the
        left+down scan) instead of the geometrically complete 2;
      * a triangle is only detected where three Voronoi regions meet, i.e. at
        its circumcenter, so boundary triangles whose circumcenter falls outside
        the image are missed entirely unless ``border_padding`` exposes them.
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
            int32 array of shape (H, W, 2) produced by RegularDelaunay.
        seed_positions:
            Seed coordinates in the same order as used to produce voronoi_grid
            (i.e. index == seed_id after the x-then-y sort applied internally
            by RegularDelaunay).  Pass the *original* seed list; this method
            re-applies the same sort so that index == seed_id.
        border_padding:
            Extend the Voronoi by this many pixels in each direction before
            triangle detection so that border triangles whose Voronoi vertex
            lies outside the original image are not missed.  The output grid
            is always the original (H, W, 3) resolution.

            None (the default) selects a value scaled to seed density via
            ``delauney.auto_border_padding``.  Pass 0 to disable padding
            explicitly — note that 0 means boundary triangles whose
            circumcenter falls outside the image are never detected, and their
            pixels degrade to nearest-seed with no error raised.
        as_arrays:
            Return the triangle vertex indices as an (N_tri, 3) int32 array
            instead of a {tid: (x, y, id_a, id_b, id_c)} dict.  Consumers doing
            array work convert the dict straight back into this form, so the
            dict is pure overhead for them.  The (x, y) detection pixel is not
            included in the array form; ask for the dict if you need it.
            Mirrors the CUDA path's as_arrays option.

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

        # Build the grid used for triangle detection.  With border_padding > 0
        # we run a fresh Voronoi on a larger canvas so that Voronoi vertices
        # outside the original image become visible and border triangles are
        # detected.  Shifting all seeds by (P, P) preserves lexicographic
        # order, so seed IDs stay consistent with sorted_seeds.
        if border_padding is None:
            from delauney import auto_border_padding
            border_padding = auto_border_padding(W, H, len(sorted_seeds))
        P = max(0, int(border_padding))
        if P > 0:
            shifted = [(int(x) + P, int(y) + P) for x, y in sorted_seeds]
            pad_vgrid = _RefVoronoi().compute(W + 2 * P, H + 2 * P, shifted)
            n_grid = pad_vgrid[:, :, 0]
        else:
            n_grid = voronoi_grid[:, :, 0]  # seed_id per cell

        # Step 2: scan all 4 L-shape orientations so that every triple-region
        # meeting is found regardless of which diagonal it sits on.
        # Deduplication by sorted seed-ID triplet keeps exactly one entry per
        # geometric triangle.
        #
        # The four L-shapes at the corner of a 2×2 block (TL,TR,BL,BR):
        #   left+down  at TR → (TL, TR, BR)
        #   right+down at TL → (TL, TR, BL)
        #   right+up   at BL → (TL, BL, BR)
        #   left+up    at BR → (TR, BL, BR)
        seen_triplets: dict[frozenset, int] = {}

        triangle_map: dict[int, tuple] = {}
        next_id = 0

        # Step 2a: cocircular (4-region) meetings.
        #
        # A 2x2 block whose four cells belong to four *different* Voronoi
        # regions is a degree-4 Voronoi vertex — i.e. four cocircular seeds.
        # Such a quad has two equally valid Delaunay triangulations (one per
        # diagonal), but the four L-shape orientations below would each report a
        # different triple and register ALL FOUR, which is over-complete: it
        # yields overlapping triangles and a pixel can then land in several at
        # once.  Pick one diagonal and blacklist the other's two triples.
        #
        # Choice rule (deterministic, and matching the CUDA implementation):
        # the shorter diagonal wins; on an exact tie — which is the usual case,
        # since the points are cocircular — take the diagonal that does *not*
        # contain the lowest seed id.  The lowest id lies in exactly one
        # diagonal, so this is always well defined.
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

        # left+down: cell=(x,y), left=(x-1,y), down=(x,y+1)
        # slice: cell=n_grid[:-1,1:], left=n_grid[:-1,:-1], down=n_grid[1:,1:]
        s_cell = n_grid[:-1, 1:]
        s_left = n_grid[:-1, :-1]
        s_down = n_grid[1:, 1:]
        mask = (s_cell != s_left) & (s_cell != s_down) & (s_left != s_down)
        for ry, rx in zip(*np.where(mask)):
            _register(rx + 1, ry, int(s_left[ry, rx]), int(s_cell[ry, rx]), int(s_down[ry, rx]))

        # right+down: cell=(x,y), right=(x+1,y), down=(x,y+1)
        # slice: cell=n_grid[:-1,:-1], right=n_grid[:-1,1:], down=n_grid[1:,:-1]
        s_cell = n_grid[:-1, :-1]
        s_right = n_grid[:-1, 1:]
        s_down = n_grid[1:, :-1]
        mask = (s_cell != s_right) & (s_cell != s_down) & (s_right != s_down)
        for ry, rx in zip(*np.where(mask)):
            _register(rx, ry, int(s_cell[ry, rx]), int(s_right[ry, rx]), int(s_down[ry, rx]))

        # right+up: cell=(x,y), right=(x+1,y), up=(x,y-1)
        # slice: cell=n_grid[1:,:-1], right=n_grid[1:,1:], up=n_grid[:-1,:-1]
        s_cell = n_grid[1:, :-1]
        s_right = n_grid[1:, 1:]
        s_up = n_grid[:-1, :-1]
        mask = (s_cell != s_right) & (s_cell != s_up) & (s_right != s_up)
        for ry, rx in zip(*np.where(mask)):
            _register(rx, ry + 1, int(s_cell[ry, rx]), int(s_right[ry, rx]), int(s_up[ry, rx]))

        # left+up: cell=(x,y), left=(x-1,y), up=(x,y-1)
        # slice: cell=n_grid[1:,1:], left=n_grid[1:,:-1], up=n_grid[:-1,1:]
        s_cell = n_grid[1:, 1:]
        s_left = n_grid[1:, :-1]
        s_up = n_grid[:-1, 1:]
        mask = (s_cell != s_left) & (s_cell != s_up) & (s_left != s_up)
        for ry, rx in zip(*np.where(mask)):
            _register(rx + 1, ry + 1, int(s_left[ry, rx]), int(s_cell[ry, rx]), int(s_up[ry, rx]))

        # Step 4: assign each cell to a triangle by testing its integer
        # coordinate (x, y) — not its centre.  See the class docstring: this
        # matches the CUDA kernel and keeps seeds, assignment and downstream
        # evaluation on one convention.
        #
        # We test ALL triangles for each pixel, not just those that share the
        # pixel's Voronoi seed_id.  Restricting to seed-id neighbours is an
        # incorrect optimisation: a pixel in Voronoi region A can geometrically
        # lie inside a triangle (B,C,D) where A is not a vertex.
        #
        # Tie between multiple containing triangles → highest id wins.
        # Pixels no triangle contains keep -1.  There is deliberately no
        # "fold into the highest id" fallback: that would make "outside the
        # convex hull" indistinguishable from "inside the hull but its triangle
        # was never detected" (see the limitations in the class docstring), and
        # both need the same nearest-seed handling downstream anyway.

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

        # Step 5: build output grid
        tgrid = np.empty((H, W, 3), dtype=np.int32)
        tgrid[:, :, 0] = voronoi_grid[:, :, 0]
        tgrid[:, :, 1] = voronoi_grid[:, :, 1]
        tgrid[:, :, 2] = tri_id_grid

        if as_arrays:
            # triangle_map keys are a dense 0..N-1 range by construction
            # (assigned by next_id in _register), so ordering by key reproduces
            # the same tid -> row correspondence the grid channel refers to.
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
