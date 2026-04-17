"""NumPy reference implementation of GridTriangulation."""
import numpy as np


class GridTriangulation:
    """Extracts an approximate Delaunay triangulation from a Voronoi grid.

    Scans the grid for left+down L-shapes where all three Voronoi regions
    differ, deduplicates by sorted seed-ID triplet, then assigns each cell
    to its containing triangle via a center-point test.

    Known limitation: 4-region meetings produce only 1 triangle (the one
    detected by the left+down scan) instead of the geometrically complete 2.
    """

    def compute(
        self,
        voronoi_grid: np.ndarray,
        seed_positions: list[tuple[int, int]],
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

        Returns
        -------
        triangle_map:
            dict mapping triangle_id (int) → (x, y, id_a, id_b, id_c).
        triangulation_grid:
            int32 array of shape (H, W, 3): channels (seed_id, distance, triangle_id).
        """
        seeds_arr = np.array(seed_positions, dtype=np.int32)
        order = np.lexsort((seeds_arr[:, 1], seeds_arr[:, 0]))
        sorted_seeds = seeds_arr[order].astype(np.float64)  # for geometric ops

        H, W = voronoi_grid.shape[:2]
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

        def _register(gx: int, gy: int, a: int, b: int, c: int) -> None:
            nonlocal next_id
            triplet = frozenset({a, b, c})
            if len(triplet) < 3:
                return
            if triplet not in seen_triplets:
                seen_triplets[triplet] = next_id
                triangle_map[next_id] = (gx, gy, a, b, c)
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

        # Step 4: assign each cell to a triangle via center-point test.
        #
        # We test ALL triangles for each pixel, not just those that share the
        # pixel's Voronoi seed_id.  Restricting to seed-id neighbours is an
        # incorrect optimisation: a pixel in Manhattan Voronoi region A can
        # geometrically lie inside a Euclidean triangle (B,C,D) where A is not
        # a vertex, because Manhattan and Euclidean Voronoi boundaries differ.
        #
        # Tie between multiple containing triangles → highest id wins.
        # Fallback for pixels outside the convex hull (no triangle contains the
        # center): assign the triangle with the globally highest id so that
        # every cell always has a valid assignment.

        all_tids = list(triangle_map.keys())
        max_tid_global = max(all_tids) if all_tids else -1

        tri_id_grid = np.full((H, W), -1, dtype=np.int32)

        for y in range(H):
            for x in range(W):
                cx = x + 0.5
                cy = y + 0.5

                best_tid = -1
                for tid, (_, _, a, b, c) in triangle_map.items():
                    if _point_in_triangle(cx, cy, sorted_seeds[a],
                                          sorted_seeds[b], sorted_seeds[c]):
                        if best_tid == -1 or tid > best_tid:
                            best_tid = tid

                tri_id_grid[y, x] = best_tid if best_tid != -1 else max_tid_global

        # Step 5: build output grid
        tgrid = np.empty((H, W, 3), dtype=np.int32)
        tgrid[:, :, 0] = voronoi_grid[:, :, 0]
        tgrid[:, :, 1] = voronoi_grid[:, :, 1]
        tgrid[:, :, 2] = tri_id_grid

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
