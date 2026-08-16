"""NumPy reference implementation of Voronoi (L2-distance Voronoi)."""
import numpy as np


_UNDEFINED = -1


class Voronoi:
    """Computes an L2-distance (Euclidean) Voronoi diagram on an integer grid.

    Seed IDs are assigned in ascending X order, tiebroken by ascending Y.

    At equidistant cells propagation *prefers* the higher seed ID, but does not
    guarantee it: propagation is local, so a cell only sees seeds owned by its
    four neighbours, and a tied seed that never reaches the cell cannot win it.
    Both outcomes are correct nearest-seed assignments, so the CUDA path -- a
    different update schedule -- may resolve an individual tie differently.
    Do not assume bit-equality at equidistant cells; see
    tests/test_voronoi_cuda.py::test_large_grid.

    Locality has a sharper consequence for the CUDA path.  This sweep updates
    in place, four directions per pass, so a better seed can travel several
    cells per iteration and in practice it settles on the exact nearest seed.
    The CUDA kernel double-buffers and moves one cell per iteration, and can
    come to rest at a fixed point that is *not* nearest-seed: measured at
    128x128 with 400 seeds, 2 of 16384 pixels, off by 1-3 in squared distance.
    It is rare and grows with seed density.  Use this reference, not the CUDA
    path, when exactness matters.

    The distance channel stores squared L2 distance (avoids sqrt; int32 safe
    for grids up to ~46000x46000).
    """

    def compute(
        self,
        width: int,
        height: int,
        seeds: list[tuple[int, int]],
    ) -> np.ndarray:
        """Return a (height, width, 2) int32 array of (seed_id, sq_l2_dist) per cell.

        Parameters
        ----------
        width, height:
            Grid dimensions. X in [0, width), Y in [0, height).
        seeds:
            Sequence of (x, y) pixel coordinates.

        Raises
        ------
        ValueError
            If seeds is empty, contains duplicates, or any coordinate is out of bounds.
        """
        if not seeds:
            raise ValueError("seeds must not be empty")
        seeds_arr = np.array(seeds, dtype=np.int32)
        if seeds_arr.ndim != 2 or seeds_arr.shape[1] != 2:
            raise ValueError("each seed must be an (x, y) pair")
        if np.any(seeds_arr[:, 0] < 0) or np.any(seeds_arr[:, 0] >= width):
            raise ValueError("seed x coordinate out of bounds")
        if np.any(seeds_arr[:, 1] < 0) or np.any(seeds_arr[:, 1] >= height):
            raise ValueError("seed y coordinate out of bounds")

        # Assign IDs: sort by x ascending, then y ascending
        order = np.lexsort((seeds_arr[:, 1], seeds_arr[:, 0]))
        sorted_seeds = seeds_arr[order]

        # Check for duplicate positions after sorting
        if len(sorted_seeds) > 1:
            diffs = np.diff(sorted_seeds, axis=0)
            if np.any((diffs[:, 0] == 0) & (diffs[:, 1] == 0)):
                raise ValueError("duplicate seed positions are not allowed")

        # grid[:,:,0] = seed_id  (-1 = undefined)
        # grid[:,:,1] = squared L2 distance
        grid = np.full((height, width, 2), _UNDEFINED, dtype=np.int32)

        for seed_id, (sx, sy) in enumerate(sorted_seeds):
            grid[sy, sx, 0] = seed_id
            grid[sy, sx, 1] = 0

        self._flood_fill(grid, sorted_seeds, height, width)
        return grid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flood_fill(grid: np.ndarray, sorted_seeds: np.ndarray,
                    H: int, W: int) -> None:
        """Iterative vectorised BFS until no cell changes.

        Each step propagates seed identity one pixel in a cardinal direction.
        Distance is recomputed as direct squared L2 from the owning seed to the
        current pixel — never accumulated along the path.
        """
        seed_id = grid[:, :, 0]
        distance = grid[:, :, 1]

        # Track the position of the owning seed for each cell so that each
        # receiving cell can compute the direct L2 distance without a lookup.
        seed_x = np.zeros((H, W), dtype=np.int32)
        seed_y = np.zeros((H, W), dtype=np.int32)
        for sid, (sx, sy) in enumerate(sorted_seeds):
            seed_x[sy, sx] = sx
            seed_y[sy, sx] = sy

        # Pixel coordinate grids for direct L2 distance computation
        px = np.arange(W, dtype=np.int32)[np.newaxis, :]   # (1, W)
        py = np.arange(H, dtype=np.int32)[:, np.newaxis]   # (H, 1)

        while True:
            changed = False

            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                n_id = np.full((H, W), _UNDEFINED, dtype=np.int32)
                n_sx = np.zeros((H, W), dtype=np.int32)
                n_sy = np.zeros((H, W), dtype=np.int32)

                if dy == 0 and dx == -1:   # neighbour from the right
                    n_id[:, 1:]  = seed_id[:, :-1]
                    n_sx[:, 1:]  = seed_x[:, :-1]
                    n_sy[:, 1:]  = seed_y[:, :-1]
                elif dy == 0 and dx == 1:  # neighbour from the left
                    n_id[:, :-1] = seed_id[:, 1:]
                    n_sx[:, :-1] = seed_x[:, 1:]
                    n_sy[:, :-1] = seed_y[:, 1:]
                elif dy == -1 and dx == 0: # neighbour from below
                    n_id[1:, :]  = seed_id[:-1, :]
                    n_sx[1:, :]  = seed_x[:-1, :]
                    n_sy[1:, :]  = seed_y[:-1, :]
                else:                      # neighbour from above
                    n_id[:-1, :] = seed_id[1:, :]
                    n_sx[:-1, :] = seed_x[1:, :]
                    n_sy[:-1, :] = seed_y[1:, :]

                neighbour_valid = n_id >= 0

                # Direct squared L2 distance from neighbour's seed to current pixel
                candidate_d = (px - n_sx) ** 2 + (py - n_sy) ** 2

                undefined = seed_id == _UNDEFINED
                lower_d = neighbour_valid & ~undefined & (candidate_d < distance)
                tie_win = (
                    neighbour_valid
                    & ~undefined
                    & (candidate_d == distance)
                    & (n_id > seed_id)
                )
                fill_new = neighbour_valid & undefined

                update_mask = fill_new | lower_d | tie_win

                if np.any(update_mask):
                    changed = True
                    seed_id[update_mask]  = n_id[update_mask]
                    distance[update_mask] = candidate_d[update_mask]
                    seed_x[update_mask]   = n_sx[update_mask]
                    seed_y[update_mask]   = n_sy[update_mask]

            if not changed:
                break
