"""NumPy reference implementation of RegularDelaunay (Manhattan-distance Voronoi)."""
import numpy as np


_UNDEFINED = -1


class RegularDelaunay:
    """Computes a Manhattan-distance Voronoi diagram on an integer grid.

    Seed IDs are assigned in ascending X order, tiebroken by ascending Y.
    At equidistant cells the higher seed ID wins.
    """

    def compute(
        self,
        width: int,
        height: int,
        seeds: list[tuple[int, int]],
    ) -> np.ndarray:
        """Return a (height, width, 2) int32 array of (seed_id, distance) per cell.

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
        # grid[:,:,1] = distance
        grid = np.full((height, width, 2), _UNDEFINED, dtype=np.int32)

        for seed_id, (sx, sy) in enumerate(sorted_seeds):
            grid[sy, sx, 0] = seed_id
            grid[sy, sx, 1] = 0

        self._flood_fill(grid, height, width)
        return grid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flood_fill(grid: np.ndarray, H: int, W: int) -> None:
        """Iterative vectorised BFS until no cell changes."""
        seed_id = grid[:, :, 0]
        distance = grid[:, :, 1]

        while True:
            changed = False

            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                # Shift the grid to bring each neighbour into alignment
                if dy == 0 and dx == -1:   # neighbour from the right
                    n_id = np.full((H, W), _UNDEFINED, np.int32)
                    n_id[:, 1:] = seed_id[:, :-1]
                    n_d = np.full((H, W), 0, np.int32)
                    n_d[:, 1:] = distance[:, :-1]
                elif dy == 0 and dx == 1:  # neighbour from the left
                    n_id = np.full((H, W), _UNDEFINED, np.int32)
                    n_id[:, :-1] = seed_id[:, 1:]
                    n_d = np.full((H, W), 0, np.int32)
                    n_d[:, :-1] = distance[:, 1:]
                elif dy == -1 and dx == 0:  # neighbour from below
                    n_id = np.full((H, W), _UNDEFINED, np.int32)
                    n_id[1:, :] = seed_id[:-1, :]
                    n_d = np.full((H, W), 0, np.int32)
                    n_d[1:, :] = distance[:-1, :]
                else:                       # dy==1: neighbour from above
                    n_id = np.full((H, W), _UNDEFINED, np.int32)
                    n_id[:-1, :] = seed_id[1:, :]
                    n_d = np.full((H, W), 0, np.int32)
                    n_d[:-1, :] = distance[1:, :]

                neighbour_valid = n_id >= 0
                candidate_d = n_d + 1  # distance through this neighbour

                # A neighbour wins over the current cell when:
                #   1. current cell is undefined, OR
                #   2. candidate distance <  current distance, OR
                #   3. candidate distance == current distance AND neighbour id > current id
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
                    seed_id[update_mask] = n_id[update_mask]
                    distance[update_mask] = candidate_d[update_mask]

            if not changed:
                break
