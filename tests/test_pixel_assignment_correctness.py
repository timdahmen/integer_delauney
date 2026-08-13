"""Correctness test: every pixel must be assigned to the triangle that
geometrically contains its integer coordinate (x, y).

Ground truth is computed by testing each pixel against EVERY triangle
(brute-force, O(H*W*T)) rather than restricting to triangles that share the
pixel's Voronoi seed_id.  When a pixel lies inside multiple triangles (on a
shared edge) the highest triangle_id wins, matching the implementation.
Pixels no triangle contains are exempt from the assertion.
"""
import numpy as np
import pytest
from delauney.reference.voronoi import RegularDelaunay
from delauney.reference.triangulation import GridTriangulation, _point_in_triangle


def _ground_truth(W, H, tri_map, sorted_seeds):
    """Return an (H, W) int32 array of correct triangle_ids.

    -1 means the pixel lies outside all triangles (no expectation).
    """
    gt = np.full((H, W), -1, dtype=np.int32)
    seed_arr = np.array(sorted_seeds, dtype=np.float64)

    for y in range(H):
        for x in range(W):
            best = -1
            for tid, (_, _, a, b, c) in tri_map.items():
                if _point_in_triangle(x, y, seed_arr[a], seed_arr[b], seed_arr[c]):
                    if best == -1 or tid > best:
                        best = tid
            gt[y, x] = best
    return gt


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    # (name, W, H, seeds)
    (
        "3-seeds-triangle",
        30, 30,
        [(3, 3), (25, 5), (14, 25)],
    ),
    (
        "4-seeds-quad",
        30, 30,
        [(3, 3), (26, 3), (3, 26), (26, 26)],
    ),
    (
        "grid-9-seeds",
        40, 40,
        [(5, 5), (20, 3), (35, 5),
         (3, 20), (20, 20), (37, 20),
         (5, 35), (20, 37), (35, 35)],
    ),
    (
        "random-15-seeds",
        50, 50,
        # fixed positions so the test is deterministic
        [(4, 7), (12, 2), (23, 5), (38, 3), (46, 8),
         (2, 18), (15, 22), (30, 19), (44, 21), (8, 33),
         (22, 35), (36, 30), (47, 38), (10, 45), (40, 44)],
    ),
]


@pytest.mark.parametrize("name,W,H,seeds", CASES, ids=[c[0] for c in CASES])
def test_every_pixel_in_correct_triangle(name, W, H, seeds):
    vgrid = RegularDelaunay().compute(W, H, seeds)
    tri_map, tgrid = GridTriangulation().compute(vgrid, seeds)

    if not tri_map:
        pytest.skip("no triangles found (degenerate seed arrangement)")

    sorted_seeds = sorted(seeds, key=lambda s: (s[0], s[1]))
    gt = _ground_truth(W, H, tri_map, sorted_seeds)

    assigned = tgrid[:, :, 2]

    # Only check pixels that are actually inside a triangle (gt >= 0)
    inside_mask = gt >= 0
    n_inside = int(inside_mask.sum())
    wrong_mask = inside_mask & (assigned != gt)
    n_wrong = int(wrong_mask.sum())

    if n_wrong:
        ys, xs = np.where(wrong_mask)
        examples = [
            f"  pixel ({xs[i]},{ys[i]}): assigned={assigned[ys[i],xs[i]]}"
            f"  expected={gt[ys[i],xs[i]]}"
            f"  voronoi_seed={vgrid[ys[i],xs[i],0]}"
            for i in range(min(8, len(xs)))
        ]
        pytest.fail(
            f"[{name}] {n_wrong}/{n_inside} inside-pixels have wrong triangle.\n"
            + "\n".join(examples)
        )
