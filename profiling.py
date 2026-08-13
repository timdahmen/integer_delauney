"""Performance profiling -- 1024x1024 grid, 100 000 seed points.

Run with:
    python profiling.py

Sections
--------
1. Seed generation
2. RegularDelaunay   -- NumPy reference  (BFS only)
3. RegularDelaunay   -- CUDA
4. GridTriangulation -- NumPy reference  (detection + dedup; assignment skipped
                        because T triangles x W*H pixels is billions of
                        containment tests, i.e. hours in a Python loop)
5. GridTriangulation -- CUDA             (full pipeline)
6. IncrementalDelaunay -- CUDA           (cold insert + warm single-seed insert)
7. Summary table
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: dict[str, float] = {}


@contextmanager
def timer(label: str, store: bool = True):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if store:
        _results[label] = elapsed
    print(f"    {label:<52} {elapsed * 1000:>10.1f} ms")


def section(title: str) -> None:
    print(f"\n  {'-'*62}")
    print(f"  {title}")
    print(f"  {'-'*62}")


def hr() -> None:
    print(f"\n  {'='*68}")


# ---------------------------------------------------------------------------
# CUDA availability
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        from delauney import _delauney_cuda  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Reference triangle detection, without the O(H*W*T) assignment loop
# ---------------------------------------------------------------------------

def _ref_detect_triangles(voronoi_grid: np.ndarray,
                           seeds: list[tuple[int, int]]) -> dict:
    """Detect + deduplicate triangles only, skipping pixel assignment.

    Timing-only stand-in for the reference: it omits the cocircular-quad
    blocking, so at a degree-4 Voronoi vertex it registers all four triples
    where GridTriangulation keeps two.  Counts are an upper bound, not a match.
    """
    seeds_arr = np.array(seeds, dtype=np.int32)
    n_grid = voronoi_grid[:, :, 0]

    seen: dict[frozenset, int] = {}
    tri_map: dict[int, tuple] = {}
    next_id = 0

    def register(gx: int, gy: int, a: int, b: int, c: int) -> None:
        nonlocal next_id
        triplet = frozenset({a, b, c})
        if len(triplet) < 3:
            return
        if triplet not in seen:
            seen[triplet] = next_id
            tri_map[next_id] = (gx, gy, a, b, c)
            next_id += 1

    for s_cell, s_nb, s_dn, ox, oy in [
        (n_grid[:-1, 1:],  n_grid[:-1, :-1], n_grid[1:, 1:],  1, 0),
        (n_grid[:-1, :-1], n_grid[:-1, 1:],  n_grid[1:, :-1], 0, 0),
        (n_grid[1:,  :-1], n_grid[1:,  1:],  n_grid[:-1,:-1], 0, 1),
        (n_grid[1:,  1:],  n_grid[1:,  :-1], n_grid[:-1, 1:], 1, 1),
    ]:
        mask = (s_cell != s_nb) & (s_cell != s_dn) & (s_nb != s_dn)
        for ry, rx in zip(*np.where(mask)):
            register(rx + ox, ry + oy,
                     int(s_nb[ry, rx]), int(s_cell[ry, rx]), int(s_dn[ry, rx]))

    return tri_map


# ---------------------------------------------------------------------------
# BFS with iteration counter
# ---------------------------------------------------------------------------

def _ref_voronoi_timed(W: int, H: int, seeds: list) -> tuple[np.ndarray, int]:
    """Return (voronoi_grid, n_bfs_iterations).

    Mirrors reference.voronoi._flood_fill -- squared L2 recomputed from the
    owning seed, never accumulated along the path -- plus an iteration counter.
    """
    from delauney.reference.voronoi import _UNDEFINED

    seeds_arr = np.array(seeds, dtype=np.int32)
    order = np.lexsort((seeds_arr[:, 1], seeds_arr[:, 0]))
    sorted_seeds = seeds_arr[order]

    grid = np.full((H, W, 2), _UNDEFINED, dtype=np.int32)
    seed_x = np.zeros((H, W), dtype=np.int32)
    seed_y = np.zeros((H, W), dtype=np.int32)
    for seed_id, (sx, sy) in enumerate(sorted_seeds):
        grid[sy, sx, 0] = seed_id
        grid[sy, sx, 1] = 0
        seed_x[sy, sx] = sx
        seed_y[sy, sx] = sy

    sid = grid[:, :, 0]
    dst = grid[:, :, 1]
    px = np.arange(W, dtype=np.int32)[np.newaxis, :]
    py = np.arange(H, dtype=np.int32)[:, np.newaxis]
    iters = 0

    while True:
        iters += 1
        changed = False
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            n_id = np.full((H, W), _UNDEFINED, dtype=np.int32)
            n_sx = np.zeros((H, W), dtype=np.int32)
            n_sy = np.zeros((H, W), dtype=np.int32)

            if dx == -1:
                n_id[:, 1:],  n_sx[:, 1:],  n_sy[:, 1:]  = sid[:, :-1], seed_x[:, :-1], seed_y[:, :-1]
            elif dx == 1:
                n_id[:, :-1], n_sx[:, :-1], n_sy[:, :-1] = sid[:, 1:],  seed_x[:, 1:],  seed_y[:, 1:]
            elif dy == -1:
                n_id[1:, :],  n_sx[1:, :],  n_sy[1:, :]  = sid[:-1, :], seed_x[:-1, :], seed_y[:-1, :]
            else:
                n_id[:-1, :], n_sx[:-1, :], n_sy[:-1, :] = sid[1:, :],  seed_x[1:, :],  seed_y[1:, :]

            valid = n_id >= 0
            cand  = (px - n_sx) ** 2 + (py - n_sy) ** 2
            undef = sid == _UNDEFINED
            lower = valid & ~undef & (cand < dst)
            tie   = valid & ~undef & (cand == dst) & (n_id > sid)
            new   = valid & undef
            mask  = new | lower | tie
            if np.any(mask):
                changed = True
                sid[mask]    = n_id[mask]
                dst[mask]    = cand[mask]
                seed_x[mask] = n_sx[mask]
                seed_y[mask] = n_sy[mask]

        if not changed:
            break

    return grid, iters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

W, H, N = 1024, 1024, 100_000

hr()
print(f"\n  Delauney profiling  |  grid {W}x{H}  |  {N:,} seed points")
hr()

# -- 1. Seed generation ------------------------------------------------------
section("1. Seed generation")
with timer("Generate unique positions"):
    rng = np.random.default_rng(42)
    coords: set[tuple[int, int]] = set()
    while len(coords) < N:
        xs = rng.integers(0, W, N).tolist()
        ys = rng.integers(0, H, N).tolist()
        coords.update(zip(xs, ys))
    seeds = list(coords)[:N]

# -- 2. RegularDelaunay -- reference -----------------------------------------
section("2. RegularDelaunay -- NumPy reference")
with timer("compute()  (full BFS)"):
    ref_vgrid, n_iter = _ref_voronoi_timed(W, H, seeds)
print(f"    {'BFS convergence iterations':<52} {n_iter:>10}")
print(f"    {'Max squared L2 distance in grid':<52} {int(ref_vgrid[:,:,1].max()):>10}")

# -- 3. RegularDelaunay -- CUDA ----------------------------------------------
section("3. RegularDelaunay -- CUDA")
if cuda_available():
    from delauney._delauney_cuda import RegularDelaunay as CudaVD
    cuda_vd = CudaVD()
    _ = cuda_vd.compute(32, 32, [(0, 0), (31, 31)])   # warm-up
    with timer("compute()  (warm-up excluded)"):
        cuda_vgrid = cuda_vd.compute(W, H, seeds)
    # Seed ids may legitimately differ at ties, so only distances are compared.
    # A nonzero count is the known CUDA local-propagation shortfall, not noise
    # -- see RegularDelaunay in reference/voronoi.py.
    n_diff = int((cuda_vgrid[:, :, 1] != ref_vgrid[:, :, 1]).sum())
    print(f"    {'Distance cells differing from NumPy reference':<52} "
          f"{'%d (%.4f%%)' % (n_diff, 100.0 * n_diff / (W * H)):>10}")
else:
    print("    [CUDA extension not available -- skipped]")
    cuda_vgrid = ref_vgrid

# -- 4. GridTriangulation -- reference (detection only) ----------------------
section("4. GridTriangulation -- NumPy reference (detection + dedup only)")
with timer("Triangle detection + deduplication"):
    ref_tri_map = _ref_detect_triangles(ref_vgrid, seeds)

T = len(ref_tri_map)
ops = W * H * T
hours_est = ops / 1e7 / 3600
print(f"    {'Unique triangles found (upper bound)':<52} {T:>10,}")
print(f"    {'Assignment ops  W x H x T  (skipped)':<52} {ops/1e9:>9.1f} G")
print(f"    ('-> ~{hours_est:.1f} h in Python; GPU required for this scale)")

# -- 5. GridTriangulation -- CUDA --------------------------------------------
section("5. GridTriangulation -- CUDA (full pipeline incl. assignment)")
if cuda_available():
    from delauney._delauney_cuda import GridTriangulation as CudaTri
    cuda_tri = CudaTri()
    with timer("compute_timed()  (detection + dedup + assignment)"):
        cuda_tri_map, cuda_tgrid, gpu_timings = cuda_tri.compute_timed(cuda_vgrid, seeds)
    # T is an upper bound (see _ref_detect_triangles), so this is a sanity
    # check on the order of magnitude, not an equality assertion.
    print(f"    {'Triangles found (vs detect-only upper bound)':<52} "
          f"{'%d / %d' % (len(cuda_tri_map), T):>10}")
    print()
    print(f"    GPU sub-phase breakdown:")
    print(f"    {'  detect (find_triangle_seeds kernel)':<52} {gpu_timings['detect_ms']:>9.1f} ms")
    print(f"    {'  dedup  (thrust sort + unique)':<52} {gpu_timings['dedup_ms']:>9.1f} ms")
    print(f"    {'  assign (assign_triangles kernel)':<52} {gpu_timings['assign_ms']:>9.1f} ms")
else:
    print("    [CUDA extension not available -- skipped]")
    gpu_timings = None

# -- 6. IncrementalDelaunay -- CUDA ------------------------------------------
section("6. IncrementalDelaunay -- CUDA")
if cuda_available():
    from delauney._delauney_cuda import IncrementalDelaunay as CudaInc

    # Cold insert: all N seeds in one shot (exercises full_triangulate_)
    inc = CudaInc(W, H, N + 20)
    with timer("insert_timed()  cold  (all seeds, full triangulate)"):
        inc_tri_map, inc_tgrid, inc_t0 = inc.insert_timed(seeds)
    print(f"    {'  Seeds inserted':<52} {inc.seed_count:>10,}")
    print(f"    {'  Triangles found':<52} {len(inc_tri_map):>10,}")
    print()
    print(f"    GPU sub-phase breakdown (cold):")
    print(f"    {'  bfs    (BFS until convergence)':<52} {inc_t0['bfs_ms']:>9.1f} ms")
    print(f"    {'  detect (find_triangle_seeds kernel)':<52} {inc_t0['detect_ms']:>9.1f} ms")
    print(f"    {'  dedup  (thrust sort + unique)':<52} {inc_t0['dedup_ms']:>9.1f} ms")
    print(f"    {'  assign (assign_triangles kernel)':<52} {inc_t0['assign_ms']:>9.1f} ms")

    # Warm insert: one additional seed (exercises partial_triangulate_)
    rng2 = np.random.default_rng(99)
    extra_seeds = []
    seed_set = set(map(tuple, seeds))
    while len(extra_seeds) < 10:
        x, y = int(rng2.integers(0, W)), int(rng2.integers(0, H))
        if (x, y) not in seed_set:
            extra_seeds.append((x, y))
            seed_set.add((x, y))

    # Warm-up: one insert to prime caches
    inc.insert_timed([extra_seeds[0]])

    # Measure average over remaining 9 single-seed inserts
    warm_totals = {"bfs_ms": 0., "detect_ms": 0., "dedup_ms": 0., "assign_ms": 0.}
    warm_wall = 0.
    WARM_N = len(extra_seeds) - 1
    for s in extra_seeds[1:]:
        t0 = time.perf_counter()
        _, _, wt = inc.insert_timed([s])
        warm_wall += time.perf_counter() - t0
        for k in warm_totals:
            warm_totals[k] += wt[k]

    warm_wall_avg = warm_wall / WARM_N * 1000
    _results["insert_timed()  warm  (single seed, partial triangulate)"] = warm_wall / WARM_N
    print()
    print(f"    {'insert_timed()  warm  (single seed, avg over ' + str(WARM_N) + ')':<52} {warm_wall_avg:>9.1f} ms")
    print()
    print(f"    GPU sub-phase breakdown (warm, avg):")
    for k in ("bfs_ms", "detect_ms", "dedup_ms", "assign_ms"):
        label = k.replace("_ms", "").ljust(6)
        print(f"    {'  ' + label:<52} {warm_totals[k]/WARM_N:>9.1f} ms")
else:
    print("    [CUDA extension not available -- skipped]")

# -- 7. Summary --------------------------------------------------------------
section("7. Summary")
COL = 46
labels = [
    ("Generate seeds",               "Generate unique positions"),
    ("RegularDelaunay  (NumPy ref)", "compute()  (full BFS)"),
    ("RegularDelaunay  (CUDA)",      "compute()  (warm-up excluded)"),
    ("GridTriang. detect (NumPy)",   "Triangle detection + deduplication"),
    ("GridTriang. full   (CUDA)",    "compute_timed()  (detection + dedup + assignment)"),
    ("Incremental cold   (CUDA)",    "insert_timed()  cold  (all seeds, full triangulate)"),
    ("Incremental warm   (CUDA)",    "insert_timed()  warm  (single seed, partial triangulate)"),
]

print(f"\n    {'Phase':<{COL}}  {'Time':>11}  {'vs ref BFS':>10}")
print(f"    {'-'*COL}  {'-'*11}  {'-'*10}")

ref_bfs_s = _results.get("compute()  (full BFS)")
for display, key in labels:
    ms = _results.get(key)
    if ms is None:
        print(f"    {display:<{COL}}  {'n/a':>11}  {'':>10}")
        continue
    speedup = ""
    if ref_bfs_s and key == "compute()  (warm-up excluded)":
        speedup = f"{ref_bfs_s / ms:.1f}x"
    print(f"    {display:<{COL}}  {ms*1000:>10.1f}ms  {speedup:>10}")

print(f"\n    Complexity note:")
print(f"      Voronoi BFS:      O(W x H x iters)  -- iters = {n_iter} here")
print(f"      Tri. detection:   O(W x H x 4)      -- vectorised NumPy")
print(f"      Tri. assignment:  O(W x H x T)      -- T <= {T:,} triangles")
print(f"        Reference Python: ~{hours_est:.1f} h")
if cuda_available() and gpu_timings is not None:
    print(f"        CUDA kernel breakdown:")
    print(f"          detect: {gpu_timings['detect_ms']:.1f} ms")
    print(f"          dedup:  {gpu_timings['dedup_ms']:.1f} ms  (Thrust sort+unique on device)")
    print(f"          assign: {gpu_timings['assign_ms']:.1f} ms")
hr()
