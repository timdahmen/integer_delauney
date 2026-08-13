# `delauney` — Python API Reference

**Package:** `delauney` | **Requires:** Python ≥ 3.10, NumPy ≥ 1.24, CUDA GPU (for top-level classes)

---

## Common Data Types

| Type alias | Description |
|---|---|
| `Seeds` | `Sequence[tuple[int, int]]` — list of `(x, y)` pixel coordinates |
| `VoronoiGrid` | `np.ndarray` shape `(H, W, 2)` dtype `int32` — channel 0: `seed_id`, channel 1: **squared** L2 (Euclidean) distance |
| `TriangleMap` | `dict[int, tuple[int, int, int, int, int]]` — `{triangle_id: (x, y, id_a, id_b, id_c)}` where x/y is the detection vertex and id_a/b/c are the three seed IDs |
| `TriangleArray` | `np.ndarray` shape `(N_tri, 3)` dtype `int32` — row `tid` holds that triangle's three seed IDs. The `as_arrays` alternative to `TriangleMap`; no `(x, y)` detection pixel. |
| `TriangulationGrid` | `np.ndarray` shape `(H, W, 3)` dtype `int32` — channel 0: `seed_id`, channel 1: squared distance, channel 2: `triangle_id`, or **`-1`** where no triangle contains the pixel |
| `Timings` | `dict[str, float]` — per-stage GPU timings in milliseconds |

> **Seed ID assignment:** Seeds are always sorted internally by `x` ascending, then `y` ascending. The index in that sorted order becomes the `seed_id`. Pass seeds in any order; the library handles normalization.

> **The `-1` sentinel:** `triangle_id` is `-1` wherever no detected triangle contains the pixel — outside the convex hull, or inside it but in a triangle that was never detected. There is no fallback to a nearest triangle; both cases need the same nearest-seed handling downstream. Always mask on `>= 0` before indexing a triangle list.

---

## `delauney.auto_border_padding`

```python
from delauney import auto_border_padding

padding: int = auto_border_padding(width: int, height: int, n_seeds: int)
```

Returns `round(sqrt(width * height / n_seeds))`, or `0` when `n_seeds <= 0`.

A triangle is only detected where three Voronoi regions meet — its circumcenter. Border triangles frequently have circumcenters outside the image, so at padding `0` they are never registered and the pixels they cover degrade to nearest-seed with no error raised. This is the value both `GridTriangulation` paths use by default. Padding reduces but never eliminates missing triangles; sub-pixel slivers are missed at any padding.

---

## `delauney.RegularDelaunay` (CUDA)

```python
from delauney import RegularDelaunay
```

One-shot, stateless computation of an L2-distance (Euclidean) Voronoi diagram on the GPU.

### Constructor

```python
RegularDelaunay()
```

No parameters.

---

### `compute`

```python
voronoi_grid = RegularDelaunay().compute(
    width: int,
    height: int,
    seeds: Seeds,
) -> np.ndarray  # shape (H, W, 2), dtype int32
```

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Grid width in pixels. X coordinates must be in `[0, width)`. |
| `height` | `int` | Grid height in pixels. Y coordinates must be in `[0, height)`. |
| `seeds` | `Seeds` | List of `(x, y)` seed positions. Must be non-empty, no duplicates. |

**Returns** `np.ndarray` shape `(height, width, 2)` dtype `int32`
- `[y, x, 0]` → `seed_id` of the nearest seed to pixel `(x, y)`
- `[y, x, 1]` → **squared** L2 distance to that seed

> **Accuracy.** Both paths propagate locally. At cells equidistant from two seeds they may pick different — equally correct — seeds, so do not assume bit-equality of channel 0 at ties.
>
> The CUDA kernel advances one cell per iteration and can additionally settle at a fixed point that is *not* nearest-seed. Measured at 128×128 with 400 seeds: 2 of 16384 pixels, off by 1–3 in squared distance. It is rare and grows with seed density. Use `delauney.reference.RegularDelaunay` when exactness matters.

**Raises** `ValueError` if seeds is empty, contains duplicates, or any coordinate is out of bounds.

---

## `delauney.GridTriangulation` (CUDA)

```python
from delauney import GridTriangulation
```

Extracts Delaunay triangles from an existing Voronoi grid. Stateless, one-shot.

### Constructor

```python
GridTriangulation()
```

No parameters.

---

### `compute`

```python
triangle_map, triangulation_grid = GridTriangulation().compute(
    voronoi_grid: np.ndarray,   # shape (H, W, 2), int32
    seed_positions: Seeds,
    border_padding: int = -1,
    as_arrays: bool = False,
) -> tuple[TriangleMap | TriangleArray, TriangulationGrid]
```

| Parameter | Type | Description |
|---|---|---|
| `voronoi_grid` | `np.ndarray (H, W, 2) int32` | Output of `RegularDelaunay.compute()`. |
| `seed_positions` | `Seeds` | The same seed list passed to `RegularDelaunay.compute()`. |
| `border_padding` | `int` | Pixels of Voronoi canvas added on each side before triangle detection, exposing border triangles whose circumcenter lies outside the image. Negative (the default) resolves to `auto_border_padding`; `0` disables padding. The output grid is always the original `(H, W, 3)`. |
| `as_arrays` | `bool` | Return a `TriangleArray` instead of a `TriangleMap`, skipping the per-triangle Python objects. |

> The reference implementation spells the auto default as `border_padding=None`; the CUDA binding cannot take `None` through an `int` argument and uses a negative sentinel instead. Both resolve to `auto_border_padding`.

**Returns** `(TriangleMap | TriangleArray, TriangulationGrid)`

**Raises** `ValueError` if `voronoi_grid` does not have shape `(H, W, 2)`.

---

### `compute_timed`

```python
triangle_map, triangulation_grid, timings = GridTriangulation().compute_timed(
    voronoi_grid: np.ndarray,
    seed_positions: Seeds,
    border_padding: int = -1,
) -> tuple[TriangleMap, TriangulationGrid, Timings]
```

Same as `compute()` with an additional `Timings` dict. Always returns the dict form of the triangle map.

| Key | Unit | Description |
|---|---|---|
| `detect_ms` | ms | Triangle-edge detection kernel |
| `dedup_ms` | ms | Deduplication pass |
| `assign_ms` | ms | Per-cell triangle assignment |

---

### `compute_debug`

```python
triangle_map, triangulation_grid, padded_voronoi_grid = GridTriangulation().compute_debug(
    voronoi_grid: np.ndarray,
    seed_positions: Seeds,
    border_padding: int = -1,
) -> tuple[TriangleMap, TriangulationGrid, np.ndarray]
```

Same as `compute()`, but also returns the padded Voronoi canvas that triangle detection actually ran on, shape `(H + 2*P, W + 2*P, 2)` where `P` is the resolved `border_padding`. Useful for visualising which border triangles padding recovered. Always returns the dict form of the triangle map.

---

## `delauney._delauney_cuda.IncrementalDelaunay` (CUDA)

```python
from delauney._delauney_cuda import IncrementalDelaunay
```

> **Note:** Not re-exported from the `delauney` top-level. Import directly from the C extension.

Stateful GPU triangulator that maintains Voronoi + triangulation state across multiple `insert()` calls. Device memory is allocated once at construction.

### Constructor

```python
tri = IncrementalDelaunay(
    width: int,
    height: int,
    max_seeds: int,
)
```

| Parameter | Type | Description |
|---|---|---|
| `width` | `int` | Grid width in pixels. |
| `height` | `int` | Grid height in pixels. |
| `max_seeds` | `int` | Hard upper bound on total seeds ever inserted. Used to pre-allocate GPU memory. |

---

### `insert`

```python
triangle_map, triangulation_grid = tri.insert(
    seeds: Seeds,
) -> tuple[TriangleMap, TriangulationGrid]
```

Inserts a batch of new seeds into the running triangulation. Seeds are appended to the existing set; do not re-pass previously inserted seeds.

| Parameter | Type | Description |
|---|---|---|
| `seeds` | `Seeds` | New `(x, y)` seed coordinates to add. |

**Returns** `(TriangleMap, TriangulationGrid)` reflecting **all** seeds inserted so far.

---

### `insert_timed`

```python
triangle_map, triangulation_grid, timings = tri.insert_timed(
    seeds: Seeds,
) -> tuple[TriangleMap, TriangulationGrid, Timings]
```

Same as `insert()` with an additional `Timings` dict:

| Key | Unit | Description |
|---|---|---|
| `bfs_ms` | ms | BFS Voronoi expansion for new seeds |
| `detect_ms` | ms | Triangle-edge detection |
| `dedup_ms` | ms | Deduplication pass |
| `assign_ms` | ms | Per-cell triangle assignment |

---

### `get_voronoi_grid`

```python
voronoi_grid = tri.get_voronoi_grid() -> np.ndarray  # shape (H, W, 2), int32
```

Returns the current Voronoi state as a `VoronoiGrid` without running a triangulation update.

---

### Properties (read-only)

| Property | Type | Description |
|---|---|---|
| `tri.width` | `int` | Grid width. |
| `tri.height` | `int` | Grid height. |
| `tri.seed_count` | `int` | Number of seeds inserted so far. |

---

## `delauney.reference` — Pure NumPy Fallback

```python
from delauney.reference import RegularDelaunay, GridTriangulation
```

CPU-only reference implementations with matching signatures. Useful for testing without a GPU. No `IncrementalDelaunay` equivalent exists.

| Class | Method | Signature |
|---|---|---|
| `reference.RegularDelaunay` | `compute` | `(width, height, seeds) → VoronoiGrid` |
| `reference.GridTriangulation` | `compute` | `(voronoi_grid, seed_positions, border_padding=None, as_arrays=False) → (TriangleMap \| TriangleArray, TriangulationGrid)` |

The one deliberate signature difference: the reference spells the auto padding default as `border_padding=None`, the CUDA path as a negative `int`. There is no `compute_timed` or `compute_debug` on the reference.

**Cocircular quads:** at a degree-4 Voronoi vertex a quad has two equally valid triangulations. Both paths resolve it the same way — shorter diagonal, tiebreaking away from the lowest seed id — so given the same Voronoi grid they produce identical triangle sets. Note that they may still differ *end to end*, because the two Voronoi implementations can disagree first (see the accuracy note under `RegularDelaunay`).

---

## Typical Usage Pattern

```python
import numpy as np
from delauney import RegularDelaunay, GridTriangulation

seeds = [(10, 20), (50, 80), (120, 40)]
W, H = 200, 150

# Step 1: Voronoi diagram
voronoi = RegularDelaunay().compute(W, H, seeds)
# voronoi[y, x, 0]  →  seed_id
# voronoi[y, x, 1]  →  squared L2 distance

# Step 2: Delaunay triangulation
tri_map, tri_grid = GridTriangulation().compute(voronoi, seeds)
# tri_map[tid]       →  (x, y, id_a, id_b, id_c)
# tri_grid[y, x, 2]  →  triangle_id, or -1 where no triangle contains the pixel

# ... or skip the dict entirely when you only need vertex indices
verts, tri_grid = GridTriangulation().compute(voronoi, seeds, as_arrays=True)
# verts[tid]  →  array([id_a, id_b, id_c])

# --- OR: incremental (GPU state persists across calls) ---
from delauney._delauney_cuda import IncrementalDelaunay

tri = IncrementalDelaunay(W, H, max_seeds=1000)
tri_map, tri_grid = tri.insert([(10, 20), (50, 80)])
tri_map, tri_grid = tri.insert([(120, 40)])  # adds to existing state
print(tri.seed_count)  # → 3
```
