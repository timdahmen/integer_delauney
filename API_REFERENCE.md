# `delauney` — Python API Reference

**Package:** `delauney` | **Requires:** Python ≥ 3.10, NumPy ≥ 1.24, CUDA GPU (for top-level classes)

---

## Common Data Types

| Type alias | Description |
|---|---|
| `Seeds` | `Sequence[tuple[int, int]]` — list of `(x, y)` pixel coordinates |
| `VoronoiGrid` | `np.ndarray` shape `(H, W, 2)` dtype `int32` — channel 0: `seed_id`, channel 1: Manhattan distance |
| `TriangleMap` | `dict[int, tuple[int, int, int, int, int]]` — `{triangle_id: (x, y, id_a, id_b, id_c)}` where x/y is the detection vertex and id_a/b/c are the three seed IDs |
| `TriangulationGrid` | `np.ndarray` shape `(H, W, 3)` dtype `int32` — channel 0: `seed_id`, channel 1: distance, channel 2: `triangle_id` |
| `Timings` | `dict[str, float]` — per-stage GPU timings in milliseconds |

> **Seed ID assignment:** Seeds are always sorted internally by `x` ascending, then `y` ascending. The index in that sorted order becomes the `seed_id`. Pass seeds in any order; the library handles normalization.

---

## `delauney.RegularDelaunay` (CUDA)

```python
from delauney import RegularDelaunay
```

One-shot, stateless computation of a Manhattan-distance Voronoi diagram on the GPU.

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
- `[y, x, 1]` → Manhattan distance to that seed

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
) -> tuple[TriangleMap, TriangulationGrid]
```

| Parameter | Type | Description |
|---|---|---|
| `voronoi_grid` | `np.ndarray (H, W, 2) int32` | Output of `RegularDelaunay.compute()`. |
| `seed_positions` | `Seeds` | The same seed list passed to `RegularDelaunay.compute()`. |

**Returns** `(TriangleMap, TriangulationGrid)`

**Raises** `ValueError` if `voronoi_grid` does not have shape `(H, W, 2)`.

---

### `compute_timed`

```python
triangle_map, triangulation_grid, timings = GridTriangulation().compute_timed(
    voronoi_grid: np.ndarray,
    seed_positions: Seeds,
) -> tuple[TriangleMap, TriangulationGrid, Timings]
```

Same as `compute()` with an additional `Timings` dict:

| Key | Unit | Description |
|---|---|---|
| `detect_ms` | ms | Triangle-edge detection kernel |
| `dedup_ms` | ms | Deduplication pass |
| `assign_ms` | ms | Per-cell triangle assignment |

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

CPU-only reference implementations with identical signatures to the CUDA classes. Useful for testing without a GPU. No `IncrementalDelaunay` equivalent exists.

| Class | Method | Signature |
|---|---|---|
| `reference.RegularDelaunay` | `compute` | `(width, height, seeds) → VoronoiGrid` |
| `reference.GridTriangulation` | `compute` | `(voronoi_grid, seed_positions) → (TriangleMap, TriangulationGrid)` |

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
# voronoi[y, x, 1]  →  Manhattan distance

# Step 2: Delaunay triangulation
tri_map, tri_grid = GridTriangulation().compute(voronoi, seeds)
# tri_map[tid]       →  (x, y, id_a, id_b, id_c)
# tri_grid[y, x, 2]  →  triangle_id

# --- OR: incremental (GPU state persists across calls) ---
from delauney._delauney_cuda import IncrementalDelaunay

tri = IncrementalDelaunay(W, H, max_seeds=1000)
tri_map, tri_grid = tri.insert([(10, 20), (50, 80)])
tri_map, tri_grid = tri.insert([(120, 40)])  # adds to existing state
print(tri.seed_count)  # → 3
```
