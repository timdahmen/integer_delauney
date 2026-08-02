# integer_delauney
A GPU Implementation of a Delauney Triangulation for Pixel Coordinates


## Performance Optimization: Duplicate Seed Validation

### Background

While profiling the incremental insertion pipeline, the duplicate seed validation step was identified as one of the largest CPU-side bottlenecks, consuming about **2000 ms** for large insertion batches.

The original approach used duplicate detection by comparing every new seed against every other seed using nested loops:

```cpp
for (int i = 0; i < k; ++i)
    for (int j = i + 1; j < k; ++j)
        ...
```

This approach has a time complexity of **O(n²)**, meaning the execution time grows quadratically with the number of inserted seeds.

```bash
 --------------------------------------------------------------
  6. IncrementalDelaunay -- CUDA
  --------------------------------------------------------------
    insert_timed()  cold  (all seeds, full triangulate)      2425.9 ms
      Seeds inserted                                        100,000
      Triangles found                                       240,261

    GPU sub-phase breakdown (cold):
      bfs    (BFS until convergence)                           1.4 ms
      detect (find_triangle_seeds kernel)                      0.2 ms
      dedup  (thrust sort + unique)                            2.1 ms
      assign (assign_triangles kernel)                        19.9 ms

    insert_timed()  warm  (single seed, avg over 9)          200.4 ms

    GPU sub-phase breakdown (warm, avg):
      bfs                                                      0.5 ms
      detect                                                   0.0 ms
      dedup                                                    0.0 ms
      assign                                                   0.0 ms

```

### Optimization

The nested comparison was replaced with an `std::unordered_set<uint64_t>`.

Each `(x, y)` coordinate pair is packed into a single 64-bit integer using `pack_xy_()` and inserted into the hash table.

```cpp
std::unordered_set<uint64_t> batch_hash;
batch_hash.reserve(k);

for (int i = 0; i < k; ++i) {
    auto key = pack_xy_(new_xs[i], new_ys[i]);
    if (!batch_hash.insert(key).second) {
        throw std::invalid_argument("duplicate seed positions within batch");
    }
}
```

The hash table provides average **O(1)** insertion and lookup, reducing the overall duplicate check to **O(n)**.

Calling `reserve(k)` allocates sufficient buckets beforehand, avoiding costly rehash operations while inserting the batch.

### Result

Replacing the quadratic duplicate search with a hash-based lookup-table
 removes the bottleneck for large insertion batches and scales linearly with the number of seeds.

| Implementation | Time Complexity |
|----------------|-----------------|
| Nested loops | O(n²) |
| `std::unordered_set` | O(n) |

This optimization significantly reduces CPU preprocessing time before the CUDA kernels are launched while preserving identical functionality.

```bash
 --------------------------------------------------------------
  6. IncrementalDelaunay -- CUDA
  --------------------------------------------------------------
    insert_timed()  cold  (all seeds, full triangulate)       203.9 ms
      Seeds inserted                                        100,000
      Triangles found                                       240,261

    GPU sub-phase breakdown (cold):
      bfs    (BFS until convergence)                           1.1 ms
      detect (find_triangle_seeds kernel)                      0.2 ms
      dedup  (thrust sort + unique)                            2.1 ms
      assign (assign_triangles kernel)                        19.8 ms

    insert_timed()  warm  (single seed, avg over 9)          184.3 ms

    GPU sub-phase breakdown (warm, avg):
      bfs                                                      0.3 ms
      detect                                                   0.0 ms
      dedup                                                    0.0 ms
      assign                                                   0.0 ms
```




## Performance Optimization: Grid Splitting (Optional)

### Background

Profiling showed that part of the preprocessing before the triangulation algorithm was spent splitting the input Voronoi grid into two separate arrays.

The input grid stores two values for every cell:

```
(seed_id, distance)
```

The original implementation first iterated over the entire grid on the CPU, copying the two values into temporary host arrays before uploading them to the GPU.

```cpp
for (int i = 0; i < N; ++i) {
    h_n[i]    = voronoi_grid[i * 2];
    h_dist[i] = voronoi_grid[i * 2 + 1];
}
```

This introduced an additional serial preprocessing step that had to finish before any GPU computation could begin.

```bash
  --------------------------------------------------------------
  5. GridTriangulation -- CUDA (full pipeline incl. assignment)
  --------------------------------------------------------------
    compute_timed()  (detection + dedup + assignment)         106.9 ms
    Triangle count matches reference                     YES (236643)

    GPU sub-phase breakdown:
      detect (find_triangle_seeds kernel)                      0.7 ms
      dedup  (thrust sort + unique)                            3.6 ms
      assign (assign_triangles kernel)                        13.7 ms
```

### Optimization

Instead of splitting the grid on the CPU, the complete `(seed_id, distance)` grid is copied directly to the GPU in a single memory transfer.

A CUDA kernel then performs the split in parallel:

```cpp
split_seed_grid_kernel<<<(N + 255) / 256, 256>>>(
    seed_distance_grid,
    d_n,
    d_dist,
    N);
```

Each GPU thread processes exactly one grid cell by reading one `(seed_id, distance)` pair and writing the values into separate device arrays.

### Is it faster?

No, not with this batch size.
The original implementation processed every grid cell sequentially on the CPU before copying the data to the GPU. If the batch size gets bigger this improvement will be noticeable.

The optimized implementation removes this CPU loop. The complete grid is transferred once, and thousands of GPU threads split the data simultaneously. This shifts the work to the GPU, which is designed for highly parallel operations, while also eliminating the need for temporary host arrays.

### Result

Moving the grid splitting operation to the GPU removes unnecessary CPU preprocessing and allows the input data to be prepared in parallel.

| Implementation | Processing |
|---------------|------------|
| CPU loop | Sequential (one cell after another) |
| CUDA kernel | Parallel (many cells processed simultaneously) |

The difference is marginal and requires higher batch sizes to accelerate the program.

```bash
--------------------------------------------------------------
  5. GridTriangulation -- CUDA (full pipeline incl. assignment)
  --------------------------------------------------------------
    compute_timed()  (detection + dedup + assignment)         104.5 ms
    Triangle count matches reference                     YES (236643)

    GPU sub-phase breakdown:
      detect (find_triangle_seeds kernel)                      0.1 ms
      dedup  (thrust sort + unique)                            2.6 ms
      assign (assign_triangles kernel)                        13.5 ms
```