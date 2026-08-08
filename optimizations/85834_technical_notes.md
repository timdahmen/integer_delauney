# 85834 — Technical notes

> **Authorship note.** This file was written by **Claude (Claude Code, Opus 5)** on
> 2026-08-08 at the author's request, as a documentation pass over work that was
> already finished. It describes and explains code the author wrote; it contains
> no optimization design of its own. The author's own account of the work — the
> reasoning, the dead ends, the ideas that were tried and rejected — is in
> [`85834.md`](85834.md) and remains the primary document.
>
> Companion files, same authorship: [`85834_measurements.md`](85834_measurements.md)
> (numbers and methodology) and [`85834_review_notes.md`](85834_review_notes.md)
> (issues found while reading the code).

---

## 1. What this covers

Nine commits on `newmaster`, from `5e1b060` (baseline + profiling harness) to
`e5f75ab` (HEAD). Files touched:

| File | Role |
| --- | --- |
| `src/cuda/incremental.cu` / `.cuh` | `IncrementalDelaunay` — cold and warm insert paths |
| `src/cuda/triangulation.cu` | `GridTriangulation` — batch pipeline |
| `src/cuda/shared_utils.cu` / `.cuh` | **new** — shared `RawTriangle`, the CSR vertex cache, and the kernel that fills it |
| `src/cuda/profiling.cpp` | host-side phase timers in the harness |
| `CMakeLists.txt` | pulls in the new translation unit |

Net effect at the default workload (1024×1024, 100 000 seeds):

| Phase | Baseline | HEAD | Speed-up |
| --- | ---: | ---: | ---: |
| `GridTriangulation` full pipeline | 159.1 ms | 85.8 ms | **1.85×** |
| `IncrementalDelaunay` cold insert | 254.7 ms | 114.0 ms | **2.23×** |
| `IncrementalDelaunay` warm insert | 122.5 ms | 58.1 ms | **2.11×** |
| `assign_triangles_kernel` (batch) | 103.8 ms | 28.9 ms | **3.59×** |
| `assign_triangles_kernel` (cold) | 148.7 ms | 30.4 ms | **4.89×** |

Triangle output is bit-identical to baseline (238 034 triangles, same set).
Full numbers, per-commit attribution and methodology are in
[`85834_measurements.md`](85834_measurements.md).

---

## 2. Where the time was going

The baseline pipeline has three cost centres, and they are of genuinely
different kinds — which is why the fixes look nothing alike:

```
  ┌─ per-pixel GPU work ────────────────────────────────────────┐
  │  assign_triangles_kernel        103 ms  (batch)             │  ← §4, §5
  │    · scans a (2R+1)² seed window per pixel                  │
  │    · then walks a CSR triangle list per unique nearby seed  │
  │    · pointer-chases triangle → 6 seed coordinate loads      │
  └─────────────────────────────────────────────────────────────┘
  ┌─ per-insert host work ──────────────────────────────────────┐
  │  registry compaction + hash-map rebuild   ~60 ms (warm)     │  ← §3
  │    · O(T) rebuild of two 238k-entry unordered_maps          │
  │    · full copy of the triangle vector into a new vector     │
  └─────────────────────────────────────────────────────────────┘
  ┌─ host↔device transfer ──────────────────────────────────────┐
  │  mask H2D / grid D2H              ~30 ms (warm)             │  ← §6
  │    · four 4 MB masks + an 8 MB grid readback per insert     │
  └─────────────────────────────────────────────────────────────┘
```

The kernel work and the host work were attacked separately, and the transfer
work turned out to be bounded by something outside the code's control (§6).

---

## 3. Host path: the triangle registry

### 3.1 A dead lookup map (`ddf31b8`)

`IncrementalDelaunay` maintained two hash maps over the triangle registry:

```cpp
std::unordered_map<uint64_t,int32_t>   h_triplet_to_tid_;   // (a,b,c) → tid
std::unordered_map<uint64_t,int32_t>   h_canon_to_tid_;     // (x,y)   → tid
```

`h_canon_to_tid_` was written on every insert and **never read anywhere**. Both
`full_triangulate_` and `partial_triangulate_` cleared it and refilled it with
one entry per triangle, so every insert paid ~238 000 hash insertions plus the
allocation and rehash traffic behind them, for nothing.

The commit deletes the member and the two loops that fill it.

The commit message calls this "genuinely not used" and `85834.md` describes it as
"so minor, it shouldn't count". Measured, it is not minor — it is the
**second-largest single win on the warm path** in the whole branch:

| | warm insert | cold insert |
| --- | ---: | ---: |
| before (`5e1b060`) | 122.5 ms | 254.7 ms |
| after (`ddf31b8`) | 88.9 ms | 228.2 ms |
| **delta** | **−33.6 ms (−27 %)** | **−26.5 ms (−10 %)** |

Worth restating in the submission: deleting a write-only data structure removed
a quarter of the warm-insert cost.

### 3.2 In-place compaction (`eab0393`)

This is the change `85834.md` §"Working on the warm insert" describes.

**Before.** `partial_triangulate_` rebuilt the registry by copying:

```cpp
std::vector<HTriangle> compacted;
compacted.reserve(old_count - (int)std::count(is_stale.begin(), is_stale.end(), true)
                  + (int)to_add.size());          // pass 1: count the survivors
for (int tid = 0; tid < old_count; ++tid) {
    if (!is_stale[tid]) {
        remap[tid] = (int32_t)compacted.size();
        compacted.push_back(h_triangles_[tid]);   // pass 2: copy every survivor
    }
}
for (const auto& t : to_add) compacted.push_back(t);
h_triangles_ = std::move(compacted);

h_triplet_to_tid_.clear();                        // pass 3: rebuild from scratch
for (int tid = 0; tid < (int)h_triangles_.size(); ++tid)
    h_triplet_to_tid_[pack_triplet_(...)] = tid;
```

Three full O(T) passes over ~238 000 triangles, one ~9 MB allocation, and a
complete hash-map teardown and rebuild — all to remove the handful of triangles
(typically a few dozen) invalidated by a single new seed. The work scales with
the size of the *triangulation*, not with the size of the *change*.

**After.** Control flow is inverted: instead of walking the survivors and
copying them forward, walk the stale entries and backfill each from the tail.

```cpp
int next = old_count;          // final size, discovered as we go
int last = old_count - 1;      // scan cursor from the tail
for (int tid = 0; tid < old_count; ++tid) {
    if (is_stale[tid]) {
        h_triplet_to_tid_.erase(pack_triplet_(...));   // targeted erase
        if (tid < next) {
            while (last > tid && is_stale[last]) --last;   // find a live donor
            if (last <= tid) {
                next = tid;                                // tail is all stale
            } else {
                h_triangles_[tid] = h_triangles_[last];    // swap-from-end
                remap[last] = tid;
                h_triplet_to_tid_[pack_triplet_(          // re-key the donor
                    h_triangles_[tid].a, ...)] = tid;
                next = last;
                --last;
            }
        }
    } else if (tid < next) {
        remap[tid] = tid;                                  // identity
    }
}
h_triangles_.resize(next);
for (const auto& t : to_add) { h_triangles_.push_back(t); ... }
```

Three things changed at once, and it is worth separating them:

1. **No second vector.** `h_triangles_` is compacted in place with a
   swap-from-end, so no ~9 MB allocation and no bulk copy of survivors. Only
   stale slots are written.
2. **The hash map is edited, not rebuilt.** `erase` for each stale triplet,
   one re-key for each donor moved. The number of map operations is now
   proportional to the number of *stale* triangles, not to `T`.
3. **New triangles need no remap entry.** They are appended after compaction, so
   their IDs follow from `h_triangles_.size()` directly.

The `remap` array stays `-1`-initialised on purpose — the in-code comment says
so, and it is load-bearing. `remap_tgrid_kernel` treats `remap[old_tid] < 0` as
"this triangle is gone" and routes the pixel to `fallback`, so entries the
compaction loop never touches (stale slots with no live donor) correctly resolve
to the fallback triangle rather than to a stale ID.

The two loop-guard subtleties are also real, not defensive noise:

- `if (tid < next)` on the *non-stale* branch: once `next` has been pulled down,
  entries at or past it have already been consumed as donors, and writing
  `remap[tid] = tid` for them would resurrect an ID that no longer exists.
- `h_triplet_to_tid_` is re-keyed from `h_triangles_[tid]` *after* the
  assignment, not from the pre-swap triplet — the slot has already been
  overwritten. The comment in the source records that this cost the author real
  debugging time; it is an easy thing to get backwards.

Measured: warm insert 88.9 ms → 58.4 ms (**−34 %**). This matches the 86.4 → 58.0
pair recorded in `85834.md` within run-to-run noise.

---

## 4. `assign_triangles_kernel`, part 1: folding the nearby scan

Commits `ab8630e` / `70d6d8a`, described in `85834.md` §"Working on
assign_triangles_kernel" rev 1 and rev 2.

**Before.** Each pixel ran the window scan twice, materialising an intermediate:

```cpp
int32_t nearby[MAX_NEARBY];        // 64 int32 = 256 B of local memory
int n_nearby = 0;
for (sy, sx in window) {           // pass 1: collect unique seed ids
    int32_t sid = n_grid[sy * W + sx];
    bool dup = false;
    for (int i = 0; i < n_nearby; ++i)
        if (nearby[i] == sid) { dup = true; break; }
    if (!dup && n_nearby < MAX_NEARBY) nearby[n_nearby++] = sid;
}
for (int i = 0; i < n_nearby; ++i) { /* pass 2: walk CSR, containment test */ }
```

The 64-entry array is the problem. With a 41×41 window it cannot live in
registers, so `nearby` spills to local memory — which on the T1000 means L1/L2
traffic per element, per comparison, per pixel. The dedup inner loop is O(n²) in
the number of unique seeds on top of that.

**After.** The two passes are fused, and exact dedup is replaced with a small
fixed-size recency filter:

```cpp
int32_t previous_sid = -1;
int32_t sid_hist_pseudostack[SID_HISTORY_STACK_SIZE];   // 16 entries

for (sy, sx in window) {
    int32_t sid = n_grid[sy * W + sx];
    if (sid == previous_sid) continue;      // run-length filter along the row
    previous_sid = sid;
    if (sid < 0 || sid >= N_seeds) continue;

    bool did_find = false;
    #pragma unroll
    for (int i = 0; i < SID_HISTORY_STACK_SIZE; ++i)
        did_find |= (sid_hist_pseudostack[i] == sid);   // branchless, unrolled
    if (did_find) continue;

    #pragma unroll
    for (int i = SID_HISTORY_STACK_SIZE - 1; i > 0; --i)
        sid_hist_pseudostack[i] = sid_hist_pseudostack[i - 1];
    sid_hist_pseudostack[0] = sid;

    for (int j = csr_ptr[sid]; j < csr_ptr[sid+1]; ++j) { /* containment test */ }
}
```

Why this is faster despite doing *more* redundant work:

- **The 16-entry array fits in registers.** Every access to it is at a
  compile-time-constant index (both loops are `#pragma unroll`-ed over a
  `constexpr` bound), which is precisely the condition `ptxas` needs to promote
  a local array into the register file. The 64-entry version could not qualify:
  its accesses are bounded by a runtime `n_nearby`.
- **The dedup test is branchless.** `did_find |= (...)` accumulates instead of
  breaking early, so all 32 lanes of a warp execute the same 16 compares with no
  divergence. The `break` in the old version diverged per lane.
- **`previous_sid` catches the common case for free.** Voronoi cells are
  spatially contiguous, so scanning a row hits long runs of the same seed id.
  One compare eliminates most of them before the history filter is consulted.

`ptxas -v` confirms the register-residency argument directly — this is the
cleanest piece of evidence in the whole branch:

| build | registers | stack frame | spills |
| --- | ---: | ---: | ---: |
| baseline `5e1b060` | 45 | **256 bytes** | 0 |
| HEAD `e5f75ab` | 63 | **0 bytes** | 0 |

256 bytes is exactly `MAX_NEARBY × sizeof(int32_t)` — the `nearby` array, sitting
in local memory. At HEAD it is gone: the 16-entry history lives entirely in
registers. The 45 → 63 register jump is the array being absorbed, and on `sm_75`
it is free: Turing allows 32 warps/SM and 65 536 registers/SM, so anything up to
64 registers/thread still permits full occupancy. The kernel got a register-file
scratchpad at no occupancy cost.

The trade-off is that the filter is *lossy*: a seed evicted from the 16-deep
history and then seen again gets its CSR list walked twice. That is safe — the
containment test is idempotent and `best` only ever increases — but it is the
reason rev 1 (fold with no history at all) was measured as *slower* than the
baseline, as `85834.md` records. The history is what makes the fold pay.

`85834.md` notes the stack size was picked empirically (32 → 16 → 8, kept 16).
The register-file argument above explains why the curve is flat between 32 and
16 and falls off at 8: 16 entries is enough to cover the working set of unique
seeds in a typical window, and the array is register-resident at all three
sizes, so the only cost that moves is redundant CSR walks.

### 4.1 Two behaviour changes worth declaring

**(a) The `MAX_NEARBY = 64` cap is gone.** The old kernel silently dropped every
unique seed past the 64th in a window. The new one has no cap — an evicted seed
is re-walked rather than discarded. In dense regions the new kernel therefore
tests a *superset* of the triangles the old one tested, which means the "before"
and "after" kernels are not strictly the same algorithm. At the benchmarked
workload the outputs are identical (238 034 triangles, and `--verify` reports the
same 150-pixel warm/cold delta as baseline), so the cap was not being hit here —
but the new version is the more correct of the two, not merely the faster one.

**(b) The containment test is now guarded by the ID test.**

```cpp
// before
if (point_in_triangle(px, py, ...))
    if (best == -1 || tid > best) best = tid;

// after
if (tid > best) {
    if (point_in_triangle(px, py, ...)) best = tid;
}
```

The spec (and `tests/test_pixel_assignment_correctness.py`) defines the winner as
the highest-id triangle containing the pixel centre. `tid` is always `>= 0` and
`best` starts at `-1`, so `tid > best` subsumes the `best == -1` case exactly.
Hoisting it skips ~9 floating-point multiplies and a pile of comparisons for
every candidate that could not have won anyway. Same result, strictly less work.

---

## 5. `assign_triangles_kernel`, part 2: the CSR vertex cache

Commit `e5f75ab`, described in `85834.md` rev 3. This is the largest single
kernel win.

**The problem.** After §4, the inner loop was still a two-level pointer chase:

```cpp
for (int j = csr_ptr[sid]; j < csr_ptr[sid+1]; ++j) {
    int32_t tid = csr_idx[j];                     // sequential in j — fine
    const RawTriangle& tri = triangles[tid];      // 32 B at a random index
    float ax = seed_xs[tri.orig_a], ay = seed_ys[tri.orig_a];   // 6 more
    float bx = seed_xs[tri.orig_b], by = seed_ys[tri.orig_b];   // random
    float cx = seed_xs[tri.orig_c], cy = seed_ys[tri.orig_c];   // gathers
}
```

Per candidate triangle: one 32-byte random-index structure load of which only
12 bytes (`orig_a/b/c`) are ever used, then six dependent scalar gathers into two
separate arrays. Seven uncoalesced accesses to obtain 24 bytes of actual
geometry — and every neighbouring pixel repeats them for the same triangles.

**The fix.** Precompute the six coordinates once per CSR *slot* and store them in
the order the kernel will read them:

```cpp
// shared_utils.cuh
struct CsrEntryVertexCache {          // indexed by CSR slot j, not by tid
    PackedCoordinate a, b, c;         // 3 × uint32 = 12 bytes
};
```

```cpp
// shared_utils.cu — one thread per CSR slot
__global__ void build_csr_verts_kernel(...) {
    int csr_slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (csr_slot < csr_size) {
        const RawTriangle& which = triangles[csr_idx[csr_slot]];
        csr_verts_cache[csr_slot] = CsrEntryVertexCache(
            seed_xs[which.orig_a], seed_ys[which.orig_a],
            seed_xs[which.orig_b], seed_ys[which.orig_b],
            seed_xs[which.orig_c], seed_ys[which.orig_c]);
    }
}
```

The assign kernel's inner loop becomes:

```cpp
for (int j = csr_ptr[sid]; j < csr_ptr[sid+1]; ++j) {
    int32_t tid = csr_idx[j];
    if (tid > best) {
        const CsrEntryVertexCache& c = csr_verts_cache[j];    // sequential in j
        if (point_in_triangle(px, py, c.a.x, c.a.y, c.b.x, c.b.y, c.c.x, c.c.y))
            best = tid;
    }
}
```

Seven scattered accesses per candidate become one sequential 12-byte load.
Critically the index is `j`, the CSR slot — the same variable the loop already
walks — so consecutive iterations touch consecutive cache lines and a line
fetched for one candidate serves the next several.

**Why the kernel and not the host.** `85834.md` notes the cache was first built
CPU-side and that this was a bottleneck. It has to be: the cache is
`3 × T ≈ 714 000` entries, i.e. ~8.6 MB that would need building *and* uploading
on every insert, on the same PCIe link §6 shows to be the constraint. Building it
on the device instead costs one kernel launch over data (`d_raw_buf_`, `d_sx_`,
`d_sy_`, `d_csr_idx_`) that is already resident, and adds nothing to the transfer
budget. It appears in the host breakdown as `rebuild_csr` at ~5.5 ms, which is
dominated by the `csr_ptr`/`csr_idx` upload it was already doing.

Ordering is guaranteed without an explicit sync: `build_csr_verts_kernel` and
`assign_triangles_kernel` are both issued on the legacy default stream, so the
first completes before the second starts. In `partial_triangulate_` the call
order `upload_triangles_()` → `rebuild_csr_and_upload_()` also matters, since the
build kernel reads the `d_raw_buf_` contents the upload just refreshed.

### 5.1 Coordinate packing, and how to turn it off

`PackedCoordinate` stores each vertex as two `uint16_t` in one `uint32_t`,
halving the cache to 12 bytes per slot. `85834.md` is explicit that this is a
bandwidth trick tuned to the author's link-limited T1000 and flags it as
unsuitable for production, because it caps usable canvas dimensions at 65 535.

The escape hatch is the `EXTEND_COORDINATE_LIMIT` macro, which switches
`CsrEntryVertexCache` to a `uint32_t`-per-axis layout (24 bytes per slot).
`85834.md` says to "define / pass it during compile" but does not give the
invocation, and there is no CMake option for it (see
[`85834_review_notes.md`](85834_review_notes.md), finding 3). It works via:

```bash
cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON -DBUILD_PYTHON_MODULE=OFF -DCMAKE_CUDA_FLAGS="-DEXTEND_COORDINATE_LIMIT"
```

Only `.cu` files include `shared_utils.cuh`, so scoping the define to
`CMAKE_CUDA_FLAGS` is sufficient and cannot produce an ODR mismatch with the
C++-only translation units.

Verified: that build compiles, passes `--verify` with identical results, and
costs ~4 ms on the cold assign kernel. Numbers in
[`85834_measurements.md`](85834_measurements.md) §5.

---

## 6. Host↔device transfers: the negative result

`85834.md` §"Working on the memory side of things" documents this honestly and
the measurements agree with it, so this section is mostly here to make the
negative result legible to a reader.

`1413499` moved the five recurring host staging buffers from `std::vector` to
`cudaHostAlloc` pinned allocations, owned by the object for its lifetime:

```cpp
cudaHostAlloc(&p_changed_,  N     * sizeof(int32_t), cudaHostAllocDefault);
cudaHostAlloc(&p_border_,   N     * sizeof(int32_t), cudaHostAllocDefault);
cudaHostAlloc(&p_reassign_, N     * sizeof(int32_t), cudaHostAllocDefault);
cudaHostAlloc(&p_t_,        N     * sizeof(int32_t), cudaHostAllocDefault);
cudaHostAlloc(&p_grid_,     N * 2 * sizeof(int32_t), cudaHostAllocDefault);
```

This is the textbook move: pinned memory lets the DMA engine transfer without
staging through a driver bounce buffer. Allocating once in the constructor also
removes a per-insert `std::vector` allocation and zero-fill.

Measured gain: **~0.9 ms** on the warm insert, nothing on cold.

The reason is visible in the current warm host breakdown, where the transfers
that remain are stuck at a flat rate regardless of what is done to them:

| transfer | size | time | effective rate |
| --- | ---: | ---: | ---: |
| `D2H changed mask` | 4 MB | 5.12 ms | ~0.8 GB/s |
| `H2D border mask` | 4 MB | 5.32 ms | ~0.8 GB/s |
| `H2D reassign mask` | 4 MB | 5.31 ms | ~0.8 GB/s |
| `D2H grids` | 12 MB | 15.32 ms | ~0.8 GB/s |

~0.8 GB/s, identical in both directions, and stable to two decimal places across
runs. A pinned transfer on a healthy PCIe 3.0 ×16 link should reach ~10 GB/s.
This is the link-width problem `85834.md` describes: the ceiling is the
interconnect, not the allocation strategy, so pinning cannot help and neither
could the stream/event pipelining that rev 2 tried and abandoned. Those four
rows are 51 % of the warm insert and are not addressable in software on this
machine.

This also explains why the warm insert plateaus at ~58 ms while its *compute*
has been reduced to almost nothing: the warm GPU sub-phases are sub-millisecond,
and what is left is transfer plus the ~9.3 ms `upload_triangles` re-upload of
the full triangle array.

---

## 7. Build

Unchanged from the project default apart from the new translation unit, which
`CMakeLists.txt` adds to `DELAUNEY_CUDA_SOURCES`:

```bash
cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON -DBUILD_PYTHON_MODULE=OFF
```

```bash
cmake --build build/native --config Release --target delauney_profiling
```

```bash
build/native/Release/delauney_profiling.exe --verify
```

`--verify` cross-checks the incremental warm path against a full rebuild; see
[`85834_measurements.md`](85834_measurements.md) §6 for what its output means and
what the expected steady state is.
