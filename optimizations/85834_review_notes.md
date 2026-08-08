# 85834 — Review notes

> **Authorship note.** This file was written by **Claude (Claude Code, Opus 5)** on
> 2026-08-08 at the author's request, as a read-through of the branch looking for
> mistakes. Nothing here was changed in the source — every item is a report, and
> whether to act on any of it is the author's call. Findings are ordered by
> consequence, and each states whether it is **new on this branch** or
> **pre-existing** in the ported baseline.
>
> Companions: [`85834.md`](85834.md),
> [`85834_technical_notes.md`](85834_technical_notes.md),
> [`85834_measurements.md`](85834_measurements.md).

**Short version:** the branch is correct at the workload it is benchmarked and
submitted against — `--verify` passes, `compute-sanitizer` is clean, triangle
output is identical to baseline at every commit. There is one real latent bug
(finding 1), one reporting gap that a grader is likely to notice (finding 2),
and a handful of nits.

---

## 1. CSR device buffers overflow at high seed density — **latent bug, partly new**

`IncrementalDelaunay`'s constructor sizes two device buffers off `max_seeds`:

```cpp
// incremental.cu
cudaMalloc(&d_csr_idx_,          (size_t)max_seeds * 8 * sizeof(int32_t));
cudaMalloc(&d_csr_verts_cache_,  (size_t)max_seeds * 8 * sizeof(CsrEntryVertexCache));
```

The size actually needed is `csr_size = 3 × T`, where `T` is the triangle count.
So the code is assuming `T ≤ 8/3 × max_seeds ≈ 2.67 × max_seeds`. Nothing checks
it, and there is no CUDA error checking to catch it after the fact.

That assumption does not hold. `T/N_seeds` is not bounded by the planar-graph
`2N` rule here, because these are grid-detected L-shape triangles rather than a
true Delaunay triangulation — and the ratio rises as seeds get dense relative to
the grid. Measured:

| grid | seeds | triangles | `T`/seeds | `csr_size` | capacity | |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2048² | 100 000 | 218 001 | 2.18 | 654 003 | 800 232 | ok |
| **1024² | 100 000 | 238 034 | 2.38 | 714 102 | 800 232** | **ok — 89 % full** |
| 512² | 20 000 | 46 542 | 2.33 | 139 626 | 160 168 | ok |
| 1024² | 300 000 | 818 343 | 2.73 | 2 455 029 | 2 400 232 | **over** |
| 512² | 150 000 | 478 066 | 3.19 | 1 434 198 | 1 200 232 | **over** |
| 256² | 40 000 | 129 398 | 3.24 | 388 194 | 320 232 | **over** |
| 128² | 12 000 | 41 042 | 3.42 | 123 126 | 96 232 | **over** |

(Capacity assumes the harness's own `cap = N + warm + 20`, i.e. `max_seeds ≈ N`.)

Confirmed with the sanitizer, which reports both halves of the failure:

```bash
compute-sanitizer --tool memcheck build/native/Release/delauney_profiling.exe --only 6 -w 128 -h 128 -n 12000 --warm 1
```

```
========= Program hit cudaErrorInvalidValue (error 1) ... on CUDA API call to cudaMemcpy.
=========     Host Frame: IncrementalDelaunay::rebuild_csr_and_upload_ in incremental.cu:454
========= CUDA API Error: Copy is larger than memobj size, for destination operand

========= Invalid __global__ write of size 4 bytes
=========     at build_csr_verts_kernel(...) in shared_utils.cu:20
=========     Host Frame: IncrementalDelaunay::rebuild_csr_and_upload_ in incremental.cu:458
========= ERROR SUMMARY: 710 errors
```

Two distinct problems:

- `incremental.cu:454` — the `d_csr_idx_` upload is rejected outright, so the CSR
  index array keeps stale contents and every subsequent result is wrong. The
  program prints a plausible-looking answer anyway. **This is pre-existing**:
  `5e1b060` sizes `d_csr_idx_` identically.
- `shared_utils.cu:20` — `build_csr_verts_kernel` writes past the end of
  `d_csr_verts_cache_`, corrupting whatever device allocation follows it. **This
  is new with `e5f75ab`**, which added the buffer under the same bound. Because a
  cache slot is 12 bytes rather than 4, the overrun is 3× the size of the
  pre-existing one, and unlike a rejected `cudaMemcpy` it actually writes.

Worth being clear about scope: **the submitted configuration is not affected.**
At 1024² / 100 000 seeds the buffer is 89 % full, the sanitizer is clean, and
every number in [`85834_measurements.md`](85834_measurements.md) is valid. But
89 % is thin headroom for a bound nothing enforces, and the batch path in
`triangulation.cu` does not share the problem — there `d_csr_idx` and
`d_csr_verts_cache` are allocated at the exact `csr_size`, because it is known by
then.

Options, in ascending order of effort: assert on `csr_size` against the capacity
so the failure is loud rather than silent; raise the multiplier; or size the two
buffers lazily from `csr_size` and grow them when it increases, matching what
`triangulation.cu` already does. Your call which is worth it before hand-in — the
one-line assert would at least make the limit visible.

## 2. The warm GPU sub-phase table always prints 0.0 — **reporting gap, pre-existing**

`partial_triangulate_` takes three out-params and never writes them:

```cpp
void IncrementalDelaunay::partial_triangulate_(float* det_ms, float* dedup_ms, float* asgn_ms)
{
    // ... det_ms, dedup_ms and asgn_ms are never assigned anywhere in the body
}
```

`insert()` copies them into `timings->detect_ms / dedup_ms / assign_ms` regardless,
so every warm insert reports zeros, and the harness prints:

```
    GPU sub-phase breakdown (warm, avg):
      bfs                                                      0.8 ms
      detect                                                   0.0 ms
      dedup                                                    0.0 ms
      assign                                                   0.0 ms
```

`bfs` is non-zero only because `run_bfs_` fills its own out-param separately.

This is pre-existing — the ported baseline had a `cudaEvent` created, recorded
and leaked in that spot without ever being read, which `9dd0d07` correctly
removed as a leak. But removing it left the parameters dead, and the practical
effect matters here: **those are exactly the four rows a reader checks to see
whether the assign-kernel work helped the warm path**, and they read as zero. In
`3003087.md` the same zeros appear in both the before and after listings,
which suggests it has been misread as "nothing to see here" before.

The information is not actually missing — the host-timer table carries it
(`partial: assign 0.34 ms`, `partial: detect 0.07 ms`, `partial: dedup 0.08 ms`) —
so this is a presentation problem rather than a measurement one. Either wiring
the events up the way `full_triangulate_` does, or dropping the three parameters
and printing "n/a" for the warm GPU rows, would stop it reading as a broken
measurement.

Related, minor: `full_triangulate_` renames its parameters to
`det_gpu / dedup_gpu / asgn_gpu` with a comment explaining that they must not
collide with the `IncrementalHostTimings` field names. `partial_triangulate_`
still uses the colliding `det_ms / dedup_ms / asgn_ms`. Not a bug — the
`DELAUNEY_HOST_TIME` macro qualifies with `ht_->` — but it breaks the convention
the comment sets out, in the one function where the collision is live.

## 3. `EXTEND_COORDINATE_LIMIT` has no build wiring — **new**

`85834.md` is emphatic that the coordinate packing *"should be removed for
production use"* and says to "Define / pass **EXTEND_COORDINATE_LIMIT** during
compile". There is no CMake `option()` for it and no documented invocation, so
the escape hatch for the one thing the document flags as production-unsafe is
undiscoverable from the build system.

It does work, and it is verified — this configures, compiles, passes `--verify`
with identical results, and costs ~4 ms on the cold assign kernel
([`85834_measurements.md`](85834_measurements.md) §5):

```bash
cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON -DBUILD_PYTHON_MODULE=OFF -DCMAKE_CUDA_FLAGS="-DEXTEND_COORDINATE_LIMIT"
```

Scoping the define to `CMAKE_CUDA_FLAGS` is safe: only `.cu` files include
`shared_utils.cuh`, so the C++-only translation units cannot see a different
`CsrEntryVertexCache` layout and there is no ODR hazard. A three-line
`option(EXTEND_COORDINATE_LIMIT ...)` in `CMakeLists.txt` would make it
self-documenting, but pasting the invocation into `85834.md` would do.

## 4. `85834.md` undersells `ddf31b8` — **documentation**

Filed under "General changes" as *"technically an optimization … but it's so
minor, it shouldn't count"*. Measured, removing the write-only `h_canon_to_tid_`
map is worth **−33.6 ms (−27 %) on the warm insert** and **−26.5 ms (−10 %) on
cold** — the second-largest single win on the branch, behind only the two
assign-kernel commits.

A grader reading only `85834.md` would skip past a change that removed a quarter
of the warm-insert cost. It is worth its own section: finding a 238 000-entry
hash map that is rebuilt every insert and read by nobody is a legitimate result,
and the "is this actually used anywhere?" reflex is the thing being demonstrated.

## 5. Some before/after pairs in `85834.md` look like single runs — **documentation**

The pairs cross-check well against a four-run-per-commit sweep, with one
exception:

| `85834.md` claim | measured (mean of 3, warm-up discarded) |
| --- | --- |
| GridTriang. 158.5 → 121.8 (rev 2) | 158.9 → 121.8 ✓ |
| GridTriang. 121.8 → 85.3 (rev 3) | 121.8 → 85.8 ✓ |
| Warm insert 86.4 → 58.0 | 88.9 → 58.4 ✓ |
| rev 1 "slowed down by about 20-30 ms" | +21.5 ms ✓ |
| **Incremental cold 159.6 → 112.0 (rev 3)** | **171.2 → 114.0** |

`Incremental cold` is the noisiest row in the harness — at `70d6d8a` a single
four-run set spanned 159.8 to 188.1 ms. The document's 159.6 sits at the bottom
of that spread, which is what a single run of a noisy measurement looks like. The
conclusion is unaffected (it is still a large win either way), but stating the
run count and the averaging next to the numbers would take the question off the
table. The protocol is written up in
[`85834_measurements.md`](85834_measurements.md) §1 if you want to cite it.

## 6. Smaller things

**No CUDA error checking anywhere.** Not one `cudaMalloc`, `cudaMemcpy` or kernel
launch is checked, project-wide and pre-existing. This is why finding 1 produces
a plausible-looking wrong answer instead of a crash. Out of scope for an
optimization submission, but it is the reason a real failure is invisible.

**`shared_utils.cuh` is listed in `DELAUNEY_CUDA_SOURCES`.** A header in a source
list. CMake treats the unknown extension as header-only so it builds fine, and it
does make the file show up in IDE project trees — but `shared_utils.cu` alone is
what actually needs to be there.

**`pack_triplet_` packs three seed IDs into 21 bits each.** IDs above 2 097 151
silently alias, which would corrupt `h_triplet_to_tid_` and therefore the
staleness decisions in the warm path. Pre-existing, unreachable at any workload
here, and `max_seeds` is not checked against it.

**Two 4 MB `memset`s at the top of `partial_triangulate_` are untimed.** The
warm host breakdown accounts for 56.43 ms against a 58.1 ms measured wall time;
the `p_border_` / `p_reassign_` clears are part of the ~1.7 ms difference. Only
matters if the breakdown is meant to close.

**`upload_triangles_` is the largest remaining addressable cost on the warm
path** — 9.27 ms, 16.4 %. It allocates a fresh `std::vector<RawTriangle>`,
converts all ~238 000 host triangles into it, and uploads the whole array, on
every single-seed insert. Flagging it as an observation only, since it is
optimization work rather than review: after this branch, the warm insert is 55 %
raw PCIe transfer (finding in [`85834_measurements.md`](85834_measurements.md) §7)
and 16 % this, and the transfer half is not addressable in software on this
machine.

---

## What I checked and found nothing wrong with

For completeness, since "no findings" is also a result:

- **The compaction loop in `partial_triangulate_`.** The swap-from-end walk, the
  `tid < next` guards on both branches, the `-1` initialisation of `remap`, and
  the re-keying of the donor triplet *after* the assignment are all correct, and
  each of them is load-bearing. The in-code comments about this are accurate.
- **`if (tid > best)` hoisted above `point_in_triangle`.** `tid` is always `≥ 0`
  and `best` starts at `-1`, so it subsumes the old `best == -1 || tid > best`
  exactly. Matches the spec in `tests/test_pixel_assignment_correctness.py`
  (highest containing ID wins).
- **Kernel ordering without explicit syncs.** `build_csr_verts_kernel` and
  `assign_triangles_kernel` are both on the legacy default stream, so the first
  completes before the second starts. The `upload_triangles_()` →
  `rebuild_csr_and_upload_()` call order in `partial_triangulate_` is also right,
  since the build kernel reads the `d_raw_buf_` contents the upload refreshes.
- **The lossy history filter.** Re-walking an evicted seed's CSR list is safe:
  the containment test is idempotent and `best` only increases.
- **Removal of the `MAX_NEARBY = 64` cap.** The new kernel tests a superset of
  what the old one tested — the old one silently dropped unique seeds past the
  64th. Worth *declaring* in the write-up (the two kernels are not strictly the
  same algorithm), but the change is in the direction of more correct, and the
  outputs are identical at every workload measured.
- **`PackedCoordinate` / `UnpackedCoordinate`.** Only ever constructed through
  the `(x, y)` constructor and read through `.x` / `.y`, so no union type-punning
  actually occurs. The `packed` members and their constructors are unused.
- **Triangle output.** Identical to baseline (238 034) at every commit on the
  branch, including the intermediate regression at `ab8630e`.
