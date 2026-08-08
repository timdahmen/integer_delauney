# 85834 — Measurements

> **Authorship note.** This file was written by **Claude (Claude Code, Opus 5)** on
> 2026-08-08 at the author's request. Every number in it was measured on the
> author's machine during that session by building each commit and running the
> project's own profiling harness; none are copied from `85834.md` or from any
> earlier run. Where a measurement disagrees with a figure in `85834.md` that is
> stated explicitly.
>
> Companions: [`85834.md`](85834.md) (the author's own account),
> [`85834_technical_notes.md`](85834_technical_notes.md) (what the code does),
> [`85834_review_notes.md`](85834_review_notes.md) (issues found while reading).

---

## 1. Setup

| | |
| --- | --- |
| GPU | NVIDIA T1000 8 GB (`sm_75`, Turing) |
| Toolkit | CUDA 13.3, `-arch=native` |
| Host compiler | MSVC 14.51.36231, Release (`/O2`) |
| OS | Windows 11 Enterprise 26200 |
| Workload | 1024 × 1024 grid, 100 000 seeds, RNG seed 42 (harness defaults) |
| Harness | `src/cuda/profiling.cpp` → `delauney_profiling.exe` |

### Protocol

Each measurement point is **four consecutive runs of the same binary; the first
is discarded and the mean of the last three is reported.** The first run of a
process consistently reads 20–35 % high while the GPU clocks up — it is the
single largest source of error in this harness and discarding it is not optional.
Illustration, from the baseline commit:

```
run 1:  assign 135.2 ms      <- discarded
run 2:  assign 103.8 ms
run 3:  assign 103.8 ms      <- mean of these three is reported
run 4:  assign 103.7 ms
```

Run-to-run spread after the warm-up run is ≤ 1 % for the GPU-event timings and
the batch pipeline, and ≤ 3 % for `Incremental cold` (which carries ~50 ms of
host-side work and is the noisiest row in the table).

Two flag notes:

- Comparisons use `--skip-ref`. The CPU reference sections are irrelevant to the
  CUDA numbers and add ~150 ms of CPU-bound work before them.
- `--only 5,6` does **not** work for this purpose: section 5 needs the Voronoi
  grid that section 3 produces and prints `[no Voronoi grid available -- skipped]`,
  after which section 6 reports an inflated cold assign time. Use plain
  `--skip-ref`.

The two `assign` columns below are distinct measurements of the same kernel:
`assign §5` is the unmasked batch call inside `GridTriangulation`, `assign §6`
is the unmasked cold call inside `IncrementalDelaunay`. Both are `cudaEvent`
timings, not wall clock.

---

## 2. Headline

Baseline is `5e1b060` — the port plus the profiling harness, before any
optimization commit. HEAD is `e5f75ab`.

| Phase | `5e1b060` | `e5f75ab` | Speed-up |
| --- | ---: | ---: | ---: |
| `GridTriangulation` full pipeline | 159.1 ms | 85.8 ms | **1.85×** |
| `IncrementalDelaunay` cold insert | 254.7 ms | 114.0 ms | **2.23×** |
| `IncrementalDelaunay` warm insert | 122.5 ms | 58.1 ms | **2.11×** |
| `assign_triangles_kernel` (§5, batch) | 103.8 ms | 28.9 ms | **3.59×** |
| `assign_triangles_kernel` (§6, cold) | 148.7 ms | 30.4 ms | **4.89×** |
| `find_triangle_seeds_kernel` | 0.3 ms | 0.3 ms | — |
| `thrust` sort + unique | 6.2 ms | 6.2 ms | — |
| Triangles produced | 238 034 | 238 034 | identical |

The two untouched phases are listed deliberately: detection and dedup were
already sub-7 ms and were correctly left alone.

These figures, and the whole progression in §3, are measured at `e5f75ab`. The
CSR capacity fix that followed it
([`85834_review_notes.md`](85834_review_notes.md) finding 1) is correctness-only
and costs ~0.9 ms on the cold insert; it does not move any other row.

---

## 3. Per-commit progression

Every optimization commit on the branch, built and measured under the protocol
in §1. Times in ms; **bold** marks the phase each commit was aimed at.

| # | Commit | Subject | assign §5 | assign §6 | GridTri full | Inc cold | Inc warm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `5e1b060` | baseline + harness | 103.8 | 148.7 | 159.1 | 254.7 | 122.5 |
| 1 | `ddf31b8` | Remove unused param | 103.5 | 148.3 | 158.8 | **228.2** | **88.9** |
| 2 | `eab0393` | Optimize warm-insert | 103.6 | 149.0 | 158.8 | 228.8 | **58.4** |
| 3 | `1413499` | Refactor to CUDA host allocations | 103.5 | 148.8 | 158.9 | 228.1 | **57.5** |
| 4 | `708586a` | Host-side phase timers | 103.6 | 148.8 | 158.6 | 228.1 | 57.4 |
| 5 | `c1e22b4` | Instrument the cold insert path | 103.5 | 149.0 | 158.9 | 228.6 | 57.3 |
| 6 | `ab8630e` | Optimize nearby assign (1/2) | **125.5** | 147.4 | **180.4** | 227.5 | 57.4 |
| 7 | `70d6d8a` | Optimize nearby assign (2/2) | **66.1** | **79.2** | **121.8** | **171.2** | 57.5 |
| 8 | `e5f75ab` | Rebuild CSR + cache on-device | **28.9** | **30.4** | **85.8** | **114.0** | 58.1 |

`9dd0d07` (removal of a leaked `cudaEvent` in the profiling path) is not listed
separately — it is a two-line fix with no measurable effect on these rows.

### Deltas, and what each commit actually bought

| Commit | Effect | Note |
| --- | --- | --- |
| `ddf31b8` | warm **−33.6 ms (−27 %)**, cold **−26.5 ms (−10 %)** | See below — this is undersold in `85834.md` |
| `eab0393` | warm **−30.5 ms (−34 %)** | Matches the 86.4 → 58.0 pair recorded in `85834.md` |
| `1413499` | warm −0.9 ms | Pinned host buffers; see §7 for why this is all it could give |
| `708586a`, `c1e22b4` | ≈ 0 | Instrumentation only; confirms the host timers are not distorting the result |
| `ab8630e` | assign §5 **+22.0 ms (+21 %)** | An intentional intermediate regression |
| `70d6d8a` | assign §5 **−59.4 ms**, assign §6 **−68.2 ms**, cold **−56.3 ms** | The history stack |
| `e5f75ab` | assign §5 **−37.2 ms**, assign §6 **−48.8 ms**, cold **−57.2 ms** | The CSR vertex cache |

Three of these deserve comment.

**`ddf31b8` is not a minor change.** `85834.md` files it under "General changes"
as *"technically an optimization … but it's so minor, it shouldn't count"*, and
the commit message says *"seems like it's genuinely not used"*. Measured, it is
the second-largest single win on the warm path in the entire branch — larger than
the pinned-memory work and larger than everything except the two assign-kernel
commits. Deleting `h_canon_to_tid_`, a 238 000-entry `unordered_map` that was
rebuilt on every insert and read by nobody, removed **27 % of the warm insert**.
This is worth promoting in the submission rather than apologising for.

**`ab8630e` is a measured regression, and it is the honest thing to show.**
Folding the nearby scan into the containment test without any dedup made the
batch pipeline 21 ms slower — `85834.md` predicts "about 20-30 ms" and the
measurement lands at 21.5 ms. Keeping it as its own commit makes the reasoning in
rev 1 → rev 2 legible instead of hiding a failed attempt inside a working one.

**The last two commits are where the bulk is.** Together they take the cold
insert from 227.5 ms to 114.0 ms — 113.5 ms of the 140.7 ms saved overall — and
they account for the entire 74.9 ms saved on the batch pipeline, having first had
to give back the 22.0 ms `ab8630e` cost.

---

## 4. Kernel-level statistics

`ptxas -v`, `sm_75`, both compiled standalone from the same sources
(the CMake build does not surface these):

| kernel / build | registers | stack frame | spills |
| --- | ---: | ---: | ---: |
| `assign_triangles_kernel`, baseline `5e1b060` | 45 | **256 B** | 0 |
| `assign_triangles_kernel`, HEAD `e5f75ab` | 63 | **0 B** | 0 |
| `assign_triangles_kernel`, HEAD + `EXTEND_COORDINATE_LIMIT` | 63 | 0 B | 0 |
| `build_csr_verts_kernel` (new) | 20 | 0 B | 0 |
| `find_triangle_seeds_kernel` | 26 | 0 B | 0 |
| `voronoi_step_kernel` | 18 | 0 B | 0 |
| `remap_tgrid_kernel` | 8 | 0 B | 0 |

The 256-byte stack frame in the baseline is exactly `MAX_NEARBY × 4` — the
`nearby[64]` array in local memory. At HEAD it is zero: the 16-entry history
array is register-resident, which was the point of fixing its size and unrolling
every access to it.

The 45 → 63 register jump costs nothing here. Turing allows 32 warps and 65 536
registers per SM, so up to 64 registers/thread still permits full occupancy — the
kernel gained a register-file scratchpad for free. This also answers the open
question in `85834.md` ("I have no means of verifying if the compiler actually
unrolls the relevant sections"): a 0-byte stack frame is only achievable if both
`#pragma unroll` loops were unrolled and the array promoted to registers. They
were.

Per-candidate memory traffic in the assign inner loop:

| | baseline | HEAD |
| --- | --- | --- |
| accesses per candidate triangle | 7 (1 × 32 B struct + 6 × 4 B gather) | 1 |
| bytes touched | 56 B, of which 24 B used | 12 B, all used |
| index pattern | random (`tid`), then dependent random | sequential (`j`, the CSR slot) |
| resident cache size | — | 8.6 MB (`3T × 12 B`) |

---

## 5. Ablation: coordinate packing

`85834.md` flags the `uint16_t` coordinate packing as a bandwidth trick specific
to this GPU, unsuitable for production, and disabled by `EXTEND_COORDINATE_LIMIT`.
Both builds were measured under the same protocol and the same flags
(`--verify`), so the cost of turning it off is quantified rather than assumed:

| | packed (default) | `EXTEND_COORDINATE_LIMIT` | cost of disabling |
| --- | ---: | ---: | ---: |
| `assign` §5 (batch) | 28.5 ms | 30.3 ms | +1.8 ms (+6.3 %) |
| `assign` §6 (cold) | 30.6 ms | 34.7 ms | **+4.1 ms (+13.4 %)** |
| `GridTriangulation` full | 84.9 ms | 87.0 ms | +2.1 ms (+2.5 %) |
| `Incremental cold` | 111.3 ms | 118.1 ms | +6.8 ms (+6.1 %) |
| `Incremental warm` | 57.5 ms | 58.0 ms | +0.5 ms |
| cache footprint (`3T` slots) | 8.6 MB | 17.1 MB | 2× |
| `--verify` result | pass, 150 px | pass, 150 px | identical |

So the packing is worth ~4 ms on the phase it targets, and disabling it is free
of correctness risk. That is a favourable trade to be able to offer: the
production-safe build costs 6 % of the cold insert and still lands at 118 ms
against a 255 ms baseline.

The invocation, which `85834.md` does not spell out:

```bash
cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON -DBUILD_PYTHON_MODULE=OFF -DCMAKE_CUDA_FLAGS="-DEXTEND_COORDINATE_LIMIT"
```

---

## 6. Correctness

`--verify` builds two `IncrementalDelaunay` instances over the same seeds in the
same order — one inserting everything in a single cold batch, one inserting the
base set cold and then nine seeds one at a time through the warm path — and
compares them. Result, identical across all four runs of the shipped build:

```
    Warm inserts applied                                          9
    Triangles (full rebuild)                                238,062
    Triangles (warm path)                                   238,062
    Triplet sets identical                                      YES
    Pixels assigned to a different triangle                     150
      as a fraction of the grid                              0.014 %
    Triangle IDs out of range (must be 0)                         0
```

The 150-pixel delta (0.014 %) is the expected steady state, not a regression: it
is present at the baseline commit too and is a property of the warm path's
bounded reassign window, not of anything in this branch. The triplet sets being
identical is the strong check — the warm path produces exactly the same
triangulation, and the disagreement is confined to which of several equally valid
triangles a handful of boundary pixels land in.

Additionally verified during this session:

- Triangle count is byte-identical to baseline (238 034) at **every** commit in
  §3, including the intermediate regression at `ab8630e`. No optimization changed
  the output.
- `GridTriangulation` matches the CPU reference triangle count (235 652) on every
  run.
- `compute-sanitizer --tool memcheck` reports **0 errors** for the whole
  incremental pipeline, at the default workload and at the high-density
  workloads that previously overran the CSR device buffers (128² / 12 000 seeds
  went from 710 invalid writes to zero). That overrun is
  [`85834_review_notes.md`](85834_review_notes.md) finding 1; it was found during
  this documentation pass and has since been fixed. It never affected the
  configuration measured here, so every number in this document stands — the fix
  adds ~0.9 ms to `full: rebuild_csr` on the cold insert and nothing to the warm
  path (§7).

---

## 7. Where the time goes now

Host wall-clock breakdown from the shipped build, representative run.

### Cold insert (112 ms)

| phase | ms | % |
| --- | ---: | ---: |
| `full: assign` | 33.16 | 29.7 |
| `full: build registry` | 17.50 | 15.7 |
| `outputs: D2H grids` | 15.33 | 13.8 |
| `full: D2H dedup triangles` | 9.33 | 8.4 |
| `full: rebuild_csr` | 6.39 | 5.7 |
| `insert: register seeds` | 6.33 | 5.7 |
| `full: dedup` | 6.21 | 5.6 |
| `insert: validate seeds` | 6.07 | 5.4 |
| `insert: bfs` | 3.83 | 3.4 |
| `insert: write_seeds kernel` | 3.15 | 2.8 |
| `outputs: interleave` | 2.41 | 2.2 |
| everything else | 1.76 | 1.6 |
| **accounted total** | **111.46** | |

`full: rebuild_csr` carries the one-time `cudaMalloc` pair added by the CSR
capacity fix ([`85834_review_notes.md`](85834_review_notes.md) finding 1); it was
5.50 ms before that change. The warm path is unaffected — see the `partial:
rebuild_csr` row below.

### Warm insert (58 ms)

| phase | ms | % |
| --- | ---: | ---: |
| `outputs: D2H grids` | 15.32 | 27.1 |
| `partial: upload_triangles` | 9.27 | 16.4 |
| `partial: H2D border mask` | 5.32 | 9.4 |
| `partial: H2D reassign mask` | 5.31 | 9.4 |
| `partial: rebuild_csr` | 5.22 | 9.2 |
| `partial: D2H changed mask` | 5.12 | 9.1 |
| `partial: remap t_grid` | 4.15 | 7.4 |
| `outputs: interleave` | 2.39 | 4.2 |
| `partial: mark stale` | 1.43 | 2.5 |
| `insert: bfs` | 0.79 | 1.4 |
| `outputs: fill tri_map` | 0.73 | 1.3 |
| `partial: expand masks` | 0.58 | 1.0 |
| `partial: assign` | 0.34 | 0.6 |
| **`partial: compact registry`** | **0.15** | **0.3** |
| everything else (detect, dedup, D2H new, collect) | 0.18 | 0.3 |
| **accounted total** | **56.43** | |

Two things this table establishes:

**The optimized paths are done.** `compact registry` — the target of `eab0393`,
and originally ~30 ms — is now 0.15 ms, 0.3 % of the insert. `assign` on the warm
path is 0.34 ms. There is nothing left to win in either.

**What remains is the interconnect.** Pull out the raw transfers:

| transfer | size | time | effective rate |
| --- | ---: | ---: | ---: |
| `D2H changed mask` | 4 MB | 5.12 ms | 0.78 GB/s |
| `H2D border mask` | 4 MB | 5.32 ms | 0.75 GB/s |
| `H2D reassign mask` | 4 MB | 5.31 ms | 0.75 GB/s |
| `D2H grids` | 12 MB | 15.32 ms | 0.78 GB/s |
| **total** | **24 MB** | **31.07 ms** | **55 % of the warm insert** |

~0.78 GB/s, symmetric in both directions, and stable to two decimals across every
run in this session. These are `cudaHostAlloc` pinned buffers on the default
stream; a healthy PCIe 3.0 ×16 link should deliver an order of magnitude more.
This is the link-width problem `85834.md` describes, and it is what bounds the
warm insert at ~58 ms: 55 % of the remaining time is transfer running at a fixed
rate the code cannot influence, plus 16 % re-uploading the full triangle array.

It also explains, quantitatively, why `1413499` returned only 0.9 ms and why the
stream/event pipelining tried in rev 2 came out *worse*. On a link this slow the
transfers are long enough that overlapping them buys little, while the CPU-side
event and stream bookkeeping is pure added cost. The negative results in
`85834.md` are the correct conclusions from the hardware, not missed
opportunities.

---

## 8. Reproducing

```bash
cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON -DBUILD_PYTHON_MODULE=OFF
```

```bash
cmake --build build/native --config Release --target delauney_profiling
```

Correctness (what to hand in):

```bash
build/native/Release/delauney_profiling.exe --verify
```

Timing, following the protocol in §1 — run four times, take the mean of the last
three:

```bash
build/native/Release/delauney_profiling.exe --skip-ref
```

Register/occupancy statistics (Git Bash; `MSYS2_ARG_CONV_EXCL` stops MSYS from
mangling the `/`-prefixed MSVC flags, and the MSVC `HostX64/x64` directory must
be on `PATH`):

```bash
MSYS2_ARG_CONV_EXCL="*" nvcc -c -o /dev/null -x cu src/cuda/triangulation.cu -Isrc/cuda -std=c++17 -arch=native --expt-relaxed-constexpr -Xptxas -v -Xcompiler=/EHsc,/O2,/MD,/nologo,/Zc:preprocessor
```

Memory checking:

```bash
compute-sanitizer --tool memcheck build/native/Release/delauney_profiling.exe --only 6 -w 256 -h 256 -n 5000 --warm 2
```
