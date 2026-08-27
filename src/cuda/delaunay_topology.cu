// Triangle topology: detect + dedup + registry + CSR, either over the whole
// grid (first insert) or scoped to what an insert actually touched.
//
// Triangle ids are slots, not a dense list: a partial update retires the
// triangles a change invalidated and appends their replacements, so
// everything else keeps its id and only the entries that actually moved touch
// the map or the device. Compacting instead would renumber every triangle on
// every insert; compact_registry_ restores density once, in finalise(),
// rather than paying for it per insert. See delaunay.cuh for the fuller
// rationale on both counts.
#include "delaunay.cuh"
#include "triangle_detect.cuh"
#include "triangle_csr.cuh"
#include "phase_timer.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>

#include <cstdint>
#include <utility>
#include <vector>

//: An entry in the old->new remap for a triangle that did not survive
//: compaction. Distinct from NO_TRIANGLE, which is what the *pixel* grid then
//: receives for those triangles: this one says "this id is gone", the other
//: says "no triangle covers this pixel".
static constexpr int32_t TID_DELETED = -1;

// ---------------------------------------------------------------------------
// Kernel: 2x2-block triangle detection with optional mask
//
// The rule itself is shared with the batch path -- see triangle_detect.cuh,
// which explains why: a purely pixel-based tie-break can make the two paths
// cut a cocircular quad differently and register overlapping triangles.
//
// What stays here is only what genuinely differs. The grid is interleaved
// (seed_id, distance) rather than plain seed ids, and detection can be scoped
// to a mask so a deferred round does not re-detect earlier rounds' regions.
// ---------------------------------------------------------------------------

__global__
void find_triangle_seeds_kernel(
    const int32_t* __restrict__ grid,   // interleaved (seed_id, dist)
    int W, int H,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys,
    RawTriangle* __restrict__ raw_buf,
    int32_t* __restrict__ counter,
    const int32_t* __restrict__ mask)   // nullptr -> all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W - 1 || y >= H - 1) return;   // 2x2 block must be in bounds
    if (mask && !mask[y * W + x]) return;

    // Interleaved (seed_id, distance) grid here; the batch path's is a plain
    // seed-id array. That layout difference, and the mask above, are the only
    // things this kernel adds to the shared rule.
    auto sid = [&](int cx, int cy) -> int32_t { return grid[(cy * W + cx) * 2]; };

    detect_block_triangles(
        sid(x,     y    ),                  // top-left
        sid(x + 1, y    ),                  // top-right
        sid(x,     y + 1),                  // bottom-left
        sid(x + 1, y + 1),                  // bottom-right
        seed_xs, seed_ys,
        [&](int32_t oa, int32_t ob, int32_t oc) {
            append_raw_triangle(raw_buf, counter, x, y, oa, ob, oc);
        });
}

// Dilate a 0/1 mask by a fixed radius. Used for the detection border, where
// the radius is 2 and a direct gather is cheaper than any prefilter.
__global__
void dilate_fixed_kernel(const int32_t* __restrict__ src,
                         int32_t* __restrict__ dst,
                         int W, int H, int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int x0 = max(0, x-r), x1 = min(W-1, x+r);
    int y0 = max(0, y-r), y1 = min(H-1, y+r);
    int32_t v = 0;
    for (int sy = y0; sy <= y1 && !v; ++sy)
        for (int sx = x0; sx <= x1; ++sx)
            if (src[sy * W + sx]) { v = 1; break; }
    dst[y * W + x] = v;
}

// Flag triangles whose canonical pixel falls inside the mask.
//
// Sampling the mask here rather than downloading it keeps the transfer
// proportional to the triangle count (~50k flags) instead of the pixel count
// (~1.5M int32). Reads d_raw_buf_, so it must run before detection overwrites
// that buffer.
__global__
void mark_stale_kernel(const RawTriangle* __restrict__ tris, int n_tri,
                       const int32_t* __restrict__ mask, int W, int H,
                       uint8_t* __restrict__ stale)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;
    const RawTriangle& r = tris[t];
    stale[t] = (r.x >= 0 && r.x < W && r.y >= 0 && r.y < H && mask[r.y * W + r.x])
             ? 1 : 0;
}

// Predicate for compact_stale_tids_: a tid belongs in the compacted list if
// mark_stale_kernel flagged it AND it was not already retired by an earlier
// round. d_dead_ reflects state as of the end of the previous round, which is
// exactly what "already retired" should mean here.
struct IsStaleAndLive {
    const uint8_t* __restrict__ stale;
    const uint8_t* __restrict__ dead;
    __device__ bool operator()(int32_t tid) const {
        return stale[tid] != 0 && dead[tid] == 0;
    }
};

// ---------------------------------------------------------------------------
// Kernels: d_centroid_index_ maintenance and the registry itself.
//
// A triangle's key is the exact (unrounded) sum of its three vertices' seed
// coordinates -- see d_centroid_index_'s doc comment in delaunay.cuh for why
// this position is provably unique per live triangle. d_raw_buf_/d_dead_ are
// the registry's source of truth; nothing here mirrors a host copy.
// ---------------------------------------------------------------------------

__device__ static __forceinline__ int32_t centroid_key(
    int32_t a, int32_t b, int32_t c,
    const int32_t* __restrict__ seed_xs, const int32_t* __restrict__ seed_ys, int CW)
{
    int32_t sx = seed_xs[a] + seed_xs[b] + seed_xs[c];
    int32_t sy = seed_ys[a] + seed_ys[b] + seed_ys[c];
    return sy * CW + sx;
}

// index[key(triangles[i])] = tid_base + i, for i in [0, count). Seeds the
// index from a contiguous, freshly-tid'd run of triangles: full_topology_
// (tid_base = 0, the whole grid) and compact_registry_ (tid_base = 0, the
// renumbered survivors), both after rebuild_centroid_index_ clears it first.
__global__
void write_centroid_index_kernel(
    int32_t* __restrict__ index,
    const RawTriangle* __restrict__ triangles, int32_t tid_base, int count,
    const int32_t* __restrict__ seed_xs, const int32_t* __restrict__ seed_ys, int CW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const RawTriangle& r = triangles[i];
    index[centroid_key(r.a, r.b, r.c, seed_xs, seed_ys, CW)] = tid_base + i;
}

// Retires each of this round's stale tids directly: marks it dead and clears
// its centroid-index slot, reading the triangle's own (not yet overwritten)
// data straight out of d_raw_buf_ -- the same buffer and tids
// mark_stale_kernel already tested. The caller updates n_live_ from n_stale
// alone.
__global__
void retire_triangles_kernel(
    uint8_t* __restrict__ dead,
    int32_t* __restrict__ centroid_index,
    const RawTriangle* __restrict__ registry,
    const int32_t* __restrict__ stale_tids, int n_stale,
    const int32_t* __restrict__ seed_xs, const int32_t* __restrict__ seed_ys, int CW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_stale) return;
    int32_t tid = stale_tids[i];
    dead[tid] = 1;
    const RawTriangle& r = registry[tid];
    centroid_index[centroid_key(r.a, r.b, r.c, seed_xs, seed_ys, CW)] = NO_TRIANGLE;
}

// Which of this round's deduped detection candidates are genuinely new (not
// already registered), flagged 1/0 for the caller's exclusive scan to rank.
__global__
void mark_new_candidates_kernel(
    const int32_t* __restrict__ centroid_index,
    const RawTriangle* __restrict__ candidates, int n_new,
    const int32_t* __restrict__ seed_xs, const int32_t* __restrict__ seed_ys, int CW,
    int32_t* __restrict__ is_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_new) return;
    const RawTriangle& r = candidates[i];
    int32_t key = centroid_key(r.a, r.b, r.c, seed_xs, seed_ys, CW);
    is_new[i] = (centroid_index[key] == NO_TRIANGLE) ? 1 : 0;
}

// rank[i], after the caller's in-place exclusive scan over
// mark_new_candidates_kernel's output, holds candidate i's count of new
// candidates before it in array order -- the same order detect_and_dedup_'s
// thrust::unique already sorted candidates into (by vertex triplet), so a
// genuinely new candidate's tid is tid_base + rank[i]. Tid assignment must
// stay deterministic in this array order: insert_deferred()+finalise() has
// to assign the same tids insert() would for the same seeds, regardless of
// how the insert was batched -- test_five_rounds_matches_immediate and
// test_matches_batch_after_several_deferred_inserts check this directly.
// Re-tests the identity check directly: the scan below overwrites rank[]'s
// original 0/1 flags with prefix counts, and nothing else touches
// centroid_index between mark_new_candidates_kernel and here, so the two
// reads still agree. Must run after retire_triangles_kernel has completed
// for this round (stream-ordered, no explicit sync needed) --
// a triangle re-detected at a stale position needs its old slot cleared
// before this check runs, or it is retired without being re-registered and
// silently disappears.
__global__
void append_triangles_kernel(
    RawTriangle* __restrict__ registry,
    uint8_t* __restrict__ dead,
    int32_t* __restrict__ centroid_index,
    const int32_t* __restrict__ rank, int32_t tid_base,
    const RawTriangle* __restrict__ candidates, int n_new,
    const int32_t* __restrict__ seed_xs, const int32_t* __restrict__ seed_ys, int CW)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_new) return;
    const RawTriangle& r = candidates[i];
    int32_t key = centroid_key(r.a, r.b, r.c, seed_xs, seed_ys, CW);
    if (centroid_index[key] != NO_TRIANGLE) return;   // already registered
    int32_t new_tid = tid_base + rank[i];
    registry[new_tid] = r;
    dead[new_tid] = 0;
    centroid_index[key] = new_tid;
}

// compact_registry_'s per-tid live flag, used both to scan (new dense tid =
// count of live slots before this one) and, combined with the scan result in
// finalize_remap_kernel below, to build the remap array in one pass.
struct IsLive {
    const uint8_t* __restrict__ dead;
    __device__ int32_t operator()(int32_t tid) const { return dead[tid] ? 0 : 1; }
};

// remap[] already holds each live tid's new dense position (the exclusive
// scan's output); this just overwrites the dead slots with TID_DELETED,
// in place -- each thread only ever touches its own index.
__global__
void finalize_remap_kernel(int32_t* __restrict__ remap,
                           const uint8_t* __restrict__ dead, int old_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= old_count) return;
    if (dead[tid]) remap[tid] = TID_DELETED;
}

// Scatters each surviving triangle from src[tid] to dst[remap[tid]].
__global__
void scatter_compact_kernel(const RawTriangle* __restrict__ src,
                            RawTriangle* __restrict__ dst,
                            const int32_t* __restrict__ remap, int old_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= old_count) return;
    int32_t new_tid = remap[tid];
    if (new_tid == TID_DELETED) return;
    dst[new_tid] = src[tid];
}

// ---------------------------------------------------------------------------
// Kernel: remap triangle IDs in t_grid (after compaction)
// ---------------------------------------------------------------------------

__global__
void remap_tgrid_kernel(int32_t* __restrict__ t_grid, int N,
                        const int32_t* __restrict__ remap, int remap_size,
                        int32_t fallback)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t old_tid = t_grid[i];
    // Already uncovered, out of range, or the triangle did not survive
    // compaction. The last case is why finalise() can reassign a pixel whose
    // own neighbourhood never changed: build_reassign_mask_kernel tests for the
    // fallback directly, since no dilation around changed cells would find it.
    if (old_tid == NO_TRIANGLE || old_tid >= remap_size
            || remap[old_tid] == TID_DELETED)
        t_grid[i] = fallback;
    else
        t_grid[i] = remap[old_tid];
}

//: Clear pixels whose triangle has been retired.
//:
//: The counterpart to remap_tgrid_kernel for the incremental path, where ids do
//: not move: a pixel is either still covered by the triangle it names, or that
//: triangle is gone and the pixel must be reassigned.
//: build_reassign_mask_kernel picks these up by testing for NO_TRIANGLE, which
//: is how a pixel is caught when the change that retired its triangle lay
//: outside its own search window.
__global__
void invalidate_dead_tgrid_kernel(int32_t* __restrict__ t_grid, int N,
                                  const uint8_t* __restrict__ dead, int n_slots)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t tid = t_grid[i];
    if (tid == NO_TRIANGLE) return;
    if (tid >= n_slots || dead[tid]) t_grid[i] = NO_TRIANGLE;
}

// ---------------------------------------------------------------------------
// rebuild_centroid_index_ / download_registry_
// ---------------------------------------------------------------------------

void Delaunay::rebuild_centroid_index_(int count)
{
    CUDA_CHECK(cudaMemset(d_centroid_index_, SENTINEL_BYTE,
               (size_t)centroid_index_w_ * centroid_index_h_ * sizeof(int32_t)));
    if (count > 0) {
        write_centroid_index_kernel<<<(count + 255) / 256, 256>>>(
            d_centroid_index_, static_cast<RawTriangle*>(d_raw_buf_), 0, count,
            d_sx_, d_sy_, centroid_index_w_);
        CUDA_CHECK_LAST_ERROR();
    }
}

void Delaunay::download_registry_(std::vector<RawTriangle>& tris,
                                  std::vector<uint8_t>& dead) const
{
    const int slots = next_tid_host_;
    tris.resize(slots);
    dead.resize(slots);
    if (slots > 0) {
        CUDA_CHECK(cudaMemcpy(tris.data(), d_raw_buf_, (size_t)slots * sizeof(RawTriangle),
                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(dead.data(), d_dead_, (size_t)slots * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost));
    }
}

int Delaunay::compact_stale_tids_(int old_count)
{
    if (old_count == 0) return 0;
    thrust::counting_iterator<int32_t> begin(0), end(old_count);
    thrust::device_ptr<int32_t> out(d_stale_tids_);
    IsStaleAndLive pred{d_stale_, d_dead_};
    auto out_end = thrust::copy_if(thrust::device, begin, end, out, pred);
    return (int)(out_end - out);
}

//: Compact once the retired slots outnumber the live ones, or the slot count
//: approaches the device buffers' bound. Amortised: each compaction is
//: O(triangles) but at least halves the slot count.
bool Delaunay::should_compact_() const
{
    const int slots = next_tid_host_;
    if (slots == 0) return false;
    if (slots > 2 * n_live_ + 1024) return true;
    return slots > max_seeds_ * 4 - 4096;
}

void Delaunay::compact_registry_()
{
    const int old_count = next_tid_host_;
    if (old_count == n_live_) return;          // already dense

    // Exclusive scan of "is this slot live" gives each survivor's new dense
    // tid directly -- that already *is* the remap array's content for live
    // slots; finalize_remap_kernel fills in TID_DELETED for the rest.
    thrust::counting_iterator<int32_t> cbegin(0), cend(old_count);
    IsLive is_live{d_dead_};
    thrust::device_ptr<int32_t> scan_out(d_remap_);
    thrust::exclusive_scan(thrust::device,
        thrust::make_transform_iterator(cbegin, is_live),
        thrust::make_transform_iterator(cend, is_live),
        scan_out);
    finalize_remap_kernel<<<(old_count + 255) / 256, 256>>>(d_remap_, d_dead_, old_count);
    CUDA_CHECK_LAST_ERROR();

    // Scatter survivors into the ping-pong buffer, then swap -- same pattern
    // d_grid_/d_tmp_ already use for BFS.
    scatter_compact_kernel<<<(old_count + 255) / 256, 256>>>(
        static_cast<RawTriangle*>(d_raw_buf_), static_cast<RawTriangle*>(d_raw_buf_compact_),
        d_remap_, old_count);
    CUDA_CHECK_LAST_ERROR();
    std::swap(d_raw_buf_, d_raw_buf_compact_);

    CUDA_CHECK(cudaMemset(d_dead_, 0, (size_t)n_live_ * sizeof(uint8_t)));
    next_tid_host_ = n_live_;
    rebuild_centroid_index_(n_live_);

    csr_dirty_ = true;
    edges_dirty_ = true;

    const int N = W_det_ * H_det_;
    remap_tgrid_kernel<<<(N+255)/256, 256>>>(d_t_grid_, N, d_remap_, old_count,
                                             NO_TRIANGLE);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// rebuild_csr_and_upload_
// ---------------------------------------------------------------------------

// One (seed, tid) pair per corner of every registry slot, live or dead --
// a dead slot's three pairs are tagged with the sentinel seed id n_seeds
// (one past the valid range), so the sort below pushes them past every
// real seed's range.
__global__
void emit_csr_pairs_kernel(
    const RawTriangle* __restrict__ registry, const uint8_t* __restrict__ dead,
    int n_tri, int32_t n_seeds,
    int32_t* __restrict__ pair_seed, int32_t* __restrict__ pair_tid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tri) return;
    int32_t s0 = n_seeds, s1 = n_seeds, s2 = n_seeds;
    if (!dead[tid]) {
        const RawTriangle& r = registry[tid];
        s0 = r.orig_a; s1 = r.orig_b; s2 = r.orig_c;
    }
    pair_seed[tid * 3]     = s0; pair_tid[tid * 3]     = tid;
    pair_seed[tid * 3 + 1] = s1; pair_tid[tid * 3 + 1] = tid;
    pair_seed[tid * 3 + 2] = s2; pair_tid[tid * 3 + 2] = tid;
}

void Delaunay::rebuild_csr_and_upload_()
{
    const int n_tri = next_tid_host_;
    if (n_tri == 0) {
        CUDA_CHECK(cudaMemset(d_csr_ptr_, 0, (size_t)(N_ + 1) * sizeof(int32_t)));
        return;
    }

    // Retired slots are skipped: assign_triangles_kernel reaches a triangle
    // only through this index, so keeping them past d_csr_ptr_[N_] (via the
    // sentinel tag below) is what keeps a dead slot from ever being tested
    // against a pixel.
    emit_csr_pairs_kernel<<<(n_tri + 255) / 256, 256>>>(
        static_cast<RawTriangle*>(d_raw_buf_), d_dead_, n_tri, N_,
        d_csr_pair_seed_, d_csr_idx_);
    CUDA_CHECK_LAST_ERROR();

    // Sorting d_csr_idx_'s tids by d_csr_pair_seed_'s keys groups every
    // seed's triangles together, in place: d_csr_idx_ becomes the CSR's
    // index array directly.
    thrust::device_ptr<int32_t> seed_keys(d_csr_pair_seed_);
    thrust::device_ptr<int32_t> tid_vals(d_csr_idx_);
    thrust::sort_by_key(thrust::device, seed_keys, seed_keys + (size_t)n_tri * 3, tid_vals);

    // d_csr_ptr_[s] = the sorted array's first position with seed id >= s,
    // for s in [0, N_] -- exactly the CSR row-start array, since the sort
    // above already grouped equal seed ids contiguously. Dead slots'
    // sentinel-tagged pairs sort past every s < N_, so d_csr_ptr_[N_] is the
    // boundary past which nothing valid is ever read.
    thrust::counting_iterator<int32_t> search_begin(0);
    thrust::device_ptr<int32_t> csr_ptr_out(d_csr_ptr_);
    thrust::lower_bound(thrust::device, seed_keys, seed_keys + (size_t)n_tri * 3,
                        search_begin, search_begin + (N_ + 1), csr_ptr_out);
}

// ---------------------------------------------------------------------------
// detect_and_dedup_: the step full_topology_ and partial_topology_ do
// identically, aside from the mask they scope detection to.
// ---------------------------------------------------------------------------

int Delaunay::detect_and_dedup_(const int32_t* mask, float* detect_ms, float* dedup_ms)
{
    PhaseTimer<4> timer(detect_ms != nullptr);

    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int32_t* d_counter = d_tri_count_;
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int32_t)));

    timer.mark(0);
    find_triangle_seeds_kernel<<<grid_dim, block>>>(
        d_grid_, W_det_, H_det_, d_sx_, d_sy_, d_raw, d_counter, mask);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
    int32_t raw_count = 0;
    CUDA_CHECK(cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost));
    timer.mark(1);

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);
    timer.mark(2);
    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    timer.mark(3);

    if (detect_ms) *detect_ms = timer.elapsed_ms(0, 1);
    if (dedup_ms)  *dedup_ms  = timer.elapsed_ms(2, 3);
    return (int)(new_end - d_ptr);
}

// ---------------------------------------------------------------------------
// full_topology_: detect -> dedup -> registry -> CSR, over the whole grid
// ---------------------------------------------------------------------------

void Delaunay::full_topology_(float* det_ms, float* dedup_ms)
{
    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int N_tri = detect_and_dedup_(nullptr, det_ms, dedup_ms);

    // Detection wrote to the scratch buffer, so the registry gets its own
    // copy, device to device. The order out of detect_and_dedup_ is already
    // dense, so tid == own index.
    if (N_tri > 0)
        CUDA_CHECK(cudaMemcpy(d_raw_buf_, d_raw, (size_t)N_tri * sizeof(RawTriangle),
                   cudaMemcpyDeviceToDevice));
    // A full build leaves no holes.
    CUDA_CHECK(cudaMemset(d_dead_, 0, (size_t)N_tri * sizeof(uint8_t)));
    n_live_ = N_tri;
    next_tid_host_ = N_tri;
    rebuild_centroid_index_(N_tri);

    // The CSR is not built here. Its only reader is assign_triangles_kernel,
    // which runs in assign_pending_, so building it per insert was O(N_tri +
    // N_seeds) of host work that a deferred round never used.
    csr_dirty_ = true;

    // Every pixel's assignment is now stale. Marking them invalidated rather
    // than assigning here lets assign_pending_ pick the mask up like any other
    // dirty region, and makes a first insert behave like the rest.
    CUDA_CHECK(cudaMemset(d_t_grid_, SENTINEL_BYTE, (size_t)W_det_ * H_det_ * sizeof(int32_t)));
}

// ---------------------------------------------------------------------------
// partial_topology_: use d_changed_ to scope detection and assignment
// ---------------------------------------------------------------------------

void Delaunay::partial_topology_(float* det_ms, float* dedup_ms)
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    // Detection border: this insert's changes expanded by 2, which is the reach
    // of the L-shaped stencil in find_triangle_seeds_kernel. Scoped to
    // d_changed_ rather than the accumulator so a deferred round does not
    // re-detect regions earlier rounds already handled.
    dilate_fixed_kernel<<<grid_dim, block>>>(d_changed_, d_mask_, W_det_, H_det_, 2);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Which existing triangles the border invalidates. Sampling the mask at the
    // triangles' own canonical pixels moves N_tri flags instead of the whole
    // W*H mask, and d_raw_buf_ already holds those positions on the device in
    // registry order -- but only until detection overwrites it below, so this
    // has to happen first.
    int old_count = next_tid_host_;
    int n_stale = 0;
    if (old_count > 0) {
        mark_stale_kernel<<<(old_count + 255) / 256, 256>>>(
            static_cast<RawTriangle*>(d_raw_buf_), old_count,
            d_mask_, W_det_, H_det_, d_stale_);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compacted on the device: old_count can be the whole registry, but a
        // change typically retires a handful, so this stays proportional to
        // that handful, not every slot.
        n_stale = compact_stale_tids_(old_count);
        if (n_stale > 0) {
            // Must complete before the append kernel's "already registered"
            // check below runs (stream-ordered, no explicit sync needed) --
            // see retire_triangles_kernel's own comment for why a
            // re-detected-but-still-valid triangle would otherwise be lost.
            retire_triangles_kernel<<<(n_stale + 255) / 256, 256>>>(
                d_dead_, d_centroid_index_, static_cast<RawTriangle*>(d_raw_buf_),
                d_stale_tids_, n_stale, d_sx_, d_sy_, centroid_index_w_);
            CUDA_CHECK_LAST_ERROR();
            n_live_ -= n_stale;
        }
    }

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int n_new = detect_and_dedup_(d_mask_, det_ms, dedup_ms);

    // Append whatever the re-detection found that d_centroid_index_ does not
    // already carry -- a retired triangle that is still valid geometry takes
    // a fresh slot here. The identity check, tid ranking and registry write
    // all happen in the two kernels below; n_appended is the one value that
    // comes back to the host, from thrust::reduce.
    if (n_new > 0) {
        mark_new_candidates_kernel<<<(n_new + 255) / 256, 256>>>(
            d_centroid_index_, d_raw, n_new, d_sx_, d_sy_, centroid_index_w_,
            d_new_rank_);
        CUDA_CHECK_LAST_ERROR();

        thrust::device_ptr<int32_t> rank(d_new_rank_);
        const int32_t n_appended = thrust::reduce(thrust::device, rank, rank + n_new, 0);
        // In-place: rank[i] becomes "how many new candidates before i", the
        // same order append_triangles_kernel re-derives from centroid_index,
        // so a stale 1/0 flag is never read after this point.
        thrust::exclusive_scan(thrust::device, rank, rank + n_new, rank);

        if (n_appended > 0) {
            append_triangles_kernel<<<(n_new + 255) / 256, 256>>>(
                static_cast<RawTriangle*>(d_raw_buf_), d_dead_, d_centroid_index_,
                d_new_rank_, next_tid_host_, d_raw, n_new, d_sx_, d_sy_, centroid_index_w_);
            CUDA_CHECK_LAST_ERROR();
        }
        n_live_ += n_appended;
        next_tid_host_ += n_appended;
    }

    csr_dirty_ = true;

    // Pixels naming a retired triangle are cleared. Ids did not move, so the
    // old remap table is unnecessary: the flags say which ids are gone.
    const int N = W_det_ * H_det_;
    invalidate_dead_tgrid_kernel<<<(N+255)/256, 256>>>(
        d_t_grid_, N, d_dead_, next_tid_host_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Amortised: holes are cheap to carry but not free to carry forever, and
    // the device buffers are sized for a bounded slot count.
    if (should_compact_()) compact_registry_();
}
