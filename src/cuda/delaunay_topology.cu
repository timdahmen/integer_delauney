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

#include <cstdint>
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
// upload_triangles_: sync h_triangles_ -> d_raw_buf_
// ---------------------------------------------------------------------------

void Delaunay::upload_triangles_()
{
    upload_triangles_range_(0, (int)h_triangles_.size());
}

void Delaunay::upload_triangles_range_(int first, int count)
{
    if (count <= 0) return;
    std::vector<RawTriangle> h_raw(count);
    for (int i = 0; i < count; ++i) {
        const auto& h = h_triangles_[first + i];
        h_raw[i] = {h.x, h.y, h.a, h.b, h.c, h.orig_a, h.orig_b, h.orig_c};
    }
    CUDA_CHECK(cudaMemcpy(static_cast<RawTriangle*>(d_raw_buf_) + first, h_raw.data(),
               (size_t)count * sizeof(RawTriangle), cudaMemcpyHostToDevice));
}

void Delaunay::upload_dead_flags_()
{
    if (h_dead_.empty()) return;
    CUDA_CHECK(cudaMemcpy(d_dead_, h_dead_.data(), h_dead_.size() * sizeof(uint8_t),
               cudaMemcpyHostToDevice));
}

//: Compact once the retired slots outnumber the live ones, or the slot count
//: approaches the device buffers' bound. Amortised: each compaction is
//: O(triangles) but at least halves the slot count.
bool Delaunay::should_compact_() const
{
    const int slots = (int)h_triangles_.size();
    if (slots == 0) return false;
    if (slots > 2 * n_live_ + 1024) return true;
    return slots > max_seeds_ * 4 - 4096;
}

void Delaunay::compact_registry_()
{
    const int old_count = (int)h_triangles_.size();
    if (old_count == n_live_) return;          // already dense

    std::vector<int32_t> remap(old_count, TID_DELETED);
    std::vector<HTriangle> live;
    live.reserve(n_live_);
    for (int tid = 0; tid < old_count; ++tid) {
        if (h_dead_[tid]) continue;
        remap[tid] = (int32_t)live.size();
        live.push_back(h_triangles_[tid]);
    }

    h_triangles_ = std::move(live);
    h_dead_.assign(h_triangles_.size(), 0);
    n_live_ = (int)h_triangles_.size();

    h_triplet_to_tid_.clear();
    h_triplet_to_tid_.reserve(h_triangles_.size() * 2);
    for (int tid = 0; tid < (int)h_triangles_.size(); ++tid) {
        const auto& t = h_triangles_[tid];
        h_triplet_to_tid_[pack_triplet_(t.a, t.b, t.c)] = tid;
    }

    upload_triangles_();
    upload_dead_flags_();
    csr_dirty_ = true;
    edges_dirty_ = true;

    const int N = W_det_ * H_det_;
    CUDA_CHECK(cudaMemcpy(d_remap_, remap.data(), (size_t)old_count * sizeof(int32_t),
               cudaMemcpyHostToDevice));
    remap_tgrid_kernel<<<(N+255)/256, 256>>>(d_t_grid_, N, d_remap_, old_count,
                                             NO_TRIANGLE);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// rebuild_csr_and_upload_
// ---------------------------------------------------------------------------

void Delaunay::rebuild_csr_and_upload_()
{
    // Retired slots are skipped: assign_triangles_kernel reaches a triangle
    // only through this index, so leaving them out is what keeps a dead slot
    // from ever being tested against a pixel.
    std::vector<int32_t> h_csr_ptr, h_csr_idx;
    build_seed_triangle_csr(h_triangles_, N_,
        [this](int tid) { return h_dead_[tid] != 0; },
        h_csr_ptr, h_csr_idx);

    CUDA_CHECK(cudaMemcpy(d_csr_ptr_, h_csr_ptr.data(), (N_+1)*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_idx_, h_csr_idx.data(), h_csr_idx.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
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

    std::vector<RawTriangle> h_dedup(N_tri);
    CUDA_CHECK(cudaMemcpy(h_dedup.data(), d_raw, N_tri * sizeof(RawTriangle), cudaMemcpyDeviceToHost));

    h_triangles_.clear(); h_triplet_to_tid_.clear();
    h_triangles_.reserve(N_tri);
    for (int32_t tid = 0; tid < N_tri; ++tid) {
        const auto& r = h_dedup[tid];
        h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
        h_triplet_to_tid_[pack_triplet_(r.a, r.b, r.c)] = tid;
    }
    // A full build leaves no holes.
    h_dead_.assign(N_tri, 0);
    n_live_ = N_tri;
    upload_dead_flags_();
    // Detection wrote to the scratch buffer, so the list gets its own copy.
    // Device to device, and the order already matches h_triangles_.
    if (N_tri > 0)
        CUDA_CHECK(cudaMemcpy(d_raw_buf_, d_raw, (size_t)N_tri * sizeof(RawTriangle),
                   cudaMemcpyDeviceToDevice));

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
    int old_count = (int)h_triangles_.size();
    std::vector<uint8_t> h_stale(old_count, 0);
    if (old_count > 0) {
        mark_stale_kernel<<<(old_count + 255) / 256, 256>>>(
            static_cast<RawTriangle*>(d_raw_buf_), old_count,
            d_mask_, W_det_, H_det_, d_stale_);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_stale.data(), d_stale_, old_count * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost));
    }
    auto is_stale = [&h_stale](int tid) { return h_stale[tid] != 0; };

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int n_new = detect_and_dedup_(d_mask_, det_ms, dedup_ms);

    std::vector<RawTriangle> h_new(n_new);
    if (n_new > 0)
        CUDA_CHECK(cudaMemcpy(h_new.data(), d_raw, n_new * sizeof(RawTriangle), cudaMemcpyDeviceToHost));

    // Retire the invalidated triangles. Their ids are not reused and nothing
    // else is renumbered, so this touches only the entries that changed --
    // which is the whole point. Renumbering instead cost a full map rebuild, a
    // full device upload and a grid remap on every insert, all proportional to
    // the total triangle count rather than to the size of the change.
    for (int tid = 0; tid < old_count; ++tid) {
        if (h_dead_[tid] || !is_stale(tid)) continue;
        const auto& t = h_triangles_[tid];
        h_triplet_to_tid_.erase(pack_triplet_(t.a, t.b, t.c));
        h_dead_[tid] = 1;
        --n_live_;
    }

    // Append whatever the re-detection found that the registry no longer holds.
    // A retired triangle that is still valid geometry comes back through here
    // and takes a fresh slot, which is what the renumbering path did too.
    const int append_first = (int)h_triangles_.size();
    for (const auto& r : h_new) {
        auto key = pack_triplet_(r.a, r.b, r.c);
        if (h_triplet_to_tid_.count(key)) continue;
        h_triplet_to_tid_[key] = (int32_t)h_triangles_.size();
        h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c,
                                r.orig_a, r.orig_b, r.orig_c});
        h_dead_.push_back(0);
        ++n_live_;
    }
    const int append_count = (int)h_triangles_.size() - append_first;

    // Only the tail goes up; the earlier slots are untouched on the device.
    // mark_stale_kernel and get_edges read d_raw_buf_, so it has to stay in
    // step. The CSR does not -- see full_topology_.
    upload_triangles_range_(append_first, append_count);
    upload_dead_flags_();
    csr_dirty_ = true;

    // Pixels naming a retired triangle are cleared. Ids did not move, so the
    // old remap table is unnecessary: the flags say which ids are gone.
    const int N = W_det_ * H_det_;
    invalidate_dead_tgrid_kernel<<<(N+255)/256, 256>>>(
        d_t_grid_, N, d_dead_, (int)h_triangles_.size());
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Amortised: holes are cheap to carry but not free to carry forever, and
    // the device buffers are sized for a bounded slot count.
    if (should_compact_()) compact_registry_();
}
