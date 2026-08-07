// CUDA implementation of IncrementalDelaunay.
//
// insert(batch) pipeline:
//   1. Write new seeds into d_grid_ at distance 0.
//   2. BFS until convergence; d_changed_ accumulates every cell that moved.
//   3. First insert → full triangulation.
//      Subsequent inserts → partial triangulation:
//        a. Expand d_changed_ by 2 px → border mask (re-detect here).
//        b. Remove triangles whose canonical pixel is in border.
//        c. Re-detect in border, merge new triplets.
//        d. Remap d_t_grid_ through old→new ID table.
//        e. Expand d_changed_ by WINDOW_CAP → reassign mask.
//        f. Re-assign only pixels in reassign mask.

#include "incremental.cuh"
#include "shared_utils.cuh"

#include "nvtx_range.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// ---------------------------------------------------------------------------
// Internal types (parallel to triangulation.cu internals)
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEF_SEED = -1;

// Scoped host wall-clock accumulator. A null sink disables it entirely, so
// non-profiling callers pay only a branch.
namespace {
struct HostTimer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0;
    float* out;
    explicit HostTimer(float* o) : out(o) { if (out) t0 = Clock::now(); }
    ~HostTimer() {
        if (out)
            *out += std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    }
    HostTimer(const HostTimer&) = delete;
    HostTimer& operator=(const HostTimer&) = delete;
};
}  // namespace

// Times the enclosing scope into ht_->field. Note that CUDA calls are async:
// these measure launch cost plus whatever the following sync absorbs.
#define DELAUNEY_HOST_TIME(field) \
    HostTimer _host_timer_(ht_ ? &ht_->field : nullptr)

struct RawLess {
    __device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
        if (p.a != q.a) return p.a < q.a;
        if (p.b != q.b) return p.b < q.b;
        return p.c < q.c;
    }
};

struct RawEqual {
    __device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
        return p.a == q.a && p.b == q.b && p.c == q.c;
    }
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
bool beats(int32_t a_id, int32_t a_d, int32_t b_id, int32_t b_d)
{
    if (b_d < a_d) return true;
    if (b_d == a_d && b_id > a_id) return true;
    return false;
}

__device__ __forceinline__
float cross2d(float ox, float oy, float ax, float ay, float bx, float by)
{
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__
bool point_in_triangle(float px, float py,
                       float ax, float ay, float bx, float by, float cx, float cy)
{
    float d1 = cross2d(px, py, ax, ay, bx, by);
    float d2 = cross2d(px, py, bx, by, cx, cy);
    float d3 = cross2d(px, py, cx, cy, ax, ay);
    bool neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
    bool pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
    return !(neg && pos);
}

// ---------------------------------------------------------------------------
// Kernel: write new seeds into the interleaved grid
// ---------------------------------------------------------------------------

__global__
void write_seeds_kernel(int32_t* grid, int32_t* changed_mask, int W,
                        const int32_t* xs, const int32_t* ys,
                        const int32_t* ids, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    int base = (ys[i] * W + xs[i]) * 2;
    grid[base]     = ids[i];
    grid[base + 1] = 0;
    // Mark the seed cell as changed so the border expansion
    // covers detection positions that touch the seed cell directly.
    atomicOr(&changed_mask[ys[i] * W + xs[i]], 1);
}

// ---------------------------------------------------------------------------
// Kernel: BFS step with per-cell change accumulation
// ---------------------------------------------------------------------------

__global__
void voronoi_step_kernel(
    const int32_t* __restrict__ src,
    int32_t* __restrict__       dst,
    int W, int H,
    int32_t* __restrict__ updated_flag,
    int32_t* __restrict__ changed_mask)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int base = (y * W + x) * 2;
    int32_t cur_id = src[base], cur_d = src[base + 1];
    int32_t best_id = cur_id, best_d = cur_d;

    const int dx[4] = {-1, 1,  0, 0};
    const int dy[4] = { 0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k], ny = y + dy[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        int nb = (ny * W + nx) * 2;
        int32_t n_id = src[nb], n_d = src[nb + 1] + 1;
        if (n_id == UNDEF_SEED) continue;
        if (best_id == UNDEF_SEED || beats(best_id, best_d, n_id, n_d)) {
            best_id = n_id; best_d = n_d;
        }
    }

    dst[base]     = best_id;
    dst[base + 1] = best_d;

    if (best_id != cur_id || best_d != cur_d) {
        atomicOr(updated_flag, 1);
        atomicOr(&changed_mask[y * W + x], 1);
    }
}

// ---------------------------------------------------------------------------
// Kernel: L-shape triangle detection with optional mask
// Grid is interleaved (seed_id, distance); only channel 0 is read.
// ---------------------------------------------------------------------------

__global__
void find_triangle_seeds_kernel(
    const int32_t* __restrict__ grid,   // interleaved (seed_id, dist)
    int W, int H,
    RawTriangle* __restrict__ raw_buf,
    int32_t* __restrict__ counter,
    const int32_t* __restrict__ mask)   // nullptr → all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask && !mask[y * W + x]) return;

    auto sid = [&](int cx, int cy) -> int32_t {
        return grid[(cy * W + cx) * 2];
    };

    auto try_reg = [&](int gx, int gy, int32_t a, int32_t b, int32_t c) {
        if (a == UNDEF_SEED || b == UNDEF_SEED || c == UNDEF_SEED) return;
        if (a == b || a == c || b == c) return;
        int32_t pos = atomicAdd(counter, 1);
        int32_t sa = a, sb = b, sc = c;
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        if (sb > sc) { int32_t t = sb; sb = sc; sc = t; }
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        raw_buf[pos] = {gx, gy, sa, sb, sc, a, b, c};
    };

    if (x >= 1   && y <= H-2) try_reg(x, y, sid(x-1,y), sid(x,y),   sid(x,  y+1));
    if (x <= W-2 && y <= H-2) try_reg(x, y, sid(x,  y), sid(x+1,y), sid(x,  y+1));
    if (x <= W-2 && y >= 1  ) try_reg(x, y, sid(x,  y), sid(x+1,y), sid(x,  y-1));
    if (x >= 1   && y >= 1  ) try_reg(x, y, sid(x-1,y), sid(x,  y), sid(x,  y-1));
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
    if (old_tid < 0 || old_tid >= remap_size || remap[old_tid] < 0)
        t_grid[i] = fallback;
    else
        t_grid[i] = remap[old_tid];
}

// ---------------------------------------------------------------------------
// Kernel: window-based assignment with optional mask
// ---------------------------------------------------------------------------

// I am beginning to wonder why this stuff is never defined as global
// constants and instead strun everywhere ... someone needs to get on that
// see also: same thing but in triangulation.cu

static constexpr int WINDOW_SLACK = 3;
static constexpr int WINDOW_CAP   = 20;
static constexpr int SID_HISTORY_STACK_SIZE = 16;

__global__
void assign_triangles_kernel(
    int32_t* __restrict__                   t_grid,
    int W, int H,
    const int32_t* __restrict__             grid,       // interleaved (seed_id, dist)
    const CsrEntryVertexCache* __restrict__ csr_verts_cache,
    const int32_t* __restrict__             csr_ptr,
    const int32_t* __restrict__             csr_idx,
    int N_seeds, int N_triangles,
    const int32_t* __restrict__             mask)       // nullptr → all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask && !mask[y * W + x]) return;

    float px = x + 0.5f, py = y + 0.5f;
    int dist = grid[(y * W + x) * 2 + 1];
    int R    = min(dist + WINDOW_SLACK, WINDOW_CAP);

    int x0 = max(0, x-R), x1 = min(W-1, x+R);
    int y0 = max(0, y-R), y1 = min(H-1, y+R);

    int32_t best = -1;
    int32_t previous_sid = -1;

    int32_t sid_hist_pseudostack[SID_HISTORY_STACK_SIZE];

    #pragma unroll // Kudos to the one time I get to use this effectively
    for (int i = 0; i < SID_HISTORY_STACK_SIZE; ++i) {
        sid_hist_pseudostack[i] = -1;
    } 

    for (int sy = y0; sy <= y1; ++sy) {
        for (int sx = x0; sx <= x1; ++sx) {
            int32_t sid = grid[(sy * W + sx) * 2];

            // Skip if same, already handeled
            if (sid != previous_sid) {
                previous_sid = sid;

                 // Skip if outside boundaries
                if (sid >= 0 && sid < N_seeds) {

                    // Find out if we have this on the "stack"
                    bool did_find = false;
                    #pragma unroll
                    for (int i = 0; i < SID_HISTORY_STACK_SIZE; ++i) {
                        did_find |= (sid_hist_pseudostack[i] == sid);
                    }

                    if (!did_find) {
                        // Push the "stack"
                        #pragma unroll
                        for (int i = SID_HISTORY_STACK_SIZE - 1; i > 0; --i) {
                            sid_hist_pseudostack[i] = sid_hist_pseudostack[i - 1];
                        }
                        sid_hist_pseudostack[0] = sid;

                        for (int j = csr_ptr[sid]; j < csr_ptr[sid+1]; ++j) {
                            int32_t tid = csr_idx[j];
                        
                            // We only kept the best (largest containing) one, anything
                            // else should be outside of the area ... if I understood it correctly
                            if (tid > best) {
                                const CsrEntryVertexCache& c = csr_verts_cache[j];

                                if (point_in_triangle(px, py, c.a.x, c.a.y, c.b.x, c.b.y, c.c.x, c.c.y)) {
                                   best = tid;
                                }
                            }
                        }
                    }
                }
            }      
        }
    }
    
    t_grid[y * W + x] = (best != -1) ? best : (N_triangles - 1);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

IncrementalDelaunay::IncrementalDelaunay(int width, int height, int max_seeds)
    : W_(width), H_(height), N_(0), max_seeds_(max_seeds)
{
    if (width <= 0 || height <= 0 || max_seeds <= 0)
        throw std::invalid_argument("dimensions and max_seeds must be positive");

    const int N = W_ * H_;
    cudaMalloc(&d_grid_,         (size_t)N * 2          * sizeof(int32_t));
    cudaMalloc(&d_tmp_,          (size_t)N * 2          * sizeof(int32_t));
    cudaMalloc(&d_changed_,      (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_sx_,           (size_t)max_seeds      * sizeof(int32_t));
    cudaMalloc(&d_sy_,           (size_t)max_seeds      * sizeof(int32_t));
    cudaMalloc(&d_raw_buf_,      (size_t)N * 4          * sizeof(RawTriangle));
    cudaMalloc(&d_t_grid_,       (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_csr_ptr_,      (size_t)(max_seeds + 1)* sizeof(int32_t));
    cudaMalloc(&d_csr_idx_,      (size_t)max_seeds * 8  * sizeof(int32_t));
    cudaMalloc(&d_updated_flag_, 1                      * sizeof(int32_t));
    cudaMalloc(&d_mask_,         (size_t)N              * sizeof(int32_t));

    cudaMemset(d_grid_,    -1, (size_t)N * 2 * sizeof(int32_t));  // UNDEF
    cudaMemset(d_t_grid_,  -1, (size_t)N     * sizeof(int32_t));
    cudaMemset(d_changed_,  0, (size_t)N     * sizeof(int32_t));

    cudaMalloc(&d_csr_verts_cache_,  (size_t)max_seeds * 8  * sizeof(CsrEntryVertexCache));

    // Pinned host allocations. Warning, these are expensive 
    cudaHostAlloc(&p_changed_,  (size_t)N     * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc(&p_border_,   (size_t)N     * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc(&p_reassign_, (size_t)N     * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc(&p_t_,        (size_t)N     * sizeof(int32_t), cudaHostAllocDefault);
    cudaHostAlloc(&p_grid_,     (size_t)N * 2 * sizeof(int32_t), cudaHostAllocDefault);
}

IncrementalDelaunay::~IncrementalDelaunay()
{
    cudaFree(d_grid_);    cudaFree(d_tmp_);      cudaFree(d_changed_);
    cudaFree(d_sx_);      cudaFree(d_sy_);        cudaFree(d_raw_buf_);
    cudaFree(d_t_grid_);  cudaFree(d_csr_ptr_);  cudaFree(d_csr_idx_);
    cudaFree(d_updated_flag_); cudaFree(d_mask_);

    cudaFree(d_csr_verts_cache_);

    cudaFreeHost(p_changed_); cudaFreeHost(p_border_); cudaFreeHost(p_reassign_);
    cudaFreeHost(p_t_); cudaFreeHost(p_grid_);
}

// ---------------------------------------------------------------------------
// run_bfs_: write new seeds already in d_grid_, BFS until stable
// ---------------------------------------------------------------------------

void IncrementalDelaunay::run_bfs_(float* bfs_ms_out)
{
    DELAUNEY_NVTX_RANGE_C("inc: bfs", delauney_nvtx::kKernel);
    DELAUNEY_HOST_TIME(bfs_ms);

    dim3 block(16, 16);
    dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (bfs_ms_out) {
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
    }

    int32_t zero = 0;
    for (;;) {
        cudaMemcpy(d_updated_flag_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
        voronoi_step_kernel<<<grid_dim, block>>>(
            d_grid_, d_tmp_, W_, H_, d_updated_flag_, d_changed_);
        cudaDeviceSynchronize();
        std::swap(d_grid_, d_tmp_);

        int32_t flag = 0;
        cudaMemcpy(&flag, d_updated_flag_, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (!flag) break;
    }

    if (bfs_ms_out) {
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(bfs_ms_out, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
}

// ---------------------------------------------------------------------------
// upload_triangles_: sync h_triangles_ → d_raw_buf_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::upload_triangles_()
{
    DELAUNEY_NVTX_RANGE_C("inc: upload_triangles", delauney_nvtx::kMemcpy);
    DELAUNEY_HOST_TIME(upload_tri_ms);

    int N_tri = (int)h_triangles_.size();
    std::vector<RawTriangle> h_raw(N_tri);
    for (int i = 0; i < N_tri; ++i) {
        const auto& h = h_triangles_[i];
        h_raw[i] = {h.x, h.y, h.a, h.b, h.c, h.orig_a, h.orig_b, h.orig_c};
    }
    cudaMemcpy(d_raw_buf_, h_raw.data(), N_tri * sizeof(RawTriangle), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// rebuild_csr_and_upload_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::rebuild_csr_and_upload_()
{
    DELAUNEY_NVTX_RANGE("inc: rebuild_csr (host)");
    DELAUNEY_HOST_TIME(csr_ms);

    int N_tri = (int)h_triangles_.size();
    std::vector<int32_t> h_csr_ptr(N_ + 1, 0);
    for (const auto& t : h_triangles_) {
        h_csr_ptr[t.orig_a + 1]++;
        h_csr_ptr[t.orig_b + 1]++;
        h_csr_ptr[t.orig_c + 1]++;
    }
    for (int s = 1; s <= N_; ++s) h_csr_ptr[s] += h_csr_ptr[s-1];

    int csr_size = h_csr_ptr[N_];
    std::vector<int32_t> h_csr_idx(csr_size);
    std::vector<int32_t> fill(N_, 0);
    for (int tid = 0; tid < N_tri; ++tid) {
        for (int32_t s : {h_triangles_[tid].orig_a,
                          h_triangles_[tid].orig_b,
                          h_triangles_[tid].orig_c}) {
            h_csr_idx[h_csr_ptr[s] + fill[s]] = tid;
            fill[s]++;
        }
    }
    cudaMemcpy(d_csr_ptr_, h_csr_ptr.data(), (N_+1)*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_idx_, h_csr_idx.data(),  csr_size*sizeof(int32_t), cudaMemcpyHostToDevice);

    if (csr_size > 0) {
        // Kernel size taken from remap / write seed kernel
        build_csr_verts_kernel<<<((csr_size + 255) / 256), 256>>>(
            static_cast<CsrEntryVertexCache*>(d_csr_verts_cache_),
            d_csr_idx_,
            static_cast<RawTriangle*>(d_raw_buf_), 
            d_sx_, 
            d_sy_, 
            csr_size
        );
    }
}

// ---------------------------------------------------------------------------
// full_triangulate_: detect → dedup → CSR → assign (no mask)
// ---------------------------------------------------------------------------

// The out-params carry the GPU (cudaEvent) timings; the DELAUNEY_HOST_TIME
// scopes record host wall time for the same phases into ht_.  They are named
// distinctly from the IncrementalHostTimings fields to keep the two apart.
void IncrementalDelaunay::full_triangulate_(float* det_gpu, float* dedup_gpu, float* asgn_gpu)
{
    DELAUNEY_NVTX_RANGE_C("inc: full_triangulate", delauney_nvtx::kPhase);

    const int N = W_ * H_;
    dim3 block(16, 16);
    dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);

    auto mk = [](cudaEvent_t* e){ cudaEventCreate(e); };
    auto rc = [](cudaEvent_t  e){ cudaEventRecord(e); };
    auto el = [](cudaEvent_t a, cudaEvent_t b) -> float {
        cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms;
    };

    cudaEvent_t e0,e1,e2,e3,e4,e5;
    if (det_gpu) { mk(&e0);mk(&e1);mk(&e2);mk(&e3);mk(&e4);mk(&e5); }

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_raw_buf_);
    int32_t* d_counter;
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaMalloc(&d_counter, sizeof(int32_t));
    }
    cudaMemset(d_counter, 0, sizeof(int32_t));

    int32_t raw_count = 0;
    {
        DELAUNEY_NVTX_RANGE_C("inc: detect", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(detect_ms);
        if (det_gpu) rc(e0);
        find_triangle_seeds_kernel<<<grid_dim, block>>>(d_grid_, W_, H_, d_raw, d_counter, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (det_gpu) rc(e1);
    }
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaFree(d_counter);
    }

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);
    int N_tri = 0;
    {
        DELAUNEY_NVTX_RANGE_C("inc: dedup (thrust)", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(dedup_ms);
        if (dedup_gpu) rc(e2);
        thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
        auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
        N_tri = (int)(new_end - d_ptr);
        if (dedup_gpu) rc(e3);
    }

    std::vector<RawTriangle> h_dedup(N_tri);
    {
        DELAUNEY_NVTX_RANGE_C("inc: D2H dedup triangles", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(d2h_new_ms);
        cudaMemcpy(h_dedup.data(), d_raw, N_tri * sizeof(RawTriangle), cudaMemcpyDeviceToHost);
    }

    {
        DELAUNEY_NVTX_RANGE("inc: build registry (host)");
        DELAUNEY_HOST_TIME(build_registry_ms);
        h_triangles_.clear(); h_triplet_to_tid_.clear();
        h_triangles_.reserve(N_tri);
        for (int32_t tid = 0; tid < N_tri; ++tid) {
            const auto& r = h_dedup[tid];
            h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
            h_triplet_to_tid_[pack_triplet_(r.a, r.b, r.c)] = tid;
        }
        // d_raw_buf_ already has the deduplicated triangles in correct order
    }

    rebuild_csr_and_upload_();

    {
        DELAUNEY_NVTX_RANGE_C("inc: assign", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(assign_ms);
        if (asgn_gpu) rc(e4);
        if (N_tri > 0) {
            assign_triangles_kernel<<<grid_dim, block>>>(
                d_t_grid_, 
                W_, H_, 
                d_grid_, 
                static_cast<CsrEntryVertexCache*>(d_csr_verts_cache_),
                d_csr_ptr_,
                d_csr_idx_,
                N_, N_tri, 
                nullptr);
            cudaDeviceSynchronize();
        } else {
            cudaMemset(d_t_grid_, -1, N * sizeof(int32_t));
        }
        if (asgn_gpu) rc(e5);
    }

    if (det_gpu) {
        *det_gpu   = el(e0,e1);
        *dedup_gpu = el(e2,e3);
        *asgn_gpu  = el(e4,e5);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
        cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    }
}

// ---------------------------------------------------------------------------
// partial_triangulate_: use d_changed_ to scope detection and assignment
// ---------------------------------------------------------------------------

void IncrementalDelaunay::partial_triangulate_(float* det_ms, float* dedup_ms, float* asgn_ms)
{
    DELAUNEY_NVTX_RANGE_C("inc: partial_triangulate", delauney_nvtx::kPhase);

    const int N = W_ * H_;
    dim3 block(16, 16);
    dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);

    std::memset(p_border_,   0, (size_t)N * sizeof(int32_t));
    std::memset(p_reassign_, 0, (size_t)N * sizeof(int32_t));

    // Download change mask
    {
        DELAUNEY_NVTX_RANGE_C("inc: D2H changed mask", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(d2h_changed_ms);
        cudaMemcpy(p_changed_, d_changed_, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    // Build border (expand by 2 for L-shapes) and reassign (expand by WINDOW_CAP)
    {
        DELAUNEY_NVTX_RANGE("inc: expand masks (host)");
        DELAUNEY_HOST_TIME(expand_ms);
        for (int y = 0; y < H_; ++y) {
            for (int x = 0; x < W_; ++x) {
                if (!p_changed_[y * W_ + x]) continue;
                for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    int bx = x+dx, by = y+dy;
                    if (bx>=0 && bx<W_ && by>=0 && by<H_) p_border_[by*W_+bx] = 1;
                }
                for (int dy = -WINDOW_CAP; dy <= WINDOW_CAP; ++dy)
                for (int dx = -WINDOW_CAP; dx <= WINDOW_CAP; ++dx) {
                    int bx = x+dx, by = y+dy;
                    if (bx>=0 && bx<W_ && by>=0 && by<H_) p_reassign_[by*W_+bx] = 1;
                }
            }
        }
    }

    // Mark stale triangles (canonical pixel in border)
    int old_count = (int)h_triangles_.size();
    std::vector<bool> is_stale(old_count, false);
    {
        DELAUNEY_NVTX_RANGE("inc: mark stale (host)");
        DELAUNEY_HOST_TIME(mark_stale_ms);
        for (int tid = 0; tid < old_count; ++tid) {
            const auto& t = h_triangles_[tid];
            if (t.x >= 0 && t.x < W_ && t.y >= 0 && t.y < H_)
                if (p_border_[t.y * W_ + t.x]) is_stale[tid] = true;
        }
    }

    // Detect new triangles in border
    {
        DELAUNEY_NVTX_RANGE_C("inc: H2D border mask", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(h2d_border_ms);
        cudaMemcpy(d_mask_, p_border_, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_raw_buf_);
    int32_t* d_counter;
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaMalloc(&d_counter, sizeof(int32_t));
    }
    cudaMemset(d_counter, 0, sizeof(int32_t));

    int32_t raw_count = 0;
    {
        DELAUNEY_NVTX_RANGE_C("inc: detect (masked)", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(detect_ms);
        find_triangle_seeds_kernel<<<grid_dim, block>>>(d_grid_, W_, H_, d_raw, d_counter, d_mask_);
        cudaDeviceSynchronize();
        cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaFree(d_counter);
    }

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);
    int n_new = 0;
    {
        DELAUNEY_NVTX_RANGE_C("inc: dedup (thrust)", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(dedup_ms);
        thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
        auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
        n_new = (int)(new_end - d_ptr);
    }

    std::vector<RawTriangle> h_new(n_new);
    if (n_new > 0) {
        DELAUNEY_NVTX_RANGE_C("inc: D2H new triangles", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(d2h_new_ms);
        cudaMemcpy(h_new.data(), d_raw, n_new * sizeof(RawTriangle), cudaMemcpyDeviceToHost);
    }

    // Collect truly new triplets: not in registry, OR in registry but stale
    // (stale triangles are re-detected here; they must be re-added since they
    // will be removed in the compaction step below).
    std::vector<HTriangle> to_add;
    {
        DELAUNEY_NVTX_RANGE("inc: collect new triplets (host)");
        DELAUNEY_HOST_TIME(collect_ms);
        for (const auto& r : h_new) {
            auto key = pack_triplet_(r.a, r.b, r.c);
            auto it = h_triplet_to_tid_.find(key);
            if (it == h_triplet_to_tid_.end() || is_stale[it->second])
                to_add.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
        }
    }

    // This is kept initialized with -1 because I could not figure out where the remap[tid] = -1 line 
    // would need to be ... it created a bevy of issuess. This is load-bearing, do not remove it.
    std::vector<int32_t> remap(old_count, -1);
    {
        DELAUNEY_NVTX_RANGE("inc: compact registry and remap (host)");
        DELAUNEY_HOST_TIME(compact_ms);

        // Fill each stale slot with the last live triangle.
        int next = old_count;
        int last = old_count - 1;
        for (int tid = 0; tid < old_count; ++tid) {
            if (is_stale[tid]) {
                // In-place erase, stale, we dont need this
                h_triplet_to_tid_.erase(pack_triplet_(
                    h_triangles_[tid].a,
                    h_triangles_[tid].b,
                    h_triangles_[tid].c
                ));

                // Skip if no in-place remapping is necessary
                if (tid < next) {      
                    // Skip over missing elements. Not a bottleneck,
                    // has a linear amount of runs total and needs to be done anywayW
                    while (last > tid && is_stale[last]) {
                        --last;
                    }

                    // Next free slot
                    if (last <= tid) { 
                        next = tid; 
                    } else {
                        // Or remap to the known last next free slot (swap)
                        h_triangles_[tid] = h_triangles_[last];
                        remap[last] = tid;

                        // No, you cannot re-use the triplet from before, it is overwritten
                        // literally two lines above. I spent way too much time here. I did not see the issue.
                        h_triplet_to_tid_[
                            pack_triplet_(h_triangles_[tid].a,
                                h_triangles_[tid].b,
                                h_triangles_[tid].c)
                            ] = tid;
                        next = last;
                        --last;
                    } 
                }
            } else {
                // Blindly doing this overwrites our logic
                // Side-effect of merging this into one loop
                if (tid < next) {
                    remap[tid] = tid;
                }
            }
        }

        h_triangles_.resize(next);

        // Add new ones to the end, no need for the index
        for (const auto& t : to_add) {
            h_triangles_.push_back(t);
            h_triplet_to_tid_[pack_triplet_(t.a, t.b, t.c)] = (int32_t)h_triangles_.size() - 1;
        }
    }

    // Upload triangle array and CSR
    upload_triangles_();
    rebuild_csr_and_upload_();

    // Remap d_t_grid_ through old→new table
    int32_t fallback = h_triangles_.empty() ? 0 : (int32_t)h_triangles_.size() - 1;
    if (old_count > 0) {
        DELAUNEY_NVTX_RANGE_C("inc: remap t_grid", delauney_nvtx::kKernel);
        int32_t* d_remap;
        {
            DELAUNEY_HOST_TIME(scratch_ms);
            cudaMalloc(&d_remap, old_count * sizeof(int32_t));
        }
        {
            DELAUNEY_HOST_TIME(remap_ms);
            cudaMemcpy(d_remap, remap.data(), old_count * sizeof(int32_t), cudaMemcpyHostToDevice);
            remap_tgrid_kernel<<<(N+255)/256, 256>>>(d_t_grid_, N, d_remap, old_count, fallback);
            cudaDeviceSynchronize();
        }
        {
            DELAUNEY_HOST_TIME(scratch_ms);
            cudaFree(d_remap);
        }
    }

    // Re-assign pixels in expanded reassign region
    {
        DELAUNEY_NVTX_RANGE_C("inc: H2D reassign mask", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(h2d_reassign_ms);
        cudaMemcpy(d_mask_, p_reassign_, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    int N_tri = (int)h_triangles_.size();
    if (N_tri > 0) {
        DELAUNEY_NVTX_RANGE_C("inc: assign (masked)", delauney_nvtx::kKernel);
        DELAUNEY_HOST_TIME(assign_ms);
        assign_triangles_kernel<<<grid_dim, block>>>(
            d_t_grid_, 
            W_, H_, 
            d_grid_,
            static_cast<CsrEntryVertexCache*>(d_csr_verts_cache_),
            d_csr_ptr_,
            d_csr_idx_,
            N_, N_tri,
            d_mask_);
        cudaDeviceSynchronize();
    }
}

// ---------------------------------------------------------------------------
// build_outputs_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::build_outputs_(std::vector<TriangleEntry>& tri_map_out,
                                         std::vector<int32_t>& tgrid_out) const
{
    DELAUNEY_NVTX_RANGE_C("inc: build_outputs", delauney_nvtx::kPhase);

    const int N = W_ * H_;
    int N_tri = (int)h_triangles_.size();

    {
        DELAUNEY_HOST_TIME(out_trimap_ms);
        tri_map_out.resize(N_tri);
        for (int tid = 0; tid < N_tri; ++tid) {
            const auto& t = h_triangles_[tid];
            tri_map_out[tid] = {t.x, t.y, t.orig_a, t.orig_b, t.orig_c};
        }
    }

    {
        DELAUNEY_NVTX_RANGE_C("inc: D2H grids", delauney_nvtx::kMemcpy);
        DELAUNEY_HOST_TIME(out_d2h_ms);
        cudaMemcpy(p_t_,    d_t_grid_, N     * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(p_grid_, d_grid_,   N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    {
        DELAUNEY_NVTX_RANGE("inc: interleave out grid (host)");
        DELAUNEY_HOST_TIME(out_interleave_ms);
        tgrid_out.resize(N * 3);
        for (int i = 0; i < N; ++i) {
            tgrid_out[i*3]   = p_grid_[i*2];
            tgrid_out[i*3+1] = p_grid_[i*2+1];
            tgrid_out[i*3+2] = p_t_[i];
        }
    }
}

// ---------------------------------------------------------------------------
// get_voronoi_grid
// ---------------------------------------------------------------------------

void IncrementalDelaunay::get_voronoi_grid(std::vector<int32_t>& out) const
{
    const int N = W_ * H_;
    out.resize(N * 2);
    cudaMemcpy(out.data(), d_grid_, N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------

void IncrementalDelaunay::insert(
    const std::vector<int32_t>& new_xs,
    const std::vector<int32_t>& new_ys,
    std::vector<TriangleEntry>&  tri_map_out,
    std::vector<int32_t>&        tgrid_out,
    IncrementalTimings*          timings)
{
    DELAUNEY_NVTX_RANGE_C("IncrementalDelaunay::insert", delauney_nvtx::kPhase);

    // Route host phase timings for this call; cleared on every exit path.
    ht_ = timings ? &timings->host : nullptr;
    struct HtGuard {
        IncrementalHostTimings** p;
        ~HtGuard() { *p = nullptr; }
    } _ht_guard{&ht_};

    int k = (int)new_xs.size();
    if (k == 0) { build_outputs_(tri_map_out, tgrid_out); return; }

    if (N_ + k > max_seeds_)
        throw std::invalid_argument("insert would exceed max_seeds capacity");

    {
        DELAUNEY_NVTX_RANGE("inc: validate seeds (host)");
        DELAUNEY_HOST_TIME(validate_ms);

        // Validate bounds
        for (int i = 0; i < k; ++i)
            if (new_xs[i] < 0 || new_xs[i] >= W_ || new_ys[i] < 0 || new_ys[i] >= H_)
                throw std::invalid_argument("seed coordinate out of bounds");

        // Duplicate check within batch
	    // Use a hash to keep this part O(k) instead of O(k^2)
	    std::unordered_set<uint64_t> batch_hash;
	    batch_hash.reserve(k);  // Reserve space to avoid rehashing
        for (int i = 0; i < k; ++i) {
            // for (int j = i+1; j < k; ++j)
		    auto key = pack_xy_(new_xs[i], new_ys[i]);  // Pack x and y into a single 64-bit key
            if (!batch_hash.insert(key).second) {
                throw std::invalid_argument("duplicate seed positions within batch");
            }
        }

        // Duplicate check against existing seeds
        for (int i = 0; i < k; ++i)
            if (h_seed_set_.count(pack_xy_(new_xs[i], new_ys[i])))
                throw std::invalid_argument("seed position already exists");
    }

    bool is_first = (N_ == 0);

    // Register seeds
    std::vector<int32_t> new_ids(k);
    {
        DELAUNEY_HOST_TIME(seed_reg_ms);
        for (int i = 0; i < k; ++i) {
            new_ids[i] = N_ + i;
            h_sx_.push_back(new_xs[i]);
            h_sy_.push_back(new_ys[i]);
            h_seed_set_.insert(pack_xy_(new_xs[i], new_ys[i]));
        }
        N_ += k;
    }

    {
        DELAUNEY_HOST_TIME(seed_h2d_ms);
        // Upload new seed positions to persistent device arrays
        cudaMemcpy(d_sx_ + N_ - k, new_xs.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sy_ + N_ - k, new_ys.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);

        // Reset change accumulator before writing seeds (so seed positions
        // are the first entries in d_changed_ — necessary for partial_triangulate_
        // to cover detection positions that touch the seed cell directly).
        cudaMemset(d_changed_, 0, (size_t)W_ * H_ * sizeof(int32_t));
    }

    // Write seeds into grid (also marks seed cells in d_changed_)
    DELAUNEY_NVTX_MARK("inc: write seeds");
    int32_t *d_kxs, *d_kys, *d_kids;
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaMalloc(&d_kxs,  k * sizeof(int32_t));
        cudaMalloc(&d_kys,  k * sizeof(int32_t));
        cudaMalloc(&d_kids, k * sizeof(int32_t));
    }
    {
        DELAUNEY_HOST_TIME(write_seeds_ms);
        cudaMemcpy(d_kxs,  new_xs.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kys,  new_ys.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_kids, new_ids.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
        write_seeds_kernel<<<(k+255)/256, 256>>>(d_grid_, d_changed_, W_, d_kxs, d_kys, d_kids, k);
        cudaDeviceSynchronize();
    }
    {
        DELAUNEY_HOST_TIME(scratch_ms);
        cudaFree(d_kxs); cudaFree(d_kys); cudaFree(d_kids);
    }

    // BFS
    float bfs_ms = 0.f;
    run_bfs_(timings ? &bfs_ms : nullptr);
    if (timings) timings->bfs_ms = bfs_ms;

    // Triangulate
    float det_ms = 0.f, dup_ms = 0.f, asgn_ms = 0.f;
    if (is_first)
        full_triangulate_(timings ? &det_ms : nullptr,
                          timings ? &dup_ms  : nullptr,
                          timings ? &asgn_ms : nullptr);
    else
        partial_triangulate_(timings ? &det_ms : nullptr,
                             timings ? &dup_ms  : nullptr,
                             timings ? &asgn_ms : nullptr);
    if (timings) {
        timings->detect_ms = det_ms;
        timings->dedup_ms  = dup_ms;
        timings->assign_ms = asgn_ms;
    }

    build_outputs_(tri_map_out, tgrid_out);
}
