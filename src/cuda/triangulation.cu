// CUDA kernels for GridTriangulation.
//
// Step 2 (GPU): scan all 4 L-shape orientations; raw hits → device buffer.
// Step 3 (GPU): thrust::sort + thrust::unique on the device buffer.
// Step 4 (GPU): window-based nearest-neighbor assign.
//   For each pixel, read its Voronoi distance d, scan a (2R+1)² window of
//   the seed-id grid (R = min(d + WINDOW_SLACK, WINDOW_CAP)), collect unique
//   nearby seed IDs, then test only the triangles adjacent to those seeds via
//   a per-seed CSR list.  Reduces per-pixel tests from O(T) to O(k·t) where
//   k ≈ nearby seeds (~10-70) and t ≈ triangles/seed (~7).
//
// Host copies:
//   • voronoi seed_id + distance channels  → device  (input, once)
//   • seed positions                        → device  (input, once)
//   • deduplicated triangles               ← device  (output, ~7 MB)
//   • CSR adjacency list                   → device  (built host-side, ~3 MB)
//   • triangle_id grid                     ← device  (output, W*H ints)

#include "triangulation.cuh"

#include "nvtx_range.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Shared structs
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEF = -1;

struct RawTriangle {
    int32_t x, y;
    int32_t a, b, c;           // sorted dedup key  (a <= b <= c)
    int32_t orig_a, orig_b, orig_c;
};

struct RawLess {
    __device__ bool operator()(const RawTriangle& x, const RawTriangle& y) const {
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.c < y.c;
    }
};

struct RawEqual {
    __device__ bool operator()(const RawTriangle& x, const RawTriangle& y) const {
        return x.a == y.a && x.b == y.b && x.c == y.c;
    }
};

// ---------------------------------------------------------------------------
// Kernel 1: detect triangle seeds (all 4 L-shape orientations)
// ---------------------------------------------------------------------------

__global__
void find_triangle_seeds_kernel(
    const int32_t* __restrict__ n_grid,
    int W, int H,
    RawTriangle* __restrict__ raw_buf,
    int32_t* __restrict__ counter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    auto idx = [&](int cx, int cy) -> int32_t {
        return n_grid[cy * W + cx];
    };

    auto try_register = [&](int gx, int gy, int32_t a, int32_t b, int32_t c) {
        if (a == UNDEF || b == UNDEF || c == UNDEF) return;
        if (a == b || a == c || b == c) return;
        int32_t pos = atomicAdd(counter, 1);
        int32_t sa = a, sb = b, sc = c;
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        if (sb > sc) { int32_t t = sb; sb = sc; sc = t; }
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        raw_buf[pos] = {gx, gy, sa, sb, sc, a, b, c};
    };

    if (x >= 1   && y <= H-2) try_register(x, y, idx(x-1,y), idx(x,y),   idx(x,  y+1));
    if (x <= W-2 && y <= H-2) try_register(x, y, idx(x,  y), idx(x+1,y), idx(x,  y+1));
    if (x <= W-2 && y >= 1  ) try_register(x, y, idx(x,  y), idx(x+1,y), idx(x,  y-1));
    if (x >= 1   && y >= 1  ) try_register(x, y, idx(x-1,y), idx(x,  y), idx(x,  y-1));
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
float cross2d(float ox, float oy, float ax, float ay, float bx, float by)
{
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__
bool point_in_triangle(float px, float py,
                       float ax, float ay,
                       float bx, float by,
                       float cx, float cy)
{
    float d1 = cross2d(px, py, ax, ay, bx, by);
    float d2 = cross2d(px, py, bx, by, cx, cy);
    float d3 = cross2d(px, py, cx, cy, ax, ay);
    bool has_neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
    bool has_pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
    return !(has_neg && has_pos);
}

// ---------------------------------------------------------------------------
// Kernel 2: window-based triangle assignment
//
// For each pixel:
//   1. Read Voronoi distance d; set search radius R = min(d+SLACK, CAP).
//   2. Scan the (2R+1)x(2R+1) seed-id window; For each match: iterate 
//      its CSR triangle list and run the
//      containment test.  Duplicate triangle tests (same tri reachable via
//      multiple seeds) are allowed — they are idempotent.
//
// Optimization note: this seems to create more potential matches but still arrives
// at the correct conclusion
// ---------------------------------------------------------------------------

static constexpr int WINDOW_SLACK = 3;   // extra radius beyond Voronoi dist
static constexpr int WINDOW_CAP   = 20;  // hard cap to bound work in sparse areas
static constexpr int SID_HISTORY_STACK_SIZE = 16;

__global__
void assign_triangles_kernel(
    int32_t* __restrict__             t_grid,
    int W, int H,
    const int32_t* __restrict__       n_grid,     // seed_id per pixel
    const int32_t* __restrict__       dist_grid,  // voronoi distance per pixel
    const RawTriangle* __restrict__   triangles,
    const int32_t* __restrict__       seed_xs,
    const int32_t* __restrict__       seed_ys,
    const int32_t* __restrict__       csr_ptr,    // [N_seeds+1]
    const int32_t* __restrict__       csr_idx,    // triangle IDs per seed
    int N_seeds,
    int N_triangles)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float px = x + 0.5f;
    float py = y + 0.5f;

    int dist = dist_grid[y * W + x];
    int R    = min(dist + WINDOW_SLACK, WINDOW_CAP);


    int x0 = max(0, x - R),     x1 = min(W - 1, x + R);
    int y0 = max(0, y - R),     y1 = min(H - 1, y + R);

    // Test triangles reachable via nearby seeds.
    // A triangle with all 3 vertices in the window appears 3 times (once per
    // seed); the containment test is idempotent so duplicates are harmless.
    int32_t best = -1;
    int32_t previous_sid = -1;

    int32_t sid_hist_pseudostack[SID_HISTORY_STACK_SIZE];

    #pragma unroll // Kudos to the one time I get to use this effectively
    for (int i = 0; i < SID_HISTORY_STACK_SIZE; ++i) {
        sid_hist_pseudostack[i] = -1;
    } 

    for (int sy = y0; sy <= y1; ++sy) {
        for (int sx = x0; sx <= x1; ++sx) {
            int32_t sid = n_grid[sy * W + sx];
           
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

                        // Test what can be seen from here, inverse from the collect then try approach
                        for (int j = csr_ptr[sid]; j < csr_ptr[sid + 1]; ++j) {
                            int32_t tid = csr_idx[j];
                            const RawTriangle& tri = triangles[tid];
                            float ax = (float)seed_xs[tri.orig_a], ay = (float)seed_ys[tri.orig_a];
                            float bx = (float)seed_xs[tri.orig_b], by = (float)seed_ys[tri.orig_b];
                            float cx = (float)seed_xs[tri.orig_c], cy = (float)seed_ys[tri.orig_c];

                            // I want to guess a few fractions of a millisecond can be saved here
                            // by doing the tid-best check first but that yielded no measurable
                            // difference for me
                            if (point_in_triangle(px, py, ax, ay, bx, by, cx, cy)) {
                                if (best == -1 || tid > best) {
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
// Kernel 3: Split an grid into separate arrays.
//
// Every GPU thread processes exactly one grid cell.
// ---------------------------------------------------------------------------

__global__
void split_seed_distance_grid_kernel(
    const int32_t* __restrict__ seed_distance_grid,
    int32_t* __restrict__ seed_ids,
    int32_t* __restrict__ distances,
    int num_cells)
{
	int cell = blockIdx.x * blockDim.x + threadIdx.x;   // cell index in the grid

	if (cell >= num_cells)  // out-of-bounds check
        return;

	seed_ids[cell] = seed_distance_grid[cell * 2];        // seed ID channel
	distances[cell] = seed_distance_grid[cell * 2 + 1];   // distance channel
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_triangulation(
    int W, int H,
    const int32_t* voronoi_grid,
    const std::vector<int32_t>& seed_xs,
    const std::vector<int32_t>& seed_ys,
    std::vector<TriangleEntry>& triangle_map_out,
    std::vector<int32_t>& out_grid,
    TriTimings* timings)
{
    DELAUNEY_NVTX_RANGE_C("cuda_compute_triangulation", delauney_nvtx::kPhase);

    const int N        = W * H;
    const int N_seeds  = (int)seed_xs.size();

    auto make_event = [](cudaEvent_t* e) { cudaEventCreate(e); };
    auto record     = [](cudaEvent_t e)  { cudaEventRecord(e); };
    auto elapsed_ms = [](cudaEvent_t a, cudaEvent_t b) -> float {
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    };

    cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5;
    if (timings) {
        make_event(&ev0); make_event(&ev1); make_event(&ev2);
        make_event(&ev3); make_event(&ev4); make_event(&ev5);
    }

    // -----------------------------------------------------------------------
    // Upload inputs: seed_id channel, distance channel, seed positions
    // -----------------------------------------------------------------------

    int32_t *d_n = nullptr, *d_dist = nullptr;
    int32_t *d_sx = nullptr, *d_sy = nullptr;
    int32_t* seed_distance_grid = nullptr;
    {
        DELAUNEY_NVTX_RANGE_C("tri: upload inputs", delauney_nvtx::kMemcpy);

        // Upload the seed grid as-is (one contiguous 8MB copy) and split
        // it on the GPU instead of looping over it on the CPU first.
        cudaMalloc(&seed_distance_grid, (size_t)N * 2 * sizeof(int32_t));
        cudaMemcpy(seed_distance_grid, voronoi_grid, (size_t)N * 2 * sizeof(int32_t),
            cudaMemcpyHostToDevice);

        cudaMalloc(&d_n,    N       * sizeof(int32_t));
        cudaMalloc(&d_dist, N       * sizeof(int32_t));
        cudaMalloc(&d_sx,   N_seeds * sizeof(int32_t));
        cudaMalloc(&d_sy,   N_seeds * sizeof(int32_t));

        split_seed_distance_grid_kernel <<<(N + 255) / 256, 256 >>> (seed_distance_grid, d_n, d_dist, N);

        cudaMemcpy(d_sx,   seed_xs.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sy,   seed_ys.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
    }


    dim3 block(16, 16);
    dim3 grid_dim((W + 15) / 16, (H + 15) / 16);

    // -----------------------------------------------------------------------
    // Step 2: detect raw triangle seeds
    // -----------------------------------------------------------------------

    const int max_raw = N * 4;
    RawTriangle* d_raw = nullptr;
    cudaMalloc(&d_raw, max_raw * sizeof(RawTriangle));

    int32_t* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int32_t));
    cudaMemset(d_counter, 0, sizeof(int32_t));

    int32_t raw_count = 0;
    {
        DELAUNEY_NVTX_RANGE_C("tri: detect", delauney_nvtx::kKernel);
        if (timings) record(ev0);

        find_triangle_seeds_kernel<<<grid_dim, block>>>(d_n, W, H, d_raw, d_counter);
        cudaDeviceSynchronize();

        cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_counter);

        if (timings) record(ev1);
    }

    // -----------------------------------------------------------------------
    // Step 3: deduplicate on device with Thrust sort + unique
    // -----------------------------------------------------------------------

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);

    int N_triangles = 0;
    {
        DELAUNEY_NVTX_RANGE_C("tri: dedup (thrust)", delauney_nvtx::kKernel);
        if (timings) record(ev2);

        thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
        auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
        N_triangles  = (int)(new_end - d_ptr);

        if (timings) record(ev3);
    }

    // Copy deduplicated entries to host (for the triangle map + CSR build)
    std::vector<RawTriangle> h_dedup(N_triangles);
    {
        DELAUNEY_NVTX_RANGE_C("tri: D2H dedup triangles", delauney_nvtx::kMemcpy);
        cudaMemcpy(h_dedup.data(), d_raw, N_triangles * sizeof(RawTriangle),
                   cudaMemcpyDeviceToHost);
    }

    {
        DELAUNEY_NVTX_RANGE("tri: build triangle map (host)");
        triangle_map_out.clear();
        triangle_map_out.reserve(N_triangles);
        for (const auto& r : h_dedup)
            triangle_map_out.push_back({r.x, r.y, r.orig_a, r.orig_b, r.orig_c});
    }

    // -----------------------------------------------------------------------
    // Build CSR: seed → triangle list  (host, then upload)
    //
    // csr_ptr[s]   = start index in csr_idx for seed s
    // csr_ptr[s+1] = end   index
    // csr_idx[i]   = triangle ID
    // -----------------------------------------------------------------------

    int32_t *d_csr_ptr = nullptr, *d_csr_idx = nullptr;
    {
        DELAUNEY_NVTX_RANGE("tri: build CSR (host)");

        std::vector<int32_t> h_csr_ptr(N_seeds + 1, 0);
        for (int tid = 0; tid < N_triangles; ++tid) {
            h_csr_ptr[h_dedup[tid].orig_a + 1]++;
            h_csr_ptr[h_dedup[tid].orig_b + 1]++;
            h_csr_ptr[h_dedup[tid].orig_c + 1]++;
        }
        for (int s = 1; s <= N_seeds; ++s)
            h_csr_ptr[s] += h_csr_ptr[s - 1];

        const int csr_size = h_csr_ptr[N_seeds];  // == 3 * N_triangles
        std::vector<int32_t> h_csr_idx(csr_size);
        std::vector<int32_t> fill(N_seeds, 0);

        for (int tid = 0; tid < N_triangles; ++tid) {
            for (int32_t s : {h_dedup[tid].orig_a, h_dedup[tid].orig_b, h_dedup[tid].orig_c}) {
                h_csr_idx[h_csr_ptr[s] + fill[s]] = tid;
                fill[s]++;
            }
        }

        cudaMalloc(&d_csr_ptr, (N_seeds + 1) * sizeof(int32_t));
        cudaMalloc(&d_csr_idx,  csr_size     * sizeof(int32_t));
        DELAUNEY_NVTX_MARK("tri: H2D CSR");
        cudaMemcpy(d_csr_ptr, h_csr_ptr.data(), (N_seeds + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_idx, h_csr_idx.data(),  csr_size     * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    // -----------------------------------------------------------------------
    // Step 4: assign pixels to triangles (window-based nearest-neighbor)
    // -----------------------------------------------------------------------

    int32_t* d_t = nullptr;
    cudaMalloc(&d_t, N * sizeof(int32_t));

    {
        DELAUNEY_NVTX_RANGE_C("tri: assign", delauney_nvtx::kKernel);
        if (timings) record(ev4);

        if (N_triangles > 0) {
            assign_triangles_kernel<<<grid_dim, block>>>(
                d_t, W, H,
                d_n, d_dist,
                d_raw, d_sx, d_sy,
                d_csr_ptr, d_csr_idx,
                N_seeds, N_triangles);
            cudaDeviceSynchronize();
        } else {
            cudaMemset(d_t, -1, N * sizeof(int32_t));
        }

        if (timings) record(ev5);
    }

    // -----------------------------------------------------------------------
    // Step 5: build output grid (H * W * 3) and fill timings
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_t(N);
    {
        DELAUNEY_NVTX_RANGE_C("tri: D2H triangle grid", delauney_nvtx::kMemcpy);
        cudaMemcpy(h_t.data(), d_t, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    {
        DELAUNEY_NVTX_RANGE("tri: build out grid (host)");
        out_grid.resize(N * 3);
        for (int i = 0; i < N; ++i) {
            out_grid[i * 3]     = voronoi_grid[i * 2];
            out_grid[i * 3 + 1] = voronoi_grid[i * 2 + 1];
            out_grid[i * 3 + 2] = h_t[i];
        }
    }

    if (timings) {
        timings->detect_ms = elapsed_ms(ev0, ev1);
        timings->sort_ms   = elapsed_ms(ev1, ev2);
        timings->dedup_ms  = elapsed_ms(ev2, ev3);
        timings->assign_ms = elapsed_ms(ev4, ev5);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
        cudaEventDestroy(ev3); cudaEventDestroy(ev4); cudaEventDestroy(ev5);
    }

    cudaFree(d_raw);
    cudaFree(d_n);
    cudaFree(d_dist);
    cudaFree(d_t);
    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_csr_ptr);
    cudaFree(d_csr_idx);
    cudaFree(seed_distance_grid);
}
