// CUDA kernels for GridTriangulation.
//
// Step 2 (GPU): 2x2-block scan for triangle detection.
//   Each pixel (x,y) is the top-left corner of a 2x2 block covering pixels
//   (x,y)=a, (x+1,y)=b, (x,y+1)=c, (x+1,y+1)=d.
//   - 3 distinct seed IDs → one triangle from the 3 distinct seeds.
//   - 4 distinct seed IDs → two triangles: (a,b,c) and (b,c,d), splitting
//     the quad along the (b,c) anti-diagonal.  Both triangles are valid since
//     4 Voronoi cells sharing a corner implies the 4 seeds are in convex
//     position.
//   Max output: 2*(W-1)*(H-1) raw entries (at most 2 per block).
// Step 3 (GPU): thrust::sort + thrust::unique on the device buffer.
// Step 4 (GPU): seed-position-scan triangle assignment.
//   For each pixel, iterate all seeds; skip any whose Manhattan distance to
//   the pixel exceeds window_cap (= max_manhattan_side + SLACK).  For seeds
//   that pass the guard, iterate their CSR triangle list and run the
//   containment test.  window_cap is derived from the longest detected
//   triangle side, so the window always covers at least one vertex of the
//   containing triangle regardless of Voronoi ownership.
//   Expected cost: O(N_seeds_in_window × tris/seed) ≈ O(16) seeds for
//   uniform distributions; scales with local seed density, not total N.
//
// Host copies:
//   - voronoi seed_id channel              -> device  (input, once)
//   - seed positions                       -> device  (input, once)
//   - deduplicated triangles               <- device  (output, ~7 MB)
//   - CSR adjacency list                   -> device  (built host-side, ~3 MB)
//   - triangle_id grid                     <- device  (output, W*H ints)

#include "triangulation.cuh"

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
// Kernel 1: detect triangle seeds via 2x2 block scan
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
    if (x >= W - 1 || y >= H - 1) return;   // 2x2 block must be in bounds

    int32_t a = n_grid[ y      * W + x    ];  // top-left
    int32_t b = n_grid[ y      * W + x + 1];  // top-right
    int32_t c = n_grid[(y + 1) * W + x    ];  // bottom-left
    int32_t d = n_grid[(y + 1) * W + x + 1];  // bottom-right

    // Sort three seed IDs and atomically write one RawTriangle entry.
    auto register_tri = [&](int32_t oa, int32_t ob, int32_t oc) {
        int32_t sa = oa, sb = ob, sc = oc;
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        if (sb > sc) { int32_t t = sb; sb = sc; sc = t; }
        if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
        int32_t pos = atomicAdd(counter, 1);
        raw_buf[pos] = {x, y, sa, sb, sc, oa, ob, oc};
    };

    // Collect distinct seed IDs from the 2x2 block.
    int32_t quad[4] = {a, b, c, d};
    int32_t s[4];
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        bool dup = false;
        for (int j = 0; j < n; ++j) if (s[j] == quad[i]) { dup = true; break; }
        if (!dup) s[n++] = quad[i];
    }

    if (n == 3) {
        // Three cells meet at this corner: one Delaunay triangle.
        register_tri(s[0], s[1], s[2]);
    } else if (n == 4) {
        // Four cells meet: split the convex quad along the (b,c) anti-diagonal.
        register_tri(a, b, c);
        register_tri(b, c, d);
    }
    // n <= 2: at most two distinct cells, no triangle.
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
// Kernel 2: seed-position-scan triangle assignment
//
// For each pixel:
//   1. Iterate all seeds; skip any whose Chebyshev distance to the pixel
//      exceeds window_cap.  window_cap = max_manhattan_side + SLACK, so it
//      always covers at least one vertex of the containing triangle.
//   2. For each nearby seed, iterate its CSR triangle list and run the
//      containment test.  A triangle is reachable via any of its 3 vertices,
//      so duplicate tests are possible but idempotent.
//
// Correctness argument: for any pixel P inside triangle (A,B,C), at least one
// vertex V satisfies Chebyshev_dist(P,V) <= Manhattan_dist(A,B) (longest side).
// Because window_cap >= max_side + SLACK, every vertex of the containing
// triangle falls within the window.
// ---------------------------------------------------------------------------

static constexpr int WINDOW_SLACK = 3;    // extra radius beyond max_side

__global__
void assign_triangles_kernel(
    int32_t* __restrict__             t_grid,
    int W, int H,
    const RawTriangle* __restrict__   triangles,
    const int32_t* __restrict__       seed_xs,
    const int32_t* __restrict__       seed_ys,
    const int32_t* __restrict__       csr_ptr,    // [N_seeds+1]
    const int32_t* __restrict__       csr_idx,    // triangle IDs per seed
    int N_seeds,
    int window_cap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float px = (float)x;
    float py = (float)y;
    int32_t best = -1;

    for (int sid = 0; sid < N_seeds; ++sid) {
        if (abs(seed_xs[sid] - x) > window_cap) continue;
        if (abs(seed_ys[sid] - y) > window_cap) continue;
        for (int j = csr_ptr[sid]; j < csr_ptr[sid + 1]; ++j) {
            int32_t tid = csr_idx[j];
            const RawTriangle& tri = triangles[tid];
            float ax = (float)seed_xs[tri.orig_a], ay = (float)seed_ys[tri.orig_a];
            float bx = (float)seed_xs[tri.orig_b], by = (float)seed_ys[tri.orig_b];
            float cx = (float)seed_xs[tri.orig_c], cy = (float)seed_ys[tri.orig_c];
            if (point_in_triangle(px, py, ax, ay, bx, by, cx, cy))
                if (best == -1 || tid > best)
                    best = tid;
        }
    }

    t_grid[y * W + x] = best;
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
    // Upload inputs: seed_id channel, seed positions
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_n(N);
    for (int i = 0; i < N; ++i)
        h_n[i] = voronoi_grid[i * 2];

    int32_t *d_n = nullptr;
    int32_t *d_sx = nullptr, *d_sy = nullptr;
    cudaMalloc(&d_n,    N       * sizeof(int32_t));
    cudaMalloc(&d_sx,   N_seeds * sizeof(int32_t));
    cudaMalloc(&d_sy,   N_seeds * sizeof(int32_t));
    cudaMemcpy(d_n,    h_n.data(),       N       * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx,   seed_xs.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy,   seed_ys.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid_dim((W + 15) / 16, (H + 15) / 16);

    // -----------------------------------------------------------------------
    // Step 2: detect raw triangle seeds
    // -----------------------------------------------------------------------

    const int max_raw = 2 * (W - 1) * (H - 1);  // at most 2 triangles per 2x2 block
    RawTriangle* d_raw = nullptr;
    cudaMalloc(&d_raw, max_raw * sizeof(RawTriangle));

    int32_t* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int32_t));
    cudaMemset(d_counter, 0, sizeof(int32_t));

    if (timings) record(ev0);

    find_triangle_seeds_kernel<<<grid_dim, block>>>(d_n, W, H, d_raw, d_counter);
    cudaDeviceSynchronize();

    int32_t raw_count = 0;
    cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_counter);

    if (timings) record(ev1);

    // -----------------------------------------------------------------------
    // Step 3: deduplicate on device with Thrust sort + unique
    // -----------------------------------------------------------------------

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);

    if (timings) record(ev2);

    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end   = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    int N_triangles = (int)(new_end - d_ptr);

    if (timings) record(ev3);

    // Copy deduplicated entries to host (for the Python dict + CSR build)
    std::vector<RawTriangle> h_dedup(N_triangles);
    cudaMemcpy(h_dedup.data(), d_raw, N_triangles * sizeof(RawTriangle),
               cudaMemcpyDeviceToHost);

    triangle_map_out.clear();
    triangle_map_out.reserve(N_triangles);
    for (const auto& r : h_dedup)
        triangle_map_out.push_back({r.x, r.y, r.orig_a, r.orig_b, r.orig_c});

    // -----------------------------------------------------------------------
    // Build CSR: seed → triangle list  (host, then upload)
    //
    // csr_ptr[s]   = start index in csr_idx for seed s
    // csr_ptr[s+1] = end   index
    // csr_idx[i]   = triangle ID
    // -----------------------------------------------------------------------

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

    int32_t *d_csr_ptr = nullptr, *d_csr_idx = nullptr;
    cudaMalloc(&d_csr_ptr, (N_seeds + 1) * sizeof(int32_t));
    cudaMalloc(&d_csr_idx,  csr_size     * sizeof(int32_t));
    cudaMemcpy(d_csr_ptr, h_csr_ptr.data(), (N_seeds + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_idx, h_csr_idx.data(),  csr_size     * sizeof(int32_t), cudaMemcpyHostToDevice);

    // -----------------------------------------------------------------------
    // Step 4: assign pixels to triangles (seed-position-scan)
    // -----------------------------------------------------------------------

    int32_t* d_t = nullptr;
    cudaMalloc(&d_t, N * sizeof(int32_t));

    if (timings) record(ev4);

    // window_cap: Chebyshev guard on seed distance.  For any pixel P inside
    // triangle (A,B,C), at least one vertex V satisfies
    //   max(|Px-Vx|, |Py-Vy|) <= max_manhattan_side
    // so setting window_cap = max_side + SLACK guarantees every vertex of the
    // containing triangle is visited, regardless of Voronoi ownership.
    int max_side = 0;
    for (int tid = 0; tid < N_triangles; ++tid) {
        const auto& r = h_dedup[tid];
        auto md = [&](int32_t i, int32_t j) {
            return abs(seed_xs[i] - seed_xs[j]) + abs(seed_ys[i] - seed_ys[j]);
        };
        int s1 = md(r.orig_a, r.orig_b);
        int s2 = md(r.orig_a, r.orig_c);
        int s3 = md(r.orig_b, r.orig_c);
        max_side = max(max_side, max(s1, max(s2, s3)));
    }
    const int window_cap = max(20, max_side + WINDOW_SLACK);

    if (N_triangles > 0) {
        assign_triangles_kernel<<<grid_dim, block>>>(
            d_t, W, H,
            d_raw, d_sx, d_sy,
            d_csr_ptr, d_csr_idx,
            N_seeds, window_cap);
        cudaDeviceSynchronize();
    } else {
        cudaMemset(d_t, -1, N * sizeof(int32_t));
    }

    if (timings) record(ev5);

    // -----------------------------------------------------------------------
    // Step 5: build output grid (H * W * 3) and fill timings
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_t(N);
    cudaMemcpy(h_t.data(), d_t, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    out_grid.resize(N * 3);
    for (int i = 0; i < N; ++i) {
        out_grid[i * 3]     = voronoi_grid[i * 2];
        out_grid[i * 3 + 1] = voronoi_grid[i * 2 + 1];
        out_grid[i * 3 + 2] = h_t[i];
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
    cudaFree(d_t);
    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_csr_ptr);
    cudaFree(d_csr_idx);
}
