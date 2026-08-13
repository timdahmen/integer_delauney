// CUDA kernels for GridTriangulation.
//
// Detect (GPU): 2x2-block scan on the (optionally padded) detection grid.
//   Each pixel (x,y) is the top-left corner of a 2x2 block covering pixels
//   (x,y)=a, (x+1,y)=b, (x,y+1)=c, (x+1,y+1)=d.
//   - 3 distinct seed IDs → one triangle from the 3 distinct seeds.
//   - 4 distinct seed IDs → two triangles: (a,b,c) and (b,c,d), splitting
//     the quad along the (b,c) anti-diagonal.  Both triangles are valid since
//     4 Voronoi cells sharing a corner implies the 4 seeds are in convex
//     position.  The NumPy reference instead picks the shorter diagonal, so
//     the two can cut a cocircular quad differently.
//   Max output: 2*(W_det-1)*(H_det-1) raw entries (at most 2 per block).
// Dedup  (GPU): thrust::sort + thrust::unique on the device buffer.
// Assign (GPU): seed-position scan; see the kernel 2 header below.

#include "triangulation.cuh"
#include "voronoi.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
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
//      exceeds window_cap.
//   2. For each surviving seed, iterate its CSR triangle list and run the
//      containment test.  A triangle is reachable via any of its 3 vertices,
//      so duplicate tests are possible but idempotent.
//
// window_cap (computed host-side) is the longest L2 side over ALL detected
// triangles, plus WINDOW_SLACK, floored at 20.  Every vertex of a containing
// triangle is within its own longest side of the pixel, so that bound covers
// them regardless of which Voronoi region the pixel belongs to.
//
// Expected cost: O(seeds in window × tris/seed); scales with local seed
// density, not total N.
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
    TriTimings* timings,
    int border_padding,
    std::vector<int32_t>* padded_voronoi_out)
{
    const int N        = W * H;
    const int N_seeds  = (int)seed_xs.size();
    const int P        = border_padding;

    // Detection grid dimensions: padded canvas captures Voronoi vertices that
    // lie outside the original image, producing the missing border triangles.
    const int W_det = W + 2 * P;
    const int H_det = H + 2 * P;
    const int N_det = W_det * H_det;

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
    // Build h_n: seed_id channel for the detection grid.
    //
    // With border_padding > 0 we run a fresh Voronoi on the padded canvas with
    // seeds shifted by (P, P).  Shifting preserves lexicographic order, so
    // cuda_compute_voronoi assigns the same IDs as the original run.
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_n(N_det);

    if (P > 0) {
        std::vector<Seed> padded_seeds(N_seeds);
        for (int i = 0; i < N_seeds; ++i)
            padded_seeds[i] = {seed_xs[i] + P, seed_ys[i] + P};

        std::vector<int32_t> padded_flat;
        cuda_compute_voronoi(W_det, H_det, padded_seeds, padded_flat);

        if (padded_voronoi_out)
            *padded_voronoi_out = padded_flat;

        for (int i = 0; i < N_det; ++i)
            h_n[i] = padded_flat[i * 2];   // seed_id at even indices
    } else {
        for (int i = 0; i < N; ++i)
            h_n[i] = voronoi_grid[i * 2];
    }

    // -----------------------------------------------------------------------
    // Upload inputs: detection seed_id grid, seed positions (always original)
    // -----------------------------------------------------------------------

    int32_t *d_n = nullptr;
    int32_t *d_sx = nullptr, *d_sy = nullptr;
    cudaMalloc(&d_n,    N_det   * sizeof(int32_t));
    cudaMalloc(&d_sx,   N_seeds * sizeof(int32_t));
    cudaMalloc(&d_sy,   N_seeds * sizeof(int32_t));
    cudaMemcpy(d_n,  h_n.data(),       N_det   * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx, seed_xs.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy, seed_ys.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid_det((W_det + 15) / 16, (H_det + 15) / 16);
    dim3 grid_out((W     + 15) / 16, (H     + 15) / 16);

    // -----------------------------------------------------------------------
    // Detect: raw triangle seeds on the (possibly padded) grid
    // -----------------------------------------------------------------------

    const int max_raw = 2 * (W_det - 1) * (H_det - 1);
    RawTriangle* d_raw = nullptr;
    cudaMalloc(&d_raw, max_raw * sizeof(RawTriangle));

    int32_t* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int32_t));
    cudaMemset(d_counter, 0, sizeof(int32_t));

    if (timings) record(ev0);

    find_triangle_seeds_kernel<<<grid_det, block>>>(d_n, W_det, H_det, d_raw, d_counter);
    cudaDeviceSynchronize();

    int32_t raw_count = 0;
    cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_counter);

    if (timings) record(ev1);

    // -----------------------------------------------------------------------
    // Dedup: on device with Thrust sort + unique
    // -----------------------------------------------------------------------

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);

    if (timings) record(ev2);

    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end   = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    int N_triangles = (int)(new_end - d_ptr);

    if (timings) record(ev3);

    // Copy deduplicated entries to host.
    // Shift detection pixel (x,y) back from padded space to original space;
    // border triangles will have x or y outside [0,W-1]/[0,H-1] which is correct
    // (their Voronoi vertex lies outside the original image).
    std::vector<RawTriangle> h_dedup(N_triangles);
    cudaMemcpy(h_dedup.data(), d_raw, N_triangles * sizeof(RawTriangle),
               cudaMemcpyDeviceToHost);

    triangle_map_out.clear();
    triangle_map_out.reserve(N_triangles);
    for (const auto& r : h_dedup)
        triangle_map_out.push_back({r.x - P, r.y - P, r.orig_a, r.orig_b, r.orig_c});

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
    // Assign: pixels to triangles (seed-position scan)
    //
    // Always runs on the original W×H with original seed coordinates.
    // Border triangles (detected via padding) cover pixels near the image edge
    // because their seed vertices are inside the original image.
    // -----------------------------------------------------------------------

    int32_t* d_t = nullptr;
    cudaMalloc(&d_t, N * sizeof(int32_t));

    if (timings) record(ev4);

    int max_side = 0;
    for (int tid = 0; tid < N_triangles; ++tid) {
        const auto& r = h_dedup[tid];
        auto l2 = [&](int32_t i, int32_t j) {
            float dx = (float)(seed_xs[i] - seed_xs[j]);
            float dy = (float)(seed_ys[i] - seed_ys[j]);
            return (int)std::sqrtf(dx * dx + dy * dy);
        };
        int s1 = l2(r.orig_a, r.orig_b);
        int s2 = l2(r.orig_a, r.orig_c);
        int s3 = l2(r.orig_b, r.orig_c);
        max_side = max(max_side, max(s1, max(s2, s3)));
    }
    const int window_cap = max(20, max_side + WINDOW_SLACK);

    if (N_triangles > 0) {
        assign_triangles_kernel<<<grid_out, block>>>(
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
    // Output: build the (H * W * 3) grid and fill timings
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
