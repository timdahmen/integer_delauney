// CUDA kernels for GridTriangulation.
//
// Detect (GPU): 2x2-block scan on the (optionally padded) detection grid.
//   Each pixel (x,y) is the top-left corner of a 2x2 block covering pixels
//   (x,y)=a, (x+1,y)=b, (x,y+1)=c, (x+1,y+1)=d.
//   - 3 distinct seed IDs → one triangle from the 3 distinct seeds.
//   - 4 distinct seed IDs → a degree-4 Voronoi vertex (four cocircular seeds):
//     two triangles sharing one diagonal of the quad, chosen by the shorter
//     diagonal and, on a tie, the one avoiding the lowest seed id.  Same rule
//     as reference/triangulation.py, so both paths cut the quad identically.
//   Max output: 2*(W_det-1)*(H_det-1) raw entries (at most 2 per block).
// Dedup  (GPU): thrust::sort + thrust::unique on the device buffer.
// Assign (GPU): seed-position scan; see the kernel 2 header below.

#include "triangulation.cuh"
#include "voronoi.cuh"
#include "triangle_detect.cuh"
#include "geometry_device.cuh"
#include "triangle_csr.cuh"
#include "phase_timer.cuh"

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

// RawTriangle, RawLess, RawEqual and the detection rule itself all live in
// triangle_detect.cuh, shared with the incremental path; UNDEF_SEED and
// NO_TRIANGLE in voronoi.cuh and triangulation.cuh respectively.

// ---------------------------------------------------------------------------
// Kernel 1: detect triangle seeds via 2x2 block scan
// ---------------------------------------------------------------------------

//: Detect the raw (pre-dedup) triangles a 2x2 block scan of the detection grid
//: witnesses. One thread per block position; see detect_block_triangles for
//: the rule and the file header above for the phase this belongs to.
__global__
void find_triangle_seeds_kernel(
    const int32_t* __restrict__ n_grid,
    int W, int H,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys,
    RawTriangle* __restrict__ raw_buf,
    int32_t* __restrict__ counter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W - 1 || y >= H - 1) return;   // 2x2 block must be in bounds

    // Plain seed-id grid here; the incremental path's is interleaved with
    // distance. That layout difference is the only thing this kernel adds.
    detect_block_triangles(
        n_grid[ y      * W + x    ],        // top-left
        n_grid[ y      * W + x + 1],        // top-right
        n_grid[(y + 1) * W + x    ],        // bottom-left
        n_grid[(y + 1) * W + x + 1],        // bottom-right
        seed_xs, seed_ys,
        [&](int32_t oa, int32_t ob, int32_t oc) {
            append_raw_triangle(raw_buf, counter, x, y, oa, ob, oc);
        });
}

// cross2d() and point_in_triangle() are shared with the incremental path;
// see geometry_device.cuh.

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

static constexpr int WINDOW_SLACK = 3;      // extra radius beyond max_side
static constexpr int MIN_WINDOW_CAP = 20;   // floor, so tiny triangles still get a searchable window

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
    int32_t best = NO_TRIANGLE;

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
                if (best == NO_TRIANGLE || tid > best)
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

    // Marks 0/1 bound detect, 2/3 dedup, 4/5 assign -- see the timings block
    // near the end of this function.
    PhaseTimer<6> timer(timings != nullptr);

    // -----------------------------------------------------------------------
    // Build h_n: seed_id channel for the detection grid.
    //
    // With border_padding > 0 we run a fresh Voronoi on the padded canvas with
    // seeds shifted by (P, P).  Shifting preserves lexicographic order, so
    // cuda_compute_voronoi assigns the same IDs as the original run.
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_n(N_det);

    // Kept alive past the branch: at P > 0 its interior window also supplies the
    // output grid's seed-id and distance channels, so no second diagram is
    // needed. Empty when a caller-supplied grid is used instead.
    std::vector<int32_t> padded_flat;

    // A Voronoi diagram computed here when the caller did not bring one. Only
    // reachable at P == 0, where detection reads the unpadded grid directly.
    std::vector<int32_t> own_flat;

    if (P > 0) {
        std::vector<Seed> padded_seeds(N_seeds);
        for (int i = 0; i < N_seeds; ++i)
            padded_seeds[i] = {seed_xs[i] + P, seed_ys[i] + P};

        cuda_compute_voronoi(W_det, H_det, padded_seeds, padded_flat);

        if (padded_voronoi_out)
            *padded_voronoi_out = padded_flat;

        for (int i = 0; i < N_det; ++i)
            h_n[i] = padded_flat[i * 2];   // seed_id at even indices
    } else {
        if (!voronoi_grid) {
            std::vector<Seed> own_seeds(N_seeds);
            for (int i = 0; i < N_seeds; ++i)
                own_seeds[i] = {seed_xs[i], seed_ys[i]};
            cuda_compute_voronoi(W, H, own_seeds, own_flat);
            voronoi_grid = own_flat.data();
        }
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

    timer.mark(0);

    find_triangle_seeds_kernel<<<grid_det, block>>>(
        d_n, W_det, H_det, d_sx, d_sy, d_raw, d_counter);
    cudaDeviceSynchronize();

    int32_t raw_count = 0;
    cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_counter);

    timer.mark(1);

    // -----------------------------------------------------------------------
    // Dedup: on device with Thrust sort + unique
    // -----------------------------------------------------------------------

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);

    timer.mark(2);

    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end   = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    int N_triangles = (int)(new_end - d_ptr);

    timer.mark(3);

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
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_csr_ptr, h_csr_idx;
    build_seed_triangle_csr(h_dedup, N_seeds, [](int) { return false; },
                            h_csr_ptr, h_csr_idx);
    const int csr_size = (int)h_csr_idx.size();  // == 3 * N_triangles

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

    timer.mark(4);

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
    const int window_cap = max(MIN_WINDOW_CAP, max_side + WINDOW_SLACK);

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

    timer.mark(5);

    // -----------------------------------------------------------------------
    // Output: build the (H * W * 3) grid and fill timings
    // -----------------------------------------------------------------------

    std::vector<int32_t> h_t(N);
    cudaMemcpy(h_t.data(), d_t, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Channels 0 and 1 are the seed id and distance. At P > 0 they come from the
    // interior window of the padded diagram, which equals an unpadded one
    // exactly -- verified bit-identical over 256x256 to 1510x1018 and up to
    // 30000 seeds. That is what lets the caller stop computing its own.
    out_grid.resize(N * 3);
    if (P > 0) {
        for (int y = 0; y < H; ++y) {
            const int32_t* src = &padded_flat[(size_t)((y + P) * W_det + P) * 2];
            int32_t* dst = &out_grid[(size_t)y * W * 3];
            for (int x = 0; x < W; ++x) {
                dst[x * 3]     = src[x * 2];
                dst[x * 3 + 1] = src[x * 2 + 1];
                dst[x * 3 + 2] = h_t[y * W + x];
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            out_grid[i * 3]     = voronoi_grid[i * 2];
            out_grid[i * 3 + 1] = voronoi_grid[i * 2 + 1];
            out_grid[i * 3 + 2] = h_t[i];
        }
    }

    if (timings) {
        timings->detect_ms = timer.elapsed_ms(0, 1);
        timings->dedup_ms  = timer.elapsed_ms(2, 3);
        timings->assign_ms = timer.elapsed_ms(4, 5);
    }

    cudaFree(d_raw);
    cudaFree(d_n);
    cudaFree(d_t);
    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_csr_ptr);
    cudaFree(d_csr_idx);
}
