// CUDA kernel for Voronoi: parallel L2-distance Voronoi BFS.
//
// Grid representation: flat int32 array of (seed_id, squared_l2_distance) pairs.
// Index layout: cell (x, y) → base index (y * W + x) * 2.
//   [base+0] = seed_id              (-1 = undefined)
//   [base+1] = squared L2 distance  (0 at seed pixel)
//
// Each BFS step propagates seed identity one pixel in a cardinal direction.
// Distance is NOT accumulated along the path — it is recomputed as the
// direct squared Euclidean distance from the owning seed to the current pixel.
// This makes the distance metric exact regardless of propagation direction.
//
// Double-buffer approach: kernel reads from `src`, writes to `dst`, then
// the host swaps pointers. An `updated` device flag is OR-ed by any thread
// that changes a cell; the loop stops when no thread sets it.

#include "voronoi.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// UNDEF_SEED comes from voronoi.cuh.

// Compare two (seed_id, distance) candidates; return true if b beats a.
// Lower distance wins; on tie, higher seed_id wins.
//
// Only among the candidates this cell actually sees: propagation is local and
// this kernel advances one cell per iteration, so it can settle at a fixed
// point that is not nearest-seed (rare, grows with seed density -- measured
// 2/16384 pixels at 128x128 with 400 seeds, off by 1-3 in squared distance).
// See Voronoi's docstring in reference/voronoi.py.
__device__ __forceinline__
bool beats(int32_t a_id, int32_t a_d, int32_t b_id, int32_t b_d)
{
    if (b_d < a_d) return true;
    if (b_d == a_d && b_id > a_id) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Voronoi step kernel
// ---------------------------------------------------------------------------

__global__
void voronoi_step_kernel(
    const int32_t* __restrict__ src,
    int32_t*       __restrict__ dst,
    int W, int H,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys,
    int32_t* __restrict__ updated_flag)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int base = (y * W + x) * 2;
    int32_t cur_id  = src[base];
    int32_t cur_d   = src[base + 1];

    int32_t best_id = cur_id;
    int32_t best_d  = cur_d;

    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = { 0, 0,-1, 1};

    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

        int nb = (ny * W + nx) * 2;
        int32_t n_id = src[nb];
        if (n_id == UNDEF_SEED) continue;

        // Recomputed from the seed position, never accumulated through
        // neighbours -- that would give a Manhattan metric.
        int32_t ddx = x - seed_xs[n_id];
        int32_t ddy = y - seed_ys[n_id];
        int32_t n_d = ddx * ddx + ddy * ddy;

        if (best_id == UNDEF_SEED || beats(best_id, best_d, n_id, n_d)) {
            best_id = n_id;
            best_d  = n_d;
        }
    }

    dst[base]     = best_id;
    dst[base + 1] = best_d;

    if (best_id != cur_id || best_d != cur_d) {
        atomicOr(updated_flag, 1);
    }
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_voronoi(
    int W, int H,
    const std::vector<Seed>& seeds,
    std::vector<int32_t>& out_grid)   // (H * W * 2) int32 result
{
    const int N       = W * H;
    const int bytes   = N * 2 * sizeof(int32_t);
    const int N_seeds = (int)seeds.size();

    // Initialise host grid: seed pixels get (seed_id, 0), rest get UNDEF_SEED
    std::vector<int32_t> h_grid(N * 2, UNDEF_SEED);
    for (int i = 0; i < N_seeds; ++i) {
        int base = (seeds[i].y * W + seeds[i].x) * 2;
        h_grid[base]     = static_cast<int32_t>(i);
        h_grid[base + 1] = 0;   // squared L2 distance = 0 at seed
    }

    // Seed position arrays for direct L2 recomputation in the kernel
    std::vector<int32_t> h_sx(N_seeds), h_sy(N_seeds);
    for (int i = 0; i < N_seeds; ++i) {
        h_sx[i] = seeds[i].x;
        h_sy[i] = seeds[i].y;
    }

    // Allocate double buffers + seed position arrays on device
    int32_t *d_a = nullptr, *d_b = nullptr, *d_flag = nullptr;
    int32_t *d_sx = nullptr, *d_sy = nullptr;
    cudaMalloc(&d_a,    bytes);
    cudaMalloc(&d_b,    bytes);
    cudaMalloc(&d_flag, sizeof(int32_t));
    cudaMalloc(&d_sx,   N_seeds * sizeof(int32_t));
    cudaMalloc(&d_sy,   N_seeds * sizeof(int32_t));

    cudaMemcpy(d_a,  h_grid.data(), bytes,                     cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx, h_sx.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy, h_sy.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    for (;;) {
        int32_t zero = 0;
        cudaMemcpy(d_flag, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

        voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, d_sx, d_sy, d_flag);
        cudaDeviceSynchronize();

        int32_t* tmp = d_a; d_a = d_b; d_b = tmp;

        int32_t flag = 0;
        cudaMemcpy(&flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (!flag) break;
    }

    out_grid.resize(N * 2);
    cudaMemcpy(out_grid.data(), d_a, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_flag);
    cudaFree(d_sx);
    cudaFree(d_sy);
}
