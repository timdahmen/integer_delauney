// CUDA kernel for RegularDelaunay: parallel Manhattan-distance Voronoi BFS.
//
// Grid representation: flat int32 array of (seed_id, distance) pairs.
// Index layout: cell (x, y) → base index (y * W + x) * 2.
//   [base+0] = seed_id   (-1 = undefined)
//   [base+1] = distance
//
// Double-buffer approach: kernel reads from `src`, writes to `dst`, then
// the host swaps pointers. An `updated` device flag is OR-ed by any thread
// that changes a cell; the loop stops when no thread sets it.

#include "voronoi.cuh"

#include "nvtx_range.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEFINED = -1;

// Compare two (seed_id, distance) candidates; return true if b beats a.
// Winner criterion: lower distance wins; on tie, higher seed_id wins.
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

    // Check 4 cardinal neighbours; candidate distance = neighbour_d + 1
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = { 0, 0,-1, 1};

    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

        int nb = (ny * W + nx) * 2;
        int32_t n_id = src[nb];
        int32_t n_d  = src[nb + 1] + 1;   // distance through this neighbour
        if (n_id == UNDEFINED) continue;

        if (best_id == UNDEFINED || beats(best_id, best_d, n_id, n_d)) {
            best_id = n_id;
            best_d  = n_d;
        }
    }

    dst[base]     = best_id;
    dst[base + 1] = best_d;

    // Mark update if the cell changed (includes first-time fills)
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
    DELAUNEY_NVTX_RANGE_C("cuda_compute_voronoi", delauney_nvtx::kPhase);

    const int N = W * H;
    const int bytes = N * 2 * sizeof(int32_t);

    // Initialise host grid
    int32_t *d_a = nullptr, *d_b = nullptr, *d_flag = nullptr;
    {
        DELAUNEY_NVTX_RANGE_C("vor: init + upload grid", delauney_nvtx::kMemcpy);

        std::vector<int32_t> h_grid(N * 2, UNDEFINED);
        for (size_t i = 0; i < seeds.size(); ++i) {
            int base = (seeds[i].y * W + seeds[i].x) * 2;
            h_grid[base]     = static_cast<int32_t>(i);
            h_grid[base + 1] = 0;
        }

        // Allocate double buffers on device
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_flag, sizeof(int32_t));

        cudaMemcpy(d_a, h_grid.data(), bytes, cudaMemcpyHostToDevice);
    }

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    {
        DELAUNEY_NVTX_RANGE_C("vor: bfs loop", delauney_nvtx::kKernel);
        for (;;) {
            int32_t zero = 0;
            cudaMemcpy(d_flag, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

            voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, d_flag);
            cudaDeviceSynchronize();

            // Swap buffers
            int32_t* tmp = d_a; d_a = d_b; d_b = tmp;

            int32_t flag = 0;
            cudaMemcpy(&flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (!flag) break;
        }
    }

    {
        DELAUNEY_NVTX_RANGE_C("vor: D2H result", delauney_nvtx::kMemcpy);
        out_grid.resize(N * 2);
        cudaMemcpy(out_grid.data(), d_a, bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_flag);
}
