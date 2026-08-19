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
#include "geometry_device.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

//: One synchronous BFS pass over the whole canvas: every pixel takes the best
//: (seed_id, distance) among itself and its 4 neighbours (voronoi_bfs_step,
//: geometry_device.cuh), and any pixel that changed sets `updated_flag` so
//: the host knows to loop again.
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

    int32_t best_id, best_d;
    bool changed = voronoi_bfs_step(x, y, W, H, src, seed_xs, seed_ys, best_id, best_d);

    int base = (y * W + x) * 2;
    dst[base]     = best_id;
    dst[base + 1] = best_d;

    if (changed) atomicOr(updated_flag, 1);
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

    // Allocate double buffers + seed position arrays on device. Per-call
    // alloc/free rather than persistent buffers -- see todo.txt for that
    // design tradeoff, which this change does not revisit.
    DeviceBuffer<int32_t> d_a_buf(N * 2), d_b_buf(N * 2);
    DeviceBuffer<int32_t> d_flag(1);
    DeviceBuffer<int32_t> d_sx(N_seeds), d_sy(N_seeds);
    int32_t* d_a = d_a_buf;
    int32_t* d_b = d_b_buf;

    CUDA_CHECK(cudaMemcpy(d_a,  h_grid.data(), bytes,                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sx, h_sx.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy, h_sy.data(),   N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    for (;;) {
        int32_t zero = 0;
        CUDA_CHECK(cudaMemcpy(d_flag, &zero, sizeof(int32_t), cudaMemcpyHostToDevice));

        voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, d_sx, d_sy, d_flag);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());

        int32_t* tmp = d_a; d_a = d_b; d_b = tmp;

        int32_t flag = 0;
        CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!flag) break;
    }

    out_grid.resize(N * 2);
    CUDA_CHECK(cudaMemcpy(out_grid.data(), d_a, bytes, cudaMemcpyDeviceToHost));
}
