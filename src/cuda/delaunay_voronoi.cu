// Seed insertion and BFS: writes new seeds into the interleaved (seed_id,
// distance) grid and propagates the Voronoi diagram until it settles.
#include "delaunay.cuh"
#include "voronoi.cuh"           // UNDEF_SEED
#include "geometry_device.cuh"   // beats
#include "phase_timer.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>
#include <utility>

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

//: voronoi.cu's step, plus marking each changed pixel in `changed_mask` so the
//: topology stage (delaunay_topology.cu) knows which detection tiles to
//: rescope to instead of rescanning the whole canvas.
__global__
void voronoi_step_kernel(
    const int32_t* __restrict__ src,
    int32_t* __restrict__       dst,
    int W, int H,
    int32_t* __restrict__ updated_flag,
    int32_t* __restrict__ changed_mask,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int32_t best_id, best_d;
    bool changed = voronoi_bfs_step(x, y, W, H, src, seed_xs, seed_ys, best_id, best_d);

    int base = (y * W + x) * 2;
    dst[base]     = best_id;
    dst[base + 1] = best_d;

    if (changed) {
        atomicOr(updated_flag, 1);
        atomicOr(&changed_mask[y * W + x], 1);
    }
}

//: BFS passes to run between convergence checks.
//:
//: A pass costs about 23us of kernel; asking whether it changed anything costs
//: a device synchronisation and two 4-byte transfers, which together run
//: longer. Batching trades up to BFS_CHECK_EVERY - 1 passes that find nothing
//: for one check instead of that many. Runs are 6 to 51 passes, so 8 keeps the
//: waste under a fifth while removing seven eighths of the checks.
static constexpr int BFS_CHECK_EVERY = 8;

void Delaunay::run_bfs_(float* bfs_ms_out, int* iters_out)
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    PhaseTimer<2> timer(bfs_ms_out != nullptr);
    timer.mark(0);

    int32_t zero = 0;
    int iters = 0;
    for (;;) {
        CUDA_CHECK(cudaMemcpy(d_updated_flag_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice));

        // The flag accumulates across the batch, so one read answers "did
        // anything move in any of these". Asking after every pass instead cost
        // a device synchronisation and two 4-byte transfers per pass, around a
        // kernel of 23us -- the question was dearer than the work.
        //
        // No synchronise between passes: the launches are ordered on one
        // stream, so each sees the previous one's output, and the pointer swap
        // is on the host and takes effect at the next launch. The memcpy below
        // blocks until the batch has drained, which is the only wait needed.
        for (int i = 0; i < BFS_CHECK_EVERY; ++i) {
            ++iters;
            voronoi_step_kernel<<<grid_dim, block>>>(
                d_grid_, d_tmp_, W_det_, H_det_, d_updated_flag_, d_changed_,
                d_sx_, d_sy_);
            CUDA_CHECK_LAST_ERROR();
            std::swap(d_grid_, d_tmp_);
        }

        int32_t flag = 0;
        CUDA_CHECK(cudaMemcpy(&flag, d_updated_flag_, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!flag) break;
    }
    // Passes performed, not passes needed: convergence is noticed at the end of
    // a batch, so up to BFS_CHECK_EVERY - 1 of them found nothing to do. That
    // is the cost being reported.
    if (iters_out) *iters_out = iters;

    timer.mark(1);
    if (bfs_ms_out) *bfs_ms_out = timer.elapsed_ms(0, 1);
}
