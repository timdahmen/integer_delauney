// CUDA kernel for RegularDelaunay: parallel Manhattan-distance Voronoi BFS.
//
// Grid representation: array of VoronoiCell structures.
// Index layout: cell (x, y) → index (y * W + x).
//   cell.id = seed_id   (-1 = undefined)
//   cell.distance = distance
//
// Double-buffer approach: kernel reads from `src`, writes to `dst`, then
// the host swaps pointers. An `updated` flag is set by any thread
// that changes a cell; the loop stops when no thread sets it.
// The double buffering is applied to the flag, to avoid stalls from copying
// and resetting it host side, when dispatching multiple iterations before checking the flag.

#include "common.h"
#include "shared_kernels.cuh"
#include "voronoi.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_voronoi(const int W,
						  const int H,
						  const std::vector<Vec2i>& seeds,
						  std::vector<int32_t>& out_grid) // (H * W * 2) int32 result (flat for compatibility)
{
	const int N = W * H;

	// Initialise host grid using VoronoiCell
	std::vector<int32_t> h_grid(N * 2);
	for (int i = 0; i < N; ++i) {
		h_grid[i * 2] = UNDEF;
		h_grid[i * 2 + 1] = UNDEF;
	}
	for (size_t i = 0; i < seeds.size(); ++i) {
		const int idx = seeds[i].y * W + seeds[i].x;
		h_grid[idx * 2] = static_cast<int32_t>(i); // seed Id
		h_grid[idx * 2 + 1] = 0; // distance
	}

	// Allocate 2 flags on device, alternate each iteration
	int32_t* d_flags;
	cudaMalloc(&d_flags, 2 * sizeof(int32_t));
	cudaMemset(d_flags, 0, 2 * sizeof(int32_t));

	// Allocate double buffers on device
	int32_t *d_a = nullptr, *d_b = nullptr;
	const size_t cell_bytes = N * 2 * sizeof(int32_t);
	cudaMalloc(&d_a, cell_bytes);
	cudaMalloc(&d_b, cell_bytes);
	cudaMemcpy(d_a, h_grid.data(), cell_bytes, cudaMemcpyHostToDevice);

	dim3 block(TILE_SIZE_BFS, TILE_SIZE_BFS);
	dim3 grid((W + TILE_SIZE_BFS - 1) / TILE_SIZE_BFS, (H + TILE_SIZE_BFS - 1) / TILE_SIZE_BFS);

	int cur_flag = 0;
	while (true) {
		// Run kernel multiple times before checking flag, to reduce MemCopy and Sync fences
		for (int i = 0; i < AGGREGATE_ITERATIONS_BFS; ++i) {
			int32_t* flag_write = d_flags + cur_flag;
			int32_t* flag_reset = d_flags + (1 - cur_flag);

			voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, flag_write, flag_reset);

			std::swap(d_a, d_b); // Swap buffers
			cur_flag = 1 - cur_flag;
		}

		int32_t h_flag = 0;
		cudaMemcpy(&h_flag, d_flags + (1 - cur_flag), sizeof(int32_t), cudaMemcpyDeviceToHost);
		if (!h_flag) break;
	}

	out_grid.resize(N * 2);
	cudaMemcpy(out_grid.data(), d_a, cell_bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_flags);
}
