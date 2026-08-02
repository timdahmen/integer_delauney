// CUDA kernel for RegularDelaunay: parallel Manhattan-distance Voronoi BFS.
//
// Grid representation: array of VoronoiCell structures.
// Index layout: cell (x, y) → index (y * W + x).
//   cell.id = seed_id   (-1 = undefined)
//   cell.distance = distance
//
// Double-buffer approach: kernel reads from `src`, writes to `dst`, then
// the host swaps pointers. An `updated` device flag is OR-ed by any thread
// that changes a cell; the loop stops when no thread sets it.

#include "voronoi.cuh"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEFINED = -1;

// Compare two VoronoiCell candidates; return true if b beats a.
// Winner criterion: lower distance wins; on tie, higher seed_id wins.
__device__ __forceinline__ bool beats(const VoronoiCell& a, const VoronoiCell& b) {
	if (b.distance < a.distance) return true;
	if (b.distance == a.distance && b.id > a.id) return true;
	return false;
}

// ---------------------------------------------------------------------------
// Voronoi step kernel
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const VoronoiCell* __restrict__ src,
									VoronoiCell* __restrict__ dst,
									int W,
									int H,
									int32_t* __restrict__ updated_flag) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int idx = y * W + x;
	VoronoiCell cur = src[idx];

	VoronoiCell best = cur;

	// Check 4 cardinal neighbours; candidate distance = neighbour.distance + 1
	const int dx[4] = {-1, 1, 0, 0};
	const int dy[4] = {0, 0, -1, 1};

	for (int k = 0; k < 4; ++k) {
		int nx = x + dx[k];
		int ny = y + dy[k];
		if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

		int n_idx = ny * W + nx;
		VoronoiCell neighbor = src[n_idx];
		if (neighbor.id == UNDEFINED) continue;

		VoronoiCell candidate = neighbor;
		candidate.distance = neighbor.distance + 1;

		if (best.id == UNDEFINED || beats(best, candidate)) {
			best = candidate;
		}
	}

	dst[idx] = best;

	// Mark update if the cell changed (includes first-time fills)
	if (best.id != cur.id || best.distance != cur.distance) {
		atomicOr(updated_flag, 1);
	}
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_voronoi(const int W,
						  const int H,
						  const std::vector<Seed>& seeds,
						  std::vector<int32_t>& out_grid) // (H * W * 2) int32 result (flat for compatibility)
{
	const int N = W * H;

	// Initialise host grid using VoronoiCell
	std::vector<VoronoiCell> h_grid(N);
	for (int i = 0; i < N; ++i) {
		h_grid[i].id = UNDEFINED;
		h_grid[i].distance = UNDEFINED;
	}
	for (size_t i = 0; i < seeds.size(); ++i) {
		const int idx = seeds[i].y * W + seeds[i].x;
		h_grid[idx].id = static_cast<int32_t>(i);
		h_grid[idx].distance = 0;
	}

	// Allocate double buffers on device
	VoronoiCell *d_a = nullptr, *d_b = nullptr;
	int32_t* d_flag = nullptr;
	const size_t cell_bytes = N * sizeof(VoronoiCell);

	cudaMalloc(&d_a, cell_bytes);
	cudaMalloc(&d_b, cell_bytes);
	cudaMalloc(&d_flag, sizeof(int32_t));

	cudaMemcpy(d_a, h_grid.data(), cell_bytes, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid((W + 15) / 16, (H + 15) / 16);

	for (;;) {
		int32_t zero = 0;
		cudaMemcpy(d_flag, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

		voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, d_flag);
		cudaDeviceSynchronize();

		// Swap buffers
		VoronoiCell* tmp = d_a;
		d_a = d_b;
		d_b = tmp;

		int32_t flag = 0;
		cudaMemcpy(&flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
		if (!flag) break;
	}

	// Convert result back to flat int32 array for compatibility with bindings
	std::vector<VoronoiCell> h_result(N);
	cudaMemcpy(h_result.data(), d_a, cell_bytes, cudaMemcpyDeviceToHost);

	// Flatten VoronoiCell array to (id, distance) pairs for output
	out_grid.resize(N * 2);
	for (int i = 0; i < N; ++i) {
		out_grid[i * 2] = h_result[i].id;
		out_grid[i * 2 + 1] = h_result[i].distance;
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_flag);
}
