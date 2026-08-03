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

// size of shared memory tile and TILE_SIZExTILE_SIZE threads per block
// (TILE_SIZE+2) * (TILE_SIZE+2) * sizeof(VoronoiCell) has to be below hardware limit of shared memory per block
// and TILE_SIZE * TILE_SIZE has to be below hardware limit of threads per block
static constexpr uint32_t TILE_SIZE = 32;

// Compare two VoronoiCell candidates; return true if b beats a.
// Winner criterion: lower distance wins; on tie, higher seed_id wins.
__device__ __forceinline__ bool beats(const VoronoiCell& a, const VoronoiCell& b) {
	if (b.distance < a.distance) return true;
	if (b.distance == a.distance && b.id > a.id) return true;
	return false;
}

// load a VoronoiCell from the source grid, returning cell, UNDEFINED if out of bounds
__device__ __forceinline__ VoronoiCell
load_cell(const int x, const int y, const int W, const int H, const VoronoiCell* src) {
	if (x < 0 || x >= W || y < 0 || y >= H) return {UNDEFINED, UNDEFINED};
	return src[y * W + x];
}

// ---------------------------------------------------------------------------
// Voronoi step kernel
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const VoronoiCell* __restrict__ src,
									VoronoiCell* __restrict__ dst,
									const int W,
									const int H,
									int32_t* __restrict__ updated_flag,
									uint8_t* __restrict__ tile_settled_flags) {
	const int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	const int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	const int tile_idx = blockIdx.y * gridDim.x + blockIdx.x;
	const int cell_idx = y * W + x;
	const bool in_bounds = (x < W && y < H);

	if (tile_settled_flags[tile_idx]) {
		if (in_bounds) dst[cell_idx] = src[cell_idx];
		return;
	}

	__shared__ VoronoiCell tile[TILE_SIZE + 2][TILE_SIZE + 2]; // +2 for halo, for cross-tile access
	const int tile_x = threadIdx.x + 1;
	const int tile_y = threadIdx.y + 1;

	// each thread loads its own cell into shared memory
	tile[tile_y][tile_x] = load_cell(x, y, W, H, src);
	// load halo cells for cross-tile access
	if (threadIdx.x == 0)
		tile[tile_y][0] = load_cell(x - 1, y, W, H, src);
	else if (threadIdx.x == TILE_SIZE - 1)
		tile[tile_y][TILE_SIZE + 1] = load_cell(x + 1, y, W, H, src);
	if (threadIdx.y == 0)
		tile[0][tile_x] = load_cell(x, y - 1, W, H, src);
	else if (threadIdx.y == TILE_SIZE - 1)
		tile[TILE_SIZE + 1][tile_x] = load_cell(x, y + 1, W, H, src);

	__syncthreads(); // wait for shared memory to be populated

	VoronoiCell cur = tile[tile_y][tile_x];
	VoronoiCell best = cur;

	if (in_bounds) {
		if (cur.id == UNDEFINED) {
			// Check 4 cardinal neighbours; candidate distance = neighbour.distance + 1
			const int dx[4] = {-1, 1, 0, 0};
			const int dy[4] = {0, 0, -1, 1};
			for (int k = 0; k < 4; ++k) {
				VoronoiCell neighbor = tile[tile_y + dy[k]][tile_x + dx[k]];
				if (neighbor.id == UNDEFINED) continue;

				VoronoiCell candidate = neighbor;
				candidate.distance = neighbor.distance + 1;

				if (best.id == UNDEFINED || beats(best, candidate)) {
					best = candidate;
				}
			}

			dst[cell_idx] = best;
			// Mark update if the cell changed (includes first-time fills)
			// No AtomicOr needed, since any thread setting it is enough
			if (best.id != cur.id || best.distance != cur.distance) *updated_flag = 1;
		} else {
			dst[cell_idx] = cur; // already settled: propagate to other buffer
		}
	}

	bool contributes_unsettled = in_bounds && (best.id == UNDEFINED);
	bool block_fully_settled = !__syncthreads_or(contributes_unsettled);
	if (block_fully_settled && threadIdx.x == 0 && threadIdx.y == 0) {
		tile_settled_flags[tile_idx] = 1;
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
	uint8_t* d_tile_settled_flags = nullptr;
	const size_t cell_bytes = N * sizeof(VoronoiCell);

	cudaMalloc(&d_a, cell_bytes);
	cudaMalloc(&d_b, cell_bytes);
	cudaMalloc(&d_flag, sizeof(int32_t));

	// TODO compress to bitfield to save mem
	cudaMemcpy(d_a, h_grid.data(), cell_bytes, cudaMemcpyHostToDevice);

	dim3 block(TILE_SIZE, TILE_SIZE);
	dim3 grid((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

	const int num_tiles = grid.x * grid.y * grid.z;
	cudaMalloc(&d_tile_settled_flags, num_tiles * sizeof(uint8_t));
	cudaMemset(d_tile_settled_flags, 0, num_tiles * sizeof(uint8_t));

	while (true) {
		cudaMemset(d_flag, 0, sizeof(int32_t));

		voronoi_step_kernel<<<grid, block>>>(d_a, d_b, W, H, d_flag, d_tile_settled_flags);
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
	cudaFree(d_tile_settled_flags);
}
