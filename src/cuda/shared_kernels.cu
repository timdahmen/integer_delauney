#include "common.h"
#include "shared_kernels.cuh"

#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ Cell load_cell(const int x, const int y, const int W, const int H, const Cell* src) {
	if (x < 0 || x >= W || y < 0 || y >= H) return {UNDEF, UNDEF};
	return src[y * W + x];
}

__device__ __forceinline__ bool beats(const int32_t a_id, const int32_t a_d, const int32_t b_id, const int32_t b_d) {
	if (b_d < a_d) return true;
	if (b_d == a_d && b_id > a_id) return true;
	return false;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

__global__ void build_output_kernel(int32_t* __restrict__ out,
									const int32_t* __restrict__ voronoi_grid,
									const int32_t* __restrict__ canvas,
									const int N,
									const int default_id) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int32_t tri_id = canvas[i];
	out[i * 3] = voronoi_grid[i * 2];
	out[i * 3 + 1] = voronoi_grid[i * 2 + 1];
	out[i * 3 + 2] = (tri_id != -1) ? tri_id : default_id;
}

__global__ void build_triangle_map_out(TriangleEntry* __restrict__ out,
									   const Vec2i* __restrict__ tri_xy,
									   const Vec3i* __restrict__ tri_og_seeds,
									   const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const auto [x, y] = tri_xy[i];
	const auto [a, b, c] = tri_og_seeds[i];
	out[i] = {x, y, a, b, c};
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float
cross2d(const float ox, const float oy, const float ax, const float ay, const float bx, const float by) {
	return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__ bool
point_in_triangle(const float px, const float py, const Vec2i a, const Vec2i b, const Vec2i c) {
	const float d1 = cross2d(px, py, a.x, a.y, b.x, b.y);
	const float d2 = cross2d(px, py, b.x, b.y, c.x, c.y);
	const float d3 = cross2d(px, py, c.x, c.y, a.x, a.y);
	const bool has_neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
	const bool has_pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
	return !(has_neg && has_pos);
}

// ---------------------------------------------------------------------------
// Kernel: Triangle centric rasterisation
// For each Triangle:
//	 1. launch 1 block with N threads
//	 2. load the corner Seeds for the triangle, blockIdx.x (same across entire block, so cheap)
//	 3. compute triangle AABB, and iterate over it in strides of blockDim.x
//	 4. each iteration each tread checks for 1 pixel, if it is inside the triangle
//		- if it is inside the Triangle, atomicMax the triangle seed so the highest ID wins
// ---------------------------------------------------------------------------

__global__ void rasterize_tri_kernel(int32_t* __restrict__ canvas,
									 const int W,
									 const Vec2i* __restrict__ seed_pos,
									 const Vec3i* __restrict__ tri_seed_ids,
									 const int n_tris) {
	const int warp_id = threadIdx.x / 32; // 32 = warp size
	const int lane_id = threadIdx.x % 32;
	const int tri_id = blockIdx.x * RASTER_TRIS_PER_BLOCK + warp_id;
	if (tri_id >= n_tris) return;

	// get Tri data
	const auto [a_idx, b_idx, c_idx] = tri_seed_ids[tri_id];
	const Vec2i a = seed_pos[a_idx];
	const Vec2i b = seed_pos[b_idx];
	const Vec2i c = seed_pos[c_idx];

	// compute Tri AABB (all are guaranteed to be within canvas)
	const int min_x = min(a.x, min(b.x, c.x));
	const int max_x = max(a.x, max(b.x, c.x));
	const int min_y = min(a.y, min(b.y, c.y));
	const int max_y = max(a.y, max(b.y, c.y));
	const int box_w = max_x - min_x + 1;
	const int box_h = max_y - min_y + 1;
	const int box_area = box_w * box_h;

	// threads within a warp stride across this Tri AABB
	for (int flat = lane_id; flat < box_area; flat += THREADS_PER_TRI) {
		const int x = min_x + (flat % box_w);
		const int y = min_y + (flat / box_w);
		if (point_in_triangle(x + 0.5f, y + 0.5f, a, b, c)) atomicMax(&canvas[y * W + x], tri_id);
	}
}

// ---------------------------------------------------------------------------
// Kernel: L-shape triangle detection with optional mask
// Grid is interleaved (seed_id, distance); only channel 0 is read.
// ---------------------------------------------------------------------------

__global__ void find_triangle_seeds_kernel(const int32_t* __restrict__ voronoi_grid,
										   const int W,
										   const int H,
										   Vec2i* __restrict__ raw_xy,
										   Vec3i* __restrict__ raw_key,
										   Vec3i* __restrict__ raw_orig,
										   int32_t* __restrict__ counter,
										   const int cap) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	auto idx = [&](const int cx, const int cy) -> int32_t { return voronoi_grid[(cy * W + cx) * 2]; };

	auto try_register = [&](const int gx, const int gy, const int32_t a, const int32_t b, const int32_t c) {
		if (a == UNDEF || b == UNDEF || c == UNDEF) return;
		if (a == b || a == c || b == c) return;
		const int32_t pos = atomicAdd(counter, 1);
		if (pos >= cap) return; // check for overflow
		int32_t sa = a, sb = b, sc = c;
		if (sa > sb) {
			const int32_t t = sa;
			sa = sb;
			sb = t;
		}
		if (sb > sc) {
			const int32_t t = sb;
			sb = sc;
			sc = t;
		}
		if (sa > sb) {
			const int32_t t = sa;
			sa = sb;
			sb = t;
		}
		raw_xy[pos] = {gx, gy};
		raw_key[pos] = {sa, sb, sc};
		raw_orig[pos] = {a, b, c};
	};

	const int32_t center = idx(x, y);
	const int32_t left = (x >= 1) ? idx(x - 1, y) : UNDEF;
	const int32_t right = (x <= W - 2) ? idx(x + 1, y) : UNDEF;
	const int32_t up = (y >= 1) ? idx(x, y - 1) : UNDEF;
	const int32_t down = (y <= H - 2) ? idx(x, y + 1) : UNDEF;

	if (y <= H - 2) try_register(x, y, left, center, down);
	if (x <= W - 2) try_register(x, y, center, right, down);
	if (y >= 1) try_register(x, y, center, right, up);
	if (x >= 1) try_register(x, y, left, center, up);
}

// ---------------------------------------------------------------------------
// Kernel: BFS step with per-cell change accumulation
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const int32_t* __restrict__ src_raw,
									int32_t* __restrict__ dst_raw,
									const int W,
									const int H,
									uint8_t* __restrict__ flag_write,
									uint8_t* __restrict__ flag_reset_for_next) {
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		*flag_reset_for_next = 0;
	}

	auto* src = reinterpret_cast<const Cell*>(src_raw);
	auto* dst = reinterpret_cast<Cell*>(dst_raw);

	__shared__ Cell tile[TILE_SIZE_BFS + 2][TILE_SIZE_BFS + 2];

	const int x = blockIdx.x * TILE_SIZE_BFS + threadIdx.x;
	const int y = blockIdx.y * TILE_SIZE_BFS + threadIdx.y;
	const int tile_x = threadIdx.x + 1;
	const int tile_y = threadIdx.y + 1;

	tile[tile_y][tile_x] = load_cell(x, y, W, H, src);
	if (threadIdx.x == 0)
		tile[tile_y][0] = load_cell(x - 1, y, W, H, src);
	else if (threadIdx.x == TILE_SIZE_BFS - 1)
		tile[tile_y][TILE_SIZE_BFS + 1] = load_cell(x + 1, y, W, H, src);
	if (threadIdx.y == 0)
		tile[0][tile_x] = load_cell(x, y - 1, W, H, src);
	else if (threadIdx.y == TILE_SIZE_BFS - 1)
		tile[TILE_SIZE_BFS + 1][tile_x] = load_cell(x, y + 1, W, H, src);

	__syncthreads();
	if (x >= W || y >= H) return;

	Cell cur = tile[tile_y][tile_x];
	Cell best = cur;

	const int dx[4] = {-1, 1, 0, 0};
	const int dy[4] = {0, 0, -1, 1};
	for (int k = 0; k < 4; ++k) {
		Cell neighbor = tile[tile_y + dy[k]][tile_x + dx[k]];
		if (neighbor.id == UNDEF) continue;
		int32_t n_d = neighbor.distance + 1;
		if (best.id == UNDEF || beats(best.id, best.distance, neighbor.id, n_d)) {
			best.id = neighbor.id;
			best.distance = n_d;
		}
	}

	dst[y * W + x] = best;

	if (best.id != cur.id || best.distance != cur.distance) {
		*flag_write = 1;
	}
}