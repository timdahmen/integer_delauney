// CUDA kernels for GridTriangulation.
// Step 1 (CPU): prepare and upload data to the GPU
// Step 2 (GPU): scan all 4 L-shape orientations; raw hits → device buffer.
// Step 3 (GPU): thrust::sort_by_key + thrust::unique_by_key on the device buffer.
// Step 4 (GPU): triangle rasterisation. 1 block per triangle, using multiple threads per 
//				 triangle so that consecutive threads access neighbouring memory.
//
// Host copies:
//   • voronoi grid (seed_id + distance, interleaved) → device  (input, once)
//   • seed positions                                 → device  (input, once)
//   • deduplicated triangles                         ← device  (output; x,y for
//                                                      triangle_map_out + orig_a/b/c
//                                                      for rasterize_tri_kernel)
//   • triangulation grid (seed_id, distance, tri_id) ← device  (output, W*H*3 ints)
//
// No CSR adjacency list: triangle → pixel assignment is now done by a
// triangle-centric rasterizer (one block per triangle, atomicMax into the
// canvas) instead of a per-pixel seed-window search, so the seed → triangle
// CSR this pipeline used to build is no longer needed.

#include "triangulation.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Shared structs
// ---------------------------------------------------------------------------


static constexpr int32_t UNDEF = -1;

// Split from the original single RawTriangle{x,y,a,b,c,orig_a,orig_b,orig_c}
// into three groups, grouped by who actually reads them together:
//   RawXY   - host-only output (triangle_map_out), never read on device after write
//   RawKey  - dedup/sort key only; discarded once unique() finishes
//   RawOrig - hot path: read by assign_triangles_kernel for every candidate test
struct Vec2i {
	int32_t x, y;
};
struct Vec3i {
	int32_t a, b, c;
};

struct KeyLess {
	__device__ bool operator()(const Vec3i& x, const Vec3i& y) const {
		if (x.a != y.a) return x.a < y.a;
		if (x.b != y.b) return x.b < y.b;
		return x.c < y.c;
	}
};

struct KeyEqual {
	__device__ bool operator()(const Vec3i& x, const Vec3i& y) const { return x.a == y.a && x.b == y.b && x.c == y.c; }
};

// ---------------------------------------------------------------------------
// Kernel 1: detect triangle seeds (all 4 L-shape orientations)
// ---------------------------------------------------------------------------

__global__ void find_triangle_seeds_kernel(const int32_t* __restrict__ n_grid,
										   const int W,
										   const int H,
										   Vec2i* __restrict__ raw_xy,
										   Vec3i* __restrict__ raw_key,
										   Vec3i* __restrict__ raw_orig,
										   int32_t* __restrict__ counter) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	auto idx = [&](const int cx, const int cy) -> int32_t { return n_grid[cy * W + cx]; };

	auto try_register = [&](const int gx, const int gy, const int32_t a, const int32_t b, const int32_t c) {
		if (a == UNDEF || b == UNDEF || c == UNDEF) return;
		if (a == b || a == c || b == c) return;
		const int32_t pos = atomicAdd(counter, 1);
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

	if (x >= 1 && y <= H - 2) try_register(x, y, idx(x - 1, y), idx(x, y), idx(x, y + 1));
	if (x <= W - 2 && y <= H - 2) try_register(x, y, idx(x, y), idx(x + 1, y), idx(x, y + 1));
	if (x <= W - 2 && y >= 1) try_register(x, y, idx(x, y), idx(x + 1, y), idx(x, y - 1));
	if (x >= 1 && y >= 1) try_register(x, y, idx(x - 1, y), idx(x, y), idx(x, y - 1));
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float
cross2d(const float ox, const float oy, const float ax, const float ay, const float bx, const float by) {
	return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__ bool
point_in_triangle(const float px, const  float py, const Vec2i a, const Vec2i b, const Vec2i c) {
	const float d1 = cross2d(px, py, a.x, a.y, b.x, b.y);
	const float d2 = cross2d(px, py, b.x, b.y, c.x, c.y);
	const float d3 = cross2d(px, py, c.x, c.y, a.x, a.y);
	const bool has_neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
	const bool has_pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
	return !(has_neg && has_pos);
}

// ---------------------------------------------------------------------------
// Kernel 2: Triangle centric rasterisation
// For each Triangle:
//	 1. launch 1 block with N threads
//	 2. load the corner Seeds for the triangle, blockIdx.x (same across entire block, so cheap)
//	 3. compute triangle AABB, and iterate over it in strides of blockDim.x
//	 4. each iteration each tread checks for 1 pixel, if it is inside the triangle
//		- if it is inside the Triangle, atomicMax the triangle seed so the highest ID wins
// ---------------------------------------------------------------------------

__global__ void rasterize_tri_kernel(
	int32_t* __restrict__ canvas, const int W, const Vec2i* seed_pos, const Vec3i* tri_seed_ids, const int32_t n_tris) {
	const int tri_id = blockIdx.x;
	if (tri_id >= n_tris) return;

	// get Tri data
	const auto [a_idx, b_idx, c_idx] = tri_seed_ids[tri_id];
	const Vec2i a = seed_pos[a_idx]; // same address for whole block
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

	// check for pixels in AABB, if they are in Tri, in strides across multiple threads
	for (int flat = threadIdx.x; flat < box_area; flat += blockDim.x) {
		const int x = min_x + (flat % box_w);
		const int y = min_y + (flat / box_w);
		const float px = x + 0.5f, py = y + 0.5f;

		if (point_in_triangle(px, py, a, b, c)) atomicMax(&canvas[y * W + x], tri_id);
	}
}

// ---------------------------------------------------------------------------
// Guard for cudaMalloc device memory.
// There are multiple throw points in cuda_compute_triangulation. 
// This automates the free, and avoids manual checking/freeing
// ---------------------------------------------------------------------------
struct CudaFreeGuard {
	void* ptr = nullptr;
	explicit CudaFreeGuard(void* p = nullptr) : ptr(p) {}
	CudaFreeGuard(const CudaFreeGuard&) = delete;
	CudaFreeGuard& operator=(const CudaFreeGuard&) = delete;
	~CudaFreeGuard() {
		if (ptr) cudaFree(ptr);
	}
};

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_triangulation(const int W,
								const int H,
								const int32_t* voronoi_grid,
								const std::vector<int32_t>& seed_xs,
								const std::vector<int32_t>& seed_ys,
								std::vector<TriangleEntry>& triangle_map_out,
								std::vector<int32_t>& out_grid,
								TriTimings* timings) {
	const int N = W * H;
	const int N_seeds = static_cast<int>(seed_xs.size());

	auto make_event = [](cudaEvent_t* e) { cudaEventCreate(e); };
	auto record = [](cudaEvent_t e) { cudaEventRecord(e); };
	auto elapsed_ms = [](cudaEvent_t a, cudaEvent_t b) -> float {
		cudaEventSynchronize(b);
		float ms = 0.f;
		cudaEventElapsedTime(&ms, a, b);
		return ms;
	};

	cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5;
	if (timings) {
		make_event(&ev0);
		make_event(&ev1);
		make_event(&ev2);
		make_event(&ev3);
		make_event(&ev4);
		make_event(&ev5);
	}

	// -----------------------------------------------------------------------
	// Upload inputs: seed_id channel, seed positions
	// -----------------------------------------------------------------------

	std::vector<int32_t> h_n(N);
	for (int i = 0; i < N; ++i) h_n[i] = voronoi_grid[i * 2];
	std::vector<Vec2i> h_seed_pos(N_seeds);
	for (int i = 0; i < N_seeds; ++i) h_seed_pos[i] = {seed_xs[i], seed_ys[i]};

	int32_t *d_seed_ids = nullptr, *d_dist = nullptr;
	Vec2i* d_seed_pos = nullptr;
	cudaMalloc(&d_seed_ids, N * sizeof(int32_t));
	cudaMalloc(&d_seed_pos, N_seeds * sizeof(Vec2i));
	CudaFreeGuard g_seed_ids(d_seed_ids);
	CudaFreeGuard g_seed_pos(d_seed_pos);
	cudaMemcpy(d_seed_ids, h_n.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_seed_pos, h_seed_pos.data(), N_seeds * sizeof(Vec2i), cudaMemcpyHostToDevice);


	// -----------------------------------------------------------------------
	// Step 2: detect raw triangle seeds
	// -----------------------------------------------------------------------

	const int max_raw = N * 4;
	Vec2i* d_tri_xy = nullptr;
	Vec3i* d_tri_key = nullptr;
	Vec3i* d_tri_orig = nullptr;
	cudaMalloc(&d_tri_xy, max_raw * sizeof(Vec2i));
	cudaMalloc(&d_tri_key, max_raw * sizeof(Vec3i));
	cudaMalloc(&d_tri_orig, max_raw * sizeof(Vec3i));
	CudaFreeGuard g_tri_xy(d_tri_xy);
	CudaFreeGuard g_tri_key(d_tri_key);
	CudaFreeGuard g_tri_orig(d_tri_orig);

	int32_t* d_counter = nullptr;
	cudaMalloc(&d_counter, sizeof(int32_t));
	CudaFreeGuard g_counter(d_counter);
	cudaMemset(d_counter, 0, sizeof(int32_t));

	dim3 block(16, 16);
	dim3 grid_dim((W + 15) / 16, (H + 15) / 16);

	if (timings) record(ev0);

	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_seed_ids, W, H, d_tri_xy, d_tri_key, d_tri_orig, d_counter);
	{
		cudaError_t sync_err = cudaDeviceSynchronize();
		if (sync_err != cudaSuccess) {
			throw std::runtime_error(std::string("find_triangle_seeds_kernel execution failed: ") +
									 cudaGetErrorString(sync_err));
		}
	}

	int32_t raw_count = 0;
	cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

	if (timings) record(ev1);

	// -----------------------------------------------------------------------
	// Step 3: deduplicate on device with Thrust sort_by_key + unique_by_key
	// Key = Vec3i(a,b,c); values = (x/y, Orig_a/b/c) zipped so they're
	// reordered/compacted in lockstep with the key, in one pass each.
	// -----------------------------------------------------------------------

	thrust::device_ptr<Vec3i> d_key_ptr(d_tri_key);
	thrust::device_ptr<Vec2i> d_xy_ptr(d_tri_xy);
	thrust::device_ptr<Vec3i> d_orig_ptr(d_tri_orig);

	auto values_begin = thrust::make_zip_iterator(thrust::make_tuple(d_xy_ptr, d_orig_ptr));

	if (timings) record(ev2);

	thrust::sort_by_key(d_key_ptr, d_key_ptr + raw_count, values_begin, KeyLess{});
	auto unique_result = thrust::unique_by_key(d_key_ptr, d_key_ptr + raw_count, values_begin, KeyEqual{});
	const int N_triangles = static_cast<int>(unique_result.first - d_key_ptr);

	if (timings) record(ev3);

	// Copy deduplicated x,y and orig_a/b/c to host (Python dict + CSR build)
	std::vector<Vec2i> h_xy(N_triangles);
	std::vector<Vec3i> h_orig(N_triangles);
	cudaMemcpy(h_xy.data(), d_tri_xy, N_triangles * sizeof(Vec2i), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_orig.data(), d_tri_orig, N_triangles * sizeof(Vec3i), cudaMemcpyDeviceToHost);

	triangle_map_out.clear();
	triangle_map_out.reserve(N_triangles);
	for (int i = 0; i < N_triangles; ++i)
		triangle_map_out.push_back({h_xy[i].x, h_xy[i].y, h_orig[i].a, h_orig[i].b, h_orig[i].c});

	// -----------------------------------------------------------------------
	// Step 4: assign pixels to triangles
	// -----------------------------------------------------------------------

	int32_t* d_canvas = nullptr;
	cudaMalloc(&d_canvas, N * sizeof(int32_t));
	CudaFreeGuard g_canvas(d_canvas);
	cudaMemset(d_canvas, -1, N * sizeof(int32_t)); // initialize Canvas with sentinal value

	if (timings) record(ev4);

	if (N_triangles > 0) {
		static constexpr int threads_per_block = 256;
		int num_blocks = (N_triangles + threads_per_block - 1) / threads_per_block;
		rasterize_tri_kernel<<<num_blocks, threads_per_block>>>(d_canvas, W, d_seed_pos, d_tri_orig, N_triangles);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("rasterize_tri_kernel launch failed: ") + cudaGetErrorString(err));
		}
		cudaError_t sync_err = cudaDeviceSynchronize();
		if (sync_err != cudaSuccess) {
			throw std::runtime_error(std::string("rasterize_tri_kernel execution failed: ") +
									 cudaGetErrorString(sync_err));
		}
	} else {
		cudaMemset(d_canvas, -1, N * sizeof(int32_t));
	}

	if (timings) record(ev5);

	// -----------------------------------------------------------------------
	// Step 5: build output grid (H * W * 3) and fill timings
	// -----------------------------------------------------------------------

	std::vector<int32_t> h_canvas(N);
	cudaMemcpy(h_canvas.data(), d_canvas, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

	const int32_t default_val = N_triangles - 1;
	out_grid.resize(N * 3);
	for (int i = 0; i < N; ++i) {
		int32_t t = h_canvas[i];
		out_grid[i * 3] = voronoi_grid[i * 2];
		out_grid[i * 3 + 1] = voronoi_grid[i * 2 + 1];
		out_grid[i * 3 + 2] = (t != -1) ? t : default_val;
	}

	if (timings) {
		timings->detect_ms = elapsed_ms(ev0, ev1);
		timings->sort_ms = elapsed_ms(ev1, ev2);
		timings->dedup_ms = elapsed_ms(ev2, ev3);
		timings->assign_ms = elapsed_ms(ev4, ev5);
		cudaEventDestroy(ev0);
		cudaEventDestroy(ev1);
		cudaEventDestroy(ev2);
		cudaEventDestroy(ev3);
		cudaEventDestroy(ev4);
		cudaEventDestroy(ev5);
	}
}
