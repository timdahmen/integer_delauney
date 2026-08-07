// CUDA kernels for GridTriangulation.
// Step 1 (CPU): prepare and upload data to the GPU
// Step 2 (GPU): scan all 4 L-shape orientations; raw hits → device buffer.
// Step 3 (GPU): thrust::sort_by_key + thrust::unique_by_key on the device buffer.
// Step 4 (GPU): triangle rasterisation. 1 block works on multiple triangles at once
//				 and multiple threads per triangle, so that consecutive threads access neighbouring memory.
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

#include "common.h"
#include "triangulation.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "shared_kernels.cuh"

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
// Host entry points
// ---------------------------------------------------------------------------

void cuda_compute_triangulation(const int W,
								const int H,
								const int32_t* voronoi_grid,
								const std::vector<Vec2i>& h_seeds,
								std::vector<TriangleEntry>& triangle_map_out,
								std::vector<int32_t>& out_grid,
								TriTimings* timings) {
	const int N = W * H;
	const int N_seeds = static_cast<int>(h_seeds.size());

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

	Vec2i* d_seed_pos = nullptr;
	cudaMalloc(&d_seed_pos, N_seeds * sizeof(Vec2i));
	CudaFreeGuard g_seed_pos(d_seed_pos);
	cudaMemcpy(d_seed_pos, h_seeds.data(), N_seeds * sizeof(Vec2i), cudaMemcpyHostToDevice);
	int32_t* d_voronoi_grid = nullptr;
	cudaMalloc(&d_voronoi_grid, N * 2 * sizeof(int32_t));
	CudaFreeGuard g_voronoi_grid(d_voronoi_grid);
	cudaMemcpy(d_voronoi_grid, voronoi_grid, N * 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

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

	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_voronoi_grid, W, H, d_tri_xy, d_tri_key, d_tri_orig, d_counter,
													max_raw);
	{
		cudaError_t launch_err = cudaGetLastError();
		if (launch_err != cudaSuccess) {
			throw std::runtime_error(std::string("find_triangle_seeds_kernel launch failed: ") +
									 cudaGetErrorString(launch_err));
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

	thrust::sort_by_key(thrust::cuda::par_nosync, d_key_ptr, d_key_ptr + raw_count, values_begin, KeyLess{});

	// resize here, while GPU sorts triangles (thrust::cuda::par_nosync -> async)
	out_grid.resize(N * 3);

	auto unique_result = thrust::unique_by_key(d_key_ptr, d_key_ptr + raw_count, values_begin, KeyEqual{});
	const int N_triangles = static_cast<int>(unique_result.first - d_key_ptr);

	if (timings) record(ev3);

	// Copy deduplicated x,y and orig_a/b/c to host (Python dict + CSR build)
	{
		TriangleEntry* d_out_map = nullptr;
		cudaMalloc(&d_out_map, N_triangles * sizeof(TriangleEntry));
		CudaFreeGuard g_out_map(d_out_map);

		static constexpr int num_threads_out_map = 1024;
		const int num_blocks = (N_triangles + num_threads_out_map - 1) / num_threads_out_map;
		build_triangle_map_out<<<num_blocks, num_threads_out_map>>>(d_out_map, d_tri_xy, d_tri_orig, N_triangles);

		triangle_map_out.clear();
		triangle_map_out.resize(N_triangles);
		cudaMemcpy(triangle_map_out.data(), d_out_map, N_triangles * sizeof(TriangleEntry), cudaMemcpyDeviceToHost);
	}

	// -----------------------------------------------------------------------
	// Step 4: assign pixels to triangles
	// -----------------------------------------------------------------------

	int32_t* d_canvas = nullptr;
	cudaMalloc(&d_canvas, N * sizeof(int32_t));
	CudaFreeGuard g_canvas(d_canvas);
	cudaMemset(d_canvas, -1, N * sizeof(int32_t)); // initialize Canvas with sentinal value

	if (timings) record(ev4);

	if (N_triangles > 0) {
		static constexpr int threads_per_block = RASTER_TRIS_PER_BLOCK * 32;
		static_assert(threads_per_block <= 1024, "threads per block exceeds max");
		const int n_blocks = (N_triangles + RASTER_TRIS_PER_BLOCK - 1) / RASTER_TRIS_PER_BLOCK;
		rasterize_tri_kernel<<<n_blocks, threads_per_block>>>(d_canvas, W, d_seed_pos, d_tri_orig, N_triangles);
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


	int32_t* d_out = nullptr;
	cudaMalloc(&d_out, N * 3 * sizeof(int32_t));
	CudaFreeGuard g_out(d_out);
	const int32_t default_id = N_triangles - 1;

	static constexpr int num_threads_output = 1024;
	const int num_blocks = (N + num_threads_output - 1) / num_threads_output;
	build_output_kernel<<<num_blocks, num_threads_output>>>(d_out, d_voronoi_grid, d_canvas, N, default_id);

	cudaMemcpy(out_grid.data(), d_out, N * 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);

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

void cuda_compute_triangulation(const int W,
								const int H,
								const int32_t* voronoi_grid,
								const std::vector<int32_t>& seed_xs,
								const std::vector<int32_t>& seed_ys,
								std::vector<TriangleEntry>& triangle_map_out,
								std::vector<int32_t>& out_grid,
								TriTimings* timings) {
	const size_t n_seeds = seed_xs.size();
	std::vector<Vec2i> seeds(n_seeds);
	for (int i = 0; i < n_seeds; ++i) seeds[i] = {seed_xs[i], seed_ys[i]};

	cuda_compute_triangulation(W, H, voronoi_grid, seeds, triangle_map_out, out_grid, timings);
}
