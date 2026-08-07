// CUDA implementation of IncrementalDelaunay.
//
// insert(batch) pipeline:
//   1. Write new seeds into d_grid_ at distance 0.
//   2. BFS until convergence;
//	 3. All inserts do a full triangulation. This is cheaper, than keeping a mask and only updating new Tris

#include "incremental.cuh"
#include "shared_kernels.cuh"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Kernel: write new seeds into the interleaved grid
// ---------------------------------------------------------------------------

__global__ void write_seeds_kernel(int32_t* __restrict__ grid,
								   uint32_t* __restrict__ row_mask,
								   const int W,
								   const int H,
								   const int tiles_x,
								   const Vec2i* __restrict__ new_pos,
								   int32_t base_id,
								   const int n_seeds) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_seeds) return;
	const auto [x, y] = new_pos[i];
	const int cell_idx = y * W + x;
	grid[cell_idx * 2] = base_id + i;
	grid[cell_idx * 2 + 1] = 0;

	const int tile_col = x / TILE_SIZE_BFS;
	const int lane = x % TILE_SIZE_BFS;
	atomicOr(&row_mask[y * tiles_x + tile_col], 1u << lane); // write updated bit
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

IncrementalDelaunay::IncrementalDelaunay(int width, int height, int max_seeds) :
	W_(width), H_(height), N_(0), max_seeds_(max_seeds), N_tri_(0) {
	if (width <= 0 || height <= 0 || max_seeds <= 0)
		throw std::invalid_argument("dimensions and max_seeds must be positive");

	const size_t N = W_ * H_;

	// theoretical max = 2*V - 5 faces, but this detection method produces
	// more than a strict planar triangulation -> generous 3x margin
	tri_cap_ = 3 * max_seeds_;
	det_cap_ = 4 * N;
	row_mask_tiles_x_ = (W_ + TILE_SIZE_BFS - 1) / TILE_SIZE_BFS;

	cudaMalloc(&d_grid_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_tmp_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_updated_flag_, 2 * sizeof(int32_t));
	cudaMalloc(&d_row_mask_a_, H_ * row_mask_tiles_x_ * sizeof(uint32_t));
	cudaMalloc(&d_row_mask_b_, H_ * row_mask_tiles_x_ * sizeof(uint32_t));
	cudaMalloc(&d_seed_pos_, max_seeds_ * sizeof(Vec2i));
	cudaMalloc(&d_t_grid_, N * sizeof(int32_t));

	cudaMalloc(&d_det_xy_, det_cap_ * sizeof(Vec2i));
	cudaMalloc(&d_det_key_, det_cap_ * sizeof(Vec3i));
	cudaMalloc(&d_det_orig_, det_cap_ * sizeof(Vec3i));
	cudaMalloc(&d_counter_, sizeof(int32_t));

	cudaMalloc(&d_out_map_, tri_cap_ * sizeof(TriangleEntry));
	cudaMalloc(&d_out_grid_, N * 3 * sizeof(int32_t));

	cudaMemset(d_grid_, UNDEF, N * 2 * sizeof(int32_t));
	cudaMemset(d_t_grid_, UNDEF, N * sizeof(int32_t));
	cudaMemset(d_updated_flag_, 0, 2 * sizeof(int32_t));

	h_seed_set_.reserve(max_seeds_);
}

IncrementalDelaunay::~IncrementalDelaunay() {
	cudaFree(d_grid_);
	cudaFree(d_tmp_);
	cudaFree(d_updated_flag_);
	cudaFree(d_row_mask_a_);
	cudaFree(d_row_mask_b_);
	cudaFree(d_seed_pos_);
	cudaFree(d_t_grid_);
	cudaFree(d_det_xy_);
	cudaFree(d_det_key_);
	cudaFree(d_det_orig_);
	cudaFree(d_counter_);
	cudaFree(d_out_map_);
	cudaFree(d_out_grid_);
}

// ---------------------------------------------------------------------------
// run_bfs_: write new seeds already in d_grid_, BFS until stable
// ---------------------------------------------------------------------------

void IncrementalDelaunay::run_bfs_(float* bfs_ms_out) {
	dim3 block(TILE_SIZE_BFS, TILE_SIZE_BFS);
	dim3 grid_dim((W_ + TILE_SIZE_BFS - 1) / TILE_SIZE_BFS, (H_ + TILE_SIZE_BFS - 1) / TILE_SIZE_BFS);

	cudaEvent_t ev0 = nullptr, ev1 = nullptr;
	if (bfs_ms_out) {
		cudaEventCreate(&ev0);
		cudaEventCreate(&ev1);
		cudaEventRecord(ev0);
	}

	cudaMemset(d_updated_flag_, 0, 2 * sizeof(int32_t));
	cudaMemset(d_updated_flag_, 0, 2 * sizeof(int32_t));

	int cur_flag = 0;
	while (true) {
		for (int i = 0; i < AGGREGATE_ITERATIONS_BFS; ++i) {
			int32_t* flag_write = d_updated_flag_ + cur_flag;
			int32_t* flag_reset = d_updated_flag_ + (1 - cur_flag);

			voronoi_step_kernel<<<grid_dim, block>>>(d_grid_, d_tmp_, W_, H_, flag_write, flag_reset, d_row_mask_a_,
													 d_row_mask_b_);
			std::swap(d_grid_, d_tmp_);
			std::swap(d_row_mask_a_, d_row_mask_b_);
			cur_flag = 1 - cur_flag;
		}

		int32_t h_flag = 0;
		cudaMemcpy(&h_flag, d_updated_flag_ + (1 - cur_flag), sizeof(int32_t), cudaMemcpyDeviceToHost);
		if (!h_flag) break;
	}

	if (bfs_ms_out) {
		cudaEventRecord(ev1);
		cudaEventSynchronize(ev1);
		cudaEventElapsedTime(bfs_ms_out, ev0, ev1);
		cudaEventDestroy(ev0);
		cudaEventDestroy(ev1);
	}
}

// ---------------------------------------------------------------------------
// full_triangulate_: detect → dedup → assign
// ---------------------------------------------------------------------------

void IncrementalDelaunay::triangulate_(float* det_ms,
									   float* dedup_ms,
									   float* asgn_ms,
									   std::vector<int32_t>& tgrid_out) {
	const int N = W_ * H_;
	dim3 block(16, 16);
	dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);

	auto mk = [](cudaEvent_t* e) { cudaEventCreate(e); };
	auto rc = [](cudaEvent_t e) { cudaEventRecord(e); };
	auto el = [](cudaEvent_t a, cudaEvent_t b) -> float {
		cudaEventSynchronize(b);
		float ms = 0;
		cudaEventElapsedTime(&ms, a, b);
		return ms;
	};

	cudaEvent_t e0, e1, e2, e3, e4, e5;
	if (det_ms) {
		mk(&e0);
		mk(&e1);
		mk(&e2);
		mk(&e3);
		mk(&e4);
		mk(&e5);
	}

	// detect triangle seeds

	if (det_ms) rc(e0);
	cudaMemset(d_counter_, 0, sizeof(int32_t));
	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_grid_, W_, H_, d_det_xy_, d_det_key_, d_det_orig_, d_counter_,
													det_cap_);
	int32_t raw_count = 0;
	cudaMemcpy(&raw_count, d_counter_, sizeof(int32_t), cudaMemcpyDeviceToHost);
	if (raw_count > det_cap_) throw std::runtime_error("triangle detection overflowed the scratch buffer");
	if (det_ms) rc(e1);

	// deduplicate new Tri seeds

	if (det_ms) rc(e2);
	thrust::device_ptr<Vec3i> det_key(d_det_key_);
	thrust::device_ptr<Vec2i> det_xy(d_det_xy_);
	thrust::device_ptr<Vec3i> det_orig(d_det_orig_);
	auto det_vals = thrust::make_zip_iterator(thrust::make_tuple(det_xy, det_orig));

	int n_new = 0;
	if (raw_count > 0) {
		thrust::sort_by_key(thrust::cuda::par_nosync, det_key, det_key + raw_count, det_vals, KeyLess{});
	}

	// resize, while waiting for sort to be done
	// this can affect the reported timings for dedup
	tgrid_out.resize(N * 3);

	if (raw_count > 0) {
		auto det_end = thrust::unique_by_key(det_key, det_key + raw_count, det_vals, KeyEqual{});
		n_new = static_cast<int>(det_end.first - det_key);
	}

	if (n_new > tri_cap_)
		throw std::runtime_error("triangle count exceeded expected capacity (3 * max_seeds); raise max_seeds");

	N_tri_ = n_new;
	if (det_ms) rc(e3);

	// rasterize

	if (det_ms) rc(e4);
	// set sentinal value which is lower than all seed ids and will be replaced in build output
	cudaMemset(d_t_grid_, -1, N * sizeof(int32_t));
	if (N_tri_ > 0) {
		static constexpr int threads_per_block = RASTER_TRIS_PER_BLOCK * 32;
		static_assert(threads_per_block <= 1024, "threads per block exceeds max");
		const int n_blocks = (N_tri_ + RASTER_TRIS_PER_BLOCK - 1) / RASTER_TRIS_PER_BLOCK;
		rasterize_tri_kernel<<<n_blocks, threads_per_block>>>(d_t_grid_, W_, d_seed_pos_, d_det_orig_, N_tri_);
	}
	if (det_ms) rc(e5);

	if (det_ms) {
		*det_ms = el(e0, e1);
		*dedup_ms = el(e2, e3);
		*asgn_ms = el(e4, e5);
		cudaEventDestroy(e0);
		cudaEventDestroy(e1);
		cudaEventDestroy(e2);
		cudaEventDestroy(e3);
		cudaEventDestroy(e4);
		cudaEventDestroy(e5);
	}
}

// ---------------------------------------------------------------------------
// build_outputs_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::build_outputs_(std::vector<TriangleEntry>& tri_map_out,
										 std::vector<int32_t>& tgrid_out) const {
	const int N = W_ * H_;
	constexpr int threads_per_block = 32;

	tri_map_out.resize(N_tri_);
	if (N_tri_ > 0) {
		const int n_blocks = (N_tri_ + threads_per_block - 1) / threads_per_block;
		build_triangle_map_out<<<n_blocks, threads_per_block>>>(d_out_map_, d_det_xy_, d_det_orig_, N_tri_);
	}

	const int32_t default_id = N_tri_ - 1;
	const int n_blocks = (N + threads_per_block - 1) / threads_per_block;
	build_output_kernel<<<n_blocks, threads_per_block>>>(d_out_grid_, d_grid_, d_t_grid_, N, default_id);

	// tgrid_out resized already
	if (N_tri_ > 0) cudaMemcpy(tri_map_out.data(), d_out_map_, N_tri_ * sizeof(TriangleEntry), cudaMemcpyDeviceToHost);
	cudaMemcpy(tgrid_out.data(), d_out_grid_, N * 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// get_voronoi_grid
// ---------------------------------------------------------------------------

void IncrementalDelaunay::get_voronoi_grid(std::vector<int32_t>& out) const {
	const int N = W_ * H_;
	out.resize(N * 2);
	cudaMemcpy(out.data(), d_grid_, N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------

void IncrementalDelaunay::insert(const std::vector<Vec2i>& new_seeds,
								 std::vector<TriangleEntry>& tri_map_out,
								 std::vector<int32_t>& tgrid_out,
								 IncrementalTimings* timings) {
	const int k = new_seeds.size();
	if (k == 0) {
		build_outputs_(tri_map_out, tgrid_out);
		return;
	}

	if (new_seeds.size() != k) throw std::invalid_argument("new_xs and new_ys must have the same length");
	if (N_ + k > max_seeds_) throw std::invalid_argument("insert would exceed max_seeds capacity");

	std::unordered_set<uint64_t> seen;
	seen.reserve(k * 2);
	for (int i = 0; i < k; ++i) {
		const auto [x, y] = new_seeds[i];
		// Validate bounds
		if (x < 0 || x >= W_ || y < 0 || y >= H_) throw std::invalid_argument("seed coordinate out of bounds");
		// Duplicate check against existing seeds
		const uint64_t key = pack_xy_(x, y);
		if (!seen.insert(key).second) throw std::invalid_argument("duplicate seed positions within batch");
		// Duplicate check within batch
		if (h_seed_set_.count(key)) throw std::invalid_argument("seed position already exists");
	}

	const int base_id = N_;

	h_seed_set_.reserve(h_seed_set_.size() + k);
	for (int i = 0; i < k; ++i) h_seed_set_.insert(pack_xy_(new_seeds[i].x, new_seeds[i].y));
	N_ += k;
	cudaMemcpy(d_seed_pos_ + base_id, new_seeds.data(), k * sizeof(Vec2i), cudaMemcpyHostToDevice);

	// reset masks
	const size_t row_mask_bytes = H_ * row_mask_tiles_x_ * sizeof(uint32_t);
	cudaMemset(d_row_mask_a_, 0, row_mask_bytes);
	cudaMemset(d_row_mask_b_, 0, row_mask_bytes);

	// Write seeds into grid
	write_seeds_kernel<<<(k + 255) / 256, 256>>>(d_grid_, d_row_mask_a_, W_, H_, row_mask_tiles_x_,
												 d_seed_pos_ + base_id, base_id, k);

	// BFS
	float bfs_ms = 0.f;
	run_bfs_(timings ? &bfs_ms : nullptr);
	if (timings) timings->bfs_ms = bfs_ms;

	// Triangulate
	float det_ms = 0.f, dup_ms = 0.f, asgn_ms = 0.f;
	// outputs buffer passed so it can be resized, while waiting for the gpu
	triangulate_(timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr, timings ? &asgn_ms : nullptr, tgrid_out);

	if (timings) {
		timings->detect_ms = det_ms;
		timings->dedup_ms = dup_ms;
		timings->assign_ms = asgn_ms;
	}

	build_outputs_(tri_map_out, tgrid_out);
}

void IncrementalDelaunay::insert(const std::vector<int32_t>& new_xs,
								 const std::vector<int32_t>& new_ys,
								 std::vector<TriangleEntry>& tri_map_out,
								 std::vector<int32_t>& tgrid_out,
								 IncrementalTimings* timings) {
	const size_t n_seeds = new_xs.size();
	std::vector<Vec2i> seeds(n_seeds);
	for (int i = 0; i < n_seeds; ++i) seeds[i] = {new_xs[i], new_ys[i]};

	insert(seeds, tri_map_out, tgrid_out, timings);
}

IncrementalDelaunay::InsertResultRef IncrementalDelaunay::insert(const std::vector<Vec2i>& new_seeds,
																 IncrementalTimings* timings) {
	insert(new_seeds, tri_map_buf_, tgrid_buf_);
	return {tri_map_buf_, tgrid_buf_};
}
