// CUDA implementation of IncrementalDelaunay.
// TODO: fix docs
// insert(batch) pipeline:
//   1. Write new seeds into d_grid_ at distance 0.
//   2. BFS until convergence; d_changed_ accumulates every cell that moved.
//   3. First insert → full triangulation.
//      Subsequent inserts → partial triangulation:
//        a. Expand d_changed_ by 2 px → border mask (re-detect here).
//        b. Remove triangles whose canonical pixel is in border.
//        c. Re-detect in border, merge new triplets.
//        d. Remap d_t_grid_ through old→new ID table.
//        e. Expand d_changed_ by WINDOW_CAP → reassign mask.
//        f. Re-assign only pixels in reassign mask.

#include "incremental.cuh"

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
// Internal types (parallel to triangulation.cu and voronoi.cu internals)
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEF = -1;

static constexpr uint32_t TILE_SIZE_BFS = 32; // tune for shared-mem vs occupancy, see voronoi.cu
static constexpr int AGGREGATE_ITERATIONS = 4; // tune empirically, same as voronoi.cu
static constexpr int RASTER_TRIS_PER_BLOCK = 32;
static constexpr int THREADS_PER_TRI = 32;

static_assert(sizeof(TriangleEntry) == 5 * sizeof(int32_t), "Unexpected TriangleEntry layout");

struct alignas(8) Cell {
	int32_t id;
	int32_t distance;
};
static_assert(sizeof(Cell) == 2 * sizeof(int32_t), "Unexpected Cell layout");

// dedup key for the detection pass (triplet only)
struct KeyLess {
	__host__ __device__ bool operator()(const Vec3i& x, const Vec3i& y) const {
		if (x.a != y.a) return x.a < y.a;
		if (x.b != y.b) return x.b < y.b;
		return x.c < y.c;
	}
};

struct KeyEqual {
	__host__ __device__ bool operator()(const Vec3i& x, const Vec3i& y) const {
		return x.a == y.a && x.b == y.b && x.c == y.c;
	}
};

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ Cell load_cell(const int x, const int y, const int W, const int H, const Cell* src) {
	if (x < 0 || x >= W || y < 0 || y >= H) return {UNDEF, UNDEF};
	return src[y * W + x];
}

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

__device__ __forceinline__ bool beats(const int32_t a_id, const int32_t a_d, const int32_t b_id, const int32_t b_d) {
	if (b_d < a_d) return true;
	if (b_d == a_d && b_id > a_id) return true;
	return false;
}

// ---------------------------------------------------------------------------
// Kernel: write new seeds into the interleaved grid
// ---------------------------------------------------------------------------

__global__ void write_seeds_kernel(
	int32_t* __restrict__ grid, const int W, const Vec2i* __restrict__ new_pos, int32_t base_id, const int k) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= k) return;
	const auto [x, y] = new_pos[i];
	const int cell_idx = y * W + x;
	grid[cell_idx * 2] = base_id + i;
	grid[cell_idx * 2 + 1] = 0;
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
// Kernel: triangle-centric rasterisation (1 block per triangle)
// ---------------------------------------------------------------------------

__global__ void rasterize_tri_kernel2(int32_t* __restrict__ canvas,
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
// Kernels: output assembly
// ---------------------------------------------------------------------------
// TODO: same as triangulation, put in shared file
__global__ void build_output_kernel2(int32_t* __restrict__ out,
									 const int32_t* __restrict__ voronoi_grid,
									 const int32_t* __restrict__ canvas,
									 const int N,
									 const int default_id) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const int32_t tri_id = canvas[i];
	out[i * 3] = voronoi_grid[i * 2];
	out[i * 3 + 1] = voronoi_grid[i * 2 + 1];
	out[i * 3 + 2] = (tri_id != -1) ? tri_id : default_id;
}

// TODO: same as triangulation, put in shared file
__global__ void build_triangle_map_out2(TriangleEntry* __restrict__ out,
										const Vec2i* __restrict__ tri_xy,
										const Vec3i* __restrict__ tri_og_seeds,
										const int N) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const auto [x, y] = tri_xy[i];
	const auto [a, b, c] = tri_og_seeds[i];
	out[i] = {x, y, a, b, c};
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

	cudaMalloc(&d_grid_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_tmp_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_updated_flag_, 2 * sizeof(uint8_t));
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
	cudaMemset(d_updated_flag_, 0, 2 * sizeof(uint8_t));

	h_seed_set_.reserve(max_seeds_);
}


IncrementalDelaunay::~IncrementalDelaunay() {
	cudaFree(d_grid_);
	cudaFree(d_tmp_);
	cudaFree(d_updated_flag_);
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

	cudaMemset(d_updated_flag_, 0, 2 * sizeof(uint8_t));

	int cur_flag = 0;
	while (true) {
		for (int i = 0; i < AGGREGATE_ITERATIONS; ++i) {
			uint8_t* flag_write = d_updated_flag_ + cur_flag;
			uint8_t* flag_reset = d_updated_flag_ + (1 - cur_flag);

			voronoi_step_kernel<<<grid_dim, block>>>(d_grid_, d_tmp_, W_, H_, flag_write, flag_reset);
			std::swap(d_grid_, d_tmp_);
			cur_flag = 1 - cur_flag;
		}

		uint8_t h_flag = 0;
		cudaMemcpy(&h_flag, d_updated_flag_ + (1 - cur_flag), sizeof(uint8_t), cudaMemcpyDeviceToHost);
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
		rasterize_tri_kernel2<<<n_blocks, threads_per_block>>>(d_t_grid_, W_, d_seed_pos_, d_det_orig_, N_tri_);
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
		build_triangle_map_out2<<<n_blocks, threads_per_block>>>(d_out_map_, d_det_xy_, d_det_orig_, N_tri_);
	}

	const int32_t default_id = N_tri_ - 1;
	const int n_blocks = (N + threads_per_block - 1) / threads_per_block;
	build_output_kernel2<<<n_blocks, threads_per_block>>>(d_out_grid_, d_grid_, d_t_grid_, N, default_id);

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

void IncrementalDelaunay::insert(const std::vector<int32_t>& new_xs,
								 const std::vector<int32_t>& new_ys,
								 std::vector<TriangleEntry>& tri_map_out,
								 std::vector<int32_t>& tgrid_out,
								 IncrementalTimings* timings) {
	const int k = new_xs.size();
	if (k == 0) {
		build_outputs_(tri_map_out, tgrid_out);
		return;
	}

	if (new_ys.size() != k) throw std::invalid_argument("new_xs and new_ys must have the same length");
	if (N_ + k > max_seeds_) throw std::invalid_argument("insert would exceed max_seeds capacity");

	std::vector<Vec2i> batch(k);
	std::unordered_set<uint64_t> seen;
	seen.reserve(k * 2);
	for (int i = 0; i < k; ++i) {
		const int32_t x = new_xs[i], y = new_ys[i];
		// Validate bounds
		if (x < 0 || x >= W_ || y < 0 || y >= H_) throw std::invalid_argument("seed coordinate out of bounds");
		// Duplicate check against existing seeds
		const uint64_t key = pack_xy_(x, y);
		if (!seen.insert(key).second) throw std::invalid_argument("duplicate seed positions within batch");
		// Duplicate check within batch
		if (h_seed_set_.count(key)) throw std::invalid_argument("seed position already exists");

		batch[i] = {x, y};
	}

	const int base_id = N_;

	for (int i = 0; i < k; ++i) h_seed_set_.insert(pack_xy_(batch[i].x, batch[i].y));
	N_ += k;
	cudaMemcpy(d_seed_pos_ + base_id, batch.data(), k * sizeof(Vec2i), cudaMemcpyHostToDevice);

	// Write seeds into grid
	write_seeds_kernel<<<(k + 255) / 256, 256>>>(d_grid_, W_, d_seed_pos_ + base_id, base_id, k);

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
