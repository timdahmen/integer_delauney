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

using MergeKey = IncrementalDelaunay::MergeKey;

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

struct MergeLess {
	__host__ __device__ bool operator()(const MergeKey& x, const MergeKey& y) const {
		if (x.a != y.a) return x.a < y.a;
		if (x.b != y.b) return x.b < y.b;
		if (x.c != y.c) return x.c < y.c;
		return x.is_new > y.is_new; // new tri first, if same a,b,c
	}
};

struct MergeEqual {
	__host__ __device__ bool operator()(const MergeKey& x, const MergeKey& y) const {
		return x.a == y.a && x.b == y.b && x.c == y.c;
	}
};

struct IsNew {
	__host__ __device__ bool operator()(const MergeKey& k) const { return k.is_new != 0; }
};

struct IsDirty {
	const int32_t* dirty_mask;
	int W;
	bool use;
	__host__ __device__ bool operator()(const thrust::tuple<Vec2i, Vec3i>& t) const {
		const Vec2i p = thrust::get<0>(t);
		const bool inside = dirty_mask[p.y * W + p.x] != 0;
		return use ? inside : !inside;
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

__global__ void write_seeds_kernel(int32_t* __restrict__ grid,
								   int32_t* __restrict__ changed_mask,
								   const int W,
								   const Vec2i* __restrict__ new_pos,
								   int32_t base_id,
								   const int k) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= k) return;
	const auto [x, y] = new_pos[i];
	const int cell_idx = y * W + x;
	grid[cell_idx * 2] = base_id + i;
	grid[cell_idx * 2 + 1] = 0;
	// Mark the seed cell as changed so the dirty mask expansion
	// covers detection positions that touch the seed cell directly.
	changed_mask[cell_idx] = 1;
}

// ---------------------------------------------------------------------------
// Kernel: BFS step with per-cell change accumulation
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const int32_t* __restrict__ src_raw,
									int32_t* __restrict__ dst_raw,
									const int W,
									const int H,
									uint8_t* __restrict__ flag_write,
									uint8_t* __restrict__ flag_reset_for_next,
									int32_t* __restrict__ changed_mask) {
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
		// No atomicOr needed here: (y*W+x) is unique per thread, address never written multiple threads,
		changed_mask[y * W + x] = 1;
	}
}

// ---------------------------------------------------------------------------
// Kernel: dirty mask = changed mask dilated by 2 in each direction
// ---------------------------------------------------------------------------

__global__ void
dilate_dirty_mask_kernel(const int32_t* __restrict__ changed, int32_t* __restrict__ dirty_mask, const int W, const int H) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	const int x0 = max(0, x - 2), x1 = min(W - 1, x + 2);
	const int y0 = max(0, y - 2), y1 = min(H - 1, y + 2);

	int32_t hit = 0;
	for (int sy = y0; sy <= y1 && !hit; ++sy)
		for (int sx = x0; sx <= x1; ++sx)
			if (changed[sy * W + sx]) {
				hit = 1;
				break;
			}

	dirty_mask[y * W + x] = hit; // only writes own pixel, this avoids need for atomic (can write 0 or 1)
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
										   const int cap,
										   const int32_t* __restrict__ mask) // nullptr -> all pixels
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	if (mask && !mask[y * W + x]) return;

	auto idx = [&](const int cx, const int cy) -> int32_t { return voronoi_grid[(cy * W + cx) * 2]; };

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
// Kernel: build MergeKeys (sorted triplet + tag) from orig array
// ---------------------------------------------------------------------------

__global__ void
build_merge_keys_kernel(MergeKey* __restrict__ keys, const Vec3i* __restrict__ orig, int n, const bool is_new) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	const auto [a, b, c] = orig[i];
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
	keys[i] = {sa, sb, sc, is_new};
}

// ---------------------------------------------------------------------------
// Kernel: triangle-centric rasterisation (1 block per triangle)
// ---------------------------------------------------------------------------

__global__ void rasterize_tri_kernel2(int32_t* __restrict__ canvas,
									 const int W,
									 const Vec2i* __restrict__ seed_pos,
									 const Vec3i* __restrict__ tri_seed_ids,
									 const int n_tris) {
	const int warp_id = threadIdx.x / 32;	// 32 = warp size
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
	int i = blockIdx.x * blockDim.x + threadIdx.x;
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
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	const auto [x, y] = tri_xy[i];
	const auto [a, b, c] = tri_og_seeds[i];
	out[i] = {x, y, a, b, c};
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

IncrementalDelaunay::IncrementalDelaunay(int width, int height, int max_seeds) :
	W_(width), H_(height), N_(0), max_seeds_(max_seeds), reg_cur_(0), N_tri_(0) {
	if (width <= 0 || height <= 0 || max_seeds <= 0)
		throw std::invalid_argument("dimensions and max_seeds must be positive");

	const size_t N = W_ * H_;

	reg_cap_ = 3 * max_seeds_; // theoretical max = 2*V - 5 faces, but triangle detection method has duplicates -> 3x
	det_cap_ = 4 * N;
	mrg_cap_ = 2 * reg_cap_;

	cudaMalloc(&d_grid_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_tmp_, N * 2 * sizeof(int32_t));
	cudaMalloc(&d_changed_, N * sizeof(int32_t));
	cudaMalloc(&d_dirty_, N * sizeof(int32_t));
	cudaMalloc(&d_updated_flag_, 2 * sizeof(uint8_t));
	cudaMalloc(&d_seed_pos_, max_seeds_ * sizeof(Vec2i));
	cudaMalloc(&d_t_grid_, N * sizeof(int32_t));

	cudaMalloc(&d_reg_xy_[0], reg_cap_ * sizeof(Vec2i));
	cudaMalloc(&d_reg_orig_[0], reg_cap_ * sizeof(Vec3i));
	cudaMalloc(&d_reg_xy_[1], reg_cap_ * sizeof(Vec2i));
	cudaMalloc(&d_reg_orig_[1], reg_cap_ * sizeof(Vec3i));

	cudaMalloc(&d_det_xy_, det_cap_ * sizeof(Vec2i));
	cudaMalloc(&d_det_key_, det_cap_ * sizeof(Vec3i));
	cudaMalloc(&d_det_orig_, det_cap_ * sizeof(Vec3i));
	cudaMalloc(&d_counter_, sizeof(int32_t));

	cudaMalloc(&d_mrg_key_, mrg_cap_ * sizeof(MergeKey));
	cudaMalloc(&d_mrg_xy_, mrg_cap_ * sizeof(Vec2i));
	cudaMalloc(&d_mrg_orig_, mrg_cap_ * sizeof(Vec3i));

	cudaMalloc(&d_out_map_, reg_cap_ * sizeof(TriangleEntry));
	cudaMalloc(&d_out_grid_, N * 3 * sizeof(int32_t));

	cudaMemset(d_grid_, UNDEF, N * 2 * sizeof(int32_t));
	cudaMemset(d_t_grid_, UNDEF, N * sizeof(int32_t));
	cudaMemset(d_changed_, 0, N * sizeof(int32_t));
	cudaMemset(d_dirty_, 0, N * sizeof(int32_t));
	cudaMemset(d_updated_flag_, 0, 2 * sizeof(uint8_t));

	h_seed_pos_.reserve(max_seeds_);
	h_seed_set_.reserve(max_seeds_ * 2);
}


IncrementalDelaunay::~IncrementalDelaunay() {
	cudaFree(d_grid_);
	cudaFree(d_tmp_);
	cudaFree(d_changed_);
	cudaFree(d_dirty_);
	cudaFree(d_updated_flag_);
	cudaFree(d_seed_pos_);
	cudaFree(d_t_grid_);
	cudaFree(d_reg_xy_[0]);
	cudaFree(d_reg_orig_[0]);
	cudaFree(d_reg_xy_[1]);
	cudaFree(d_reg_orig_[1]);
	cudaFree(d_det_xy_);
	cudaFree(d_det_key_);
	cudaFree(d_det_orig_);
	cudaFree(d_counter_);
	cudaFree(d_mrg_key_);
	cudaFree(d_mrg_xy_);
	cudaFree(d_mrg_orig_);
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

			voronoi_step_kernel<<<grid_dim, block>>>(d_grid_, d_tmp_, W_, H_, flag_write, flag_reset, d_changed_);
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

void IncrementalDelaunay::triangulate_(const bool is_first, float* det_ms, float* dedup_ms, float* asgn_ms) {
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
													det_cap_, is_first ? nullptr : d_dirty_);
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
		auto det_end = thrust::unique_by_key(det_key, det_key + raw_count, det_vals, KeyEqual{});
		n_new = static_cast<int>(det_end.first - det_key);
	}

	// split the active triangle registry for tris in dirty mask or not

	Vec2i* src_xy = d_reg_xy_[reg_cur_];
	Vec3i* src_orig = d_reg_orig_[reg_cur_];
	Vec2i* dst_xy = d_reg_xy_[1 - reg_cur_];
	Vec3i* dst_orig = d_reg_orig_[1 - reg_cur_];

	thrust::device_ptr<Vec2i> reg_xy(src_xy);
	thrust::device_ptr<Vec3i> reg_orig(src_orig);
	auto reg_begin = thrust::make_zip_iterator(thrust::make_tuple(reg_xy, reg_orig));

	thrust::device_ptr<Vec2i> out_xy(dst_xy);
	thrust::device_ptr<Vec3i> out_orig(dst_orig);
	auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(out_xy, out_orig));

	thrust::device_ptr<Vec2i> mrg_xy(d_mrg_xy_);
	thrust::device_ptr<Vec3i> mrg_orig(d_mrg_orig_);
	auto mrg_begin = thrust::make_zip_iterator(thrust::make_tuple(mrg_xy, mrg_orig));

	int n_kept = 0, n_masked = 0;
	if (N_tri_ > 0) {
		// copy every triangle whose corner is outside the dirty mask into dst
		auto kept_end = thrust::copy_if(reg_begin, reg_begin + N_tri_, out_begin, IsDirty{d_dirty_, W_, false});
		n_kept = static_cast<int>(kept_end - out_begin);

		// copy every triangle whose corner is inside the dirty mask into merge buffer
		auto masked_end =
			thrust::copy_if(reg_begin, reg_begin + N_tri_, mrg_begin, IsDirty{d_dirty_, W_, true});
		n_masked = static_cast<int>(masked_end - mrg_begin);
	}

	if (n_masked + n_new > mrg_cap_) throw std::runtime_error("triangle merge overflowed the temp buffer");

	if (n_new > 0) {
		// append new Tris, after the in dirty mask survivors
		cudaMemcpy(d_mrg_xy_ + n_masked, d_det_xy_, n_new * sizeof(Vec2i), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_mrg_orig_ + n_masked, d_det_orig_, n_new * sizeof(Vec3i), cudaMemcpyDeviceToDevice);
	}

	const int n_merge = n_masked + n_new;
	int n_survivors = 0;

	if (n_merge > 0) {
		// build keys to sort on (add is new tag)
		MergeKey* mrg_key = d_mrg_key_;
		constexpr int n_threads = 256;
		// TODO mby pass is_new as separate array?
		if (n_masked > 0) {
			const int n_blocks = (n_masked + n_threads - 1) / n_threads;
			build_merge_keys_kernel<<<n_blocks, n_threads>>>(mrg_key, d_mrg_orig_, n_masked, false);
		}
		if (n_new > 0) {
			const int n_blocks = (n_new + n_threads - 1) / n_threads;
			build_merge_keys_kernel<<<n_blocks, n_threads>>>(mrg_key + n_masked, d_mrg_orig_ + n_masked, n_new, true);
		}

		thrust::device_ptr<MergeKey> key_ptr(mrg_key);
		thrust::sort_by_key(thrust::cuda::par_nosync, key_ptr, key_ptr + n_merge, mrg_begin, MergeLess{});
		auto uniq_end = thrust::unique_by_key(key_ptr, key_ptr + n_merge, mrg_begin, MergeEqual{});
		const int n_groups = uniq_end.first - key_ptr;

		// append valid tris
		auto surv_end =
			thrust::copy_if(mrg_begin, mrg_begin + n_groups, key_ptr, out_begin + n_kept, IsNew{});
		n_survivors = surv_end - (out_begin + n_kept);
	}

	const int n_total = n_kept + n_survivors;
	if (n_total > reg_cap_)
		throw std::runtime_error("triangle count exceeded registry capacity (2 * max_seeds); "
								 "raise max_seeds");

	reg_cur_ = 1 - reg_cur_; // swap active tri registry
	N_tri_ = n_total;
	if (det_ms) rc(e3);

	// rasterize

	if (det_ms) rc(e4);
	// set sentinal value which is lower than all seed ids and will be replaced in build output
	cudaMemset(d_t_grid_, -1, N * sizeof(int32_t));
	if (N_tri_ > 0) {
		static constexpr int threads_per_block = RASTER_TRIS_PER_BLOCK * 32;
		static_assert(threads_per_block <= 1024, "threads per block exceeds max");
		const int n_blocks = (N_tri_ + RASTER_TRIS_PER_BLOCK - 1) / RASTER_TRIS_PER_BLOCK;
		rasterize_tri_kernel2<<<n_blocks, threads_per_block>>>(d_t_grid_, W_, d_seed_pos_, d_reg_orig_[reg_cur_], N_tri_);
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
	constexpr int threads_per_block = 1024;

	tri_map_out.resize(N_tri_);
	if (N_tri_ > 0) {
		const int n_blocks = (N_tri_ + threads_per_block - 1) / threads_per_block;
		build_triangle_map_out2<<<n_blocks, threads_per_block>>>(d_out_map_, d_reg_xy_[reg_cur_], d_reg_orig_[reg_cur_],
																 N_tri_);
	}

	const int32_t default_id = N_tri_ - 1;
	const int n_blocks = (N + threads_per_block - 1) / threads_per_block;
	build_output_kernel2<<<n_blocks, threads_per_block>>>(d_out_grid_, d_grid_, d_t_grid_, N, default_id);

	tgrid_out.resize(N * 3);

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

	const bool is_first = (N_ == 0);
	const int base_id = N_;

	for (int i = 0; i < k; ++i) {
		h_seed_pos_.push_back(batch[i]);
		h_seed_set_.insert(pack_xy_(batch[i].x, batch[i].y));
	}
	N_ += k;

	cudaMemcpy(d_seed_pos_ + base_id, batch.data(), k * sizeof(Vec2i), cudaMemcpyHostToDevice);

	// Reset change accumulator before writing seeds (so seed positions
	// are the first entries in d_changed_ — necessary for partial_triangulate_
	// to cover detection positions that touch the seed cell directly).
	const size_t N = W_ * H_;
	cudaMemset(d_changed_, 0, N * sizeof(int32_t));

	// Write seeds into grid (also marks seed cells in d_changed_)
	write_seeds_kernel<<<(k + 255) / 256, 256>>>(d_grid_, d_changed_, W_, d_seed_pos_ + base_id, base_id, k);

	// BFS
	float bfs_ms = 0.f;
	run_bfs_(timings ? &bfs_ms : nullptr);

	if (!is_first) {
		cudaEvent_t d0 = nullptr, d1 = nullptr;
		if (timings) {
			cudaEventCreate(&d0);
			cudaEventCreate(&d1);
			cudaEventRecord(d0);
		}

		dim3 block(16, 16);
		dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);
		// only need mask dilation for non first insert
		dilate_dirty_mask_kernel<<<grid_dim, block>>>(d_changed_, d_dirty_, W_, H_);

		if (timings) {
			cudaEventRecord(d1);
			cudaEventSynchronize(d1);
			float ms = 0.f;
			cudaEventElapsedTime(&ms, d0, d1);
			bfs_ms += ms; // dilation is charged to the BFS phase
			cudaEventDestroy(d0);
			cudaEventDestroy(d1);
		}
	}
	if (timings) timings->bfs_ms = bfs_ms;

	// Triangulate
	float det_ms = 0.f, dup_ms = 0.f, asgn_ms = 0.f;
	triangulate_(is_first, timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr, timings ? &asgn_ms : nullptr);

	if (timings) {
		timings->detect_ms = det_ms;
		timings->dedup_ms = dup_ms;
		timings->assign_ms = asgn_ms;
	}

	build_outputs_(tri_map_out, tgrid_out);
}
