// CUDA implementation of IncrementalDelaunay.
//
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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Internal types (parallel to triangulation.cu internals)
// ---------------------------------------------------------------------------

static constexpr int32_t UNDEF_SEED = -1;

struct RawTriangle {
	int32_t x, y;
	int32_t a, b, c; // sorted key
	int32_t orig_a, orig_b, orig_c;
};

struct RawLess {
	__device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
		if (p.a != q.a) return p.a < q.a;
		if (p.b != q.b) return p.b < q.b;
		return p.c < q.c;
	}
};

struct RawEqual {
	__device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
		return p.a == q.a && p.b == q.b && p.c == q.c;
	}
};

// ---------------------------------------------------------------------------
// Internal types (like in voronoi.cu / .cuh)
// ---------------------------------------------------------------------------

struct alignas(8) Cell {
	int32_t id;
	int32_t distance;
};
static_assert(sizeof(Cell) == 2 * sizeof(int32_t), "Unexpected Cell layout");

static constexpr uint32_t TILE_SIZE_BFS = 32; // tune for shared-mem vs occupancy, see voronoi.cu
static constexpr int AGGREGATE_ITERATIONS = 4; // tune empirically, same as voronoi.cu

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ Cell load_cell(int x, int y, int W, int H, const Cell* src) {
	if (x < 0 || x >= W || y < 0 || y >= H) return {UNDEF_SEED, UNDEF_SEED};
	return src[y * W + x];
}

__device__ __forceinline__ bool beats(int32_t a_id, int32_t a_d, int32_t b_id, int32_t b_d) {
	if (b_d < a_d) return true;
	if (b_d == a_d && b_id > a_id) return true;
	return false;
}

__device__ __forceinline__ float cross2d(float ox, float oy, float ax, float ay, float bx, float by) {
	return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__ bool
point_in_triangle(float px, float py, float ax, float ay, float bx, float by, float cx, float cy) {
	float d1 = cross2d(px, py, ax, ay, bx, by);
	float d2 = cross2d(px, py, bx, by, cx, cy);
	float d3 = cross2d(px, py, cx, cy, ax, ay);
	bool neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
	bool pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
	return !(neg && pos);
}

// ---------------------------------------------------------------------------
// Kernel: write new seeds into the interleaved grid
// ---------------------------------------------------------------------------

__global__ void write_seeds_kernel(
	int32_t* grid, int32_t* changed_mask, int W, const int32_t* xs, const int32_t* ys, const int32_t* ids, int k) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= k) return;
	int base = (ys[i] * W + xs[i]) * 2;
	grid[base] = ids[i];
	grid[base + 1] = 0;
	// Mark the seed cell as changed so the border expansion
	// covers detection positions that touch the seed cell directly.
	atomicOr(&changed_mask[ys[i] * W + xs[i]], 1);
}

// ---------------------------------------------------------------------------
// Kernel: BFS step with per-cell change accumulation
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const int32_t* __restrict__ src_raw,
									int32_t* __restrict__ dst_raw,
									int W,
									int H,
									uint8_t* __restrict__ flag_write,
									uint8_t* __restrict__ flag_reset_for_next,
									int32_t* __restrict__ changed_mask) {
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
		*flag_reset_for_next = 0;
	}

	const Cell* src = reinterpret_cast<const Cell*>(src_raw);
	Cell* dst = reinterpret_cast<Cell*>(dst_raw);

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
		if (neighbor.id == UNDEF_SEED) continue;
		int32_t n_d = neighbor.distance + 1;
		if (best.id == UNDEF_SEED || beats(best.id, best.distance, neighbor.id, n_d)) {
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
// Kernel: L-shape triangle detection with optional mask
// Grid is interleaved (seed_id, distance); only channel 0 is read.
// ---------------------------------------------------------------------------

__global__ void find_triangle_seeds_kernel(const int32_t* __restrict__ grid, // interleaved (seed_id, dist)
										   int W,
										   int H,
										   RawTriangle* __restrict__ raw_buf,
										   int32_t* __restrict__ counter,
										   const int32_t* __restrict__ mask) // nullptr → all pixels
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	if (mask && !mask[y * W + x]) return;

	auto sid = [&](int cx, int cy) -> int32_t { return grid[(cy * W + cx) * 2]; };

	auto try_reg = [&](int gx, int gy, int32_t a, int32_t b, int32_t c) {
		if (a == UNDEF_SEED || b == UNDEF_SEED || c == UNDEF_SEED) return;
		if (a == b || a == c || b == c) return;
		int32_t pos = atomicAdd(counter, 1);
		int32_t sa = a, sb = b, sc = c;
		if (sa > sb) {
			int32_t t = sa;
			sa = sb;
			sb = t;
		}
		if (sb > sc) {
			int32_t t = sb;
			sb = sc;
			sc = t;
		}
		if (sa > sb) {
			int32_t t = sa;
			sa = sb;
			sb = t;
		}
		raw_buf[pos] = {gx, gy, sa, sb, sc, a, b, c};
	};

	if (x >= 1 && y <= H - 2) try_reg(x, y, sid(x - 1, y), sid(x, y), sid(x, y + 1));
	if (x <= W - 2 && y <= H - 2) try_reg(x, y, sid(x, y), sid(x + 1, y), sid(x, y + 1));
	if (x <= W - 2 && y >= 1) try_reg(x, y, sid(x, y), sid(x + 1, y), sid(x, y - 1));
	if (x >= 1 && y >= 1) try_reg(x, y, sid(x - 1, y), sid(x, y), sid(x, y - 1));
}

// ---------------------------------------------------------------------------
// Kernel: remap triangle IDs in t_grid (after compaction)
// ---------------------------------------------------------------------------

__global__ void remap_tgrid_kernel(
	int32_t* __restrict__ t_grid, int N, const int32_t* __restrict__ remap, int remap_size, int32_t fallback) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int32_t old_tid = t_grid[i];
	if (old_tid < 0 || old_tid >= remap_size || remap[old_tid] < 0)
		t_grid[i] = fallback;
	else
		t_grid[i] = remap[old_tid];
}

// ---------------------------------------------------------------------------
// Kernel: window-based assignment with optional mask
// ---------------------------------------------------------------------------

static constexpr int WINDOW_SLACK = 3;
static constexpr int WINDOW_CAP = 20;
static constexpr int MAX_NEARBY = 64;

__global__ void assign_triangles_kernel(int32_t* __restrict__ t_grid,
										int W,
										int H,
										const int32_t* __restrict__ grid, // interleaved (seed_id, dist)
										const RawTriangle* __restrict__ triangles,
										const int32_t* __restrict__ seed_xs,
										const int32_t* __restrict__ seed_ys,
										const int32_t* __restrict__ csr_ptr,
										const int32_t* __restrict__ csr_idx,
										int N_seeds,
										int N_triangles,
										const int32_t* __restrict__ mask) // nullptr → all pixels
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;
	if (mask && !mask[y * W + x]) return;

	float px = x + 0.5f, py = y + 0.5f;
	int dist = grid[(y * W + x) * 2 + 1];
	int R = min(dist + WINDOW_SLACK, WINDOW_CAP);

	int32_t nearby[MAX_NEARBY];
	int n_nearby = 0;

	int x0 = max(0, x - R), x1 = min(W - 1, x + R);
	int y0 = max(0, y - R), y1 = min(H - 1, y + R);

	for (int sy = y0; sy <= y1; ++sy)
		for (int sx = x0; sx <= x1; ++sx) {
			int32_t sid = grid[(sy * W + sx) * 2];
			bool dup = false;
			for (int i = 0; i < n_nearby; ++i)
				if (nearby[i] == sid) {
					dup = true;
					break;
				}
			if (!dup && n_nearby < MAX_NEARBY) nearby[n_nearby++] = sid;
		}

	int32_t best = -1;
	for (int i = 0; i < n_nearby; ++i) {
		int32_t sid = nearby[i];
		if (sid < 0 || sid >= N_seeds) continue;
		for (int j = csr_ptr[sid]; j < csr_ptr[sid + 1]; ++j) {
			int32_t tid = csr_idx[j];
			const RawTriangle& tri = triangles[tid];
			float ax = (float)seed_xs[tri.orig_a], ay = (float)seed_ys[tri.orig_a];
			float bx = (float)seed_xs[tri.orig_b], by = (float)seed_ys[tri.orig_b];
			float cx = (float)seed_xs[tri.orig_c], cy = (float)seed_ys[tri.orig_c];
			if (point_in_triangle(px, py, ax, ay, bx, by, cx, cy))
				if (best == -1 || tid > best) best = tid;
		}
	}
	t_grid[y * W + x] = (best != -1) ? best : (N_triangles - 1);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

IncrementalDelaunay::IncrementalDelaunay(int width, int height, int max_seeds) :
	W_(width), H_(height), N_(0), max_seeds_(max_seeds) {
	if (width <= 0 || height <= 0 || max_seeds <= 0)
		throw std::invalid_argument("dimensions and max_seeds must be positive");

	const int N = W_ * H_;
	cudaMalloc(&d_grid_, (size_t)N * 2 * sizeof(int32_t));
	cudaMalloc(&d_tmp_, (size_t)N * 2 * sizeof(int32_t));
	cudaMalloc(&d_changed_, (size_t)N * sizeof(int32_t));
	cudaMalloc(&d_sx_, (size_t)max_seeds * sizeof(int32_t));
	cudaMalloc(&d_sy_, (size_t)max_seeds * sizeof(int32_t));
	cudaMalloc(&d_raw_buf_, (size_t)N * 4 * sizeof(RawTriangle));
	cudaMalloc(&d_t_grid_, (size_t)N * sizeof(int32_t));
	cudaMalloc(&d_csr_ptr_, (size_t)(max_seeds + 1) * sizeof(int32_t));
	cudaMalloc(&d_csr_idx_, (size_t)max_seeds * 8 * sizeof(int32_t));
	cudaMalloc(&d_updated_flag_, 2 * sizeof(uint8_t));
	cudaMalloc(&d_mask_, (size_t)N * sizeof(int32_t));

	cudaMemset(d_grid_, -1, (size_t)N * 2 * sizeof(int32_t)); // UNDEF
	cudaMemset(d_t_grid_, -1, (size_t)N * sizeof(int32_t));
	cudaMemset(d_changed_, 0, (size_t)N * sizeof(int32_t));
	cudaMemset(d_updated_flag_, 0, 2 * sizeof(uint8_t));
}

IncrementalDelaunay::~IncrementalDelaunay() {
	cudaFree(d_grid_);
	cudaFree(d_tmp_);
	cudaFree(d_changed_);
	cudaFree(d_sx_);
	cudaFree(d_sy_);
	cudaFree(d_raw_buf_);
	cudaFree(d_t_grid_);
	cudaFree(d_csr_ptr_);
	cudaFree(d_csr_idx_);
	cudaFree(d_updated_flag_);
	cudaFree(d_mask_);
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
// upload_triangles_: sync h_triangles_ → d_raw_buf_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::upload_triangles_() {
	int N_tri = (int)h_triangles_.size();
	// TODO: Host tri and raw tri are the same type struct
	//	 however, can be even simpler, with new raster tri kernel
	std::vector<RawTriangle> h_raw(N_tri);
	for (int i = 0; i < N_tri; ++i) {
		const auto& h = h_triangles_[i];
		h_raw[i] = {h.x, h.y, h.a, h.b, h.c, h.orig_a, h.orig_b, h.orig_c};
	}
	cudaMemcpy(d_raw_buf_, h_raw.data(), N_tri * sizeof(RawTriangle), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// rebuild_csr_and_upload_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::rebuild_csr_and_upload_() {
	int N_tri = (int)h_triangles_.size();
	std::vector<int32_t> h_csr_ptr(N_ + 1, 0);
	for (const auto& t : h_triangles_) {
		h_csr_ptr[t.orig_a + 1]++;
		h_csr_ptr[t.orig_b + 1]++;
		h_csr_ptr[t.orig_c + 1]++;
	}
	for (int s = 1; s <= N_; ++s) h_csr_ptr[s] += h_csr_ptr[s - 1];

	int csr_size = h_csr_ptr[N_];
	std::vector<int32_t> h_csr_idx(csr_size);
	std::vector<int32_t> fill(N_, 0);
	for (int tid = 0; tid < N_tri; ++tid) {
		for (int32_t s : {h_triangles_[tid].orig_a, h_triangles_[tid].orig_b, h_triangles_[tid].orig_c}) {
			h_csr_idx[h_csr_ptr[s] + fill[s]] = tid;
			fill[s]++;
		}
	}
	cudaMemcpy(d_csr_ptr_, h_csr_ptr.data(), (N_ + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_idx_, h_csr_idx.data(), csr_size * sizeof(int32_t), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// full_triangulate_: detect → dedup → CSR → assign (no mask)
// ---------------------------------------------------------------------------

void IncrementalDelaunay::full_triangulate_(float* det_ms, float* dedup_ms, float* asgn_ms) {
	// TODO replace with optimized triangulation from triangulation.cu

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

	RawTriangle* d_raw = static_cast<RawTriangle*>(d_raw_buf_);
	int32_t* d_counter;
	cudaMalloc(&d_counter, sizeof(int32_t));
	cudaMemset(d_counter, 0, sizeof(int32_t));

	if (det_ms) rc(e0);
	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_grid_, W_, H_, d_raw, d_counter, nullptr);
	cudaDeviceSynchronize();
	int32_t raw_count = 0;
	cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaFree(d_counter);
	if (det_ms) rc(e1);

	thrust::device_ptr<RawTriangle> d_ptr(d_raw);
	if (dedup_ms) rc(e2);
	thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
	auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
	int N_tri = (int)(new_end - d_ptr);
	if (dedup_ms) rc(e3);

	std::vector<RawTriangle> h_dedup(N_tri);
	cudaMemcpy(h_dedup.data(), d_raw, N_tri * sizeof(RawTriangle), cudaMemcpyDeviceToHost);

	h_triangles_.clear();
	h_triplet_to_tid_.clear();
	h_triangles_.reserve(N_tri);
	for (int32_t tid = 0; tid < N_tri; ++tid) {
		const auto& r = h_dedup[tid];
		h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
		h_triplet_to_tid_[pack_triplet_(r.a, r.b, r.c)] = tid;
	}
	// d_raw_buf_ already has the deduplicated triangles in correct order

	rebuild_csr_and_upload_();

	if (asgn_ms) rc(e4);
	if (N_tri > 0) {
		assign_triangles_kernel<<<grid_dim, block>>>(d_t_grid_, W_, H_, d_grid_, d_raw, d_sx_, d_sy_, d_csr_ptr_,
													 d_csr_idx_, N_, N_tri, nullptr);
		cudaDeviceSynchronize();
	} else {
		cudaMemset(d_t_grid_, -1, N * sizeof(int32_t));
	}
	if (asgn_ms) rc(e5);

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
// partial_triangulate_: use d_changed_ to scope detection and assignment
// ---------------------------------------------------------------------------

void IncrementalDelaunay::partial_triangulate_(float* det_ms, float* dedup_ms, float* asgn_ms) {
	const int N = W_ * H_;
	dim3 block(16, 16);
	dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);

	// Download change mask
	std::vector<int32_t> h_changed(N);
	cudaMemcpy(h_changed.data(), d_changed_, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

	// Build border (expand by 2 for L-shapes) and reassign (expand by WINDOW_CAP)
	// TODO: h_reassign not needed, if doing full triangulation each time
	//    Instead of downloading border to Host, expanding it by 2 and reuploading it later, just make insert_seed
	//    kernel write the expanded by2 area directly. That kernel is super cheap anyway and saves a lot of mem copy and CPU mem loading
	std::vector<int32_t> h_border(N, 0), h_reassign(N, 0);
	for (int y = 0; y < H_; ++y) {
		for (int x = 0; x < W_; ++x) {
			if (!h_changed[y * W_ + x]) continue;
			for (int dy = -2; dy <= 2; ++dy)
				for (int dx = -2; dx <= 2; ++dx) {
					int bx = x + dx, by = y + dy;
					if (bx >= 0 && bx < W_ && by >= 0 && by < H_) h_border[by * W_ + bx] = 1;
				}
			for (int dy = -WINDOW_CAP; dy <= WINDOW_CAP; ++dy)
				for (int dx = -WINDOW_CAP; dx <= WINDOW_CAP; ++dx) {
					int bx = x + dx, by = y + dy;
					if (bx >= 0 && bx < W_ && by >= 0 && by < H_) h_reassign[by * W_ + bx] = 1;
				}
		}
	}

	// Mark stale triangles (canonical pixel in border)
	int old_count = (int)h_triangles_.size();
	std::vector<bool> is_stale(old_count, false);
	// TODO: Are there any stale Tris?
	//  The find_tri kernel only finds triangles for which x/y are within W/H already, so only non-stale ones should
	//  exist. It then writes the tested x/y as the tri xy which we check here.
	for (int tid = 0; tid < old_count; ++tid) {
		const auto& t = h_triangles_[tid];
		if (t.x >= 0 && t.x < W_ && t.y >= 0 && t.y < H_)
			if (h_border[t.y * W_ + t.x]) is_stale[tid] = true;
	}

	// Detect new triangles in border
	cudaMemcpy(d_mask_, h_border.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

	RawTriangle* d_raw = static_cast<RawTriangle*>(d_raw_buf_);
	int32_t* d_counter;
	cudaMalloc(&d_counter, sizeof(int32_t));
	cudaMemset(d_counter, 0, sizeof(int32_t));

	if (det_ms) {
		cudaEvent_t e;
		cudaEventCreate(&e);
		cudaEventRecord(e);
	}

	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_grid_, W_, H_, d_raw, d_counter, d_mask_);
	cudaDeviceSynchronize();
	int32_t raw_count = 0;
	cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaFree(d_counter);

	thrust::device_ptr<RawTriangle> d_ptr(d_raw);
	thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
	auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
	int n_new = (int)(new_end - d_ptr);

	std::vector<RawTriangle> h_new(n_new);
	if (n_new > 0) cudaMemcpy(h_new.data(), d_raw, n_new * sizeof(RawTriangle), cudaMemcpyDeviceToHost);

	// Collect truly new triplets: not in registry, OR in registry but stale
	// (stale triangles are re-detected here; they must be re-added since they
	// will be removed in the compaction step below).
	std::vector<HTriangle> to_add;
	for (const auto& r : h_new) {
		auto key = pack_triplet_(r.a, r.b, r.c);
		auto it = h_triplet_to_tid_.find(key);
		// TODO: assuming we can remove is_stale can remove that check here and
		//   also make h_triplet_to_tid_ a set instead of map (assuming tid not needed later)
		if (it == h_triplet_to_tid_.end() || is_stale[it->second])
			to_add.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
	}

	// Compact: keep non-stale, append new; build remap old→new
	std::vector<int32_t> remap(old_count, -1);
	std::vector<HTriangle> compacted;
	// XXX Compated.size = n_old_tri - n_stale_old_Tris
	compacted.reserve(old_count - (int)std::count(is_stale.begin(), is_stale.end(), true) + (int)to_add.size());
	for (int tid = 0; tid < old_count; ++tid) {
		if (!is_stale[tid]) {
			// XXX push tid of non stale Tris and in remap (tid -> idx Tri in compacted)
			remap[tid] = (int32_t)compacted.size();
			compacted.push_back(h_triangles_[tid]);
		}
	}
	for (const auto& t : to_add) compacted.push_back(t); // XXX add new Tris to compacted

	h_triangles_ = std::move(compacted);

	// Rebuild lookup maps
	h_triplet_to_tid_.clear();
	for (int tid = 0; tid < (int)h_triangles_.size(); ++tid) {
		const auto& t = h_triangles_[tid];
		h_triplet_to_tid_[pack_triplet_(t.a, t.b, t.c)] = tid;
	}

	// Upload triangle array and CSR
	upload_triangles_();
	rebuild_csr_and_upload_(); // TODO: not needed with new raster tir kernel

	// Remap d_t_grid_ through old→new table
	// TODO: does this just remap the old Tri IDs to the new ones. Should be faster, to just re rasterize all Tris.
	//   Removes this kernel entirely and the CPU side bookkeeping
	int32_t fallback = h_triangles_.empty() ? 0 : (int32_t)h_triangles_.size() - 1;
	if (old_count > 0) {
		int32_t* d_remap;
		cudaMalloc(&d_remap, old_count * sizeof(int32_t));
		cudaMemcpy(d_remap, remap.data(), old_count * sizeof(int32_t), cudaMemcpyHostToDevice);
		remap_tgrid_kernel<<<(N + 255) / 256, 256>>>(d_t_grid_, N, d_remap, old_count, fallback);
		cudaDeviceSynchronize();
		cudaFree(d_remap);
	}

	// Re-assign pixels in expanded reassign region
	cudaMemcpy(d_mask_, h_reassign.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);

	int N_tri = (int)h_triangles_.size();
	if (N_tri > 0) {
		assign_triangles_kernel<<<grid_dim, block>>>(d_t_grid_, W_, H_, d_grid_, static_cast<RawTriangle*>(d_raw_buf_),
													 d_sx_, d_sy_, d_csr_ptr_, d_csr_idx_, N_, N_tri, d_mask_);
		cudaDeviceSynchronize();
	}

	// TODO: SUMMARY:
	//   Always doing the full triangulation, makes this almost the same as full_triangulate_().
	//   Saves a lot of Host side bookkeeping and temp allocations
	//   Only difference to full_triangulate(): for the find_triangle_seeds, doing it with a mask is faster.
	//   However can use a coarser mask (1 byte/bit per block), which can then also be used to reduce number of loads in
	//   voronoi kernel (stale tile detection)
}

// ---------------------------------------------------------------------------
// build_outputs_
// ---------------------------------------------------------------------------

void IncrementalDelaunay::build_outputs_(std::vector<TriangleEntry>& tri_map_out,
										 std::vector<int32_t>& tgrid_out) const {
	// TODO: replace with kernels like in triangulation.cu
	const int N = W_ * H_;
	int N_tri = (int)h_triangles_.size();

	tri_map_out.resize(N_tri);
	for (int tid = 0; tid < N_tri; ++tid) {
		const auto& t = h_triangles_[tid];
		tri_map_out[tid] = {t.x, t.y, t.orig_a, t.orig_b, t.orig_c};
	}

	std::vector<int32_t> h_t(N), h_grid(N * 2);
	cudaMemcpy(h_t.data(), d_t_grid_, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grid.data(), d_grid_, N * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);

	tgrid_out.resize(N * 3);
	for (int i = 0; i < N; ++i) {
		tgrid_out[i * 3] = h_grid[i * 2];
		tgrid_out[i * 3 + 1] = h_grid[i * 2 + 1];
		tgrid_out[i * 3 + 2] = h_t[i];
	}
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
	int k = (int)new_xs.size();
	if (k == 0) {
		build_outputs_(tri_map_out, tgrid_out);
		return;
	}

	// TODO: use Vec2i for x/y together, like in triangulation.cu

	if (N_ + k > max_seeds_) throw std::invalid_argument("insert would exceed max_seeds capacity");

	// Validate bounds
	for (int i = 0; i < k; ++i)
		if (new_xs[i] < 0 || new_xs[i] >= W_ || new_ys[i] < 0 || new_ys[i] >= H_)
			throw std::invalid_argument("seed coordinate out of bounds");

	// Duplicate check within batch
	// TODO: for large number of inserted seeds: use a set to track duplicates so not O(N^2). For small number, linear
	// search should be better (cache)
	for (int i = 0; i < k; ++i)
		for (int j = i + 1; j < k; ++j)
			if (new_xs[i] == new_xs[j] && new_ys[i] == new_ys[j])
				throw std::invalid_argument("duplicate seed positions within batch");

	// Duplicate check against existing seeds
	// TODO: interleave with Validate bounds loop, for fewer loads (x/y already in cache)
	for (int i = 0; i < k; ++i)
		if (h_seed_set_.count(pack_xy_(new_xs[i], new_ys[i])))
			throw std::invalid_argument("seed position already exists");

	bool is_first = (N_ == 0);

	// Register seeds
	std::vector<int32_t> new_ids(k);
	for (int i = 0; i < k; ++i) {
		new_ids[i] = N_ + i;
		h_sx_.push_back(new_xs[i]);
		h_sy_.push_back(new_ys[i]);
		h_seed_set_.insert(pack_xy_(new_xs[i], new_ys[i]));
	}
	N_ += k;

	// Upload new seed positions to persistent device arrays
	cudaMemcpy(d_sx_ + N_ - k, new_xs.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sy_ + N_ - k, new_ys.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);

	// Reset change accumulator before writing seeds (so seed positions
	// are the first entries in d_changed_ — necessary for partial_triangulate_
	// to cover detection positions that touch the seed cell directly).
	cudaMemset(d_changed_, 0, (size_t)W_ * H_ * sizeof(int32_t));

	// Write seeds into grid (also marks seed cells in d_changed_)
	int32_t *d_kxs, *d_kys, *d_kids;
	cudaMalloc(&d_kxs, k * sizeof(int32_t));
	cudaMalloc(&d_kys, k * sizeof(int32_t));
	cudaMalloc(&d_kids, k * sizeof(int32_t));
	cudaMemcpy(d_kxs, new_xs.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kys, new_ys.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kids, new_ids.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
	write_seeds_kernel<<<(k + 255) / 256, 256>>>(d_grid_, d_changed_, W_, d_kxs, d_kys, d_kids, k);
	cudaDeviceSynchronize();
	cudaFree(d_kxs);
	cudaFree(d_kys);
	cudaFree(d_kids);

	// BFS
	float bfs_ms = 0.f;
	run_bfs_(timings ? &bfs_ms : nullptr);
	if (timings) timings->bfs_ms = bfs_ms;

	// Triangulate
	float det_ms = 0.f, dup_ms = 0.f, asgn_ms = 0.f;
	if (is_first)
		full_triangulate_(timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr, timings ? &asgn_ms : nullptr);
	else
		partial_triangulate_(timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr, timings ? &asgn_ms : nullptr);
	if (timings) {
		timings->detect_ms = det_ms;
		timings->dedup_ms = dup_ms;
		timings->assign_ms = asgn_ms;
	}

	build_outputs_(tri_map_out, tgrid_out);
}
