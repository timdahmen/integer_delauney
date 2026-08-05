// CUDA kernels for GridTriangulation.
//
// Step 2 (GPU): scan all 4 L-shape orientations; raw hits → device buffer.
// Step 3 (GPU): thrust::sort + thrust::unique on the device buffer.
// Step 4 (GPU): window-based nearest-neighbor assign.
//   For each pixel, read its Voronoi distance d, scan a (2R+1)² window of
//   the seed-id grid (R = min(d + WINDOW_SLACK, WINDOW_CAP)), collect unique
//   nearby seed IDs, then test only the triangles adjacent to those seeds via
//   a per-seed CSR list.  Reduces per-pixel tests from O(T) to O(k·t) where
//   k ≈ nearby seeds (~10-70) and t ≈ triangles/seed (~7).
//
// Host copies:
//   • voronoi seed_id + distance channels  → device  (input, once)
//   • seed positions                        → device  (input, once)
//   • deduplicated triangles               ← device  (output, ~7 MB)
//   • CSR adjacency list                   → device  (built host-side, ~3 MB)
//   • triangle_id grid                     ← device  (output, W*H ints)

#include "triangulation.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
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
struct RawXY {
	int32_t x, y;
};
struct RawKey {
	int32_t a, b, c;
};
struct RawOrig {
	int32_t orig_a, orig_b, orig_c;
};

struct RawKeyLess {
	__device__ bool operator()(const RawKey& x, const RawKey& y) const {
		if (x.a != y.a) return x.a < y.a;
		if (x.b != y.b) return x.b < y.b;
		return x.c < y.c;
	}
};

struct RawKeyEqual {
	__device__ bool operator()(const RawKey& x, const RawKey& y) const {
		return x.a == y.a && x.b == y.b && x.c == y.c;
	}
};

// ---------------------------------------------------------------------------
// Kernel 1: detect triangle seeds (all 4 L-shape orientations)
// ---------------------------------------------------------------------------

__global__ void find_triangle_seeds_kernel(const int32_t* __restrict__ n_grid,
										   const int W,
										   const int H,
										   RawXY* __restrict__ raw_xy,
										   RawKey* __restrict__ raw_key,
										   RawOrig* __restrict__ raw_orig,
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

__device__ __forceinline__ bool point_in_triangle(const float px,
												  const float py,
												  const float ax,
												  const float ay,
												  const float bx,
												  const float by,
												  const float cx,
												  const float cy) {
	const float d1 = cross2d(px, py, ax, ay, bx, by);
	const float d2 = cross2d(px, py, bx, by, cx, cy);
	const float d3 = cross2d(px, py, cx, cy, ax, ay);
	const bool has_neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
	const bool has_pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
	return !(has_neg && has_pos);
}

// ---------------------------------------------------------------------------
// Kernel 2: window-based triangle assignment
//
// For each pixel:
//   1. Read Voronoi distance d; set search radius R = min(d+SLACK, CAP).
//   2. Scan the (2R+1)x(2R+1) seed-id window; collect unique seed IDs.
//   3. For each nearby seed, iterate its CSR triangle list and run the
//      containment test.  Duplicate triangle tests (same tri reachable via
//      multiple seeds) are allowed — they are idempotent.
// ---------------------------------------------------------------------------

static constexpr int WINDOW_SLACK = 3; // extra radius beyond Voronoi dist
static constexpr int WINDOW_CAP = 20; // hard cap to bound work in sparse areas
static constexpr int MAX_NEARBY = 64; // max unique seeds collected per pixel

__global__ void assign_triangles_kernel(int32_t* __restrict__ t_grid,
										const int W,
										const int H,
										const int32_t* __restrict__ n_grid,
										const int32_t* __restrict__ dist_grid,
										const RawOrig* __restrict__ tri_orig, // only orig_a/b/c needed here now
										const int32_t* __restrict__ seed_xs,
										const int32_t* __restrict__ seed_ys,
										const int32_t* __restrict__ csr_ptr,
										const int32_t* __restrict__ csr_idx,
										const int N_seeds,
										const int N_triangles) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	const float px = x + 0.5f;
	const float py = y + 0.5f;

	const int dist = dist_grid[y * W + x];
	const int R = min(dist + WINDOW_SLACK, WINDOW_CAP);

	int32_t nearby[MAX_NEARBY];
	int n_nearby = 0;

	const int x0 = max(0, x - R), x1 = min(W - 1, x + R);
	const int y0 = max(0, y - R), y1 = min(H - 1, y + R);

	for (int sy = y0; sy <= y1; ++sy) {
		for (int sx = x0; sx <= x1; ++sx) {
			int32_t sid = n_grid[sy * W + sx];
			bool dup = false;
			for (int i = 0; i < n_nearby; ++i)
				if (nearby[i] == sid) {
					dup = true;
					break;
				}
			if (!dup && n_nearby < MAX_NEARBY) nearby[n_nearby++] = sid;
		}
	}

	int32_t best = -1;
	for (int i = 0; i < n_nearby; ++i) {
		const int32_t sid = nearby[i];
		if (sid < 0 || sid >= N_seeds) continue;
		for (int j = csr_ptr[sid]; j < csr_ptr[sid + 1]; ++j) {
			const int32_t tid = csr_idx[j];
			const auto& [a, b, c] = tri_orig[tid];
			const float ax = static_cast<float>(seed_xs[a]), ay = static_cast<float>(seed_ys[a]);
			const float bx = static_cast<float>(seed_xs[b]), by = static_cast<float>(seed_ys[b]);
			const float cx = static_cast<float>(seed_xs[c]), cy = static_cast<float>(seed_ys[c]);
			if (point_in_triangle(px, py, ax, ay, bx, by, cx, cy))
				if (best == -1 || tid > best) best = tid;
		}
	}

	t_grid[y * W + x] = (best != -1) ? best : (N_triangles - 1);
}

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------

void cuda_compute_triangulation(int W,
								int H,
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
	// Upload inputs: seed_id channel, distance channel, seed positions
	// -----------------------------------------------------------------------

	std::vector<int32_t> h_n(N), h_dist(N);
	for (int i = 0; i < N; ++i) {
		h_n[i] = voronoi_grid[i * 2];
		h_dist[i] = voronoi_grid[i * 2 + 1];
	}

	int32_t *d_n = nullptr, *d_dist = nullptr;
	int32_t *d_sx = nullptr, *d_sy = nullptr;
	cudaMalloc(&d_n, N * sizeof(int32_t));
	cudaMalloc(&d_dist, N * sizeof(int32_t));
	cudaMalloc(&d_sx, N_seeds * sizeof(int32_t));
	cudaMalloc(&d_sy, N_seeds * sizeof(int32_t));
	cudaMemcpy(d_n, h_n.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, h_dist.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sx, seed_xs.data(), N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sy, seed_ys.data(), N_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	dim3 grid_dim((W + 15) / 16, (H + 15) / 16);

	// -----------------------------------------------------------------------
	// Step 2: detect raw triangle seeds
	// -----------------------------------------------------------------------

	const int max_raw = N * 4;
	RawXY* d_raw_xy = nullptr;
	RawKey* d_raw_key = nullptr;
	RawOrig* d_raw_orig = nullptr;
	cudaMalloc(&d_raw_xy, max_raw * sizeof(RawXY));
	cudaMalloc(&d_raw_key, max_raw * sizeof(RawKey));
	cudaMalloc(&d_raw_orig, max_raw * sizeof(RawOrig));

	int32_t* d_counter = nullptr;
	cudaMalloc(&d_counter, sizeof(int32_t));
	cudaMemset(d_counter, 0, sizeof(int32_t));

	if (timings) record(ev0);

	find_triangle_seeds_kernel<<<grid_dim, block>>>(d_n, W, H, d_raw_xy, d_raw_key, d_raw_orig, d_counter);
	cudaDeviceSynchronize();

	int32_t raw_count = 0;
	cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
	cudaFree(d_counter);

	if (timings) record(ev1);

	// -----------------------------------------------------------------------
	// Step 3: deduplicate on device with Thrust sort_by_key + unique_by_key
	// Key = RawKey (a,b,c); values = (RawXY, RawOrig) zipped so they're
	// reordered/compacted in lockstep with the key, in one pass each.
	// -----------------------------------------------------------------------

	thrust::device_ptr<RawKey> d_key_ptr(d_raw_key);
	thrust::device_ptr<RawXY> d_xy_ptr(d_raw_xy);
	thrust::device_ptr<RawOrig> d_orig_ptr(d_raw_orig);

	auto values_begin = thrust::make_zip_iterator(thrust::make_tuple(d_xy_ptr, d_orig_ptr));

	if (timings) record(ev2);

	thrust::sort_by_key(d_key_ptr, d_key_ptr + raw_count, values_begin, RawKeyLess{});
	auto unique_result = thrust::unique_by_key(d_key_ptr, d_key_ptr + raw_count, values_begin, RawKeyEqual{});
	const int N_triangles = static_cast<int>(unique_result.first - d_key_ptr);

	if (timings) record(ev3);

	// Copy deduplicated x,y and orig_a/b/c to host (Python dict + CSR build)
	std::vector<RawXY> h_xy(N_triangles);
	std::vector<RawOrig> h_orig(N_triangles);
	cudaMemcpy(h_xy.data(), d_raw_xy, N_triangles * sizeof(RawXY), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_orig.data(), d_raw_orig, N_triangles * sizeof(RawOrig), cudaMemcpyDeviceToHost);

	triangle_map_out.clear();
	triangle_map_out.reserve(N_triangles);
	for (int i = 0; i < N_triangles; ++i)
		triangle_map_out.push_back({h_xy[i].x, h_xy[i].y, h_orig[i].orig_a, h_orig[i].orig_b, h_orig[i].orig_c});

	// -----------------------------------------------------------------------
	// Build CSR: seed → triangle list  (host, then upload)
	// -----------------------------------------------------------------------

	std::vector<int32_t> h_csr_ptr(N_seeds + 1, 0);
	for (int tid = 0; tid < N_triangles; ++tid) {
		h_csr_ptr[h_orig[tid].orig_a + 1]++;
		h_csr_ptr[h_orig[tid].orig_b + 1]++;
		h_csr_ptr[h_orig[tid].orig_c + 1]++;
	}
	for (int s = 1; s <= N_seeds; ++s) h_csr_ptr[s] += h_csr_ptr[s - 1];

	const int csr_size = h_csr_ptr[N_seeds];
	std::vector<int32_t> h_csr_idx(csr_size);
	std::vector<int32_t> fill(N_seeds, 0);

	for (int tid = 0; tid < N_triangles; ++tid) {
		for (int32_t s : {h_orig[tid].orig_a, h_orig[tid].orig_b, h_orig[tid].orig_c}) {
			h_csr_idx[h_csr_ptr[s] + fill[s]] = tid;
			fill[s]++;
		}
	}

	int32_t *d_csr_ptr = nullptr, *d_csr_idx = nullptr;
	cudaMalloc(&d_csr_ptr, (N_seeds + 1) * sizeof(int32_t));
	cudaMalloc(&d_csr_idx, csr_size * sizeof(int32_t));
	cudaMemcpy(d_csr_ptr, h_csr_ptr.data(), (N_seeds + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_idx, h_csr_idx.data(), csr_size * sizeof(int32_t), cudaMemcpyHostToDevice);

	// -----------------------------------------------------------------------
	// Step 4: assign pixels to triangles (window-based nearest-neighbor)
	// -----------------------------------------------------------------------

	int32_t* d_t = nullptr;
	cudaMalloc(&d_t, N * sizeof(int32_t));

	if (timings) record(ev4);

	if (N_triangles > 0) {
		assign_triangles_kernel<<<grid_dim, block>>>(d_t, W, H, d_n, d_dist, d_raw_orig, d_sx, d_sy, d_csr_ptr,
													 d_csr_idx, N_seeds, N_triangles);
		cudaDeviceSynchronize();
	} else {
		cudaMemset(d_t, -1, N * sizeof(int32_t));
	}

	if (timings) record(ev5);

	// -----------------------------------------------------------------------
	// Step 5: build output grid (H * W * 3) and fill timings
	// -----------------------------------------------------------------------

	std::vector<int32_t> h_t(N);
	cudaMemcpy(h_t.data(), d_t, N * sizeof(int32_t), cudaMemcpyDeviceToHost);

	out_grid.resize(N * 3);
	for (int i = 0; i < N; ++i) {
		out_grid[i * 3] = voronoi_grid[i * 2];
		out_grid[i * 3 + 1] = voronoi_grid[i * 2 + 1];
		out_grid[i * 3 + 2] = h_t[i];
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

	cudaFree(d_raw_xy);
	cudaFree(d_raw_key);
	cudaFree(d_raw_orig);
	cudaFree(d_n);
	cudaFree(d_dist);
	cudaFree(d_t);
	cudaFree(d_sx);
	cudaFree(d_sy);
	cudaFree(d_csr_ptr);
	cudaFree(d_csr_idx);
}
