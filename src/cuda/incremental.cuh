#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "common.h"

struct IncrementalTimings {
	float bfs_ms = 0.f;
	float detect_ms = 0.f;
	float dedup_ms = 0.f;
	float assign_ms = 0.f;
};

class IncrementalDelaunay {
public:
	IncrementalDelaunay(int width, int height, int max_seeds);
	~IncrementalDelaunay();

	// Appends a batch of seeds (insertion-order IDs).
	// Returns current full triangle_map and triangulation grid (H,W,3).
	void insert(const std::vector<Vec2i>& new_seeds,
				std::vector<TriangleEntry>& tri_map_out,
				std::vector<int32_t>& tgrid_out,
				IncrementalTimings* timings = nullptr);

	struct InsertResultRef {
		const std::vector<TriangleEntry>& tri_map;
		const std::vector<int32_t>& tgrid;
	};

	// wrapper: uses the persistent internal buffers and
	// returns them by reference
	// !!! Only valid until the next call to insert() on this object
	// copy out if needed for longer
	InsertResultRef insert(const std::vector<Vec2i>& new_seeds, IncrementalTimings* timings = nullptr);

	// wrapper to keep same API with x,y as split inputs
	void insert(const std::vector<int32_t>& new_xs,
				const std::vector<int32_t>& new_ys,
				std::vector<TriangleEntry>& tri_map_out,
				std::vector<int32_t>& tgrid_out,
				IncrementalTimings* timings = nullptr);

	void get_voronoi_grid(std::vector<int32_t>& out) const;
	int seed_count() const { return N_; }
	int width() const { return W_; }
	int height() const { return H_; }

private:
	int W_, H_, N_, max_seeds_;

	// --- persistent output buffers (no reallocation for each insert) ---
	std::vector<TriangleEntry> tri_map_buf_;
	std::vector<int32_t> tgrid_buf_;

	int tri_cap_;
	int det_cap_; // 4 * W * H       max tri seeds that can be found
	int N_tri_; // triangles currently in the live registry

	// ---- Voronoi state ----
	int32_t* d_grid_; // (H*W*2) interleaved (seed_id, distance)
	int32_t* d_tmp_; // (H*W*2) BFS ping-pong
	int32_t* d_updated_flag_; // (2)  BFS convergence flags (ping-pong)

	// ---- seeds ----
	Vec2i* d_seed_pos_; // (max_seeds)

	// ---- rasterised assignment ----
	int32_t* d_t_grid_; // (H*W) triangle_id per pixel, -1 = uncovered

	// ---- tri seed detection ----
	Vec2i* d_det_xy_;
	Vec3i* d_det_key_; // sorted triplet, dedup key only
	Vec3i* d_det_orig_;
	int32_t* d_counter_;

	// ---- output ----
	TriangleEntry* d_out_map_;
	int32_t* d_out_grid_; // (H*W*3)

	// ---- host-side seed registry ----
	std::unordered_set<uint64_t> h_seed_set_; // fast duplicate check

	// ---- private helpers ----
	void run_bfs_(float* bfs_ms_out);
	void triangulate_(float* det_ms, float* dedup_ms, float* asgn_ms, std::vector<int32_t>& tgrid_out);
	void build_outputs_(std::vector<TriangleEntry>& tri_map_out, std::vector<int32_t>& tgrid_out) const;

	static uint64_t pack_triplet_(const int32_t a, const int32_t b, const int32_t c) {
		return (static_cast<uint64_t>(a) << 42) | (static_cast<uint64_t>(b) << 21) | static_cast<uint64_t>(c);
	}
	static uint64_t pack_xy_(const int32_t x, const int32_t y) {
		return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
			static_cast<uint64_t>(static_cast<uint32_t>(y));
	}
};
