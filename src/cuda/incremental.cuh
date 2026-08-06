#pragma once
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "triangulation.cuh" // TriangleEntry

struct IncrementalTimings {
	float bfs_ms = 0.f;
	float detect_ms = 0.f;
	float dedup_ms = 0.f;
	float assign_ms = 0.f;
};

// TODO: same as in triangulation.cu, split into separate file
struct Vec2i {
	int32_t x, y;
};
struct Vec3i {
	int32_t a, b, c;
};

class IncrementalDelaunay {
public:
	IncrementalDelaunay(int width, int height, int max_seeds);
	~IncrementalDelaunay();

	// Appends a batch of seeds (insertion-order IDs).
	// Returns current full triangle_map and triangulation grid (H,W,3).
	void insert(const std::vector<int32_t>& new_xs,
				const std::vector<int32_t>& new_ys,
				std::vector<TriangleEntry>& tri_map_out,
				std::vector<int32_t>& tgrid_out,
				IncrementalTimings* timings = nullptr);

	void get_voronoi_grid(std::vector<int32_t>& out) const;
	int seed_count() const { return N_; }
	int width() const { return W_; }
	int height() const { return H_; }

	// --- Merging new and old Triangles on device ---
	struct MergeKey {
		int32_t a, b, c; // ascending sorted triplet (a, b, c)
		bool is_new; // is_new descending
	};

private:
	int W_, H_, N_, max_seeds_;

	int reg_cap_; // 3 * max_seeds   max triangle count in registry
	int det_cap_; // 4 * W * H       max tri seeds that can be found
	int mrg_cap_; // 2 * reg_cap_    in dirty mask survivors + new detections

	// ---- Voronoi state ----
	int32_t* d_grid_; // (H*W*2) interleaved (seed_id, distance)
	int32_t* d_tmp_; // (H*W*2) BFS ping-pong
	int32_t* d_changed_; // (H*W)   cells updated during BFS (accumulated)
	int32_t* d_dirty_; // (H*W)   d_changed_ dilated by +-2 (detect scope)
	uint8_t* d_updated_flag_; // (2)     BFS convergence flags (ping-pong)

	// ---- seeds ----
	Vec2i* d_seed_pos_; // (max_seeds)

	// ---- rasterised assignment ----
	int32_t* d_t_grid_; // (H*W) triangle_id per pixel, -1 = uncovered

	// ---- triangle registry (ping-pong, GPU resident) ----
	Vec2i* d_reg_xy_[2]; // (reg_cap_) canonical detection pixel
	Vec3i* d_reg_orig_[2]; // (reg_cap_) seed triplet, in detection order
	int reg_cur_; // which side of the ping-pong is live
	int N_tri_; // triangles currently in the live registry

	// ---- tri seed detection ----
	Vec2i* d_det_xy_;
	Vec3i* d_det_key_; // sorted triplet, dedup key only
	Vec3i* d_det_orig_;
	int32_t* d_counter_;

	// ---- tri merge tmep ----
	MergeKey* d_mrg_key_;
	Vec2i* d_mrg_xy_;
	Vec3i* d_mrg_orig_;

	// ---- output ----
	TriangleEntry* d_out_map_;
	int32_t* d_out_grid_; // (H*W*3)

	// ---- host-side seed registry ----
	std::vector<Vec2i> h_seed_pos_; // TODO: remove, not read
	std::unordered_set<uint64_t> h_seed_set_; // fast duplicate check

	// ---- private helpers ----
	void run_bfs_(float* bfs_ms_out);
	void triangulate_(bool is_first, float* detect_ms, float* dedup_ms, float* assign_ms);
	void build_outputs_(std::vector<TriangleEntry>& tri_map_out, std::vector<int32_t>& tgrid_out) const;

	static uint64_t pack_triplet_(const int32_t a, const int32_t b, const int32_t c) {
		return (static_cast<uint64_t>(a) << 42) | (static_cast<uint64_t>(b) << 21) | static_cast<uint64_t>(c);
	}
	static uint64_t pack_xy_(const int32_t x, const int32_t y) {
		return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
			static_cast<uint64_t>(static_cast<uint32_t>(y));
	}
};
