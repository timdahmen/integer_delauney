#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "triangulation.cuh"   // TriangleEntry

struct IncrementalTimings {
    float bfs_ms    = 0.f;
    float detect_ms = 0.f;
    float dedup_ms  = 0.f;
    float assign_ms = 0.f;
};

class IncrementalDelaunay {
public:
    IncrementalDelaunay(int width, int height, int max_seeds);
    ~IncrementalDelaunay();

    // Appends a batch of seeds (insertion-order IDs).
    // Returns current full triangle_map and triangulation grid (H,W,3).
    void insert(
        const std::vector<int32_t>& new_xs,
        const std::vector<int32_t>& new_ys,
        std::vector<TriangleEntry>&  tri_map_out,
        std::vector<int32_t>&        tgrid_out,
        IncrementalTimings*          timings = nullptr);

    void get_voronoi_grid(std::vector<int32_t>& out) const;
    int  seed_count() const { return N_; }
    int  width()      const { return W_; }
    int  height()     const { return H_; }

private:
    int W_, H_, N_, max_seeds_;

    // ---- persistent device buffers ----
    int32_t* d_grid_;        // (H*W*2) Voronoi: interleaved (seed_id, distance)
    int32_t* d_tmp_;         // (H*W*2) BFS ping-pong
    int32_t* d_changed_;     // (H*W)   cells updated during BFS (accumulated)
    int32_t* d_sx_;          // (max_seeds) seed x
    int32_t* d_sy_;          // (max_seeds) seed y
    void*    d_raw_buf_;     // (H*W*4 * sizeof(RawTriangle)) detection scratch
    int32_t* d_t_grid_;      // (H*W)   triangle_id per pixel
    int32_t* d_csr_ptr_;     // (max_seeds+1) CSR row starts
    int32_t* d_csr_idx_;     // (max_seeds*8) CSR triangle IDs
    int32_t* d_updated_flag_;// (1)     BFS convergence flag
    int32_t* d_mask_;        // (H*W)   reused for border / reassign masks

    // ---- host-side triangle registry ----
    struct HTriangle {
        int32_t x, y;
        int32_t a, b, c;            // sorted key (a<=b<=c)
        int32_t orig_a, orig_b, orig_c;
    };
    std::vector<HTriangle>                 h_triangles_;
    std::unordered_map<uint64_t,int32_t>   h_triplet_to_tid_;
    std::unordered_map<uint64_t,int32_t>   h_canon_to_tid_;

    // ---- host-side seed registry ----
    std::vector<int32_t>            h_sx_, h_sy_;
    std::unordered_set<uint64_t>    h_seed_set_;   // fast duplicate check

    // ---- private helpers ----
    void run_bfs_(float* bfs_ms_out);
    void full_triangulate_(float* detect_ms, float* dedup_ms, float* assign_ms);
    void partial_triangulate_(float* detect_ms, float* dedup_ms, float* assign_ms);
    void rebuild_csr_and_upload_();
    void upload_triangles_();
    void build_outputs_(std::vector<TriangleEntry>& tri_map_out,
                        std::vector<int32_t>& tgrid_out) const;

    static uint64_t pack_triplet_(int32_t a, int32_t b, int32_t c) {
        return (uint64_t(a) << 42) | (uint64_t(b) << 21) | uint64_t(c);
    }
    static uint64_t pack_xy_(int32_t x, int32_t y) {
        return (uint64_t(uint32_t(x)) << 32) | uint64_t(uint32_t(y));
    }
};
