#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "triangulation.cuh"   // TriangleEntry

// Host-side wall-clock breakdown of one insert(), in milliseconds.
// Phases are non-overlapping and mirror the NVTX ranges, except scratch_ms,
// which is an "of which" sub-measure counted inside other phases.
//
// An insert() takes exactly one of the two triangulation paths, so the phases
// they have in common (detect / dedup / d2h_new / csr / assign) share fields.
// Fields marked "cold" or "warm" are only ever written by that one path.
struct IncrementalHostTimings {
    // insert()
    float validate_ms      = 0.f;  // bounds + duplicate checks
    float seed_reg_ms      = 0.f;  // host seed registry update
    float seed_h2d_ms      = 0.f;  // d_sx_/d_sy_ upload + d_changed_ clear
    float write_seeds_ms   = 0.f;  // scratch seed arrays + write_seeds kernel
    float bfs_ms           = 0.f;  // run_bfs_ wall time (incl. per-iter syncs)
    // full_triangulate_() / partial_triangulate_()
    float d2h_changed_ms   = 0.f;  // warm
    float expand_ms        = 0.f;  // warm: border/reassign dilation loop
    float mark_stale_ms    = 0.f;  // warm
    float h2d_border_ms    = 0.f;  // warm
    float detect_ms        = 0.f;
    float dedup_ms         = 0.f;
    float d2h_new_ms       = 0.f;  // D2H of the deduplicated triangles
    float build_registry_ms= 0.f;  // cold: h_triangles_ + h_triplet_to_tid_
    float collect_ms       = 0.f;  // warm
    float compact_ms       = 0.f;  // warm
    float upload_tri_ms    = 0.f;  // warm
    float csr_ms           = 0.f;
    float remap_ms         = 0.f;  // warm
    float h2d_reassign_ms  = 0.f;  // warm
    float assign_ms        = 0.f;
    // build_outputs_()
    float out_trimap_ms    = 0.f;
    float out_d2h_ms       = 0.f;
    float out_interleave_ms= 0.f;
    // overlapping sub-measure
    float scratch_ms       = 0.f;  // cudaMalloc/cudaFree of per-insert scratch
};

struct IncrementalTimings {
    float bfs_ms    = 0.f;
    float detect_ms = 0.f;
    float dedup_ms  = 0.f;
    float assign_ms = 0.f;
    IncrementalHostTimings host;
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
    int32_t* d_grid_;              // (H*W*2) Voronoi: interleaved (seed_id, distance)
    int32_t* d_tmp_;               // (H*W*2) BFS ping-pong
    int32_t* d_changed_;           // (H*W)   cells updated during BFS (accumulated)
    int32_t* d_sx_;                // (max_seeds) seed x
    int32_t* d_sy_;                // (max_seeds) seed y
    void*    d_raw_buf_;           // (H*W*4 * sizeof(RawTriangle)) detection scratch
    int32_t* d_t_grid_;            // (H*W)   triangle_id per pixel
    int32_t* d_csr_ptr_;           // (max_seeds+1) CSR row starts
    int32_t* d_csr_idx_;           // (max_seeds*8) CSR triangle IDs
    int32_t* d_updated_flag_;      // (1)     BFS convergence flag
    int32_t* d_mask_;              // (H*W)   reused for border / reassign masks
    void*    d_csr_verts_cache_;   // (max_seeds*8 * sizeof(CsrEntryVertexCache)) vert for each CSR entry

    // ---- persistent pinned host staging buffers ----
    int32_t* p_changed_  = nullptr;  // (N)    D2H  BFS change mask
    int32_t* p_border_   = nullptr;  // (N)    H2D  detect mask (host-read)
    int32_t* p_reassign_ = nullptr;  // (N)    H2D  assign mask (host-write-only)
    int32_t* p_t_        = nullptr;  // (N)    D2H  triangle id per pixel
    int32_t* p_grid_     = nullptr;  // (N*2)  D2H  voronoi grid

    // Host timing sink for the in-flight insert(); null when not profiling.
    IncrementalHostTimings* ht_ = nullptr;

    // ---- host-side triangle registry ----
    struct HTriangle {
        int32_t x, y;
        int32_t a, b, c;            // sorted key (a<=b<=c)
        int32_t orig_a, orig_b, orig_c;
    };
    std::vector<HTriangle>                 h_triangles_;
    std::unordered_map<uint64_t,int32_t>   h_triplet_to_tid_;

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
