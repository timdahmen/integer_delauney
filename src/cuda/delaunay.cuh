#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "triangulation.cuh"   // TriangleEntry

struct InsertTimings {
    float bfs_ms    = 0.f;
    float detect_ms = 0.f;
    float dedup_ms  = 0.f;
    float assign_ms = 0.f;
};

class Delaunay {
public:
    // border_padding < 0 uses DEFAULT_BORDER_PADDING.
    //
    // Fixed at construction, because the padded canvas is the persistent device
    // state. That used to matter a great deal: the default was a density
    // estimate that shrank as seeds were added, so a state below max_seeds was
    // under-padded relative to what the batch path would pick for it. With a
    // constant default both paths agree by construction.
    //
    // The padding a seed set needs is bounded only when the caller controls how
    // densely the convex hull boundary is sampled. See BORDER_PADDING_BOUND.md.
    Delaunay(int width, int height, int max_seeds,
                        int border_padding = -1);
    ~Delaunay();

    // Appends a batch of seeds (insertion-order IDs).
    // Returns current full triangle_map and triangulation grid (H,W,3).
    // Exactly equivalent to insert_deferred() followed by finalise().
    void insert(
        const std::vector<int32_t>& new_xs,
        const std::vector<int32_t>& new_ys,
        std::vector<TriangleEntry>&  tri_map_out,
        std::vector<int32_t>&        tgrid_out,
        InsertTimings*          timings = nullptr);

    // Appends a batch, updating the Voronoi diagram and the triangle topology
    // but NOT the per-pixel triangle assignment, and materialising no output.
    //
    // The per-pixel assignment is the expensive stage and the only one whose
    // dirty region saturates for large scattered batches, so a caller that
    // inserts repeatedly before it needs a raster should defer it.  Changes
    // accumulate until finalise() assigns them in one pass.
    //
    // get_triangles() is valid between deferred inserts; the triangulation
    // grid is not, and neither is get_voronoi_grid()'s triangle content.
    void insert_deferred(
        const std::vector<int32_t>& new_xs,
        const std::vector<int32_t>& new_ys,
        InsertTimings*          timings = nullptr);

    // Assigns pixels for everything deferred since the last finalise, then
    // materialises the outputs.  Chooses a masked or a full assignment by
    // whichever covers less work, so a small accumulated change stays cheap and
    // a large one does not pay for masking it cannot benefit from.
    // Calling this with nothing pending only rebuilds the outputs.
    void finalise(
        std::vector<TriangleEntry>& tri_map_out,
        std::vector<int32_t>&       tgrid_out,
        InsertTimings*         timings = nullptr);

    // Triangle topology alone, with no raster copied back.  Seed ids are
    // INSERTION-order, not the sorted numbering insert()/finalise() report:
    // a caller refining across several inserts keeps its own per-seed arrays
    // aligned by appending, which sorted ids would invalidate on every call.
    // Translate with sorted_rank() when handing results to the batch API.
    void get_triangles(std::vector<TriangleEntry>& out) const;

    // internal insertion-order id -> batch pipeline's sorted (x asc, y asc) id
    const std::vector<int32_t>& sorted_rank() const
    { ensure_sorted_rank_(); return h_sorted_rank_; }

    void get_voronoi_grid(std::vector<int32_t>& out) const;
    int  seed_count()    const { return N_; }
    int  width()         const { return W_; }
    int  height()        const { return H_; }
    int  border_padding() const { return P_; }
    bool has_pending()   const { return pending_; }

private:
    int W_, H_, N_, max_seeds_;
    // Detection canvas. All device grids live in padded coordinates: a triangle
    // is registered where three Voronoi regions meet, i.e. at its circumcentre,
    // and boundary triangles frequently have circumcentres outside the image,
    // so at P_ = 0 they are never detected at all. Working padded throughout
    // keeps every kernel on one coordinate system; the interior is extracted
    // only when building outputs, and canonical triangle positions are shifted
    // back by P_ there. Seed coordinates are stored padded on the device and
    // unpadded on the host.
    int P_, W_det_, H_det_;

    // ---- persistent device buffers, all sized on the padded canvas ----
    int32_t* d_grid_;        // (H*W*2) Voronoi: interleaved (seed_id, distance)
    int32_t* d_tmp_;         // (H*W*2) BFS ping-pong
    int32_t* d_changed_;     // (H*W)   cells updated during BFS (accumulated)
    int32_t* d_sx_;          // (max_seeds) seed x
    int32_t* d_sy_;          // (max_seeds) seed y
    void*    d_raw_buf_;     // detection scratch; see max_raw_triangles()
    int32_t* d_t_grid_;      // (H*W)   triangle_id per pixel
    int32_t* d_csr_ptr_;     // (max_seeds+1) CSR row starts
    int32_t* d_csr_idx_;     // (max_seeds*8) CSR triangle IDs
    int32_t* d_updated_flag_;// (1)     BFS convergence flag
    int32_t* d_mask_;        // (H*W)   reused for border / reassign masks
    // Changes since the last finalise, as opposed to d_changed_, which holds
    // only the current insert's.  Detection is scoped by the latter so a
    // deferred round does not re-detect earlier rounds' regions; assignment is
    // scoped by the former because it has not run for any of them yet.
    int32_t* d_dirty_accum_; // (H*W)   union of d_changed_ since last finalise
    int32_t* d_tile_dirty_;  // (tiles) tile-level dirty flags, mask prefilter
    int32_t* d_count_;       // (1)     dirty-pixel counter for the cost switch
    uint8_t* d_stale_;       // (max triangles) per-triangle invalidation flags
    int      tiles_x_, tiles_y_;
    bool     pending_;       // deferred inserts awaiting a finalise
    // Derived structures whose only consumers run at assignment or output
    // time. Rebuilding them per insert cost O(N_tri + N_seeds) of host work
    // that a deferred round never read; they are now rebuilt on demand.
    bool             csr_dirty_;
    mutable bool     sorted_rank_dirty_;

    // ---- host-side triangle registry ----
    struct HTriangle {
        int32_t x, y;
        int32_t a, b, c;            // sorted key (a<=b<=c)
        int32_t orig_a, orig_b, orig_c;
    };
    std::vector<HTriangle>                 h_triangles_;
    std::unordered_map<uint64_t,int32_t>   h_triplet_to_tid_;
    // (A canonical-pixel -> tid map used to live here. It was written on every
    // registry rebuild and never read; with the 2x2 detection two triangles can
    // share a canonical pixel anyway, so it could not have been a key.)

    // ---- host-side seed registry ----
    std::vector<int32_t>            h_sx_, h_sy_;
    std::unordered_set<uint64_t>    h_seed_set_;   // fast duplicate check

    // Internal seed ids are assigned in INSERTION order, because previously
    // inserted seeds must keep their ids for the incremental device state
    // (d_grid_, d_t_grid_, the triangle registry and the CSR) to stay valid.
    // The batch pipeline instead numbers seeds in sorted (x asc, y asc) order.
    // h_sorted_rank_[internal_id] gives that sorted id, and outputs are
    // translated through it so both pipelines expose the same numbering.
    mutable std::vector<int32_t>    h_sorted_rank_;

    // ---- private helpers ----
    void rebuild_sorted_rank_() const;
    void ensure_sorted_rank_() const;
    void ensure_csr_();
    void run_bfs_(float* bfs_ms_out);
    // Topology only: detect + dedup + registry + CSR. Pixel assignment is
    // separate so it can be deferred across several inserts and run once.
    void full_topology_(float* detect_ms, float* dedup_ms);
    void partial_topology_(float* detect_ms, float* dedup_ms);
    // Registration + seed write + BFS, shared by insert() and insert_deferred().
    void apply_batch_(const std::vector<int32_t>& new_xs,
                      const std::vector<int32_t>& new_ys,
                      float* bfs_ms_out);
    // Pixel assignment over d_dirty_accum_, masked or full by measured cost.
    void assign_pending_(float* assign_ms);
    void build_reassign_mask_();
    int  count_mask_();
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
