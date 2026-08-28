#pragma once
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "triangulation.cuh"   // TriangleEntry

// Declared in triangle_detect.cuh, a CUDA-only header (uses atomicAdd etc.)
// that host-only translation units like bindings.cpp cannot include. This
// class only ever names RawTriangle in reference parameters below, never
// needs it complete, so a forward declaration is enough here.
struct RawTriangle;

struct InsertTimings {
    float bfs_ms    = 0.f;
    int   bfs_iters = 0;     // BFS trip count; each trip is a full-canvas pass
    float detect_ms = 0.f;
    float dedup_ms  = 0.f;
    float assign_ms = 0.f;
};

//: Tile edge for the dirty prefilter in delaunay_assign.cu's reassign mask.
//: Small enough that a tile rarely covers unchanged ground, large enough that
//: a WINDOW_CAP-sized window (delaunay_locate.cuh) spans only a handful of
//: tiles. Shared with the constructor's tiles_x_/tiles_y_ sizing below, so it
//: lives here rather than in the .cu file that uses it most.
static constexpr int MASK_TILE = 8;

class Delaunay {
public:
    // border_padding < 0 uses DEFAULT_BORDER_PADDING.
    //
    // Fixed at construction, because the padded canvas is the persistent device
    // state, and DEFAULT_BORDER_PADDING is a constant rather than a per-call
    // density estimate -- so this and the batch path agree by construction,
    // regardless of how far below max_seeds the state currently sits.
    //
    // The padding a seed set needs is bounded only when the caller controls how
    // densely the convex hull boundary is sampled. See BORDER_PADDING_BOUND.md.
    Delaunay(int width, int height, int max_seeds,
                        int border_padding = -1);
    ~Delaunay();

    // Restores the object to its just-constructed state -- every seed and
    // triangle discarded, generation() bumped -- without freeing or
    // reallocating any device buffer. width()/height()/max_seeds()/
    // border_padding() are unchanged, so a caller inserting a same-canvas,
    // same-or-smaller seed set next can reuse this object instead of paying
    // for a fresh Delaunay's cudaMalloc calls.
    void reset();

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
    // new_values, when non-empty, must have one entry per seed: it is the
    // scalar field sampled at them, appended in step so it cannot drift.
    void insert_deferred(
        const std::vector<int32_t>& new_xs,
        const std::vector<int32_t>& new_ys,
        InsertTimings*          timings = nullptr,
        const std::vector<float>*   new_values = nullptr);

    // Assigns pixels for everything deferred since the last finalise, then
    // materialises the outputs.  Chooses a masked or a full assignment by
    // whichever covers less work, so a small accumulated change stays cheap and
    // a large one does not pay for masking it cannot benefit from.
    // Calling this with nothing pending only rebuilds the outputs.
    void finalise(
        std::vector<TriangleEntry>& tri_map_out,
        std::vector<int32_t>&       tgrid_out,
        InsertTimings*         timings = nullptr);

    // Same as finalise(), except every output stays on the device: the
    // (H*W) triangle-id, seed-id and outside-hull arrays are written
    // directly into device_pixel_tids()/device_pixel_seed_ids()/
    // device_outside_mask(), and the (triangle_count(), 3) vertex-id array
    // into device_triangle_verts(), all by kernels, never copied to host
    // memory here.
    //
    // Those four views are valid from the moment this call returns until
    // generation() next changes, i.e. until the next call to insert(),
    // insert_deferred(), finalise(), finalise_device() or reset() on this
    // object, or until the object is destroyed. A caller holding a view
    // across such a call is reading memory that has moved on.
    void finalise_device();

    const int32_t* device_pixel_tids()     const { return d_pixel_tids_; }
    const int32_t* device_pixel_seed_ids() const { return d_pixel_seed_ids_; }
    const uint8_t* device_outside_mask()   const { return d_outside_mask_; }
    // (triangle_count(), 3) vertex ids, sorted-rank numbered like
    // device_pixel_seed_ids(). Valid under the same rule as the three
    // buffers above; see finalise_device()'s doc comment.
    const int32_t* device_triangle_verts() const { return d_triangle_verts_; }
    // Rows in device_triangle_verts() as of the last finalise_device()
    // call -- n_live_ at that point, since finalise_device() compacts the
    // registry first.
    int triangle_count() const { return n_live_; }

    // Bumped by every call that can change what the three device_* buffers
    // above contain (see finalise_device()). A snapshot taken at one
    // generation and compared against a later one tells a consumer its view
    // is stale before it reads through a pointer that has been reused.
    uint64_t generation() const { return generation_; }

    // Triangle topology alone, with no raster copied back.  Seed ids are
    // INSERTION-order, not the sorted numbering insert()/finalise() report:
    // a caller refining across several inserts keeps its own per-seed arrays
    // aligned by appending, which sorted ids would invalidate on every call.
    // Translate with sorted_rank() when handing results to the batch API.
    void get_triangles(std::vector<TriangleEntry>& out) const;

    // The distinct undirected edges of the current triangulation, as flat
    // (a0, b0, a1, b1, ...) with a < b. Seed ids are INSERTION-order, matching
    // get_triangles().
    //
    // Every interior edge is shared by two triangles, so the 3T edges a
    // triangulation spells out hold each one about twice; deduplicating them is
    // a sort and a unique over 3T keys, which is what the device is for and
    // what a caller would otherwise pull 3T triangle indices across the bus to
    // do. Consumers score edges rather than triangles, so this is the shape the
    // data is wanted in.
    void get_edges(std::vector<int32_t>& out) const;

    // ---- scalar field on the vertices, and the edge metric over it ----
    //
    // A caller that refines a mesh carries one number per vertex -- a measured
    // value, a field sample -- and picks edges to subdivide by how much that
    // number changes along them. Holding the field here rather than on the host
    // keeps it beside the coordinates and the edge list, so scoring an edge
    // reads three device arrays and moves nothing.
    //
    // Values arrive with the seeds they belong to, so the field cannot drift
    // out of step with the vertex set.

    //: score = |value[a] - value[b]| * |p[a] - p[b]|, zero for edges shorter
    //: than min_length.
    //:
    //: |dv| is about |grad| * |ab|, so the score behaves like |grad| * |ab|^2,
    //: which is how linear interpolation error scales over a triangle. Edges
    //: below min_length are excluded because their midpoint rounds onto an
    //: endpoint. Order matches get_edges().
    void edge_scores(double min_length, std::vector<float>& out) const;

    //: Midpoints of the highest-scoring edges, as flat (x0, y0, x1, y1, ...).
    //:
    //: An edge is taken when its score beats (min_score, tie_index) under the
    //: ordering "higher score first, lower edge index on a tie". Passing the
    //: k-th largest such key therefore takes exactly k edges, with no ties to
    //: resolve and the same answer every run -- which is why selection can stay
    //: with the caller while everything around it runs here: what crosses is
    //: one score and one index, not a list.
    //:
    //: Midpoints landing on a pixel that is already a seed are dropped, which
    //: needs no separate record of what has been sampled: a seed is exactly a
    //: cell at distance zero in the Voronoi diagram. Midpoints colliding with
    //: each other are dropped down to the one from the lowest edge index.
    //: `count` is exact, not a threshold: the edges are ordered best-first by
    //: (score descending, index ascending), which is a total order, and the
    //: front of it is taken. Nothing about the selection crosses to the caller.
    //: `threshold` excludes edges scoring at or below it, so fewer than `count`
    //: may come back.
    void select_midpoints(double min_length, int count, float threshold,
                          std::vector<int32_t>& out) const;

    // The seed positions in insertion order, as flat (x0, y0, x1, y1, ...),
    // and the scalar field beside them. The mesh holds both, so a caller has
    // no reason to keep a second copy that could disagree with it.
    void get_seeds(std::vector<int32_t>& out) const;
    void get_values(std::vector<float>& out) const;

    //: The triangle containing each query point, or NO_TRIANGLE, in image
    //: coordinates. Ids index get_triangles(), which finalise()'s triangle map
    //: agrees with -- the registry is compacted first, so both are dense and in
    //: the same order.
    //:
    //: For a caller that wants containment for a list of positions rather than
    //: a raster. finalise() answers the same question for all 1.5M pixels of a
    //: real canvas; asking it about 70k points and reading the rest back out is
    //: about twenty times the work the question needs.
    void locate(const std::vector<int32_t>& qx,
                const std::vector<int32_t>& qy,
                std::vector<int32_t>& out);

    //: The in-circle predicate, lifted out of the plane by t: is the point
    //: inside the circumsphere of the triangle containing it, when the triangle
    //: lies at t = 0 and the point at t?
    //:
    //:     dx^2 + dy^2 + t^2 < R^2
    //:
    //: A caller using t for elapsed time reads this as "close enough in space
    //: and recent enough in time"; a triangle of circumradius R admits nothing
    //: beyond t = R. Reports the containing triangle alongside, since a caller
    //: asking this almost always wants it and it is found on the way.
    void in_circumsphere(const std::vector<int32_t>& qx,
                         const std::vector<int32_t>& qy,
                         const std::vector<double>& qt,
                         std::vector<uint8_t>& mask_out,
                         std::vector<int32_t>& tid_out);

    // internal insertion-order id -> batch pipeline's sorted (x asc, y asc) id
    const std::vector<int32_t>& sorted_rank() const
    { ensure_sorted_rank_(); return h_sorted_rank_; }

    void get_voronoi_grid(std::vector<int32_t>& out) const;
    int  seed_count()    const { return N_; }
    int  width()         const { return W_; }
    int  height()        const { return H_; }
    int  border_padding() const { return P_; }
    int  max_seeds()     const { return max_seeds_; }
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
    // Default-initialised to null so a mid-construction cudaMalloc failure
    // (see the constructor) can safely free whatever was allocated before it
    // -- cudaFree(nullptr) is a documented no-op, an uninitialised pointer is not.
    int32_t* d_grid_ = nullptr;        // (H*W*2) Voronoi: interleaved (seed_id, distance)
    int32_t* d_tmp_ = nullptr;         // (H*W*2) BFS ping-pong
    int32_t* d_changed_ = nullptr;     // (H*W)   cells updated during BFS (accumulated)
    int32_t* d_sx_ = nullptr;          // (max_seeds) seed x
    int32_t* d_sy_ = nullptr;          // (max_seeds) seed y
    // The triangle registry itself -- the source of truth, not a mirror of a
    // host copy. Written directly by the retire/append kernels
    // (partial_topology_) and by full_topology_/compact_registry_; read by
    // everything else that needs triangle data (get_edges, get_triangles,
    // build_tri_map_, the CSR build). See next_tid_host_ below for the
    // "how many slots" bookkeeping.
    void*    d_raw_buf_ = nullptr;
    // Ping-pong target for compact_registry_'s device-side compaction, same
    // size and layout as d_raw_buf_ -- swapped in, mirroring d_grid_/d_tmp_'s
    // BFS ping-pong, rather than allocated and freed per compaction.
    void*    d_raw_buf_compact_ = nullptr;
    // Detection scratch, written from index 0 on every detect. Separate from
    // the registry above because detection would otherwise overwrite it.
    void*    d_detect_buf_ = nullptr;  // see max_raw_triangles()
    // Scratch for partial_topology_'s retire step: the stale tids compacted
    // out of d_stale_ on the device, so the retire kernel touches only the
    // ones that changed, not every slot in the registry.
    int32_t* d_stale_tids_ = nullptr;       // (max_seeds * 4)
    // Triangle identity, keyed by geometry: a live triangle's three vertices
    // are immutable once inserted, so the exact (unrounded) sum of their seed
    // coordinates is a position no other live triangle can also claim --
    // triangulation gives every triangle a disjoint interior, and a
    // non-degenerate triangle's centroid always lies strictly inside its own
    // interior, so two distinct triangles can never share one. Sized at 3x
    // resolution (centroid_index_w_/_h_ below) because that sum is exact, not
    // divided by 3 and rounded: rounding after the divide can merge two
    // close-but-distinct centroids (two triangles sharing an edge, for
    // instance), which the exact sum cannot. NO_TRIANGLE where no live
    // triangle claims that position.
    int32_t* d_centroid_index_ = nullptr;   // (centroid_index_w_ * _h_)
    // Scratch for the append kernel: 1/0 "is this candidate new" flags,
    // exclusive-scanned in place into each new candidate's rank among the
    // new ones -- new_tid = next_tid_host_ + rank, deterministic in the
    // array order detect_and_dedup_'s thrust::unique already sorted
    // candidates into (by vertex triplet). See append_triangles_kernel's
    // doc comment for why this order has to be deterministic.
    int32_t* d_new_rank_ = nullptr;         // (max_raw_triangles bound)
    // (3 * max triangles) packed undirected edge keys for get_edges(). Sized
    // like d_stale_, off the planarity bound of under 2n triangles for n seeds.
    void*    d_edge_keys_ = nullptr;
    int32_t* d_t_grid_ = nullptr;      // (H*W)   triangle_id per pixel
    // ---- device-resident finalise_device() outputs, see the header above ----
    int32_t* d_sorted_rank_ = nullptr;    // (max_seeds) device mirror of h_sorted_rank_
    int32_t* d_pixel_tids_ = nullptr;     // (H*W) triangle id per pixel, NO_TRIANGLE -> 0
    int32_t* d_pixel_seed_ids_ = nullptr; // (H*W) nearest seed id per pixel, sorted numbering
    uint8_t* d_outside_mask_ = nullptr;   // (H*W) 1 where the pixel has no containing triangle
    // (max_seeds*4 slots * 3) vertex ids, sorted-rank numbered; only
    // [0, triangle_count()*3) is meaningful, written fresh by
    // finalise_device() each call.
    int32_t* d_triangle_verts_ = nullptr;
    uint64_t generation_;       // see generation() above
    int32_t* d_csr_ptr_ = nullptr;     // (max_seeds+1) CSR row starts
    // Sized for one (seed, tid) pair per corner of every registry slot
    // (max_seeds*4 slots x 3), including dead ones: rebuild_csr_and_upload_
    // sentinel-tags a dead slot's pairs so they sort past every live seed's
    // range, see that function.
    int32_t* d_csr_idx_ = nullptr;     // (max_seeds*12) CSR triangle IDs
    // Scratch sort keys for rebuild_csr_and_upload_, paired 1:1 with
    // d_csr_idx_'s values during the sort; not meaningful afterwards.
    int32_t* d_csr_pair_seed_ = nullptr; // (max_seeds*12)
    int32_t* d_updated_flag_ = nullptr;// (1)     BFS convergence flag
    int32_t* d_mask_ = nullptr;        // (H*W)   reused for border / reassign masks
    // Changes since the last finalise, as opposed to d_changed_, which holds
    // only the current insert's.  Detection is scoped by the latter so a
    // deferred round does not re-detect earlier rounds' regions; assignment is
    // scoped by the former because it has not run for any of them yet.
    int32_t* d_dirty_accum_ = nullptr; // (H*W)   union of d_changed_ since last finalise
    int32_t* d_tile_dirty_ = nullptr;  // (tiles) tile-level dirty flags, mask prefilter
    int32_t* d_count_ = nullptr;       // (1)     dirty-pixel counter for the cost switch
    // Persistent scratch, allocated once here rather than cudaMalloc'd and
    // freed per call: all are bounded by max_seeds or the triangle bound, so
    // the size is known at construction, and a cudaFree synchronises the
    // device -- a steep price to pay per call for a fixed-size buffer.
    int32_t* d_tri_count_ = nullptr;   // (1)     detection output counter
    int32_t* d_seed_stage_ = nullptr;  // (3 * max_seeds) staged x, y, id for an insert
    int32_t* d_remap_ = nullptr;       // (max triangles) old id -> new, for compaction
    int32_t* d_edge_out_ = nullptr;    // (2 * 3 * max triangles) unpacked edge pairs
    uint8_t* d_stale_ = nullptr;       // (max triangles) per-triangle invalidation flags
    uint8_t* d_dead_ = nullptr;        // (max triangles) retired-slot flags, authoritative
    float*   d_values_ = nullptr;      // (max_seeds) scalar field, one per seed
    uint64_t* d_score_keys_ = nullptr; // (3 * max triangles) packed (score, edge index)
    float*   d_scores_ = nullptr;      // (3 * max triangles) one per edge
    int64_t* d_mid_keys_ = nullptr;    // (max_seeds) packed midpoint pixel + edge index
    int32_t* d_mid_count_ = nullptr;   // (1)
    int      tiles_x_, tiles_y_;
    // d_centroid_index_'s dimensions: 3x the padded canvas per axis, since it
    // is indexed by an unrounded x3-scaled coordinate sum. See d_centroid_index_.
    int      centroid_index_w_, centroid_index_h_;
    // "How many slots the registry has handed out", live or dead -- what
    // h_triangles_.size() used to mean, the next tid append_triangles_kernel
    // will hand out. Only ever adjusted by cheap arithmetic (+= n_appended,
    // -= n_stale, or set directly by full_topology_/compact_registry_),
    // never by inspecting individual triangles.
    int      next_tid_host_;
    bool     pending_;       // deferred inserts awaiting a finalise
    // Derived structures whose only consumers run at assignment or output
    // time. Rebuilding them per insert cost O(N_tri + N_seeds) of host work
    // that a deferred round never read; they are now rebuilt on demand.
    bool             csr_dirty_;
    mutable bool     sorted_rank_dirty_;
    // The deduplicated edge list, cached in d_edge_keys_. get_edges,
    // edge_scores and select_midpoints all index it, so they must see one
    // list; rebuilding it per call would also sort 3T keys three times a round.
    mutable bool     edges_dirty_;
    mutable int      n_edges_;
    bool             have_values_;

    // Live slot count. d_raw_buf_/d_dead_ (device) are the registry itself --
    // see their own comments above -- this is the one piece of triangle-count
    // bookkeeping still kept host-side, and it is cheap arithmetic only
    // (+= n_appended, -= n_stale), never a per-triangle read.
    //
    // Slots are not a dense list. A partial update retires the triangles the
    // change invalidated and appends their replacements; retiring is a flag,
    // so the ids of everything else stay put and d_centroid_index_ only sees
    // the entries that actually moved. Compacting instead would renumber
    // every triangle, which forces a full index rebuild, a full device
    // upload and a remap of the pixel grid -- all O(total triangles) on a
    // change that touched a handful of them. Density is restored in
    // compact_registry_(), which finalise() calls once per frame rather
    // than once per insert.
    int                                    n_live_;

    // ---- host-side seed registry ----
    std::vector<int32_t>            h_sx_, h_sy_;
    // Mirrored rather than read back: the values pass through the host on
    // their way in, so keeping them costs a copy and saves a transfer.
    std::vector<float>              h_values_;
    std::unordered_set<uint64_t>    h_seed_set_;   // fast duplicate check

    // Internal seed ids are assigned in INSERTION order, because previously
    // inserted seeds must keep their ids for the incremental device state
    // (d_grid_, d_t_grid_, the triangle registry and the CSR) to stay valid.
    // The batch pipeline instead numbers seeds in sorted (x asc, y asc) order.
    // h_sorted_rank_[internal_id] gives that sorted id, and outputs are
    // translated through it so both pipelines expose the same numbering.
    mutable std::vector<int32_t>    h_sorted_rank_;

    // ---- private helpers ----
    // Frees every buffer above via CUDA_CHECK_NOTHROW and is safe to call on a
    // partially-populated set (unallocated members are still null, and
    // cudaFree(nullptr) is a no-op). Shared by the destructor and by the
    // constructor's catch block, so a cudaMalloc failure partway through
    // construction frees what already succeeded instead of leaking it.
    void free_device_buffers_() noexcept;
    void rebuild_sorted_rank_() const;
    void ensure_sorted_rank_() const;
    void ensure_csr_();
    void run_bfs_(float* bfs_ms_out, int* iters_out = nullptr);
    // Topology only: detect + dedup + registry + CSR. Pixel assignment is
    // separate so it can be deferred across several inserts and run once.
    void full_topology_(float* detect_ms, float* dedup_ms);
    void partial_topology_(float* detect_ms, float* dedup_ms);
    // Detect into d_detect_buf_ and deduplicate on the device, scoped to mask
    // (nullptr for the whole grid); returns the surviving count. The one part
    // full_topology_ and partial_topology_ do identically, aside from that scope.
    int  detect_and_dedup_(const int32_t* mask, float* detect_ms, float* dedup_ms);
    // Registration + seed write + BFS, shared by insert() and insert_deferred().
    void apply_batch_(const std::vector<int32_t>& new_xs,
                      const std::vector<int32_t>& new_ys,
                      float* bfs_ms_out, int* iters_out = nullptr);
    // Pixel assignment over d_dirty_accum_, masked or full by measured cost.
    void assign_pending_(float* assign_ms);
    void build_reassign_mask_();
    int  count_mask_();
    void rebuild_csr_and_upload_();
    // Squeeze the retired slots out and renumber, entirely device-side: an
    // exclusive scan over live/dead gives each survivor's new dense tid
    // (which doubles as d_remap_'s content for remap_tgrid_kernel), a scatter
    // kernel writes survivors into d_raw_buf_compact_, then the two buffers
    // swap. O(total triangles), so it runs when the holes are worth the
    // pass, and before outputs are built -- tri_map is indexed by triangle id
    // and must be dense for callers.
    void compact_registry_();
    bool should_compact_() const;
    // Streams the tids in [0, old_count) that mark_stale_kernel flagged and
    // are not already dead into d_stale_tids_, on the device. Returns the
    // count. partial_topology_'s retire kernel touches only that many,
    // instead of every slot in the registry.
    int  compact_stale_tids_(int old_count);
    // Clears d_centroid_index_ and repopulates it from d_raw_buf_[0, count),
    // tid == own index. Used wherever a contiguous, freshly-tid'd run of
    // triangles just replaced the registry: full_topology_ (a first insert
    // has nothing else to seed it from) and compact_registry_ (renumbering
    // invalidates every tid the index was holding).
    void rebuild_centroid_index_(int count);
    // Host snapshot of d_raw_buf_/d_dead_[0, next_tid_host_), downloaded
    // fresh on every call, for the output builders not yet reading the
    // device registry directly (build_tri_map_, get_triangles). These run
    // once per finalise()/get_triangles() call.
    void download_registry_(std::vector<RawTriangle>& tris,
                            std::vector<uint8_t>& dead) const;
    void ensure_edges_() const;
    // build_outputs_()'s per-triangle vertex ids, translated through
    // sorted_rank(), downloading the registry (download_registry_) to do
    // it host-side.
    void build_tri_map_(std::vector<TriangleEntry>& tri_map_out) const;
    void build_outputs_(std::vector<TriangleEntry>& tri_map_out,
                        std::vector<int32_t>& tgrid_out) const;
    // As build_outputs_(), but launches kernels into the persistent
    // device_pixel_*()/device_triangle_verts() buffers instead of
    // downloading and cropping on the host.
    void build_outputs_device_() const;
    // insertion-order seed/vertex id -> batch pipeline's sorted (x asc, y asc)
    // id, or the id unchanged if out of range. Shared by build_tri_map_,
    // build_outputs_ and get_voronoi_grid, which all translate through
    // h_sorted_rank_ the same way; each still calls ensure_sorted_rank_()
    // itself first, since this is a pure lookup.
    int32_t translate_to_sorted_rank_(int32_t internal) const;

    static uint64_t pack_xy_(int32_t x, int32_t y) {
        return (uint64_t(uint32_t(x)) << 32) | uint64_t(uint32_t(y));
    }
};
