// CUDA implementation of Delaunay: construction, buffer lifetime, and the
// insert/finalise orchestration. Everything each stage actually does lives
// beside its own kernels -- delaunay_voronoi.cu (seed write + BFS),
// delaunay_topology.cu (detect/dedup/registry/CSR), delaunay_assign.cu
// (pixel assignment + its masks), delaunay_scalar_field.cu (edge scores and
// midpoint selection) and delaunay_query.cu (locate/in_circumsphere) -- this
// file is the spine that calls them in the right order.
//
// The work splits into topology (cheap, scoped to the changed region) and pixel
// assignment (expensive, and the only stage whose dirty region saturates once a
// batch is large and scattered). They are separate calls so that a caller
// inserting repeatedly pays assignment once rather than per insert:
//
//   insert_deferred(batch)  ->  topology only, changes accumulate
//   finalise()              ->  one assignment pass + outputs
//   insert(batch)            ->  the two together, unchanged behaviour
//
// insert_deferred(batch):
//   1. Write new seeds into d_grid_ at distance 0.
//   2. BFS until convergence; d_changed_ collects every cell that moved, and is
//      OR-ed into d_dirty_accum_, the union awaiting assignment.
//   3. First insert -> full topology (detect over the whole grid).
//      Subsequent inserts -> partial topology:
//        a. Dilate d_changed_ by 2 px -> border mask (re-detect here).
//        b. Flag triangles whose canonical pixel is in border, by sampling the
//           mask on the device at those positions.
//        c. Re-detect in border, merge new triplets, compact the registry.
//        d. Remap d_t_grid_ through old->new; deleted triangles leave
//           NO_TRIANGLE.
//
// finalise():
//   5. Build the reassign mask from d_dirty_accum_: per-pixel radius
//      R = min(sqrt(dist)+SLACK, CAP) via a tile prefilter, plus every pixel
//      left at NO_TRIANGLE by the remap.
//   6. Assign, masked or full according to how much of the image is dirty.
//   7. Materialise tri_map and the (H,W,3) grid.

#include "delaunay.cuh"
#include "triangle_detect.cuh"

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Defined in delaunay_voronoi.cu and delaunay_assign.cu respectively;
// apply_batch_ below is their only caller in this file.
__global__ void write_seeds_kernel(int32_t* grid, int32_t* changed_mask, int W,
                                   const int32_t* xs, const int32_t* ys,
                                   const int32_t* ids, int k);
__global__ void or_mask_kernel(const int32_t* __restrict__ src,
                               int32_t* __restrict__ dst, int N);

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Delaunay::Delaunay(int width, int height, int max_seeds,
                                         int border_padding)
    : W_(width), H_(height), N_(0), max_seeds_(max_seeds), pending_(false),
      n_live_(0), csr_dirty_(true), sorted_rank_dirty_(true),
      edges_dirty_(true), n_edges_(0), have_values_(false), generation_(0)
{
    if (width <= 0 || height <= 0 || max_seeds <= 0)
        throw std::invalid_argument("dimensions and max_seeds must be positive");

    // Fixed default rather than a density estimate; see the header and
    // BORDER_PADDING_BOUND.md.
    P_ = border_padding >= 0 ? border_padding : DEFAULT_BORDER_PADDING;
    W_det_ = W_ + 2 * P_;
    H_det_ = H_ + 2 * P_;

    const int N = W_det_ * H_det_;
    tiles_x_ = (W_det_ + MASK_TILE - 1) / MASK_TILE;
    tiles_y_ = (H_det_ + MASK_TILE - 1) / MASK_TILE;

    // Every member pointer above defaults to nullptr, so on a mid-construction
    // failure free_device_buffers_() can safely free whichever of these
    // succeeded before the one that threw -- cudaFree(nullptr) is a no-op.
    try {
        CUDA_CHECK(cudaMalloc(&d_grid_,         (size_t)N * 2          * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tmp_,          (size_t)N * 2          * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_changed_,      (size_t)N              * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sx_,           (size_t)max_seeds      * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sy_,           (size_t)max_seeds      * sizeof(int32_t)));
        // This buffer serves two purposes and must satisfy both bounds. Detection
        // writes at most two triangles per 2x2 block over (W-1)*(H-1) blocks -- the
        // same exact bound the batch path uses. It then also holds the
        // deduplicated triangle list, under 2n for n seeds by planarity, which
        // only exceeds the detection bound on a canvas smaller than the seed
        // budget.
        CUDA_CHECK(cudaMalloc(&d_raw_buf_, (size_t)max_seeds * 4 * sizeof(RawTriangle)));
        CUDA_CHECK(cudaMalloc(&d_detect_buf_,
                   max_raw_triangles(W_det_, H_det_) * sizeof(RawTriangle)));
        CUDA_CHECK(cudaMalloc(&d_t_grid_,       (size_t)N              * sizeof(int32_t)));
        // Sized on the unpadded image, unlike the buffers above: these are the
        // finalise_device() outputs, addressed in image space by the crop kernel.
        CUDA_CHECK(cudaMalloc(&d_sorted_rank_,     (size_t)max_seeds       * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_pixel_tids_,      (size_t)W_ * H_         * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_pixel_seed_ids_,  (size_t)W_ * H_         * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_outside_mask_,    (size_t)W_ * H_         * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_csr_ptr_,      (size_t)(max_seeds + 1)* sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_csr_idx_,      (size_t)max_seeds * 8  * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_updated_flag_, 1                      * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_mask_,         (size_t)N              * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_dirty_accum_,  (size_t)N              * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tile_dirty_,
                   (size_t)tiles_x_ * tiles_y_ * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_count_,        1                      * sizeof(int32_t)));
        // A planar triangulation of n points has under 2n triangles; the detection
        // buffer is sized the same way the CSR is, so match that bound.
        CUDA_CHECK(cudaMalloc(&d_stale_,        (size_t)max_seeds * 4  * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_dead_,         (size_t)max_seeds * 4  * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_values_,       (size_t)max_seeds      * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_score_keys_,   (size_t)max_seeds * 12 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_tri_count_,    1                      * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_seed_stage_,   (size_t)max_seeds * 3  * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_remap_,        (size_t)max_seeds * 4  * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_edge_out_,     (size_t)max_seeds * 24 * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_scores_,       (size_t)max_seeds * 12 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mid_keys_,     (size_t)max_seeds      * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_mid_count_,    1                      * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_dead_, 0,       (size_t)max_seeds * 4  * sizeof(uint8_t)));
        // Three edge keys per triangle, over the same triangle bound as d_stale_.
        CUDA_CHECK(cudaMalloc(&d_edge_keys_,    (size_t)max_seeds * 4 * 3 * sizeof(int64_t)));

        CUDA_CHECK(cudaMemset(d_grid_,   SENTINEL_BYTE, (size_t)N * 2 * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_t_grid_, SENTINEL_BYTE, (size_t)N     * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_changed_,  0, (size_t)N     * sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d_dirty_accum_, 0, (size_t)N  * sizeof(int32_t)));
    } catch (...) {
        free_device_buffers_();
        throw;
    }
}

Delaunay::~Delaunay()
{
    free_device_buffers_();
}

void Delaunay::free_device_buffers_() noexcept
{
    CUDA_CHECK_NOTHROW(cudaFree(d_grid_));    CUDA_CHECK_NOTHROW(cudaFree(d_tmp_));      CUDA_CHECK_NOTHROW(cudaFree(d_changed_));
    CUDA_CHECK_NOTHROW(cudaFree(d_sx_));      CUDA_CHECK_NOTHROW(cudaFree(d_sy_));        CUDA_CHECK_NOTHROW(cudaFree(d_raw_buf_));
    CUDA_CHECK_NOTHROW(cudaFree(d_detect_buf_));
    CUDA_CHECK_NOTHROW(cudaFree(d_t_grid_));  CUDA_CHECK_NOTHROW(cudaFree(d_csr_ptr_));  CUDA_CHECK_NOTHROW(cudaFree(d_csr_idx_));
    CUDA_CHECK_NOTHROW(cudaFree(d_sorted_rank_));    CUDA_CHECK_NOTHROW(cudaFree(d_pixel_tids_));
    CUDA_CHECK_NOTHROW(cudaFree(d_pixel_seed_ids_)); CUDA_CHECK_NOTHROW(cudaFree(d_outside_mask_));
    CUDA_CHECK_NOTHROW(cudaFree(d_edge_keys_));
    CUDA_CHECK_NOTHROW(cudaFree(d_dead_));
    CUDA_CHECK_NOTHROW(cudaFree(d_values_));   CUDA_CHECK_NOTHROW(cudaFree(d_scores_));  CUDA_CHECK_NOTHROW(cudaFree(d_score_keys_));
    CUDA_CHECK_NOTHROW(cudaFree(d_mid_keys_)); CUDA_CHECK_NOTHROW(cudaFree(d_mid_count_));
    CUDA_CHECK_NOTHROW(cudaFree(d_updated_flag_)); CUDA_CHECK_NOTHROW(cudaFree(d_mask_));
    CUDA_CHECK_NOTHROW(cudaFree(d_dirty_accum_));  CUDA_CHECK_NOTHROW(cudaFree(d_tile_dirty_)); CUDA_CHECK_NOTHROW(cudaFree(d_count_));
    CUDA_CHECK_NOTHROW(cudaFree(d_stale_));
    CUDA_CHECK_NOTHROW(cudaFree(d_tri_count_));  CUDA_CHECK_NOTHROW(cudaFree(d_seed_stage_));
    CUDA_CHECK_NOTHROW(cudaFree(d_remap_));      CUDA_CHECK_NOTHROW(cudaFree(d_edge_out_));

    d_grid_ = d_tmp_ = d_changed_ = d_sx_ = d_sy_ = nullptr;
    d_raw_buf_ = d_detect_buf_ = d_edge_keys_ = nullptr;
    d_t_grid_ = d_csr_ptr_ = d_csr_idx_ = nullptr;
    d_sorted_rank_ = d_pixel_tids_ = d_pixel_seed_ids_ = nullptr;
    d_outside_mask_ = nullptr;
    d_dead_ = nullptr;
    d_values_ = nullptr;
    d_scores_ = nullptr;
    d_score_keys_ = nullptr;
    d_mid_keys_ = nullptr;
    d_mid_count_ = d_updated_flag_ = d_mask_ = nullptr;
    d_dirty_accum_ = d_tile_dirty_ = d_count_ = nullptr;
    d_stale_ = nullptr;
    d_tri_count_ = d_seed_stage_ = d_remap_ = d_edge_out_ = nullptr;
}

// ---------------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------------

void Delaunay::apply_batch_(
    const std::vector<int32_t>& new_xs,
    const std::vector<int32_t>& new_ys,
    float* bfs_ms_out, int* iters_out)
{
    int k = (int)new_xs.size();

    if (N_ + k > max_seeds_)
        throw std::invalid_argument("insert would exceed max_seeds capacity");

    // Validate bounds
    for (int i = 0; i < k; ++i)
        if (new_xs[i] < 0 || new_xs[i] >= W_ || new_ys[i] < 0 || new_ys[i] >= H_)
            throw std::invalid_argument("seed coordinate out of bounds");

    // Duplicate checks, within the batch and against existing seeds, both
    // through a hash set: an O(k^2) pairwise check would be invisible for the
    // handful of seeds the tests insert but quadratic for the thousands a
    // refinement round adds.
    {
        std::unordered_set<uint64_t> batch_seen;
        batch_seen.reserve(k * 2);
        for (int i = 0; i < k; ++i) {
            uint64_t key = pack_xy_(new_xs[i], new_ys[i]);
            if (!batch_seen.insert(key).second)
                throw std::invalid_argument("duplicate seed positions within batch");
            if (h_seed_set_.count(key))
                throw std::invalid_argument("seed position already exists");
        }
    }

    // Register seeds
    std::vector<int32_t> new_ids(k);
    for (int i = 0; i < k; ++i) {
        new_ids[i] = N_ + i;
        h_sx_.push_back(new_xs[i]);
        h_sy_.push_back(new_ys[i]);
        h_seed_set_.insert(pack_xy_(new_xs[i], new_ys[i]));
    }
    N_ += k;
    // Not rebuilt here. h_sorted_rank_ is read only by build_outputs_ and
    // get_voronoi_grid, so an O(N log N) sort of every seed on every insert
    // was work no deferred round could observe.
    sorted_rank_dirty_ = true;

    // Upload new seed positions, shifted onto the padded canvas: every device
    // grid is in padded coordinates, so the seed coordinates the kernels
    // compare against must be too. The host copies stay unpadded, which is what
    // rebuild_sorted_rank_ and the outputs want -- and a uniform shift preserves
    // lexicographic order, so the sorted ids are the same either way.
    std::vector<int32_t> padded_xs(k), padded_ys(k);
    for (int i = 0; i < k; ++i) {
        padded_xs[i] = new_xs[i] + P_;
        padded_ys[i] = new_ys[i] + P_;
    }
    CUDA_CHECK(cudaMemcpy(d_sx_ + N_ - k, padded_xs.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy_ + N_ - k, padded_ys.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Reset change accumulator before writing seeds (so seed positions
    // are the first entries in d_changed_ — necessary for partial_topology_
    // to cover detection positions that touch the seed cell directly).
    // d_changed_ is per-insert and scopes detection; d_dirty_accum_ below is
    // the union since the last finalise and scopes assignment.
    CUDA_CHECK(cudaMemset(d_changed_, 0, (size_t)W_det_ * H_det_ * sizeof(int32_t)));

    // Write seeds into grid (also marks seed cells in d_changed_)
    int32_t* d_kxs  = d_seed_stage_;
    int32_t* d_kys  = d_seed_stage_ + max_seeds_;
    int32_t* d_kids = d_seed_stage_ + 2 * max_seeds_;
    CUDA_CHECK(cudaMemcpy(d_kxs,  padded_xs.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kys,  padded_ys.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kids, new_ids.data(),    k * sizeof(int32_t), cudaMemcpyHostToDevice));
    write_seeds_kernel<<<(k+255)/256, 256>>>(d_grid_, d_changed_, W_det_, d_kxs, d_kys, d_kids, k);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // BFS
    run_bfs_(bfs_ms_out, iters_out);

    // Fold this insert's changes into the set awaiting assignment.
    const int N = W_det_ * H_det_;
    or_mask_kernel<<<(N + 255) / 256, 256>>>(d_changed_, d_dirty_accum_, N);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// insert_deferred / finalise / insert
// ---------------------------------------------------------------------------

void Delaunay::insert_deferred(
    const std::vector<int32_t>& new_xs,
    const std::vector<int32_t>& new_ys,
    InsertTimings*          timings,
    const std::vector<float>*   new_values)
{
    if (new_values && new_values->size() != new_xs.size())
        throw std::invalid_argument(
            "values must have one entry per seed in the batch");
    const int seeds_before = N_;
    int k = (int)new_xs.size();
    if (k == 0) return;

    // Invalidates any finalise_device() view outstanding from before this
    // call: the source grids it read are about to change under it.
    ++generation_;

    bool is_first = (N_ == 0);

    float bfs_ms = 0.f;
    int bfs_iters = 0;
    apply_batch_(new_xs, new_ys, timings ? &bfs_ms : nullptr,
                 timings ? &bfs_iters : nullptr);
    if (timings) { timings->bfs_ms = bfs_ms; timings->bfs_iters = bfs_iters; }

    float det_ms = 0.f, dup_ms = 0.f;
    if (is_first)
        full_topology_(timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr);
    else
        partial_topology_(timings ? &det_ms : nullptr, timings ? &dup_ms : nullptr);

    if (new_values && !new_values->empty()) {
        CUDA_CHECK(cudaMemcpy(d_values_ + seeds_before, new_values->data(),
                   new_values->size() * sizeof(float), cudaMemcpyHostToDevice));
        h_values_.insert(h_values_.end(), new_values->begin(), new_values->end());
        have_values_ = true;
    }

    // The triangulation moved, so the cached edge list no longer describes it.
    edges_dirty_ = true;

    if (timings) {
        timings->detect_ms = det_ms;
        timings->dedup_ms  = dup_ms;
        timings->assign_ms = 0.f;
    }
    pending_ = true;
}

void Delaunay::finalise(
    std::vector<TriangleEntry>& tri_map_out,
    std::vector<int32_t>&       tgrid_out,
    InsertTimings*         timings)
{
    // tri_map is indexed by triangle id, so the registry has to be dense before
    // it is built. This is the one place that requires it, which is why inserts
    // are free to leave holes: a refinement pays this once per frame instead of
    // once per round. Runs before assignment so the CSR the assignment reads is
    // rebuilt against the compacted ids.
    compact_registry_();

    if (pending_) {
        float asgn_ms = 0.f;
        assign_pending_(timings ? &asgn_ms : nullptr);
        if (timings) timings->assign_ms = asgn_ms;
        CUDA_CHECK(cudaMemset(d_dirty_accum_, 0, (size_t)W_det_ * H_det_ * sizeof(int32_t)));
        pending_ = false;
        // The grids finalise_device()'s buffers were cropped from just moved.
        ++generation_;
    }
    build_outputs_(tri_map_out, tgrid_out);
}

void Delaunay::finalise_device(std::vector<TriangleEntry>& tri_map_out)
{
    // One generation per call: even a call with nothing pending re-launches
    // the crop kernel and produces a view a caller should treat as new.
    ++generation_;

    compact_registry_();
    if (pending_) {
        assign_pending_(nullptr);
        CUDA_CHECK(cudaMemset(d_dirty_accum_, 0, (size_t)W_det_ * H_det_ * sizeof(int32_t)));
        pending_ = false;
    }
    build_outputs_device_(tri_map_out);
}

void Delaunay::insert(
    const std::vector<int32_t>& new_xs,
    const std::vector<int32_t>& new_ys,
    std::vector<TriangleEntry>&  tri_map_out,
    std::vector<int32_t>&        tgrid_out,
    InsertTimings*          timings)
{
    if (timings) *timings = InsertTimings{};
    insert_deferred(new_xs, new_ys, timings);
    finalise(tri_map_out, tgrid_out, timings);
}
