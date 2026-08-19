// Materialising results: the triangle map and (H,W,3) grid finalise() and
// finalise_device() return, get_voronoi_grid(), and the plain getters that
// read straight off the host registry.
#include "delaunay.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Lazy rebuilds of the derived structures
//
// The CSR is read only by assign_triangles_kernel and the sorted ranks only
// by the two output builders, so a deferred round pays for neither until one
// is actually needed.
// ---------------------------------------------------------------------------

void Delaunay::ensure_csr_()
{
    if (!csr_dirty_) return;
    rebuild_csr_and_upload_();
    csr_dirty_ = false;
}

void Delaunay::ensure_sorted_rank_() const
{
    if (!sorted_rank_dirty_) return;
    rebuild_sorted_rank_();
    sorted_rank_dirty_ = false;
}

// ---------------------------------------------------------------------------
// rebuild_sorted_rank_: internal (insertion) id -> batch (sorted x,y) id
// ---------------------------------------------------------------------------

void Delaunay::rebuild_sorted_rank_() const
{
    std::vector<int32_t> order(N_);
    for (int i = 0; i < N_; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [this](int32_t p, int32_t q) {
        if (h_sx_[p] != h_sx_[q]) return h_sx_[p] < h_sx_[q];
        return h_sy_[p] < h_sy_[q];
    });

    h_sorted_rank_.assign(N_, 0);
    for (int rank = 0; rank < N_; ++rank)
        h_sorted_rank_[order[rank]] = rank;

    // Mirrored for crop_pixel_arrays_kernel, which remaps seed ids on the
    // device instead of walking the padded canvas back through the host.
    if (N_ > 0)
        cudaMemcpy(d_sorted_rank_, h_sorted_rank_.data(),
                   (size_t)N_ * sizeof(int32_t), cudaMemcpyHostToDevice);
}

int32_t Delaunay::translate_to_sorted_rank_(int32_t internal) const
{
    return (internal >= 0 && internal < (int32_t)h_sorted_rank_.size())
         ? h_sorted_rank_[internal] : internal;
}

// ---------------------------------------------------------------------------
// build_outputs_
// ---------------------------------------------------------------------------

void Delaunay::build_tri_map_(std::vector<TriangleEntry>& tri_map_out) const
{
    ensure_sorted_rank_();
    int N_tri = (int)h_triangles_.size();

    tri_map_out.resize(N_tri);
    for (int tid = 0; tid < N_tri; ++tid) {
        const auto& t = h_triangles_[tid];
        // Canonical positions come back in image coordinates. A border triangle
        // lands outside [0,W)x[0,H) once shifted, which is correct and matches
        // the batch path: its circumcentre genuinely lies outside the image.
        tri_map_out[tid] = {t.x - P_, t.y - P_,
                            translate_to_sorted_rank_(t.orig_a),
                            translate_to_sorted_rank_(t.orig_b),
                            translate_to_sorted_rank_(t.orig_c)};
    }
}

void Delaunay::build_outputs_(std::vector<TriangleEntry>& tri_map_out,
                                         std::vector<int32_t>& tgrid_out) const
{
    build_tri_map_(tri_map_out);
    ensure_sorted_rank_();
    const int N     = W_ * H_;
    const int N_det = W_det_ * H_det_;

    std::vector<int32_t> h_t(N_det), h_grid(N_det * 2);
    cudaMemcpy(h_t.data(),    d_t_grid_, N_det     * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid.data(), d_grid_,   N_det * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Crop the padded canvas back to the image.
    tgrid_out.resize(N * 3);
    for (int y = 0; y < H_; ++y) {
        const int src_row = (y + P_) * W_det_ + P_;
        const int dst_row = y * W_;
        for (int x = 0; x < W_; ++x) {
            const int s = src_row + x, d = dst_row + x;
            tgrid_out[d*3]   = translate_to_sorted_rank_(h_grid[s*2]);
            tgrid_out[d*3+1] = h_grid[s*2+1];
            tgrid_out[d*3+2] = h_t[s];
        }
    }
}

//: The device counterpart of build_outputs_()'s crop loop: one thread per
//: image pixel, reading the padded d_grid_/d_t_grid_ exactly as the host loop
//: does and writing straight into the three finalise_device() buffers instead
//: of a host vector. Also bakes in the outside-hull substitution
//: (NO_TRIANGLE -> 0) at the pixel that needs it, since the mask is already
//: known there and a host-side consumer would otherwise redo it in NumPy.
__global__
void crop_pixel_arrays_kernel(const int32_t* __restrict__ grid,
                              const int32_t* __restrict__ t_grid,
                              const int32_t* __restrict__ sorted_rank,
                              int sorted_rank_size,
                              int W, int H, int P, int W_det,
                              int32_t* __restrict__ pixel_tids_out,
                              int32_t* __restrict__ pixel_seed_ids_out,
                              uint8_t* __restrict__ outside_mask_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const int s = (y + P) * W_det + P + x;
    const int d = y * W + x;

    const int32_t seed_id = grid[s * 2];
    pixel_seed_ids_out[d] = (seed_id >= 0 && seed_id < sorted_rank_size)
                           ? sorted_rank[seed_id] : seed_id;

    const int32_t tid = t_grid[s];
    const bool outside = (tid == NO_TRIANGLE);
    outside_mask_out[d] = outside ? 1 : 0;
    pixel_tids_out[d]   = outside ? 0 : tid;
}

void Delaunay::build_outputs_device_(std::vector<TriangleEntry>& tri_map_out) const
{
    build_tri_map_(tri_map_out);
    ensure_sorted_rank_();

    dim3 block(16, 16);
    dim3 grid_dim((W_ + 15) / 16, (H_ + 15) / 16);
    crop_pixel_arrays_kernel<<<grid_dim, block>>>(
        d_grid_, d_t_grid_, d_sorted_rank_, (int)h_sorted_rank_.size(),
        W_, H_, P_, W_det_,
        d_pixel_tids_, d_pixel_seed_ids_, d_outside_mask_);
}

// ---------------------------------------------------------------------------
// get_voronoi_grid
// ---------------------------------------------------------------------------

void Delaunay::get_voronoi_grid(std::vector<int32_t>& out) const
{
    ensure_sorted_rank_();
    const int N     = W_ * H_;
    const int N_det = W_det_ * H_det_;
    out.resize(N * 2);
    std::vector<int32_t> h_grid(N_det * 2);
    cudaMemcpy(h_grid.data(), d_grid_, N_det * 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Same insertion->sorted id translation as build_outputs_, cropping the
    // padded canvas back to the image.
    for (int y = 0; y < H_; ++y) {
        const int src_row = (y + P_) * W_det_ + P_;
        const int dst_row = y * W_;
        for (int x = 0; x < W_; ++x) {
            const int s = src_row + x, d = dst_row + x;
            out[d*2]   = translate_to_sorted_rank_(h_grid[s*2]);
            out[d*2+1] = h_grid[s*2+1];
        }
    }
}

// ---------------------------------------------------------------------------
// get_triangles: topology without touching the raster
// ---------------------------------------------------------------------------

void Delaunay::get_triangles(std::vector<TriangleEntry>& out) const
{
    const int slots = (int)h_triangles_.size();
    out.clear();
    out.reserve(n_live_);
    for (int tid = 0; tid < slots; ++tid) {
        if (h_dead_[tid]) continue;
        const auto& t = h_triangles_[tid];
        // Insertion-order ids, deliberately untranslated -- see the header.
        // Canonical position in image coordinates, as build_outputs_ reports it.
        out.push_back({t.x - P_, t.y - P_, t.orig_a, t.orig_b, t.orig_c});
    }
}

void Delaunay::get_seeds(std::vector<int32_t>& out) const
{
    // Already on the host and in insertion order: the registry keeps them so
    // rebuild_sorted_rank_ and the outputs can read them, so this is a copy
    // rather than a transfer.
    out.resize((size_t)N_ * 2);
    for (int i = 0; i < N_; ++i) {
        out[(size_t)i * 2]     = h_sx_[i];
        out[(size_t)i * 2 + 1] = h_sy_[i];
    }
}

void Delaunay::get_values(std::vector<float>& out) const
{
    out.assign(h_values_.begin(), h_values_.end());
}
