// Pixel assignment: which triangle covers each pixel, one pass over
// everything deferred since the last finalise(). The expensive stage, and the
// only one whose dirty region saturates once a batch is large and scattered
// -- see delaunay.cu's file comment for why it is decoupled from topology.
#include "delaunay.cuh"
#include "delaunay_locate.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

//: Above this share of the image a masked assignment costs more than an
//: unmasked one -- the per-thread mask read and the scattered writes outweigh
//: the work skipped -- so assign_pending_ switches to a full pass.
static constexpr float ASSIGN_FULL_FRACTION = 0.5f;

__global__
void assign_triangles_kernel(
    int32_t* __restrict__           t_grid,
    int W, int H,
    const int32_t* __restrict__     grid,       // interleaved (seed_id, dist)
    const RawTriangle* __restrict__ triangles,
    const int32_t* __restrict__     seed_xs,
    const int32_t* __restrict__     seed_ys,
    const int32_t* __restrict__     csr_ptr,
    const int32_t* __restrict__     csr_idx,
    int N_seeds, int N_triangles,
    const int32_t* __restrict__     mask)       // nullptr -> all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask && !mask[y * W + x]) return;

    t_grid[y * W + x] = locate_at(x, y, W, H, grid, triangles,
                                  seed_xs, seed_ys, csr_ptr, csr_idx, N_seeds);
}

// ---------------------------------------------------------------------------
// Kernels: mask construction
//
// All GPU-side rather than a host double loop over all W*H pixels: dilating on
// the CPU would need d_changed_ downloaded, processed, and two masks uploaded
// again -- every insert, whatever the batch size.
// ---------------------------------------------------------------------------

// dst |= src
__global__
void or_mask_kernel(const int32_t* __restrict__ src,
                    int32_t* __restrict__ dst, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && src[i]) dst[i] = 1;
}

// One flag per MASK_TILE x MASK_TILE block: set if any pixel in it changed.
__global__
void build_tile_dirty_kernel(const int32_t* __restrict__ src,
                             int32_t* __restrict__ tiles,
                             int W, int H, int TX)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (!src[y * W + x]) return;
    atomicOr(&tiles[(y / MASK_TILE) * TX + (x / MASK_TILE)], 1);
}

// Which pixels need their triangle re-assigned.
//
// Dilating by the uniform WINDOW_CAP would be far wider than necessary:
// assign_triangles_kernel searches a window of
// R = min(sqrt(dist) + SLACK, CAP), which at realistic seed densities is
// nearer 7 than 20 -- roughly nine times the area. Each pixel is therefore
// tested against its OWN R, via a tile prefilter so the scan stays cheap.
//
// That alone would not be safe: a pixel's triangle can be deleted by a change
// outside its search window, and no amount of dilation around changed cells
// catches that. remap_tgrid_kernel writes NO_TRIANGLE at exactly those pixels
// when the registry compacts, so the sentinel is tested directly and closes
// the gap. The cost is retesting the hull exterior, which carries it always.
__global__
void build_reassign_mask_kernel(const int32_t* __restrict__ grid,
                                const int32_t* __restrict__ t_grid,
                                const int32_t* __restrict__ tiles,
                                int32_t* __restrict__ mask,
                                int W, int H, int TX)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    if (t_grid[idx] == NO_TRIANGLE) { mask[idx] = 1; return; }

    int dist = grid[idx * 2 + 1];
    int R    = min((int)sqrtf((float)max(dist, 0)) + WINDOW_SLACK, WINDOW_CAP);

    int tx0 = max(0, x-R) / MASK_TILE, tx1 = min(W-1, x+R) / MASK_TILE;
    int ty0 = max(0, y-R) / MASK_TILE, ty1 = min(H-1, y+R) / MASK_TILE;
    int32_t v = 0;
    for (int ty = ty0; ty <= ty1 && !v; ++ty)
        for (int tx = tx0; tx <= tx1; ++tx)
            if (tiles[ty * TX + tx]) { v = 1; break; }
    mask[idx] = v;
}

void Delaunay::build_reassign_mask_()
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    cudaMemset(d_tile_dirty_, 0,
               (size_t)tiles_x_ * tiles_y_ * sizeof(int32_t));
    build_tile_dirty_kernel<<<grid_dim, block>>>(
        d_dirty_accum_, d_tile_dirty_, W_det_, H_det_, tiles_x_);
    cudaDeviceSynchronize();

    build_reassign_mask_kernel<<<grid_dim, block>>>(
        d_grid_, d_t_grid_, d_tile_dirty_, d_mask_, W_det_, H_det_, tiles_x_);
    cudaDeviceSynchronize();
}

int Delaunay::count_mask_()
{
    thrust::device_ptr<int32_t> p(d_mask_);
    return (int)thrust::reduce(p, p + (size_t)W_det_ * H_det_, (int32_t)0);
}

void Delaunay::assign_pending_(float* asgn_ms)
{
    const int N = W_det_ * H_det_;
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    int N_tri = (int)h_triangles_.size();
    if (N_tri == 0) {
        cudaMemset(d_t_grid_, SENTINEL_BYTE, (size_t)N * sizeof(int32_t));
        if (asgn_ms) *asgn_ms = 0.f;
        return;
    }

    ensure_csr_();
    build_reassign_mask_();

    // Masking is only worth its own overhead while it excludes enough pixels.
    // Past that the mask read and the scattered writes cost more than the work
    // they save, so hand the kernel a null mask and let it run coalesced.
    const int dirty = count_mask_();
    const bool use_mask = dirty < (int)(ASSIGN_FULL_FRACTION * (float)N);

    cudaEvent_t e0, e1;
    if (asgn_ms) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0); }

    assign_triangles_kernel<<<grid_dim, block>>>(
        d_t_grid_, W_det_, H_det_, d_grid_,
        static_cast<RawTriangle*>(d_raw_buf_),
        d_sx_, d_sy_, d_csr_ptr_, d_csr_idx_, N_, N_tri,
        use_mask ? d_mask_ : nullptr);
    cudaDeviceSynchronize();

    if (asgn_ms) {
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(asgn_ms, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
}
