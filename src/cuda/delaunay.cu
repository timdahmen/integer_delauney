// CUDA implementation of Delaunay.
//
// The work splits into topology (cheap, scoped to the changed region) and pixel
// assignment (expensive, and the only stage whose dirty region saturates once a
// batch is large and scattered). They are separate calls so that a caller
// inserting repeatedly pays assignment once rather than per insert:
//
//   insert_deferred(batch)  →  topology only, changes accumulate
//   finalise()              →  one assignment pass + outputs
//   insert(batch)           →  the two together, unchanged behaviour
//
// insert_deferred(batch):
//   1. Write new seeds into d_grid_ at distance 0.
//   2. BFS until convergence; d_changed_ collects every cell that moved, and is
//      OR-ed into d_dirty_accum_, the union awaiting assignment.
//   3. First insert → full topology (detect over the whole grid).
//      Subsequent inserts → partial topology:
//        a. Dilate d_changed_ by 2 px → border mask (re-detect here).
//        b. Flag triangles whose canonical pixel is in border, by sampling the
//           mask on the device at those positions.
//        c. Re-detect in border, merge new triplets, compact the registry.
//        d. Remap d_t_grid_ through old→new; deleted triangles leave
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

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// UNDEF_SEED, RawTriangle, RawLess, RawEqual and the detection rule are shared
// with the batch path; see triangle_detect.cuh.

//: An entry in the old->new remap for a triangle that did not survive
//: compaction. Distinct from NO_TRIANGLE, which is what the *pixel* grid then
//: receives for those triangles: this one says "this id is gone", the other
//: says "no triangle covers this pixel".
static constexpr int32_t TID_DELETED = -1;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__
bool beats(int32_t a_id, int32_t a_d, int32_t b_id, int32_t b_d)
{
    if (b_d < a_d) return true;
    if (b_d == a_d && b_id > a_id) return true;
    return false;
}

__device__ __forceinline__
float cross2d(float ox, float oy, float ax, float ay, float bx, float by)
{
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ __forceinline__
bool point_in_triangle(float px, float py,
                       float ax, float ay, float bx, float by, float cx, float cy)
{
    float d1 = cross2d(px, py, ax, ay, bx, by);
    float d2 = cross2d(px, py, bx, by, cx, cy);
    float d3 = cross2d(px, py, cx, cy, ax, ay);
    bool neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
    bool pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
    return !(neg && pos);
}

// ---------------------------------------------------------------------------
// Kernel: write new seeds into the interleaved grid
// ---------------------------------------------------------------------------

__global__
void write_seeds_kernel(int32_t* grid, int32_t* changed_mask, int W,
                        const int32_t* xs, const int32_t* ys,
                        const int32_t* ids, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    int base = (ys[i] * W + xs[i]) * 2;
    grid[base]     = ids[i];
    grid[base + 1] = 0;
    // Mark the seed cell as changed so the border expansion
    // covers detection positions that touch the seed cell directly.
    atomicOr(&changed_mask[ys[i] * W + xs[i]], 1);
}

// ---------------------------------------------------------------------------
// Kernel: BFS step with per-cell change accumulation
// ---------------------------------------------------------------------------

__global__
void voronoi_step_kernel(
    const int32_t* __restrict__ src,
    int32_t* __restrict__       dst,
    int W, int H,
    int32_t* __restrict__ updated_flag,
    int32_t* __restrict__ changed_mask,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int base = (y * W + x) * 2;
    int32_t cur_id = src[base], cur_d = src[base + 1];
    int32_t best_id = cur_id, best_d = cur_d;

    const int dx[4] = {-1, 1,  0, 0};
    const int dy[4] = { 0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        int nx = x + dx[k], ny = y + dy[k];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
        int nb = (ny * W + nx) * 2;
        int32_t n_id = src[nb];
        if (n_id == UNDEF_SEED) continue;
        // Recomputed from the neighbour's owning seed, never accumulated along
        // the BFS path -- that would give a Manhattan metric.
        int32_t sdx = x - seed_xs[n_id];
        int32_t sdy = y - seed_ys[n_id];
        int32_t n_d = sdx * sdx + sdy * sdy;
        if (best_id == UNDEF_SEED || beats(best_id, best_d, n_id, n_d)) {
            best_id = n_id; best_d = n_d;
        }
    }

    dst[base]     = best_id;
    dst[base + 1] = best_d;

    if (best_id != cur_id || best_d != cur_d) {
        atomicOr(updated_flag, 1);
        atomicOr(&changed_mask[y * W + x], 1);
    }
}

// ---------------------------------------------------------------------------
// Kernel: 2x2-block triangle detection with optional mask
//
// The rule itself is shared with the batch path -- see triangle_detect.cuh,
// which also records why: this file once scanned four L-shaped stencils per
// pixel and emitted four overlapping triangles at a cocircular vertex where the
// quad admits two.
//
// What stays here is only what genuinely differs. The grid is interleaved
// (seed_id, distance) rather than plain seed ids, and detection can be scoped
// to a mask so a deferred round does not re-detect earlier rounds' regions.
// ---------------------------------------------------------------------------

__global__
void find_triangle_seeds_kernel(
    const int32_t* __restrict__ grid,   // interleaved (seed_id, dist)
    int W, int H,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys,
    RawTriangle* __restrict__ raw_buf,
    int32_t* __restrict__ counter,
    const int32_t* __restrict__ mask)   // nullptr → all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W - 1 || y >= H - 1) return;   // 2x2 block must be in bounds
    if (mask && !mask[y * W + x]) return;

    // Interleaved (seed_id, distance) grid here; the batch path's is a plain
    // seed-id array. That layout difference, and the mask above, are the only
    // things this kernel adds to the shared rule.
    auto sid = [&](int cx, int cy) -> int32_t { return grid[(cy * W + cx) * 2]; };

    detect_block_triangles(
        sid(x,     y    ),                  // top-left
        sid(x + 1, y    ),                  // top-right
        sid(x,     y + 1),                  // bottom-left
        sid(x + 1, y + 1),                  // bottom-right
        seed_xs, seed_ys,
        [&](int32_t oa, int32_t ob, int32_t oc) {
            append_raw_triangle(raw_buf, counter, x, y, oa, ob, oc);
        });
}

// ---------------------------------------------------------------------------
// Kernels: undirected edge extraction
//
// One triangle contributes three edges. Orienting each as (min, max) makes the
// two triangles sharing an edge produce the same key, so a sort and a unique
// collapse them -- the same shape as the triangle dedup above.
//
// The key packs both endpoints into one int64 so the sort is over a scalar.
// n_seeds is under 2^21 in any workable canvas and the product stays far inside
// int64, so the packing is exact and its order is lexicographic in (min, max).
// ---------------------------------------------------------------------------

//: Retired slots emit a key above every real one, so the sort parks them at the
//: end and unique collapses them to a single entry the caller drops. Simpler
//: than compacting the input, and the branch is uniform across a warp.
static constexpr int64_t EDGE_KEY_DEAD = INT64_MAX;

__global__
void build_edge_keys_kernel(const RawTriangle* __restrict__ tris, int n_tri,
                            int64_t n_seeds, const uint8_t* __restrict__ dead,
                            int64_t* __restrict__ keys)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;
    if (dead && dead[t]) {
        keys[(size_t)t * 3]     = EDGE_KEY_DEAD;
        keys[(size_t)t * 3 + 1] = EDGE_KEY_DEAD;
        keys[(size_t)t * 3 + 2] = EDGE_KEY_DEAD;
        return;
    }
    const RawTriangle& r = tris[t];
    const int32_t v[3] = {r.orig_a, r.orig_b, r.orig_c};
    for (int e = 0; e < 3; ++e) {
        int32_t p = v[e], q = v[(e + 1) % 3];
        if (p > q) { int32_t s = p; p = q; q = s; }
        keys[(size_t)t * 3 + e] = (int64_t)p * n_seeds + (int64_t)q;
    }
}

__global__
void unpack_edge_keys_kernel(const int64_t* __restrict__ keys, int n_edges,
                             int64_t n_seeds, int32_t* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;
    int64_t k = keys[i];
    out[(size_t)i * 2]     = (int32_t)(k / n_seeds);
    out[(size_t)i * 2 + 1] = (int32_t)(k % n_seeds);
}

// ---------------------------------------------------------------------------
// Kernels: the scalar field on the vertices
// ---------------------------------------------------------------------------

//: Two midpoint keys land on the same pixel. A functor rather than a lambda so
//: the file needs no --extended-lambda.
struct SamePixel {
    int64_t stride;
    __device__ bool operator()(int64_t x, int64_t y) const {
        return x / stride == y / stride;
    }
};

__device__ __forceinline__
void unpack_edge(int64_t key, int64_t n_seeds, int32_t& a, int32_t& b)
{
    a = (int32_t)(key / n_seeds);
    b = (int32_t)(key % n_seeds);
}

//: |dv| * |ab|, or zero for an edge too short to subdivide.
//:
//: Squared lengths are compared so the eligibility test needs no root; the
//: score itself does, since it is a length rather than a comparison.
__global__
void score_edges_kernel(const int64_t* __restrict__ keys, int n_edges,
                        int64_t n_seeds,
                        const int32_t* __restrict__ sx,
                        const int32_t* __restrict__ sy,
                        const float* __restrict__ values,
                        double min_len2,
                        float* __restrict__ scores)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    int32_t a, b;
    unpack_edge(keys[e], n_seeds, a, b);
    const double dx = (double)sx[a] - sx[b];
    const double dy = (double)sy[a] - sy[b];
    const double len2 = dx * dx + dy * dy;
    if (len2 < min_len2) { scores[e] = 0.f; return; }
    const double dv = fabs((double)values[a] - (double)values[b]);
    scores[e] = (float)(dv * sqrt(len2));
}

//: Pack (score, edge index) into one key that sorts best-first.
//:
//: A score is non-negative, and the IEEE-754 bit pattern of a non-negative
//: float increases monotonically read as an unsigned integer. Complementing it
//: reverses that, so ascending order on the key is descending score, and the
//: index in the low half breaks ties towards the lower edge -- a total order,
//: which is what makes "the best k" an exact count rather than a threshold with
//: ties to settle. Edges at or below the threshold get the largest possible
//: key and sort to the end, out of reach.
__global__
void pack_score_keys_kernel(const float* __restrict__ scores, int n,
                            float threshold, uint64_t* __restrict__ keys)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = scores[i];
    uint32_t bits = 0xFFFFFFFFu;                 // ineligible
    if (s > threshold && s > 0.f) {
        uint32_t b;
        memcpy(&b, &s, sizeof(b));
        bits = ~b;
    }
    keys[i] = ((uint64_t)bits << 32) | (uint32_t)i;
}

//: Midpoints of the first `take` edges of the sorted key array.
//:
//: The sort already chose them, so there is no predicate here -- the caller's
//: k arrives as a count and the selection is exact by construction.
__global__
void midpoints_from_sorted_kernel(const uint64_t* __restrict__ sorted, int take,
                                  const int64_t* __restrict__ edge_keys,
                                  int64_t n_seeds,
                                  const int32_t* __restrict__ sx,
                                  const int32_t* __restrict__ sy,
                                  int W_det, int H_det, int P,
                                  const int32_t* __restrict__ grid,
                                  int64_t* __restrict__ out,
                                  int32_t* __restrict__ count)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= take) return;
    const int32_t e = (int32_t)(sorted[r] & 0xFFFFFFFFu);

    int32_t a, b;
    unpack_edge(edge_keys[e], n_seeds, a, b);
    int mx = (int)lrint(0.5 * ((double)sx[a] + sx[b]));
    int my = (int)lrint(0.5 * ((double)sy[a] + sy[b]));
    mx = min(max(mx, P), W_det - 1 - P);
    my = min(max(my, P), H_det - 1 - P);

    if (grid[(my * W_det + mx) * 2 + 1] == 0) return;   // already a seed

    const int64_t pixel = (int64_t)my * W_det + mx;
    // Rank, not edge index: the sort ordered them, so keeping the lowest rank
    // among colliding midpoints keeps the better-scoring edge.
    out[atomicAdd(count, 1)] = pixel * (int64_t)take + r;
}

//: Count the edges a threshold admits, so `take` never exceeds them.
struct AboveThreshold {
    float t;
    __device__ bool operator()(float s) const { return s > t && s > 0.f; }
};

//: Take the edges clearing (min_score, tie_index), emit their midpoints.
//:
//: The ordering is score descending, edge index ascending on a tie, so the key
//: is unique and exactly as many edges clear it as the caller intended.
//:
//: A midpoint on a cell at distance zero is already a seed and is dropped here:
//: the Voronoi diagram is the record of what has been sampled, so no parallel
//: list of taken positions is needed. Survivors are packed with their edge
//: index so the collision pass that follows can keep a deterministic one.
__global__
void select_midpoints_kernel(const int64_t* __restrict__ keys, int n_edges,
                             int64_t n_seeds,
                             const int32_t* __restrict__ sx,
                             const int32_t* __restrict__ sy,
                             const float* __restrict__ values,
                             const float* __restrict__ scores,
                             float min_score, int32_t tie_index,
                             int W_det, int H_det, int P,
                             const int32_t* __restrict__ grid,
                             int64_t* __restrict__ out, int32_t* __restrict__ count)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;

    const float s = scores[e];
    if (s <= 0.f) return;
    if (!(s > min_score || (s == min_score && e <= tie_index))) return;

    int32_t a, b;
    unpack_edge(keys[e], n_seeds, a, b);
    // Padded coordinates throughout, like every other kernel here.
    int mx = (int)lrint(0.5 * ((double)sx[a] + sx[b]));
    int my = (int)lrint(0.5 * ((double)sy[a] + sy[b]));
    mx = min(max(mx, P), W_det - 1 - P);
    my = min(max(my, P), H_det - 1 - P);

    if (grid[(my * W_det + mx) * 2 + 1] == 0) return;   // already a seed

    const int64_t pixel = (int64_t)my * W_det + mx;
    out[atomicAdd(count, 1)] = pixel * (int64_t)n_edges + e;
}

//: Clear pixels whose triangle has been retired.
//:
//: The counterpart to remap_tgrid_kernel for the incremental path, where ids do
//: not move: a pixel is either still covered by the triangle it names, or that
//: triangle is gone and the pixel must be reassigned.
//: build_reassign_mask_kernel picks these up by testing for NO_TRIANGLE, which
//: is how a pixel is caught when the change that retired its triangle lay
//: outside its own search window.
__global__
void invalidate_dead_tgrid_kernel(int32_t* __restrict__ t_grid, int N,
                                  const uint8_t* __restrict__ dead, int n_slots)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t tid = t_grid[i];
    if (tid == NO_TRIANGLE) return;
    if (tid >= n_slots || dead[tid]) t_grid[i] = NO_TRIANGLE;
}

// ---------------------------------------------------------------------------
// Kernel: remap triangle IDs in t_grid (after compaction)
// ---------------------------------------------------------------------------

__global__
void remap_tgrid_kernel(int32_t* __restrict__ t_grid, int N,
                        const int32_t* __restrict__ remap, int remap_size,
                        int32_t fallback)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t old_tid = t_grid[i];
    // Already uncovered, out of range, or the triangle did not survive
    // compaction. The last case is why finalise() can reassign a pixel whose
    // own neighbourhood never changed: build_reassign_mask_kernel tests for the
    // fallback directly, since no dilation around changed cells would find it.
    if (old_tid == NO_TRIANGLE || old_tid >= remap_size
            || remap[old_tid] == TID_DELETED)
        t_grid[i] = fallback;
    else
        t_grid[i] = remap[old_tid];
}

// ---------------------------------------------------------------------------
// Kernel: window-based assignment with optional mask
// ---------------------------------------------------------------------------

static constexpr int WINDOW_SLACK = 3;
static constexpr int WINDOW_CAP   = 20;
static constexpr int MAX_NEARBY   = 64;

//: The triangle containing pixel (x, y), or NO_TRIANGLE.
//:
//: Shared by the per-pixel assignment and the point query below, which differ
//: only in which positions they ask about. Testing every triangle would be
//: hopeless, so the search narrows first: the distance channel gives the radius
//: within which a containing triangle's vertices must lie, the seed ids in that
//: window name the candidates, and the CSR turns each into its few triangles.
__device__ __forceinline__
int32_t locate_at(int x, int y, int W, int H,
                  const int32_t* __restrict__ grid,
                  const RawTriangle* __restrict__ triangles,
                  const int32_t* __restrict__ seed_xs,
                  const int32_t* __restrict__ seed_ys,
                  const int32_t* __restrict__ csr_ptr,
                  const int32_t* __restrict__ csr_idx,
                  int N_seeds)
{
    // Integer pixel convention, matching triangulation.cu and the NumPy
    // reference: pixel (x, y) IS the point (x, y), not its centre.
    const float px = (float)x, py = (float)y;
    // The distance channel holds SQUARED L2, so take a root to get back a
    // linear search radius in pixels.
    const int dist = grid[(y * W + x) * 2 + 1];
    const int R = min((int)sqrtf((float)max(dist, 0)) + WINDOW_SLACK, WINDOW_CAP);

    int32_t nearby[MAX_NEARBY];
    int n_nearby = 0;

    const int x0 = max(0, x-R), x1 = min(W-1, x+R);
    const int y0 = max(0, y-R), y1 = min(H-1, y+R);

    for (int sy = y0; sy <= y1; ++sy)
    for (int sx = x0; sx <= x1; ++sx) {
        int32_t sid = grid[(sy * W + sx) * 2];
        bool dup = false;
        for (int i = 0; i < n_nearby; ++i)
            if (nearby[i] == sid) { dup = true; break; }
        if (!dup && n_nearby < MAX_NEARBY)
            nearby[n_nearby++] = sid;
    }

    int32_t best = NO_TRIANGLE;
    for (int i = 0; i < n_nearby; ++i) {
        int32_t sid = nearby[i];
        if (sid < 0 || sid >= N_seeds) continue;
        for (int j = csr_ptr[sid]; j < csr_ptr[sid+1]; ++j) {
            int32_t tid = csr_idx[j];
            const RawTriangle& tri = triangles[tid];
            float ax = (float)seed_xs[tri.orig_a], ay = (float)seed_ys[tri.orig_a];
            float bx = (float)seed_xs[tri.orig_b], by = (float)seed_ys[tri.orig_b];
            float cx = (float)seed_xs[tri.orig_c], cy = (float)seed_ys[tri.orig_c];
            if (point_in_triangle(px, py, ax, ay, bx, by, cx, cy))
                if (best == NO_TRIANGLE || tid > best) best = tid;
        }
    }
    return best;
}

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
    const int32_t* __restrict__     mask)       // nullptr → all pixels
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    if (mask && !mask[y * W + x]) return;

    t_grid[y * W + x] = locate_at(x, y, W, H, grid, triangles,
                                  seed_xs, seed_ys, csr_ptr, csr_idx, N_seeds);
}

//: Which triangle contains each of a list of points.
//:
//: The same search, restricted to the points asked about instead of every
//: pixel. A caller that wants the containing triangle for a few thousand
//: positions would otherwise assign the whole canvas and read the answers back
//: out of it, which is the same work per point over ~20x more points.
//:
//: Coordinates are in image space; the padding shift is applied here, since the
//: grids live on the padded canvas and callers do not know about it.
__global__
void locate_points_kernel(const int32_t* __restrict__ qx,
                          const int32_t* __restrict__ qy,
                          int n, int P,
                          int W, int H,
                          const int32_t* __restrict__ grid,
                          const RawTriangle* __restrict__ triangles,
                          const int32_t* __restrict__ seed_xs,
                          const int32_t* __restrict__ seed_ys,
                          const int32_t* __restrict__ csr_ptr,
                          const int32_t* __restrict__ csr_idx,
                          int N_seeds,
                          int32_t* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = qx[i] + P;
    const int y = qy[i] + P;
    if (x < 0 || x >= W || y < 0 || y >= H) { out[i] = NO_TRIANGLE; return; }
    out[i] = locate_at(x, y, W, H, grid, triangles,
                       seed_xs, seed_ys, csr_ptr, csr_idx, N_seeds);
}

// ---------------------------------------------------------------------------
// Kernels: mask construction
//
// These replace a serial host double loop over all W*H pixels that copied
// d_changed_ down, dilated it on the CPU, and copied two masks back -- 18 MB of
// traffic and a megapixel scan per insert, whatever the batch size.
// ---------------------------------------------------------------------------

//: Tile edge for the dirty prefilter below. Small enough that a tile rarely
//: covers unchanged ground, large enough that a WINDOW_CAP-sized window spans
//: only a handful of tiles.
static constexpr int MASK_TILE = 8;

//: Above this share of the image a masked assignment costs more than an
//: unmasked one -- the per-thread mask read and the scattered writes outweigh
//: the work skipped -- so finalise() switches to a full pass.
static constexpr float ASSIGN_FULL_FRACTION = 0.5f;

// Dilate a 0/1 mask by a fixed radius. Used for the detection border, where
// the radius is 2 and a direct gather is cheaper than any prefilter.
__global__
void dilate_fixed_kernel(const int32_t* __restrict__ src,
                         int32_t* __restrict__ dst,
                         int W, int H, int r)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int x0 = max(0, x-r), x1 = min(W-1, x+r);
    int y0 = max(0, y-r), y1 = min(H-1, y+r);
    int32_t v = 0;
    for (int sy = y0; sy <= y1 && !v; ++sy)
        for (int sx = x0; sx <= x1; ++sx)
            if (src[sy * W + sx]) { v = 1; break; }
    dst[y * W + x] = v;
}

// Flag triangles whose canonical pixel falls inside the mask.
//
// Sampling the mask here rather than downloading it keeps the transfer
// proportional to the triangle count (~50k flags) instead of the pixel count
// (~1.5M int32). Reads d_raw_buf_, so it must run before detection overwrites
// that buffer.
__global__
void mark_stale_kernel(const RawTriangle* __restrict__ tris, int n_tri,
                       const int32_t* __restrict__ mask, int W, int H,
                       uint8_t* __restrict__ stale)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tri) return;
    const RawTriangle& r = tris[t];
    stale[t] = (r.x >= 0 && r.x < W && r.y >= 0 && r.y < H && mask[r.y * W + r.x])
             ? 1 : 0;
}

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
// Dilating by the uniform WINDOW_CAP, as the host version did, is far wider
// than necessary: assign_triangles_kernel searches a window of
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

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

Delaunay::Delaunay(int width, int height, int max_seeds,
                                         int border_padding)
    : W_(width), H_(height), N_(0), max_seeds_(max_seeds), pending_(false),
      n_live_(0), csr_dirty_(true), sorted_rank_dirty_(true),
      edges_dirty_(true), n_edges_(0), have_values_(false)
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
    cudaMalloc(&d_grid_,         (size_t)N * 2          * sizeof(int32_t));
    cudaMalloc(&d_tmp_,          (size_t)N * 2          * sizeof(int32_t));
    cudaMalloc(&d_changed_,      (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_sx_,           (size_t)max_seeds      * sizeof(int32_t));
    cudaMalloc(&d_sy_,           (size_t)max_seeds      * sizeof(int32_t));
    // This buffer serves two purposes and must satisfy both bounds. Detection
    // writes at most two triangles per 2x2 block over (W-1)*(H-1) blocks -- the
    // same exact bound the batch path uses, and half the 4-per-pixel it was
    // allocated at before, which is 200 MB at a real frame size against 100 MB.
    // It then also holds the deduplicated triangle list, under 2n for n seeds
    // by planarity, which only exceeds the detection bound on a canvas smaller
    // than the seed budget.
    cudaMalloc(&d_raw_buf_, (size_t)max_seeds * 4 * sizeof(RawTriangle));
    cudaMalloc(&d_detect_buf_,
               max_raw_triangles(W_det_, H_det_) * sizeof(RawTriangle));
    cudaMalloc(&d_t_grid_,       (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_csr_ptr_,      (size_t)(max_seeds + 1)* sizeof(int32_t));
    cudaMalloc(&d_csr_idx_,      (size_t)max_seeds * 8  * sizeof(int32_t));
    cudaMalloc(&d_updated_flag_, 1                      * sizeof(int32_t));
    cudaMalloc(&d_mask_,         (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_dirty_accum_,  (size_t)N              * sizeof(int32_t));
    cudaMalloc(&d_tile_dirty_,
               (size_t)tiles_x_ * tiles_y_ * sizeof(int32_t));
    cudaMalloc(&d_count_,        1                      * sizeof(int32_t));
    // A planar triangulation of n points has under 2n triangles; the detection
    // buffer is sized the same way the CSR is, so match that bound.
    cudaMalloc(&d_stale_,        (size_t)max_seeds * 4  * sizeof(uint8_t));
    cudaMalloc(&d_dead_,         (size_t)max_seeds * 4  * sizeof(uint8_t));
    cudaMalloc(&d_values_,       (size_t)max_seeds      * sizeof(float));
    cudaMalloc(&d_score_keys_,   (size_t)max_seeds * 12 * sizeof(uint64_t));
    cudaMalloc(&d_tri_count_,    1                      * sizeof(int32_t));
    cudaMalloc(&d_seed_stage_,   (size_t)max_seeds * 3  * sizeof(int32_t));
    cudaMalloc(&d_remap_,        (size_t)max_seeds * 4  * sizeof(int32_t));
    cudaMalloc(&d_edge_out_,     (size_t)max_seeds * 24 * sizeof(int32_t));
    cudaMalloc(&d_scores_,       (size_t)max_seeds * 12 * sizeof(float));
    cudaMalloc(&d_mid_keys_,     (size_t)max_seeds      * sizeof(int64_t));
    cudaMalloc(&d_mid_count_,    1                      * sizeof(int32_t));
    cudaMemset(d_dead_, 0,       (size_t)max_seeds * 4  * sizeof(uint8_t));
    // Three edge keys per triangle, over the same triangle bound as d_stale_.
    cudaMalloc(&d_edge_keys_,    (size_t)max_seeds * 4 * 3 * sizeof(int64_t));

    cudaMemset(d_grid_,   SENTINEL_BYTE, (size_t)N * 2 * sizeof(int32_t));
    cudaMemset(d_t_grid_, SENTINEL_BYTE, (size_t)N     * sizeof(int32_t));
    cudaMemset(d_changed_,  0, (size_t)N     * sizeof(int32_t));
    cudaMemset(d_dirty_accum_, 0, (size_t)N  * sizeof(int32_t));
}

Delaunay::~Delaunay()
{
    cudaFree(d_grid_);    cudaFree(d_tmp_);      cudaFree(d_changed_);
    cudaFree(d_sx_);      cudaFree(d_sy_);        cudaFree(d_raw_buf_);
    cudaFree(d_detect_buf_);
    cudaFree(d_t_grid_);  cudaFree(d_csr_ptr_);  cudaFree(d_csr_idx_);
    cudaFree(d_edge_keys_);
    cudaFree(d_dead_);
    cudaFree(d_values_);   cudaFree(d_scores_);  cudaFree(d_score_keys_);
    cudaFree(d_mid_keys_); cudaFree(d_mid_count_);
    cudaFree(d_updated_flag_); cudaFree(d_mask_);
    cudaFree(d_dirty_accum_);  cudaFree(d_tile_dirty_); cudaFree(d_count_);
    cudaFree(d_stale_);
    cudaFree(d_tri_count_);  cudaFree(d_seed_stage_);
    cudaFree(d_remap_);      cudaFree(d_edge_out_);
}

// ---------------------------------------------------------------------------
// run_bfs_: write new seeds already in d_grid_, BFS until stable
// ---------------------------------------------------------------------------

//: BFS passes to run between convergence checks.
//:
//: A pass costs about 23us of kernel; asking whether it changed anything costs
//: a device synchronisation and two 4-byte transfers, which together run
//: longer. Batching trades up to BFS_CHECK_EVERY - 1 passes that find nothing
//: for one check instead of that many. Runs are 6 to 51 passes, so 8 keeps the
//: waste under a fifth while removing seven eighths of the checks.
static constexpr int BFS_CHECK_EVERY = 8;

void Delaunay::run_bfs_(float* bfs_ms_out, int* iters_out)
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (bfs_ms_out) {
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
    }

    int32_t zero = 0;
    int iters = 0;
    for (;;) {
        cudaMemcpy(d_updated_flag_, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

        // The flag accumulates across the batch, so one read answers "did
        // anything move in any of these". Asking after every pass instead cost
        // a device synchronisation and two 4-byte transfers per pass, around a
        // kernel of 23us -- the question was dearer than the work.
        //
        // No synchronise between passes: the launches are ordered on one
        // stream, so each sees the previous one's output, and the pointer swap
        // is on the host and takes effect at the next launch. The memcpy below
        // blocks until the batch has drained, which is the only wait needed.
        for (int i = 0; i < BFS_CHECK_EVERY; ++i) {
            ++iters;
            voronoi_step_kernel<<<grid_dim, block>>>(
                d_grid_, d_tmp_, W_det_, H_det_, d_updated_flag_, d_changed_,
                d_sx_, d_sy_);
            std::swap(d_grid_, d_tmp_);
        }

        int32_t flag = 0;
        cudaMemcpy(&flag, d_updated_flag_, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (!flag) break;
    }
    // Passes performed, not passes needed: convergence is noticed at the end of
    // a batch, so up to BFS_CHECK_EVERY - 1 of them found nothing to do. That
    // is the cost being reported.
    if (iters_out) *iters_out = iters;

    if (bfs_ms_out) {
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(bfs_ms_out, ev0, ev1);
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
}

// ---------------------------------------------------------------------------
// upload_triangles_: sync h_triangles_ → d_raw_buf_
// ---------------------------------------------------------------------------

void Delaunay::upload_triangles_()
{
    upload_triangles_range_(0, (int)h_triangles_.size());
}

void Delaunay::upload_triangles_range_(int first, int count)
{
    if (count <= 0) return;
    std::vector<RawTriangle> h_raw(count);
    for (int i = 0; i < count; ++i) {
        const auto& h = h_triangles_[first + i];
        h_raw[i] = {h.x, h.y, h.a, h.b, h.c, h.orig_a, h.orig_b, h.orig_c};
    }
    cudaMemcpy(static_cast<RawTriangle*>(d_raw_buf_) + first, h_raw.data(),
               (size_t)count * sizeof(RawTriangle), cudaMemcpyHostToDevice);
}

void Delaunay::upload_dead_flags_()
{
    if (h_dead_.empty()) return;
    cudaMemcpy(d_dead_, h_dead_.data(), h_dead_.size() * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
}

//: Compact once the retired slots outnumber the live ones, or the slot count
//: approaches the device buffers' bound. Amortised: each compaction is
//: O(triangles) but at least halves the slot count.
bool Delaunay::should_compact_() const
{
    const int slots = (int)h_triangles_.size();
    if (slots == 0) return false;
    if (slots > 2 * n_live_ + 1024) return true;
    return slots > max_seeds_ * 4 - 4096;
}

void Delaunay::compact_registry_()
{
    const int old_count = (int)h_triangles_.size();
    if (old_count == n_live_) return;          // already dense

    std::vector<int32_t> remap(old_count, TID_DELETED);
    std::vector<HTriangle> live;
    live.reserve(n_live_);
    for (int tid = 0; tid < old_count; ++tid) {
        if (h_dead_[tid]) continue;
        remap[tid] = (int32_t)live.size();
        live.push_back(h_triangles_[tid]);
    }

    h_triangles_ = std::move(live);
    h_dead_.assign(h_triangles_.size(), 0);
    n_live_ = (int)h_triangles_.size();

    h_triplet_to_tid_.clear();
    h_triplet_to_tid_.reserve(h_triangles_.size() * 2);
    for (int tid = 0; tid < (int)h_triangles_.size(); ++tid) {
        const auto& t = h_triangles_[tid];
        h_triplet_to_tid_[pack_triplet_(t.a, t.b, t.c)] = tid;
    }

    upload_triangles_();
    upload_dead_flags_();
    csr_dirty_ = true;
    edges_dirty_ = true;

    const int N = W_det_ * H_det_;
    cudaMemcpy(d_remap_, remap.data(), (size_t)old_count * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    remap_tgrid_kernel<<<(N+255)/256, 256>>>(d_t_grid_, N, d_remap_, old_count,
                                             NO_TRIANGLE);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// rebuild_csr_and_upload_
// ---------------------------------------------------------------------------

void Delaunay::rebuild_csr_and_upload_()
{
    int N_tri = (int)h_triangles_.size();
    std::vector<int32_t> h_csr_ptr(N_ + 1, 0);
    // Retired slots are skipped throughout: assign_triangles_kernel reaches a
    // triangle only through this index, so leaving them out is what keeps a
    // dead slot from ever being tested against a pixel.
    for (int tid = 0; tid < N_tri; ++tid) {
        if (h_dead_[tid]) continue;
        const auto& t = h_triangles_[tid];
        h_csr_ptr[t.orig_a + 1]++;
        h_csr_ptr[t.orig_b + 1]++;
        h_csr_ptr[t.orig_c + 1]++;
    }
    for (int s = 1; s <= N_; ++s) h_csr_ptr[s] += h_csr_ptr[s-1];

    int csr_size = h_csr_ptr[N_];
    std::vector<int32_t> h_csr_idx(csr_size);
    std::vector<int32_t> fill(N_, 0);
    for (int tid = 0; tid < N_tri; ++tid) {
        if (h_dead_[tid]) continue;
        for (int32_t s : {h_triangles_[tid].orig_a,
                          h_triangles_[tid].orig_b,
                          h_triangles_[tid].orig_c}) {
            h_csr_idx[h_csr_ptr[s] + fill[s]] = tid;
            fill[s]++;
        }
    }
    cudaMemcpy(d_csr_ptr_, h_csr_ptr.data(), (N_+1)*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_idx_, h_csr_idx.data(),  csr_size*sizeof(int32_t), cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// full_topology_: detect → dedup → registry → CSR, over the whole grid
// ---------------------------------------------------------------------------

void Delaunay::full_topology_(float* det_ms, float* dedup_ms)
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    auto mk = [](cudaEvent_t* e){ cudaEventCreate(e); };
    auto rc = [](cudaEvent_t  e){ cudaEventRecord(e); };
    auto el = [](cudaEvent_t a, cudaEvent_t b) -> float {
        cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms;
    };

    cudaEvent_t e0,e1,e2,e3;
    if (det_ms) { mk(&e0);mk(&e1);mk(&e2);mk(&e3); }

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int32_t* d_counter = d_tri_count_;
    cudaMemset(d_counter, 0, sizeof(int32_t));

    if (det_ms) rc(e0);
    find_triangle_seeds_kernel<<<grid_dim, block>>>(
        d_grid_, W_det_, H_det_, d_sx_, d_sy_, d_raw, d_counter, nullptr);
    cudaDeviceSynchronize();
    int32_t raw_count = 0;
    cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (det_ms) rc(e1);

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);
    if (dedup_ms) rc(e2);
    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    int N_tri = (int)(new_end - d_ptr);
    if (dedup_ms) rc(e3);

    std::vector<RawTriangle> h_dedup(N_tri);
    cudaMemcpy(h_dedup.data(), d_raw, N_tri * sizeof(RawTriangle), cudaMemcpyDeviceToHost);

    h_triangles_.clear(); h_triplet_to_tid_.clear();
    h_triangles_.reserve(N_tri);
    for (int32_t tid = 0; tid < N_tri; ++tid) {
        const auto& r = h_dedup[tid];
        h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c, r.orig_a, r.orig_b, r.orig_c});
        h_triplet_to_tid_[pack_triplet_(r.a, r.b, r.c)] = tid;
    }
    // A full build leaves no holes.
    h_dead_.assign(N_tri, 0);
    n_live_ = N_tri;
    upload_dead_flags_();
    // Detection wrote to the scratch buffer, so the list gets its own copy.
    // Device to device, and the order already matches h_triangles_.
    if (N_tri > 0)
        cudaMemcpy(d_raw_buf_, d_raw, (size_t)N_tri * sizeof(RawTriangle),
                   cudaMemcpyDeviceToDevice);

    // The CSR is not built here. Its only reader is assign_triangles_kernel,
    // which runs in assign_pending_, so building it per insert was O(N_tri +
    // N_seeds) of host work that a deferred round never used.
    csr_dirty_ = true;

    // Every pixel's assignment is now stale. Marking them invalidated rather
    // than assigning here lets assign_pending_ pick the mask up like any other
    // dirty region, and makes a first insert behave like the rest.
    cudaMemset(d_t_grid_, SENTINEL_BYTE, (size_t)W_det_ * H_det_ * sizeof(int32_t));

    if (det_ms) {
        *det_ms   = el(e0,e1);
        *dedup_ms = el(e2,e3);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
    }
}

// ---------------------------------------------------------------------------
// partial_triangulate_: use d_changed_ to scope detection and assignment
// ---------------------------------------------------------------------------

void Delaunay::partial_topology_(float* det_ms, float* dedup_ms)
{
    dim3 block(16, 16);
    dim3 grid_dim((W_det_ + 15) / 16, (H_det_ + 15) / 16);

    auto mk = [](cudaEvent_t* e){ cudaEventCreate(e); };
    auto rc = [](cudaEvent_t  e){ cudaEventRecord(e); };
    auto el = [](cudaEvent_t a, cudaEvent_t b) -> float {
        cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); return ms;
    };

    cudaEvent_t e0,e1,e2,e3;
    if (det_ms) { mk(&e0);mk(&e1);mk(&e2);mk(&e3); }

    // Detection border: this insert's changes expanded by 2, which is the reach
    // of the L-shaped stencil in find_triangle_seeds_kernel. Scoped to
    // d_changed_ rather than the accumulator so a deferred round does not
    // re-detect regions earlier rounds already handled.
    dilate_fixed_kernel<<<grid_dim, block>>>(d_changed_, d_mask_, W_det_, H_det_, 2);
    cudaDeviceSynchronize();

    // Which existing triangles the border invalidates. Sampling the mask at the
    // triangles' own canonical pixels moves N_tri flags instead of the whole
    // W*H mask, and d_raw_buf_ already holds those positions on the device in
    // registry order -- but only until detection overwrites it below, so this
    // has to happen first.
    int old_count = (int)h_triangles_.size();
    std::vector<uint8_t> h_stale(old_count, 0);
    if (old_count > 0) {
        mark_stale_kernel<<<(old_count + 255) / 256, 256>>>(
            static_cast<RawTriangle*>(d_raw_buf_), old_count,
            d_mask_, W_det_, H_det_, d_stale_);
        cudaDeviceSynchronize();
        cudaMemcpy(h_stale.data(), d_stale_, old_count * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost);
    }
    auto is_stale = [&h_stale](int tid) { return h_stale[tid] != 0; };

    RawTriangle* d_raw = static_cast<RawTriangle*>(d_detect_buf_);
    int32_t* d_counter = d_tri_count_;
    cudaMemset(d_counter, 0, sizeof(int32_t));

    if (det_ms) rc(e0);
    find_triangle_seeds_kernel<<<grid_dim, block>>>(
        d_grid_, W_det_, H_det_, d_sx_, d_sy_, d_raw, d_counter, d_mask_);
    cudaDeviceSynchronize();
    int32_t raw_count = 0;
    cudaMemcpy(&raw_count, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (det_ms) rc(e1);

    thrust::device_ptr<RawTriangle> d_ptr(d_raw);
    if (dedup_ms) rc(e2);
    thrust::sort(d_ptr, d_ptr + raw_count, RawLess{});
    auto new_end = thrust::unique(d_ptr, d_ptr + raw_count, RawEqual{});
    int n_new = (int)(new_end - d_ptr);
    if (dedup_ms) rc(e3);

    std::vector<RawTriangle> h_new(n_new);
    if (n_new > 0)
        cudaMemcpy(h_new.data(), d_raw, n_new * sizeof(RawTriangle), cudaMemcpyDeviceToHost);

    // Retire the invalidated triangles. Their ids are not reused and nothing
    // else is renumbered, so this touches only the entries that changed --
    // which is the whole point. Renumbering instead cost a full map rebuild, a
    // full device upload and a grid remap on every insert, all proportional to
    // the total triangle count rather than to the size of the change.
    for (int tid = 0; tid < old_count; ++tid) {
        if (h_dead_[tid] || !is_stale(tid)) continue;
        const auto& t = h_triangles_[tid];
        h_triplet_to_tid_.erase(pack_triplet_(t.a, t.b, t.c));
        h_dead_[tid] = 1;
        --n_live_;
    }

    // Append whatever the re-detection found that the registry no longer holds.
    // A retired triangle that is still valid geometry comes back through here
    // and takes a fresh slot, which is what the renumbering path did too.
    const int append_first = (int)h_triangles_.size();
    for (const auto& r : h_new) {
        auto key = pack_triplet_(r.a, r.b, r.c);
        if (h_triplet_to_tid_.count(key)) continue;
        h_triplet_to_tid_[key] = (int32_t)h_triangles_.size();
        h_triangles_.push_back({r.x, r.y, r.a, r.b, r.c,
                                r.orig_a, r.orig_b, r.orig_c});
        h_dead_.push_back(0);
        ++n_live_;
    }
    const int append_count = (int)h_triangles_.size() - append_first;

    // Only the tail goes up; the earlier slots are untouched on the device.
    // mark_stale_kernel and get_edges read d_raw_buf_, so it has to stay in
    // step. The CSR does not -- see full_topology_.
    upload_triangles_range_(append_first, append_count);
    upload_dead_flags_();
    csr_dirty_ = true;

    // Pixels naming a retired triangle are cleared. Ids did not move, so the
    // old remap table is unnecessary: the flags say which ids are gone.
    const int N = W_det_ * H_det_;
    invalidate_dead_tgrid_kernel<<<(N+255)/256, 256>>>(
        d_t_grid_, N, d_dead_, (int)h_triangles_.size());
    cudaDeviceSynchronize();

    // Amortised: holes are cheap to carry but not free to carry forever, and
    // the device buffers are sized for a bounded slot count.
    if (should_compact_()) compact_registry_();

    if (det_ms) {
        *det_ms   = el(e0,e1);
        *dedup_ms = el(e2,e3);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaEventDestroy(e2); cudaEventDestroy(e3);
    }
}

// ---------------------------------------------------------------------------
// assign_pending_: one pixel-assignment pass covering everything deferred
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Lazy rebuilds of the derived structures
//
// Both used to run on every insert. The CSR is read only by
// assign_triangles_kernel and the sorted ranks only by the two output
// builders, so a deferred round paid for neither.
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
// rebuild_sorted_rank_: internal (insertion) id → batch (sorted x,y) id
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
}

// ---------------------------------------------------------------------------
// build_outputs_
// ---------------------------------------------------------------------------

void Delaunay::build_outputs_(std::vector<TriangleEntry>& tri_map_out,
                                         std::vector<int32_t>& tgrid_out) const
{
    ensure_sorted_rank_();
    const int N     = W_ * H_;
    const int N_det = W_det_ * H_det_;
    int N_tri = (int)h_triangles_.size();

    // Translate internal insertion-order ids to the batch pipeline's sorted
    // numbering (see h_sorted_rank_).
    auto ext = [this](int32_t internal) -> int32_t {
        return (internal >= 0 && internal < (int32_t)h_sorted_rank_.size())
             ? h_sorted_rank_[internal] : internal;
    };

    tri_map_out.resize(N_tri);
    for (int tid = 0; tid < N_tri; ++tid) {
        const auto& t = h_triangles_[tid];
        // Canonical positions come back in image coordinates. A border triangle
        // lands outside [0,W)x[0,H) once shifted, which is correct and matches
        // the batch path: its circumcentre genuinely lies outside the image.
        tri_map_out[tid] = {t.x - P_, t.y - P_,
                            ext(t.orig_a), ext(t.orig_b), ext(t.orig_c)};
    }

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
            tgrid_out[d*3]   = ext(h_grid[s*2]);
            tgrid_out[d*3+1] = h_grid[s*2+1];
            tgrid_out[d*3+2] = h_t[s];
        }
    }
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

    // Same insertion→sorted id translation as build_outputs_, cropping the
    // padded canvas back to the image.
    for (int y = 0; y < H_; ++y) {
        const int src_row = (y + P_) * W_det_ + P_;
        const int dst_row = y * W_;
        for (int x = 0; x < W_; ++x) {
            const int s = src_row + x, d = dst_row + x;
            int32_t sid = h_grid[s*2];
            out[d*2]   = (sid >= 0 && sid < (int32_t)h_sorted_rank_.size())
                       ? h_sorted_rank_[sid] : sid;
            out[d*2+1] = h_grid[s*2+1];
        }
    }
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

    // Duplicate checks, within the batch and against existing seeds. Both go
    // through a hash set: the intra-batch check used to be O(k^2), which is
    // invisible for the handful of seeds the tests insert and quadratic for the
    // thousands a refinement round adds.
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
    cudaMemcpy(d_sx_ + N_ - k, padded_xs.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy_ + N_ - k, padded_ys.data(), k * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Reset change accumulator before writing seeds (so seed positions
    // are the first entries in d_changed_ — necessary for partial_topology_
    // to cover detection positions that touch the seed cell directly).
    // d_changed_ is per-insert and scopes detection; d_dirty_accum_ below is
    // the union since the last finalise and scopes assignment.
    cudaMemset(d_changed_, 0, (size_t)W_det_ * H_det_ * sizeof(int32_t));

    // Write seeds into grid (also marks seed cells in d_changed_)
    int32_t* d_kxs  = d_seed_stage_;
    int32_t* d_kys  = d_seed_stage_ + max_seeds_;
    int32_t* d_kids = d_seed_stage_ + 2 * max_seeds_;
    cudaMemcpy(d_kxs,  padded_xs.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kys,  padded_ys.data(),  k * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kids, new_ids.data(),    k * sizeof(int32_t), cudaMemcpyHostToDevice);
    write_seeds_kernel<<<(k+255)/256, 256>>>(d_grid_, d_changed_, W_det_, d_kxs, d_kys, d_kids, k);
    cudaDeviceSynchronize();

    // BFS
    run_bfs_(bfs_ms_out, iters_out);

    // Fold this insert's changes into the set awaiting assignment.
    const int N = W_det_ * H_det_;
    or_mask_kernel<<<(N + 255) / 256, 256>>>(d_changed_, d_dirty_accum_, N);
    cudaDeviceSynchronize();
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
        cudaMemcpy(d_values_ + seeds_before, new_values->data(),
                   new_values->size() * sizeof(float), cudaMemcpyHostToDevice);
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
        cudaMemset(d_dirty_accum_, 0, (size_t)W_det_ * H_det_ * sizeof(int32_t));
        pending_ = false;
    }
    build_outputs_(tri_map_out, tgrid_out);
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

void Delaunay::ensure_edges_() const
{
    if (!edges_dirty_) return;
    n_edges_ = 0;
    edges_dirty_ = false;

    const int n_tri = (int)h_triangles_.size();
    if (n_tri == 0 || N_ == 0 || n_live_ == 0) return;

    const RawTriangle* d_tris = static_cast<const RawTriangle*>(d_raw_buf_);
    int64_t* d_keys = static_cast<int64_t*>(d_edge_keys_);

    build_edge_keys_kernel<<<(n_tri + 255) / 256, 256>>>(
        d_tris, n_tri, (int64_t)N_, d_dead_, d_keys);
    cudaDeviceSynchronize();

    thrust::device_ptr<int64_t> p(d_keys);
    thrust::sort(p, p + (size_t)n_tri * 3);
    auto end = thrust::unique(p, p + (size_t)n_tri * 3);
    int n = (int)(end - p);
    // Retired slots all emit EDGE_KEY_DEAD, which sorts last and uniques down
    // to the single trailing entry dropped here.
    if (n > 0) {
        int64_t last = 0;
        cudaMemcpy(&last, d_keys + (n - 1), sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        if (last == EDGE_KEY_DEAD) --n;
    }
    n_edges_ = n;
}

void Delaunay::edge_scores(double min_length, std::vector<float>& out) const
{
    ensure_edges_();
    out.clear();
    if (n_edges_ == 0) return;
    if (!have_values_)
        throw std::logic_error(
            "edge_scores needs a value per seed; pass values to insert_deferred");

    score_edges_kernel<<<(n_edges_ + 255) / 256, 256>>>(
        static_cast<const int64_t*>(d_edge_keys_), n_edges_, (int64_t)N_,
        d_sx_, d_sy_, d_values_, min_length * min_length, d_scores_);
    cudaDeviceSynchronize();

    out.resize(n_edges_);
    cudaMemcpy(out.data(), d_scores_, (size_t)n_edges_ * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void Delaunay::select_midpoints(double min_length, int count, float threshold,
                                std::vector<int32_t>& out) const
{
    ensure_edges_();
    out.clear();
    if (n_edges_ == 0 || count <= 0) return;
    if (!have_values_)
        throw std::logic_error(
            "select_midpoints needs a value per seed; pass values to insert_deferred");

    score_edges_kernel<<<(n_edges_ + 255) / 256, 256>>>(
        static_cast<const int64_t*>(d_edge_keys_), n_edges_, (int64_t)N_,
        d_sx_, d_sy_, d_values_, min_length * min_length, d_scores_);
    cudaDeviceSynchronize();

    // The whole of the selection: order the edges best-first and take the
    // front. No threshold crosses to the caller and none comes back, because
    // the key is a total order and `count` is therefore exact.
    pack_score_keys_kernel<<<(n_edges_ + 255) / 256, 256>>>(
        d_scores_, n_edges_, threshold, d_score_keys_);
    cudaDeviceSynchronize();

    thrust::device_ptr<float> sc(d_scores_);
    const int n_eligible = (int)thrust::count_if(sc, sc + n_edges_,
                                                 AboveThreshold{threshold});
    if (n_eligible == 0) return;
    const int take = std::min(count, n_eligible);

    thrust::device_ptr<uint64_t> kp(d_score_keys_);
    thrust::sort(kp, kp + n_edges_);

    cudaMemset(d_mid_count_, 0, sizeof(int32_t));
    midpoints_from_sorted_kernel<<<(take + 255) / 256, 256>>>(
        d_score_keys_, take, static_cast<const int64_t*>(d_edge_keys_),
        (int64_t)N_, d_sx_, d_sy_, W_det_, H_det_, P_, d_grid_,
        d_mid_keys_, d_mid_count_);
    cudaDeviceSynchronize();

    int32_t n = 0;
    cudaMemcpy(&n, d_mid_count_, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (n == 0) return;

    // Sorting by (pixel, rank) puts colliding midpoints together with the
    // best-scoring one first, so uniquing on the pixel keeps a deterministic
    // choice -- the emission order above is not, being an atomic race.
    thrust::device_ptr<int64_t> p(d_mid_keys_);
    thrust::sort(p, p + n);
    auto end_it = thrust::unique(p, p + n, SamePixel{(int64_t)take});
    const int m = (int)(end_it - p);

    std::vector<int64_t> h_keys(m);
    cudaMemcpy(h_keys.data(), d_mid_keys_, (size_t)m * sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    out.resize((size_t)m * 2);
    for (int i = 0; i < m; ++i) {
        const int64_t pixel = h_keys[i] / (int64_t)take;
        out[(size_t)i * 2]     = (int32_t)(pixel % W_det_) - P_;
        out[(size_t)i * 2 + 1] = (int32_t)(pixel / W_det_) - P_;
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

void Delaunay::locate(const std::vector<int32_t>& qx,
                      const std::vector<int32_t>& qy,
                      std::vector<int32_t>& out)
{
    const int n = (int)qx.size();
    out.assign(n, NO_TRIANGLE);
    if (n == 0) return;
    if (qy.size() != qx.size())
        throw std::invalid_argument("locate: x and y must be the same length");

    // Dense ids, so the answers index get_triangles() and finalise()'s map
    // alike; and the CSR, which is how a candidate triangle is reached.
    compact_registry_();
    ensure_csr_();
    if (h_triangles_.empty()) return;

    // The queries reuse the seed staging buffer: an insert is the only other
    // user and cannot be in flight here.
    int32_t* d_qx = d_seed_stage_;
    int32_t* d_qy = d_seed_stage_ + max_seeds_;
    const int chunk = max_seeds_;

    int32_t* d_out = nullptr;
    cudaMalloc(&d_out, (size_t)std::min(n, chunk) * sizeof(int32_t));

    for (int off = 0; off < n; off += chunk) {
        const int k = std::min(chunk, n - off);
        cudaMemcpy(d_qx, qx.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_qy, qy.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice);
        locate_points_kernel<<<(k + 255) / 256, 256>>>(
            d_qx, d_qy, k, P_, W_det_, H_det_, d_grid_,
            static_cast<const RawTriangle*>(d_raw_buf_),
            d_sx_, d_sy_, d_csr_ptr_, d_csr_idx_, N_, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(out.data() + off, d_out, (size_t)k * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
    }
    cudaFree(d_out);
}

void Delaunay::get_edges(std::vector<int32_t>& out) const
{
    ensure_edges_();
    out.clear();
    if (n_edges_ == 0) return;

    // Unpacked on the device so the transfer is the (E, 2) result rather than
    // the 3T keys that produced it.
    unpack_edge_keys_kernel<<<(n_edges_ + 255) / 256, 256>>>(
        static_cast<const int64_t*>(d_edge_keys_), n_edges_, (int64_t)N_,
        d_edge_out_);
    cudaDeviceSynchronize();

    out.resize((size_t)n_edges_ * 2);
    cudaMemcpy(out.data(), d_edge_out_, (size_t)n_edges_ * 2 * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
}
