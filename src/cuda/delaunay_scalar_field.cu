// The scalar field on the vertices, and the edge metric over it: edge_scores,
// select_midpoints and get_edges, and the deduplicated edge list they all
// three read via ensure_edges_. See delaunay.cuh's "scalar field" section for
// why this lives beside the coordinates rather than on the host.
#include "delaunay.cuh"
#include "triangle_detect.cuh"
#include "cuda_check.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Kernels: undirected edge extraction
//
// One triangle contributes three edges. Orienting each as (min, max) makes the
// two triangles sharing an edge produce the same key, so a sort and a unique
// collapse them -- the same shape as the triangle dedup in delaunay_topology.cu.
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

// ---------------------------------------------------------------------------
// ensure_edges_ / edge_scores / select_midpoints / get_edges
// ---------------------------------------------------------------------------

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
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<int64_t> p(d_keys);
    thrust::sort(p, p + (size_t)n_tri * 3);
    auto end = thrust::unique(p, p + (size_t)n_tri * 3);
    int n = (int)(end - p);
    // Retired slots all emit EDGE_KEY_DEAD, which sorts last and uniques down
    // to the single trailing entry dropped here.
    if (n > 0) {
        int64_t last = 0;
        CUDA_CHECK(cudaMemcpy(&last, d_keys + (n - 1), sizeof(int64_t),
                   cudaMemcpyDeviceToHost));
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
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    out.resize(n_edges_);
    CUDA_CHECK(cudaMemcpy(out.data(), d_scores_, (size_t)n_edges_ * sizeof(float),
               cudaMemcpyDeviceToHost));
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
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    // The whole of the selection: order the edges best-first and take the
    // front. No threshold crosses to the caller and none comes back, because
    // the key is a total order and `count` is therefore exact.
    pack_score_keys_kernel<<<(n_edges_ + 255) / 256, 256>>>(
        d_scores_, n_edges_, threshold, d_score_keys_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<float> sc(d_scores_);
    const int n_eligible = (int)thrust::count_if(sc, sc + n_edges_,
                                                 AboveThreshold{threshold});
    if (n_eligible == 0) return;
    const int take = std::min(count, n_eligible);

    thrust::device_ptr<uint64_t> kp(d_score_keys_);
    thrust::sort(kp, kp + n_edges_);

    CUDA_CHECK(cudaMemset(d_mid_count_, 0, sizeof(int32_t)));
    midpoints_from_sorted_kernel<<<(take + 255) / 256, 256>>>(
        d_score_keys_, take, static_cast<const int64_t*>(d_edge_keys_),
        (int64_t)N_, d_sx_, d_sy_, W_det_, H_det_, P_, d_grid_,
        d_mid_keys_, d_mid_count_);
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    int32_t n = 0;
    CUDA_CHECK(cudaMemcpy(&n, d_mid_count_, sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (n == 0) return;

    // Sorting by (pixel, rank) puts colliding midpoints together with the
    // best-scoring one first, so uniquing on the pixel keeps a deterministic
    // choice -- the emission order above is not, being an atomic race.
    thrust::device_ptr<int64_t> p(d_mid_keys_);
    thrust::sort(p, p + n);
    auto end_it = thrust::unique(p, p + n, SamePixel{(int64_t)take});
    const int m = (int)(end_it - p);

    std::vector<int64_t> h_keys(m);
    CUDA_CHECK(cudaMemcpy(h_keys.data(), d_mid_keys_, (size_t)m * sizeof(int64_t),
               cudaMemcpyDeviceToHost));

    out.resize((size_t)m * 2);
    for (int i = 0; i < m; ++i) {
        const int64_t pixel = h_keys[i] / (int64_t)take;
        out[(size_t)i * 2]     = (int32_t)(pixel % W_det_) - P_;
        out[(size_t)i * 2 + 1] = (int32_t)(pixel / W_det_) - P_;
    }
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
    CUDA_CHECK_LAST_ERROR();
    CUDA_CHECK(cudaDeviceSynchronize());

    out.resize((size_t)n_edges_ * 2);
    CUDA_CHECK(cudaMemcpy(out.data(), d_edge_out_, (size_t)n_edges_ * 2 * sizeof(int32_t),
               cudaMemcpyDeviceToHost));
}
