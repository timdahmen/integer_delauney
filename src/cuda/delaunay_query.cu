// Point queries: which triangle contains a given position, and whether a
// position lies inside a triangle's circumsphere lifted by t. Both are
// locate_at (delaunay_locate.cuh) restricted to a caller-supplied list of
// positions instead of every pixel -- see locate()'s docstring for why that
// is cheaper than reading the answers back out of a full assignment.
#include "delaunay.cuh"
#include "delaunay_locate.cuh"
#include "cuda_check.cuh"
#include "device_buffer.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

//: Is a point inside the circumsphere of the triangle containing it?
//:
//: The in-circle predicate Delaunay triangulation is defined by, lifted out of
//: the plane: the triangle's vertices lie at t = 0 and the query point at t,
//: so the test is
//:
//:     dx^2 + dy^2 + t^2  <  R^2
//:
//: and t eats into the circumradius. A caller using t for elapsed time gets
//: "close enough in space, recent enough in time"; a triangle with circumradius
//: R admits nothing beyond t = R.
//:
//: Double throughout. The radius decides a strict comparison, and rounding it
//: to float flips the answer for points sitting near a circumsphere boundary,
//: which is exactly where the interesting ones are.
__global__
void in_circumsphere_kernel(const int32_t* __restrict__ qx,
                            const int32_t* __restrict__ qy,
                            const double* __restrict__ qt,
                            int n, int P, int W, int H,
                            const int32_t* __restrict__ grid,
                            const RawTriangle* __restrict__ triangles,
                            const int32_t* __restrict__ sx,
                            const int32_t* __restrict__ sy,
                            const int32_t* __restrict__ csr_ptr,
                            const int32_t* __restrict__ csr_idx,
                            int N_seeds,
                            uint8_t* __restrict__ out,
                            int32_t* __restrict__ tid_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0;

    const int x = qx[i] + P, y = qy[i] + P;
    if (x < 0 || x >= W || y < 0 || y >= H) {
        if (tid_out) tid_out[i] = NO_TRIANGLE;
        return;
    }
    const int32_t tid = locate_at(x, y, W, H, grid, triangles,
                                  sx, sy, csr_ptr, csr_idx, N_seeds);
    if (tid_out) tid_out[i] = tid;
    if (tid == NO_TRIANGLE) return;

    const RawTriangle& tr = triangles[tid];
    const double ax = sx[tr.orig_a], ay = sy[tr.orig_a];
    const double bx = sx[tr.orig_b], by = sy[tr.orig_b];
    const double cx = sx[tr.orig_c], cy = sy[tr.orig_c];

    const double abx = bx - ax, aby = by - ay;
    const double acx = cx - ax, acy = cy - ay;
    const double nz = abx * acy - aby * acx;      // z of AB x AC; the rest is 0

    double ox, oy;
    if (nz * nz < 1e-12) {
        // Collinear: no circumcircle. The reference falls back to the centroid
        // and the farthest vertex, so this does too.
        ox = (ax + bx + cx) / 3.0;
        oy = (ay + by + cy) / 3.0;
    } else {
        // O = A + (|AC|^2 (N x AB) + |AB|^2 (AC x N)) / (2 |N|^2), N = AB x AC.
        //
        // Written in that order, and not simplified to a division by 2*nz,
        // although the two are the same number. They are not the same rounding,
        // and this predicate is decided on the boundary: cancelling the nz
        // disagreed with the reference on a query in a few thousand.
        const double ab2 = abx * abx + aby * aby;
        const double ac2 = acx * acx + acy * acy;
        const double den = 2.0 * (nz * nz);
        const double num_x = ac2 * (-(nz * aby)) + ab2 * (acy * nz);
        const double num_y = ac2 * (nz * abx) + ab2 * (-(acx * nz));
        ox = ax + num_x / den;
        oy = ay + num_y / den;
    }

    // Radius as the farthest vertex, not one of them: for an exact circumcentre
    // the three agree, and taking the max is what the reference does.
    const double r2 = fmax(fmax((ax-ox)*(ax-ox) + (ay-oy)*(ay-oy),
                                (bx-ox)*(bx-ox) + (by-oy)*(by-oy)),
                           (cx-ox)*(cx-ox) + (cy-oy)*(cy-oy));

    const double dx = (double)x - ox, dy = (double)y - oy, t = qt[i];

    // Distances, not squared distances. The two disagree by an ulp for a point
    // sitting on the boundary, and those are precisely the points this decides:
    // measured against the reference, comparing squares flipped one query in
    // four thousand at t near the circumradius.
    out[i] = (sqrt(dx * dx + dy * dy + t * t) < sqrt(r2)) ? 1 : 0;
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

void Delaunay::in_circumsphere(const std::vector<int32_t>& qx,
                               const std::vector<int32_t>& qy,
                               const std::vector<double>& qt,
                               std::vector<uint8_t>& mask_out,
                               std::vector<int32_t>& tid_out)
{
    const int n = (int)qx.size();
    mask_out.assign(n, 0);
    tid_out.assign(n, NO_TRIANGLE);
    if (n == 0) return;
    if (qy.size() != qx.size() || qt.size() != qx.size())
        throw std::invalid_argument(
            "in_circumsphere: x, y and t must be the same length");

    compact_registry_();
    ensure_csr_();
    if (h_triangles_.empty()) return;

    int32_t* d_qx = d_seed_stage_;
    int32_t* d_qy = d_seed_stage_ + max_seeds_;
    const int chunk = max_seeds_;

    const int cap = std::min(n, chunk);
    DeviceBuffer<double>  d_qt(cap);
    DeviceBuffer<uint8_t> d_m(cap);
    DeviceBuffer<int32_t> d_t(cap);

    for (int off = 0; off < n; off += chunk) {
        const int k = std::min(chunk, n - off);
        CUDA_CHECK(cudaMemcpy(d_qx, qx.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qy, qy.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qt, qt.data() + off, (size_t)k * sizeof(double),
                   cudaMemcpyHostToDevice));
        in_circumsphere_kernel<<<(k + 255) / 256, 256>>>(
            d_qx, d_qy, d_qt, k, P_, W_det_, H_det_, d_grid_,
            static_cast<const RawTriangle*>(d_raw_buf_),
            d_sx_, d_sy_, d_csr_ptr_, d_csr_idx_, N_, d_m, d_t);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(mask_out.data() + off, d_m, (size_t)k * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tid_out.data() + off, d_t, (size_t)k * sizeof(int32_t),
                   cudaMemcpyDeviceToHost));
    }
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

    DeviceBuffer<int32_t> d_out(std::min(n, chunk));

    for (int off = 0; off < n; off += chunk) {
        const int k = std::min(chunk, n - off);
        CUDA_CHECK(cudaMemcpy(d_qx, qx.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qy, qy.data() + off, (size_t)k * sizeof(int32_t),
                   cudaMemcpyHostToDevice));
        locate_points_kernel<<<(k + 255) / 256, 256>>>(
            d_qx, d_qy, k, P_, W_det_, H_det_, d_grid_,
            static_cast<const RawTriangle*>(d_raw_buf_),
            d_sx_, d_sy_, d_csr_ptr_, d_csr_idx_, N_, d_out);
        CUDA_CHECK_LAST_ERROR();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data() + off, d_out, (size_t)k * sizeof(int32_t),
                   cudaMemcpyDeviceToHost));
    }
}
