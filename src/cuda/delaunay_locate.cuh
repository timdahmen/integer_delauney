// Point-in-triangle-set search shared by pixel assignment (delaunay_assign.cu)
// and the point queries (delaunay_query.cu): both ask "which triangle contains
// this position", differing only in which positions they ask about.
#pragma once

#include "triangle_detect.cuh"   // RawTriangle
#include "geometry_device.cuh"   // point_in_triangle

static constexpr int WINDOW_SLACK = 3;
static constexpr int WINDOW_CAP   = 20;
static constexpr int MAX_NEARBY   = 64;

//: The triangle containing pixel (x, y), or NO_TRIANGLE.
//:
//: Testing every triangle would be hopeless, so the search narrows first: the
//: distance channel gives the radius within which a containing triangle's
//: vertices must lie, the seed ids in that window name the candidates, and the
//: CSR turns each into its few triangles.
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
