// ---------------------------------------------------------------------------
// Shared 2x2-block triangle detection.
//
// Both pipelines find triangles the same way: scan every 2x2 block of the
// Voronoi seed-id grid, and where three or four distinct regions meet, the
// block witnesses a Voronoi vertex and therefore a Delaunay triangle.
//
// The rule lives here once, shared by both pipelines, rather than as separate
// copies that could drift apart: at a degree-4 vertex the tie-break must stay
// geometric (see detect_block_triangles below), because a purely pixel-based
// rule can make the two paths cut the same cocircular quad differently and
// register overlapping triangles.
//
// What differs between the callers is only how a seed id is fetched from
// their grid layout and whether a mask applies; neither is part of the rule,
// so both stay in the kernels.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

#include "voronoi.cuh"   // UNDEF_SEED

//: One detected triangle, before deduplication.
//:
//: (a, b, c) is the sorted triple and serves as the dedup key: the same
//: triangle is witnessed by every block around its circumcentre, so the same
//: key arrives many times. (orig_a, orig_b, orig_c) keeps the order the block
//: saw, and (x, y) the block that saw it, which is what later maps a triangle
//: back to a position on the grid.
struct RawTriangle {
    int32_t x, y;
    int32_t a, b, c;            // sorted dedup key  (a <= b <= c)
    int32_t orig_a, orig_b, orig_c;
};

struct RawLess {
    __device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
        if (p.a != q.a) return p.a < q.a;
        if (p.b != q.b) return p.b < q.b;
        return p.c < q.c;
    }
};

struct RawEqual {
    __device__ bool operator()(const RawTriangle& p, const RawTriangle& q) const {
        return p.a == q.a && p.b == q.b && p.c == q.c;
    }
};

//: Upper bound on raw detections for a W x H detection canvas.
//:
//: Each 2x2 block emits at most two triangles and there are (W-1)*(H-1) blocks,
//: so this is exact rather than a guess.
__host__ __device__ inline size_t max_raw_triangles(int W, int H)
{
    if (W < 2 || H < 2) return 0;
    return (size_t)2 * (W - 1) * (H - 1);
}

// ---------------------------------------------------------------------------
// The rule itself
// ---------------------------------------------------------------------------

//: Decide which triangles a 2x2 block witnesses, and hand them to `emit`.
//:
//: `a`, `b`, `c`, `d` are the seed ids at the block's top-left, top-right,
//: bottom-left and bottom-right corners. `emit(oa, ob, oc)` receives each
//: triangle in the order the block saw it; the caller sorts and stores.
//:
//: Templated on the functor so the emit path inlines -- this is the innermost
//: loop of detection, running once per pixel of the padded canvas.
template <typename Emit>
__device__ inline void detect_block_triangles(
    int32_t a, int32_t b, int32_t c, int32_t d,
    const int32_t* __restrict__ seed_xs,
    const int32_t* __restrict__ seed_ys,
    Emit emit)
{
    const int32_t quad[4] = {a, b, c, d};

    // Distinct seed ids in the block. An unreached cell cannot take part in a
    // triangle, and it would also index the seed coordinate arrays out of
    // bounds below, so the whole block is skipped.
    int32_t s[4];
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        if (quad[i] == UNDEF_SEED) return;
        bool dup = false;
        for (int j = 0; j < n; ++j) if (s[j] == quad[i]) { dup = true; break; }
        if (!dup) s[n++] = quad[i];
    }

    if (n == 3) {
        // Three regions meet at this corner: exactly one Delaunay triangle.
        emit(s[0], s[1], s[2]);
        return;
    }
    if (n != 4) return;         // one or two regions: no vertex, no triangle

    // Four regions meet: a degree-4 Voronoi vertex, so the four seeds are
    // cocircular and the quad has two equally valid Delaunay triangulations,
    // one per diagonal. Exactly one must be chosen -- registering both would
    // produce overlapping triangles. This mirrors reference/triangulation.py,
    // so all three implementations cut a given quad identically.
    //
    // Splitting on the pixel block's anti-diagonal instead would make the
    // result depend on image orientation rather than on the seed geometry:
    // rotating the input 90 degrees could change the answer.

    // Cyclic order around the centroid, so "diagonal" means opposite corners
    // rather than adjacent ones. Double precision and a stable sort, to match
    // the reference's np.arctan2 + sorted().
    double cx = 0.0, cy = 0.0;
    for (int i = 0; i < 4; ++i) { cx += seed_xs[quad[i]]; cy += seed_ys[quad[i]]; }
    cx *= 0.25; cy *= 0.25;

    double ang[4];
    for (int i = 0; i < 4; ++i)
        ang[i] = atan2((double)seed_ys[quad[i]] - cy,
                       (double)seed_xs[quad[i]] - cx);

    int ord[4] = {0, 1, 2, 3};
    for (int i = 1; i < 4; ++i) {                 // insertion sort, stable
        int k = ord[i];
        int j = i - 1;
        while (j >= 0 && ang[ord[j]] > ang[k]) { ord[j + 1] = ord[j]; --j; }
        ord[j + 1] = k;
    }

    // Exact integer lengths: seed coordinates are integral, so this reproduces
    // the reference's float comparison with no rounding risk.
    auto len2 = [&](int i, int j) -> long long {
        long long dx = (long long)seed_xs[quad[i]] - seed_xs[quad[j]];
        long long dy = (long long)seed_ys[quad[i]] - seed_ys[quad[j]];
        return dx * dx + dy * dy;
    };

    const int d0a = ord[0], d0b = ord[2];
    const int d1a = ord[1], d1b = ord[3];
    const long long l0 = len2(d0a, d0b), l1 = len2(d1a, d1b);

    int pa, pb;                                    // the chosen diagonal
    if (l0 < l1)      { pa = d0a; pb = d0b; }
    else if (l1 < l0) { pa = d1a; pb = d1b; }
    else {
        // Exact tie, the usual case for cocircular points: take the diagonal
        // that does NOT contain the lowest seed id.
        int lowest = 0;
        for (int i = 1; i < 4; ++i) if (quad[i] < quad[lowest]) lowest = i;
        if (lowest == d0a || lowest == d0b) { pa = d1a; pb = d1b; }
        else                                { pa = d0a; pb = d0b; }
    }

    // The two triangles sharing that diagonal.
    int ra = -1, rb = -1;
    for (int i = 0; i < 4; ++i) {
        if (i == pa || i == pb) continue;
        if (ra < 0) ra = i; else rb = i;
    }

    emit(quad[pa], quad[pb], quad[ra]);
    emit(quad[pa], quad[pb], quad[rb]);
}

//: Sort three ids and append one RawTriangle at `(x, y)` via an atomic bump.
//:
//: Shared for the same reason as the rule above: both callers stored the
//: identical struct and neither had a reason of its own to.
__device__ inline void append_raw_triangle(
    RawTriangle* __restrict__ raw_buf, int32_t* __restrict__ counter,
    int x, int y, int32_t oa, int32_t ob, int32_t oc)
{
    int32_t sa = oa, sb = ob, sc = oc;
    if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
    if (sb > sc) { int32_t t = sb; sb = sc; sc = t; }
    if (sa > sb) { int32_t t = sa; sa = sb; sb = t; }
    int32_t pos = atomicAdd(counter, 1);
    raw_buf[pos] = {x, y, sa, sb, sc, oa, ob, oc};
}
