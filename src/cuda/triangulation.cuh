#pragma once
#include <cstdint>
#include <vector>

//: Pixels of Voronoi canvas added on each side when a caller does not choose.
//: Must stay in step with delauney.DEFAULT_BORDER_PADDING.
//:
//: A fixed value, not a density estimate. Circumcentre excursion depends on
//: triangle shape rather than on seed count, so no per-call formula bounds it;
//: it is bounded only when the caller controls how densely the convex hull
//: boundary is sampled. See BORDER_PADDING_BOUND.md.
static constexpr int DEFAULT_BORDER_PADDING = 16;

struct TriangleEntry {
    int32_t x, y;
    int32_t id_a, id_b, id_c;
};

// Optional sub-phase timings filled when pointer is non-null.
struct TriTimings {
    float detect_ms = 0.f;   // find_triangle_seeds kernel
    float dedup_ms  = 0.f;   // thrust::sort + thrust::unique
    float assign_ms = 0.f;   // assign_triangles kernel
};

// border_padding is a real pixel count here -- resolve the "pick one for me"
// sentinel (see bindings.cpp) before calling.  Deliberately no default: 0 is a
// valid but lossy choice, so callers must make it explicitly.
void cuda_compute_triangulation(
    int W, int H,
    const int32_t* voronoi_grid,
    const std::vector<int32_t>& seed_xs,
    const std::vector<int32_t>& seed_ys,
    std::vector<TriangleEntry>& triangle_map_out,
    std::vector<int32_t>& out_grid,
    TriTimings* timings,
    int border_padding,
    std::vector<int32_t>* padded_voronoi_out = nullptr);
