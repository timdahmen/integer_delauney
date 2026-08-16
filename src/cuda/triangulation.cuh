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
//
// voronoi_grid may be nullptr, which means "compute it yourself". That is the
// preferred call: at border_padding > 0 this function builds a padded Voronoi
// diagram for detection anyway, and the interior window of that diagram is
// bit-identical to an unpadded one -- nearest-seed does not depend on where the
// canvas ends. A caller passing its own grid at padding > 0 therefore pays for
// a second diagram whose only use is to fill two output channels the padded one
// already holds. Only at padding == 0 does a supplied grid drive detection.
void cuda_compute_triangulation(
    int W, int H,
    const int32_t* voronoi_grid,          // nullptr → computed internally
    const std::vector<int32_t>& seed_xs,
    const std::vector<int32_t>& seed_ys,
    std::vector<TriangleEntry>& triangle_map_out,
    std::vector<int32_t>& out_grid,
    TriTimings* timings,
    int border_padding,
    std::vector<int32_t>* padded_voronoi_out = nullptr);
