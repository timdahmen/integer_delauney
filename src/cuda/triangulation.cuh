#pragma once
#include <cstdint>
#include <vector>

struct TriangleEntry {
    int32_t x, y;
    int32_t id_a, id_b, id_c;
};

// Optional sub-phase timings filled when pointer is non-null.
struct TriTimings {
    float detect_ms = 0.f;   // find_triangle_seeds kernel
    float dedup_ms  = 0.f;   // thrust::sort + thrust::unique
    float assign_ms = 0.f;   // assign_triangles kernel
    float sort_ms   = 0.f;   // (internal gap, not meaningful to caller)
};

void cuda_compute_triangulation(
    int W, int H,
    const int32_t* voronoi_grid,
    const std::vector<int32_t>& seed_xs,
    const std::vector<int32_t>& seed_ys,
    std::vector<TriangleEntry>& triangle_map_out,
    std::vector<int32_t>& out_grid,
    TriTimings* timings = nullptr,
    int border_padding = 0);
