// Small __device__ helpers shared by the batch (voronoi.cu, triangulation.cu)
// and incremental (delaunay.cu) pipelines.
//
// CUDA-only: never included by bindings.cpp, which is compiled as plain C++
// and does not understand __device__/__forceinline__. Keeping that boundary
// is why this is a separate header from voronoi.cuh/triangulation.cuh, which
// bindings.cpp does include.
#pragma once

#include <cstdint>

// Compare two (seed_id, distance) BFS candidates; true if b beats a. Lower
// distance wins; on a tie, higher seed_id wins.
//
// Only among the candidates a cell actually sees: propagation is local, and
// both callers advance one cell per iteration, so a run can settle at a fixed
// point that is not nearest-seed (rare, grows with seed density -- measured
// 2/16384 pixels at 128x128 with 400 seeds, off by 1-3 in squared distance).
// See Voronoi's docstring in reference/voronoi.py.
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
                       float ax, float ay,
                       float bx, float by,
                       float cx, float cy)
{
    float d1 = cross2d(px, py, ax, ay, bx, by);
    float d2 = cross2d(px, py, bx, by, cx, cy);
    float d3 = cross2d(px, py, cx, cy, ax, ay);
    bool has_neg = (d1 < 0.f) || (d2 < 0.f) || (d3 < 0.f);
    bool has_pos = (d1 > 0.f) || (d2 > 0.f) || (d3 > 0.f);
    return !(has_neg && has_pos);
}
