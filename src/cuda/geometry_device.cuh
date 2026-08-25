// Small __device__ helpers shared by the batch (voronoi.cu, triangulation.cu)
// and incremental (delaunay_voronoi.cu, delaunay_assign.cu) pipelines.
//
// CUDA-only: never included by bindings.cpp, which is compiled as plain C++
// and does not understand __device__/__forceinline__. Keeping that boundary
// is why this is a separate header from voronoi.cuh/triangulation.cuh, which
// bindings.cpp does include.
#pragma once

#include <cstdint>

#include "voronoi.cuh"   // UNDEF_SEED

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

//: One jump-flood propagation step for pixel (x, y) at jump distance `step`:
//: scan its 8 neighbours offset by (+-step, +-step) (including the 4 purely
//: cardinal ones, step in one axis and 0 in the other) in `src`, and return
//: the best (seed_id, distance) among the pixel's current owner and each
//: neighbour's owner. Returns whether that differs from the pixel's current
//: value, so the caller knows to set its own convergence flag.
//:
//: step=1 is what used to be the only propagation rule here (plain BFS, one
//: cell per pass); run_bfs_() (delaunay_voronoi.cu) and cuda_compute_voronoi()
//: (voronoi.cu) now open with several passes at halving step sizes (jump
//: flooding: https://en.wikipedia.org/wiki/Jump_flooding_algorithm) so a
//: seed's influence can cross the canvas in O(log n) passes instead of one
//: pixel at a time, then finish with step=1 passes until nothing changes --
//: same convergence criterion as before, just reached in far fewer passes.
//:
//: Distance is recomputed directly from the seed position, never accumulated
//: through neighbours -- that would give a Manhattan metric, not L2. Shared by
//: voronoi.cu's and delaunay_voronoi.cu's step kernels, which differ only in
//: what they do with the changed/unchanged result (an incremental dirty mask
//: vs. nothing).
__device__ __forceinline__
bool voronoi_jfa_step(int x, int y, int W, int H,
                      const int32_t* __restrict__ src,
                      const int32_t* __restrict__ seed_xs,
                      const int32_t* __restrict__ seed_ys,
                      int step,
                      int32_t& best_id, int32_t& best_d)
{
    const int base = (y * W + x) * 2;
    const int32_t cur_id = src[base], cur_d = src[base + 1];
    best_id = cur_id; best_d = cur_d;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx * step, ny = y + dy * step;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            int32_t n_id = src[(ny * W + nx) * 2];
            if (n_id == UNDEF_SEED) continue;
            int32_t ndx = x - seed_xs[n_id];
            int32_t ndy = y - seed_ys[n_id];
            int32_t n_d = ndx * ndx + ndy * ndy;
            if (best_id == UNDEF_SEED || beats(best_id, best_d, n_id, n_d)) {
                best_id = n_id; best_d = n_d;
            }
        }
    }
    return best_id != cur_id || best_d != cur_d;
}

//: Starting jump distance for the descent above: the largest power of two
//: that does not exceed the canvas's longer side. Halving this down to 1
//: covers the whole canvas in ceil(log2(max(W,H)))+1 passes. Host-side
//: scheduling helper, not a kernel -- shared so voronoi.cu's and
//: delaunay_voronoi.cu's independent host loops use the same schedule.
inline int jfa_start_step(int W, int H)
{
    int n = W > H ? W : H;
    int step = 1;
    while (step * 2 <= n) step *= 2;
    return step;
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
