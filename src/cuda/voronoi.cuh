#pragma once
#include <cstdint>
#include <vector>

//: A Voronoi site: an (x, y) pixel position and, implicitly, its index in the
//: seeds array that owns it.
struct Seed {
    int32_t x, y;
};

//: A Voronoi cell no seed owns yet, i.e. one the BFS has not reached.
//:
//: The batch path never exposes one - it runs the BFS to completion before
//: anything reads the grid -- but the incremental path detects into a partially
//: filled grid, so consumers must tolerate it. Lives here rather than beside the
//: triangulation code because it is a property of the Voronoi grid itself.
static constexpr int32_t UNDEF_SEED = -1;

//: Byte value for cudaMemset when filling an int32 buffer with a -1 sentinel.
//:
//: cudaMemset sets *bytes*, not words. It happens to produce the right result
//: for -1 because every byte of -1 is 0xFF -- which is exactly why -1 is the
//: sentinel and not some other negative number. Spelled out so the coincidence
//: is deliberate rather than something a reader has to rederive.
static constexpr int SENTINEL_BYTE = 0xFF;

//: Nearest-seed Voronoi diagram of a W x H canvas, by parallel BFS to a fixed
//: point (see voronoi.cu's file header for the propagation rule).
//:
//: returns (H * W * 2) int32: interleaved (seed_id, squared L2 distance) per
//: pixel, seed_id indexing `seeds`.
void cuda_compute_voronoi(
    int W, int H,
    const std::vector<Seed>& seeds,
    std::vector<int32_t>& out_grid);
