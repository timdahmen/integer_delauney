#pragma once
#include <cstdint>
#include <vector>

struct Seed {
    int32_t x, y;
};

void cuda_compute_voronoi(
    int W, int H,
    const std::vector<Seed>& seeds,
    std::vector<int32_t>& out_grid);
