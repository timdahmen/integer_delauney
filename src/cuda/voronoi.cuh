#pragma once
#include "common.h"

#include <cstdint>
#include <vector>

void cuda_compute_voronoi(const int W, const int H, const std::vector<Vec2i>& seeds, std::vector<int32_t>& out_grid);
