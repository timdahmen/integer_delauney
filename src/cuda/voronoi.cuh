#pragma once
#include <cstdint>
#include <vector>

struct alignas(8) VoronoiCell {
	int32_t id;
	int32_t distance;
};

struct Seed {
	int32_t x, y;
};
	
void cuda_compute_voronoi(const int W, const int H, const std::vector<Seed>& seeds, std::vector<int32_t>& out_grid);
