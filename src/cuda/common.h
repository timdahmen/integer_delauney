#pragma once
#include <cstdint>

struct Vec2i {
	int32_t x, y;
};

struct Vec3i {
	int32_t a, b, c;
};

struct Seed {
	int32_t x, y;
};

struct alignas(8) Cell {
	int32_t id;
	int32_t distance;
};
static_assert(sizeof(Cell) == 2 * sizeof(int32_t), "Unexpected Cell layout");

struct TriangleEntry {
	int32_t x, y;
	int32_t id_a, id_b, id_c;
};
static_assert(sizeof(TriangleEntry) == 5 * sizeof(int32_t), "Unexpected TriangleEntry layout");
