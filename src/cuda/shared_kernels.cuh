#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// size of shared memory tile and TILE_SIZExTILE_SIZE threads per block
// HAS to be 32 == threads/warp
static constexpr uint32_t TILE_SIZE_BFS = 32;

// number of kernel invocations that are run, before the flag is checked.
// Value should be tuned, based on expected input data, to balance MemCpy-Time vs Kernel-Runtime.
// Making it too small leads to a lot of memcpy stalls, making it too big to unnecessary Kernel Runs at the end.
static constexpr int AGGREGATE_ITERATIONS_BFS = 4; // tune empirically, same as voronoi.cu

static constexpr int RASTER_TRIS_PER_BLOCK = 32;
static constexpr int THREADS_PER_TRI = 32;

static constexpr int32_t UNDEF = -1;


// ---------------------------------------------------------------------------
// Shared structs
// ---------------------------------------------------------------------------

struct KeyLess {
	__device__ bool operator()(const Vec3i& x, const Vec3i& y) const {
		if (x.a != y.a) return x.a < y.a;
		if (x.b != y.b) return x.b < y.b;
		return x.c < y.c;
	}
};

struct KeyEqual {
	__device__ bool operator()(const Vec3i& x, const Vec3i& y) const { return x.a == y.a && x.b == y.b && x.c == y.c; }
};

// ---------------------------------------------------------------------------
// Shared kernels triangulate
// ---------------------------------------------------------------------------

__global__ void build_output_kernel(int32_t* __restrict__ out,
									const int32_t* __restrict__ voronoi_grid,
									const int32_t* __restrict__ canvas,
									const int N,
									const int default_id);

__global__ void build_triangle_map_out(TriangleEntry* __restrict__ out,
									   const Vec2i* __restrict__ tri_xy,
									   const Vec3i* __restrict__ tri_og_seeds,
									   const int N);

__global__ void find_triangle_seeds_kernel(const int32_t* __restrict__ voronoi_grid,
										   const int W,
										   const int H,
										   Vec2i* __restrict__ raw_xy,
										   Vec3i* __restrict__ raw_key,
										   Vec3i* __restrict__ raw_orig,
										   int32_t* __restrict__ counter,
										   const int cap);

__global__ void rasterize_tri_kernel(int32_t* __restrict__ canvas,
									 const int W,
									 const Vec2i* __restrict__ seed_pos,
									 const Vec3i* __restrict__ tri_seed_ids,
									 const int n_tris);

// ---------------------------------------------------------------------------
// Shared kernels voronoi
// ---------------------------------------------------------------------------

__global__ void voronoi_step_kernel(const int32_t* __restrict__ src_raw,
									int32_t* __restrict__ dst_raw,
									const int W,
									const int H,
									int32_t* __restrict__ flag_write,
									int32_t* __restrict__ flag_reset_for_next,
									const uint32_t* __restrict__ row_mask_read,
									uint32_t* __restrict__ row_mask_write);
