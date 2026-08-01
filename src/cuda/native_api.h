// Plain C++ mirror of the pybind11 bindings in bindings.cpp.
//
// Same three classes, same semantics (seed sorting / validation, output
// layouts), but with std:: types instead of NumPy arrays and dicts so the
// CUDA code can be driven from a standalone executable — no Python, no
// pybind11, no wheel build.
//
//   delauney::RegularDelaunay
//     .compute(width, height, seeds) -> VoronoiGrid          (H, W, 2)
//   delauney::GridTriangulation
//     .compute(voronoi_grid, seed_positions) -> TriangulationResult
//     .compute_timed(voronoi_grid, seed_positions, timings) -> TriangulationResult
//   delauney::IncrementalDelaunay
//     .insert(seeds) / .insert_timed(seeds, timings) -> TriangulationResult
//
// The Python dict {tid: (x, y, id_a, id_b, id_c)} becomes a
// std::vector<TriangleEntry> whose index is the triangle id.

#pragma once

#include <cstdint>
#include <vector>

#include "incremental.cuh"
#include "triangulation.cuh"
#include "voronoi.cuh"

namespace delauney {

// ---------------------------------------------------------------------------
// Grid3 — row-major (H, W, C) int32 buffer, the equivalent of the NumPy
// arrays the Python bindings hand back.  Element (x, y, ch) lives at
// ((y * W + x) * C + ch), matching the flat layout the kernels produce.
// ---------------------------------------------------------------------------

class Grid3 {
public:
    Grid3() = default;

    Grid3(int width, int height, int channels)
        : w_(width), h_(height), c_(channels),
          data_(static_cast<size_t>(width) * height * channels, 0) {}

    Grid3(int width, int height, int channels, std::vector<int32_t>&& data)
        : w_(width), h_(height), c_(channels), data_(std::move(data)) {}

    int width()    const { return w_; }
    int height()   const { return h_; }
    int channels() const { return c_; }
    bool empty()   const { return data_.empty(); }

    int32_t& at(int x, int y, int ch)
    {
        return data_[(static_cast<size_t>(y) * w_ + x) * c_ + ch];
    }
    int32_t at(int x, int y, int ch) const
    {
        return data_[(static_cast<size_t>(y) * w_ + x) * c_ + ch];
    }

    std::vector<int32_t>&       data()       { return data_; }
    const std::vector<int32_t>& data() const { return data_; }

    bool operator==(const Grid3& o) const
    {
        return w_ == o.w_ && h_ == o.h_ && c_ == o.c_ && data_ == o.data_;
    }
    bool operator!=(const Grid3& o) const { return !(*this == o); }

private:
    int w_ = 0, h_ = 0, c_ = 0;
    std::vector<int32_t> data_;
};

using VoronoiGrid       = Grid3;   // (H, W, 2): [seed_id, distance]
using TriangulationGrid = Grid3;   // (H, W, 3)

struct TriangulationResult {
    std::vector<TriangleEntry> triangle_map;   // index == triangle id
    TriangulationGrid          grid;           // (H, W, 3)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Sort seeds by (x asc, y asc) — this is what assigns the seed IDs.
// Throws std::invalid_argument on an empty list, an out-of-bounds coordinate
// or a duplicate position, exactly like the pybind path.
std::vector<Seed> sort_seeds(const std::vector<Seed>& seeds, int W, int H);

// True when at least one CUDA device is visible.
bool cuda_available();

// Name of device 0 (empty string when no device is available).
const char* cuda_device_name();

// ---------------------------------------------------------------------------
// RegularDelaunay
// ---------------------------------------------------------------------------

class RegularDelaunay {
public:
    // Manhattan-distance Voronoi diagram: (H, W, 2) = (seed_id, distance).
    VoronoiGrid compute(int width, int height, const std::vector<Seed>& seeds);
};

// ---------------------------------------------------------------------------
// GridTriangulation
// ---------------------------------------------------------------------------

class GridTriangulation {
public:
    TriangulationResult compute(const VoronoiGrid& vgrid,
                                const std::vector<Seed>& seed_positions);

    // Same as compute() but also fills the GPU sub-phase timings.
    TriangulationResult compute_timed(const VoronoiGrid& vgrid,
                                      const std::vector<Seed>& seed_positions,
                                      TriTimings& timings);
};

// ---------------------------------------------------------------------------
// IncrementalDelaunay — thin wrapper over the device-resident core class,
// mirroring PyIncrementalDelaunay.  Seeds keep insertion-order IDs and are
// *not* sorted here (same as the binding).
// ---------------------------------------------------------------------------

class IncrementalDelaunay {
public:
    IncrementalDelaunay(int width, int height, int max_seeds)
        : impl_(width, height, max_seeds) {}

    TriangulationResult insert(const std::vector<Seed>& seeds);
    TriangulationResult insert_timed(const std::vector<Seed>& seeds,
                                     IncrementalTimings& timings);

    VoronoiGrid get_voronoi_grid() const;

    int seed_count() const { return impl_.seed_count(); }
    int width()      const { return impl_.width(); }
    int height()     const { return impl_.height(); }

private:
    ::IncrementalDelaunay impl_;
};

}  // namespace delauney
