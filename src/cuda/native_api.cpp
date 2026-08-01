// Implementation of the plain C++ interface declared in native_api.h.
// Mirrors bindings.cpp one-to-one, minus the pybind11 marshalling.

#include "native_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace delauney {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<Seed> sort_seeds(const std::vector<Seed>& seeds_in, int W, int H)
{
    if (seeds_in.empty())
        throw std::invalid_argument("seeds must not be empty");

    std::vector<Seed> seeds = seeds_in;
    for (const auto& s : seeds) {
        if (s.x < 0 || s.x >= W || s.y < 0 || s.y >= H)
            throw std::invalid_argument("seed coordinate out of bounds");
    }

    std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
    });

    for (size_t i = 1; i < seeds.size(); ++i) {
        if (seeds[i].x == seeds[i - 1].x && seeds[i].y == seeds[i - 1].y)
            throw std::invalid_argument("duplicate seed positions are not allowed");
    }
    return seeds;
}

bool cuda_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
        return false;
    return count > 0;
}

const char* cuda_device_name()
{
    static cudaDeviceProp props;
    if (!cuda_available())
        return "";
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess)
        return "";
    return props.name;
}

// ---------------------------------------------------------------------------
// RegularDelaunay
// ---------------------------------------------------------------------------

VoronoiGrid RegularDelaunay::compute(int width, int height,
                                     const std::vector<Seed>& seeds_in)
{
    auto seeds = sort_seeds(seeds_in, width, height);

    std::vector<int32_t> flat;
    cuda_compute_voronoi(width, height, seeds, flat);

    // flat layout: (y * W + x) * 2 -> [seed_id, distance]
    return VoronoiGrid(width, height, 2, std::move(flat));
}

// ---------------------------------------------------------------------------
// GridTriangulation
// ---------------------------------------------------------------------------

static TriangulationResult run_triangulation(const VoronoiGrid& vgrid,
                                             const std::vector<Seed>& seeds_in,
                                             TriTimings* timings)
{
    if (vgrid.channels() != 2)
        throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

    const int W = vgrid.width();
    const int H = vgrid.height();

    auto seeds = sort_seeds(seeds_in, W, H);
    const int N_seeds = static_cast<int>(seeds.size());

    std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
    for (int i = 0; i < N_seeds; ++i) {
        seed_xs[i] = seeds[i].x;
        seed_ys[i] = seeds[i].y;
    }

    TriangulationResult res;
    std::vector<int32_t> flat_out;

    cuda_compute_triangulation(W, H, vgrid.data().data(),
                               seed_xs, seed_ys,
                               res.triangle_map, flat_out, timings);

    res.grid = TriangulationGrid(W, H, 3, std::move(flat_out));
    return res;
}

TriangulationResult GridTriangulation::compute(const VoronoiGrid& vgrid,
                                               const std::vector<Seed>& seeds)
{
    return run_triangulation(vgrid, seeds, nullptr);
}

TriangulationResult GridTriangulation::compute_timed(const VoronoiGrid& vgrid,
                                                     const std::vector<Seed>& seeds,
                                                     TriTimings& timings)
{
    return run_triangulation(vgrid, seeds, &timings);
}

// ---------------------------------------------------------------------------
// IncrementalDelaunay
// ---------------------------------------------------------------------------

static void split_seeds(const std::vector<Seed>& seeds,
                        std::vector<int32_t>& xs, std::vector<int32_t>& ys)
{
    xs.resize(seeds.size());
    ys.resize(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i) {
        xs[i] = seeds[i].x;
        ys[i] = seeds[i].y;
    }
}

TriangulationResult IncrementalDelaunay::insert(const std::vector<Seed>& seeds)
{
    std::vector<int32_t> xs, ys;
    split_seeds(seeds, xs, ys);

    TriangulationResult res;
    std::vector<int32_t> tgrid;
    impl_.insert(xs, ys, res.triangle_map, tgrid);

    res.grid = TriangulationGrid(impl_.width(), impl_.height(), 3, std::move(tgrid));
    return res;
}

TriangulationResult IncrementalDelaunay::insert_timed(const std::vector<Seed>& seeds,
                                                      IncrementalTimings& timings)
{
    std::vector<int32_t> xs, ys;
    split_seeds(seeds, xs, ys);

    TriangulationResult res;
    std::vector<int32_t> tgrid;
    impl_.insert(xs, ys, res.triangle_map, tgrid, &timings);

    res.grid = TriangulationGrid(impl_.width(), impl_.height(), 3, std::move(tgrid));
    return res;
}

VoronoiGrid IncrementalDelaunay::get_voronoi_grid() const
{
    std::vector<int32_t> flat;
    impl_.get_voronoi_grid(flat);
    return VoronoiGrid(impl_.width(), impl_.height(), 2, std::move(flat));
}

}  // namespace delauney
