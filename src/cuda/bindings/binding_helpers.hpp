// Internal helpers shared by the Python wrapper classes. Nothing here is
// bound to the module directly -- see bindings.cpp for the public API.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../voronoi.cuh"   // Seed
#include "../delaunay.cuh"  // DEFAULT_BORDER_PADDING, TriangleEntry

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Sentinel for a border_padding argument meaning "use the library default" --
// the int-typed spelling of the reference's border_padding=None.
static constexpr int AUTO_BORDER_PADDING = -1;

// Sort seeds by (x asc, y asc) and return the sorted list.
inline std::vector<Seed> sort_seeds(const py::object& seeds_obj, int W, int H)
{
    std::vector<Seed> seeds;
    for (auto item : seeds_obj) {
        auto pair = item.cast<py::sequence>();
        int x = pair[0].cast<int>();
        int y = pair[1].cast<int>();
        if (x < 0 || x >= W || y < 0 || y >= H)
            throw std::invalid_argument("seed coordinate out of bounds");
        seeds.push_back({x, y});
    }
    if (seeds.empty())
        throw std::invalid_argument("seeds must not be empty");

    std::sort(seeds.begin(), seeds.end(), [](const Seed& a, const Seed& b) {
        return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
    });

    for (size_t i = 1; i < seeds.size(); ++i) {
        if (seeds[i].x == seeds[i-1].x && seeds[i].y == seeds[i-1].y)
            throw std::invalid_argument("duplicate seed positions are not allowed");
    }
    return seeds;
}

// See BORDER_PADDING_BOUND.md for how DEFAULT_BORDER_PADDING was chosen.
inline int resolve_border_padding(int border_padding)
{
    if (border_padding != AUTO_BORDER_PADDING) return border_padding;
    return DEFAULT_BORDER_PADDING;
}

// Pack a triangulation into the (triangle_map, grid) pair Python sees.
//
// Triangles are held C++-side as a contiguous vector whose index is the
// triangle id. The dict form rebuilds that as one Python int plus one 5-tuple
// per triangle; as_arrays hands back the (N_tri, 3) indices instead, skipping
// the per-triangle Python objects.
inline py::tuple build_output(const std::vector<TriangleEntry>& tri_map,
                              const std::vector<int32_t>& flat_out,
                              int H, int W, bool as_arrays = false)
{
    py::array_t<int32_t> out_arr({H, W, 3});
    std::memcpy(out_arr.mutable_data(), flat_out.data(),
                flat_out.size() * sizeof(int32_t));

    const int32_t n_tri = (int32_t)tri_map.size();
    if (as_arrays) {
        py::array_t<int32_t> verts({(int)n_tri, 3});
        auto* p = verts.mutable_data();
        for (int32_t tid = 0; tid < n_tri; ++tid) {
            p[tid * 3]     = tri_map[tid].id_a;
            p[tid * 3 + 1] = tri_map[tid].id_b;
            p[tid * 3 + 2] = tri_map[tid].id_c;
        }
        return py::make_tuple(verts, out_arr);
    }

    py::dict py_map;
    for (int32_t tid = 0; tid < n_tri; ++tid) {
        const auto& e = tri_map[tid];
        py_map[py::int_(tid)] = py::make_tuple(e.x, e.y, e.id_a, e.id_b, e.id_c);
    }
    return py::make_tuple(py_map, out_arr);
}
