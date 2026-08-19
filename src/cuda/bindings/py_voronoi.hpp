// Python wrapper for the standalone Voronoi kernel. See bindings.cpp for docs.
#pragma once

#include "binding_helpers.hpp"
#include "../voronoi.cuh"

#include <cstring>

class PyVoronoi {
public:
    py::array_t<int32_t> compute(int width, int height, const py::object& seeds_obj)
    {
        auto seeds = sort_seeds(seeds_obj, width, height);

        std::vector<int32_t> flat;
        cuda_compute_voronoi(width, height, seeds, flat);

        // flat layout: (y * W + x) * 2 -> [seed_id, distance]
        // NumPy layout: (H, W, 2)
        py::array_t<int32_t> result({height, width, 2});
        std::memcpy(result.mutable_data(), flat.data(),
                    flat.size() * sizeof(int32_t));
        return result;
    }
};
