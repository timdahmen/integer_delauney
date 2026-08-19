// The batch triangulate() entry point. See bindings.cpp for docs.
#pragma once

#include "binding_helpers.hpp"
#include "../triangulation.cuh"

// The whole pipeline in one call, and the one callers should reach for.
//
// Voronoi().compute() followed by GridTriangulation().compute() was the usual
// pairing, and at any padding above 0 it built the diagram twice: once in the
// caller, and once inside the triangulator on the padded canvas, because
// detection needs the padded one. The caller's copy then served only to fill
// two output channels that the padded diagram already contains in its interior
// window. Going through here computes it once.
inline py::tuple triangulate(
    int width, int height,
    const py::object& seeds_obj,
    int border_padding = AUTO_BORDER_PADDING,
    bool as_arrays = false)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("width and height must be positive");

    auto seeds = sort_seeds(seeds_obj, width, height);
    border_padding = resolve_border_padding(border_padding);

    int N_seeds = static_cast<int>(seeds.size());
    std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
    for (int i = 0; i < N_seeds; ++i) {
        seed_xs[i] = seeds[i].x;
        seed_ys[i] = seeds[i].y;
    }

    std::vector<TriangleEntry> tri_map;
    std::vector<int32_t> flat_out;

    // nullptr is the point of this function: the triangulator builds whichever
    // diagram it actually needs and nothing else.
    cuda_compute_triangulation(width, height, nullptr,
        seed_xs, seed_ys, tri_map, flat_out, nullptr, border_padding);

    return build_output(tri_map, flat_out, height, width, as_arrays);
}
