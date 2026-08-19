// Python wrapper for the Voronoi-grid-to-triangulation step. See bindings.cpp
// for docs.
#pragma once

#include "binding_helpers.hpp"
#include "../triangulation.cuh"

class PyGridTriangulation {
public:
    py::tuple compute(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = AUTO_BORDER_PADDING,
        bool as_arrays = false)
    {
        auto p = _prepare(vgrid, seeds_obj, border_padding);

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;
        cuda_compute_triangulation(p.W, p.H, p.grid_ptr,
            p.seed_xs, p.seed_ys, tri_map, flat_out, nullptr, p.border_padding);

        return build_output(tri_map, flat_out, p.H, p.W, as_arrays);
    }

    py::tuple compute_timed(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = AUTO_BORDER_PADDING)
    {
        auto p = _prepare(vgrid, seeds_obj, border_padding);

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;
        TriTimings timings;
        cuda_compute_triangulation(p.W, p.H, p.grid_ptr,
            p.seed_xs, p.seed_ys, tri_map, flat_out, &timings, p.border_padding);

        py::dict py_timings;
        py_timings["detect_ms"] = timings.detect_ms;
        py_timings["dedup_ms"]  = timings.dedup_ms;
        py_timings["assign_ms"] = timings.assign_ms;

        // Timing/debug variants keep the dict form; not a hot path.
        auto result = build_output(tri_map, flat_out, p.H, p.W, /*as_arrays=*/false);
        return py::make_tuple(result[0], result[1], py_timings);
    }

    py::tuple compute_debug(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = AUTO_BORDER_PADDING)
    {
        auto p = _prepare(vgrid, seeds_obj, border_padding);

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;
        std::vector<int32_t> padded_flat;
        cuda_compute_triangulation(p.W, p.H, p.grid_ptr,
            p.seed_xs, p.seed_ys, tri_map, flat_out,
            nullptr, p.border_padding, &padded_flat);

        int H_det = p.H + 2 * p.border_padding;
        int W_det = p.W + 2 * p.border_padding;
        py::array_t<int32_t> padded_arr({H_det, W_det, 2});
        if (!padded_flat.empty())
            std::memcpy(padded_arr.mutable_data(), padded_flat.data(),
                        padded_flat.size() * sizeof(int32_t));

        // Timing/debug variants keep the dict form; not a hot path.
        auto result = build_output(tri_map, flat_out, p.H, p.W, /*as_arrays=*/false);
        return py::make_tuple(result[0], result[1], padded_arr);
    }

private:
    // Preamble shared by compute/compute_timed/compute_debug: validate the
    // grid, sort the seeds and resolve the padding.
    struct Prepared {
        int H, W, border_padding;
        const int32_t* grid_ptr;
        std::vector<int32_t> seed_xs, seed_ys;
    };

    static Prepared _prepare(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        Prepared p;
        p.H = static_cast<int>(info.shape[0]);
        p.W = static_cast<int>(info.shape[1]);
        p.grid_ptr = static_cast<const int32_t*>(info.ptr);

        auto seeds = sort_seeds(seeds_obj, p.W, p.H);
        p.border_padding = resolve_border_padding(border_padding);

        int N_seeds = static_cast<int>(seeds.size());
        p.seed_xs.resize(N_seeds);
        p.seed_ys.resize(N_seeds);
        for (int i = 0; i < N_seeds; ++i) {
            p.seed_xs[i] = seeds[i].x;
            p.seed_ys[i] = seeds[i].y;
        }
        return p;
    }
};
