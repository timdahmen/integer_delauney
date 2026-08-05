// pybind11 bindings for the CUDA Voronoi + triangulation kernels.
//
// Exposed module: _delauney_cuda
//   RegularDelaunay:
//     .compute(width, height, seeds) -> np.ndarray shape (H, W, 2) int32
//   GridTriangulation:
//     .compute(voronoi_grid, seed_positions) -> (dict, np.ndarray shape (H,W,3) int32)
//
// `seeds` / `seed_positions` must be a sequence of (x, y) pairs.
// Seed ordering (ascending x, tiebreak ascending y) is applied here so that
// the CUDA API matches the Python reference exactly.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "voronoi.cuh"
#include "triangulation.cuh"
#include "incremental.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Sort seeds by (x asc, y asc) and return the sorted list.
static std::vector<Seed> sort_seeds(const py::object& seeds_obj, int W, int H)
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

    // Check for duplicates
    for (size_t i = 1; i < seeds.size(); ++i) {
        if (seeds[i].x == seeds[i-1].x && seeds[i].y == seeds[i-1].y)
            throw std::invalid_argument("duplicate seed positions are not allowed");
    }
    return seeds;
}

// ---------------------------------------------------------------------------
// RegularDelaunay wrapper
// ---------------------------------------------------------------------------

class PyRegularDelaunay {
public:
    py::array_t<int32_t> compute(int width, int height, const py::object& seeds_obj)
    {
        auto seeds = sort_seeds(seeds_obj, width, height);

        std::vector<int32_t> flat;
        cuda_compute_voronoi(width, height, seeds, flat);

        // flat layout: (y * W + x) * 2 → [seed_id, distance]
        // NumPy layout: (H, W, 2)
        py::array_t<int32_t> result({height, width, 2});
        std::memcpy(result.mutable_data(), flat.data(),
                    flat.size() * sizeof(int32_t));
        return result;
    }
};

// ---------------------------------------------------------------------------
// GridTriangulation wrapper
// ---------------------------------------------------------------------------

class PyGridTriangulation {
public:
    py::tuple compute(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        int H = static_cast<int>(info.shape[0]);
        int W = static_cast<int>(info.shape[1]);

        auto seeds = sort_seeds(seeds_obj, W, H);
        int N_seeds = static_cast<int>(seeds.size());

        std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
        for (int i = 0; i < N_seeds; ++i) {
            seed_xs[i] = seeds[i].x;
            seed_ys[i] = seeds[i].y;
        }

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;

        cuda_compute_triangulation(W, H,
            static_cast<const int32_t*>(info.ptr),
            seed_xs, seed_ys, tri_map, flat_out, nullptr);

        return _build_output(tri_map, flat_out, H, W);
    }

    // compute_timed() — same as compute() but also returns a timings dict.
    // Signature: (voronoi_grid, seed_positions) -> (tri_map, grid, timings)
    py::tuple compute_timed(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        int H = static_cast<int>(info.shape[0]);
        int W = static_cast<int>(info.shape[1]);

        auto seeds = sort_seeds(seeds_obj, W, H);
        int N_seeds = static_cast<int>(seeds.size());
        std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
        for (int i = 0; i < N_seeds; ++i) {
            seed_xs[i] = seeds[i].x;
            seed_ys[i] = seeds[i].y;
        }

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;
        TriTimings timings;

        cuda_compute_triangulation(W, H,
            static_cast<const int32_t*>(info.ptr),
            seed_xs, seed_ys, tri_map, flat_out, &timings);

        py::dict py_timings;
        py_timings["detect_ms"] = timings.detect_ms;
        py_timings["dedup_ms"]  = timings.dedup_ms;
        py_timings["assign_ms"] = timings.assign_ms;

        auto result = _build_output(tri_map, flat_out, H, W);
        return py::make_tuple(result[0], result[1], py_timings);
    }

private:
    static py::tuple _build_output(const std::vector<TriangleEntry>& tri_map,
                                   const std::vector<int32_t>& flat_out,
                                   int H, int W)
    {
        py::dict py_map;
        for (int32_t tid = 0; tid < (int32_t)tri_map.size(); ++tid) {
            const auto& e = tri_map[tid];
            py_map[py::int_(tid)] = py::make_tuple(e.x, e.y, e.id_a, e.id_b, e.id_c);
        }
        py::array_t<int32_t> out_arr({H, W, 3});
        std::memcpy(out_arr.mutable_data(), flat_out.data(),
                    flat_out.size() * sizeof(int32_t));
        return py::make_tuple(py_map, out_arr);
    }
};

// ---------------------------------------------------------------------------
// IncrementalDelaunay wrapper
// ---------------------------------------------------------------------------

class PyIncrementalDelaunay {
public:
    PyIncrementalDelaunay(int width, int height, int max_seeds)
        : impl_(width, height, max_seeds) {}

    // insert(seeds) -> (triangle_map, triangulation_grid)  or
    // insert_timed(seeds) -> (triangle_map, triangulation_grid, timings_dict)
    py::tuple insert(const py::object& seeds_obj)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        impl_.insert(xs, ys, tri_map, tgrid);
        return _build_output(tri_map, tgrid, impl_.height(), impl_.width());
    }

    py::tuple insert_timed(const py::object& seeds_obj)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        IncrementalTimings t;
        impl_.insert(xs, ys, tri_map, tgrid, &t);

        py::dict py_t;
        py_t["bfs_ms"]    = t.bfs_ms;
        py_t["detect_ms"] = t.detect_ms;
        py_t["dedup_ms"]  = t.dedup_ms;
        py_t["assign_ms"] = t.assign_ms;

        auto out = _build_output(tri_map, tgrid, impl_.height(), impl_.width());
        return py::make_tuple(out[0], out[1], py_t);
    }

    py::array_t<int32_t> get_voronoi_grid() const
    {
        std::vector<int32_t> flat;
        impl_.get_voronoi_grid(flat);
        int H = impl_.height(), W = impl_.width();
        py::array_t<int32_t> arr({H, W, 2});
        std::memcpy(arr.mutable_data(), flat.data(), flat.size() * sizeof(int32_t));
        return arr;
    }

    int seed_count() const { return impl_.seed_count(); }
    int width()      const { return impl_.width(); }
    int height()     const { return impl_.height(); }

private:
    IncrementalDelaunay impl_;

    static void _parse_seeds(const py::object& seeds_obj,
                             std::vector<int32_t>& xs, std::vector<int32_t>& ys)
    {
        for (auto item : seeds_obj) {
            auto pair = item.cast<py::sequence>();
            xs.push_back(pair[0].cast<int32_t>());
            ys.push_back(pair[1].cast<int32_t>());
        }
    }

    static py::tuple _build_output(const std::vector<TriangleEntry>& tri_map,
                                   const std::vector<int32_t>& flat_out,
                                   int H, int W)
    {
        py::dict py_map;
        for (int32_t tid = 0; tid < (int32_t)tri_map.size(); ++tid) {
            const auto& e = tri_map[tid];
            py_map[py::int_(tid)] = py::make_tuple(e.x, e.y, e.id_a, e.id_b, e.id_c);
        }
        py::array_t<int32_t> out_arr({H, W, 3});
        std::memcpy(out_arr.mutable_data(), flat_out.data(),
                    flat_out.size() * sizeof(int32_t));
        return py::make_tuple(py_map, out_arr);
    }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_delauney_cuda, m)
{
	m.doc() = "CUDA-accelerated Voronoi + Delaunay triangulation";

	py::class_<PyRegularDelaunay>(m, "RegularDelaunay")
        .def(py::init<>())
        .def("compute", &PyRegularDelaunay::compute,
             py::arg("width"), py::arg("height"), py::arg("seeds"),
             "Compute Manhattan-distance Voronoi diagram.\n\n"
             "Returns int32 array of shape (height, width, 2): "
             "(seed_id, distance) per cell.");

    py::class_<PyGridTriangulation>(m, "GridTriangulation")
        .def(py::init<>())
        .def("compute", &PyGridTriangulation::compute,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             "Extract Delaunay triangulation from Voronoi grid.\n\n"
             "Returns (triangle_map, triangulation_grid) where "
             "triangle_map is {int: (x,y,id_a,id_b,id_c)} and "
             "triangulation_grid has shape (H, W, 3).")
        .def("compute_timed", &PyGridTriangulation::compute_timed,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             "Same as compute() but also returns a timings dict.\n\n"
             "Returns (triangle_map, triangulation_grid, timings) where "
             "timings has keys detect_ms, dedup_ms, assign_ms (float, ms).");

    py::class_<PyIncrementalDelaunay>(m, "IncrementalDelaunay")
        .def(py::init<int, int, int>(),
             py::arg("width"), py::arg("height"), py::arg("max_seeds"),
             "Create an incremental Delaunay triangulator with device-resident state.\n\n"
             "max_seeds: upper bound on total seeds ever inserted.")
        .def("insert", &PyIncrementalDelaunay::insert,
             py::arg("seeds"),
             "Insert a batch of (x, y) seeds and update the triangulation.\n\n"
             "Returns (triangle_map, triangulation_grid) where triangle_map is\n"
             "{int: (x,y,id_a,id_b,id_c)} and triangulation_grid has shape (H,W,3).")
        .def("insert_timed", &PyIncrementalDelaunay::insert_timed,
             py::arg("seeds"),
             "Same as insert() but also returns a timings dict with keys\n"
             "bfs_ms, detect_ms, dedup_ms, assign_ms.")
        .def("get_voronoi_grid", &PyIncrementalDelaunay::get_voronoi_grid,
             "Return the current Voronoi grid as int32 array of shape (H, W, 2).")
        .def_property_readonly("seed_count", &PyIncrementalDelaunay::seed_count)
        .def_property_readonly("width",  &PyIncrementalDelaunay::width)
        .def_property_readonly("height", &PyIncrementalDelaunay::height);
}
