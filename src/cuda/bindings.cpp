// pybind11 bindings for the CUDA Voronoi + triangulation kernels.
//
// Exposed module: _delauney_cuda -- Voronoi, GridTriangulation and
// Delaunay.  Per-method signatures live in the docstrings below;
// API_REFERENCE.md documents the Python-facing contract.
//
// `seeds` / `seed_positions` must be a sequence of (x, y) pairs.
// Seed ordering (ascending x, tiebreak ascending y) is applied here so that
// the CUDA API matches the Python reference exactly.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "voronoi.cuh"
#include "triangulation.cuh"
#include "delaunay.cuh"

#include <algorithm>
#include <cmath>
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

    for (size_t i = 1; i < seeds.size(); ++i) {
        if (seeds[i].x == seeds[i-1].x && seeds[i].y == seeds[i-1].y)
            throw std::invalid_argument("duplicate seed positions are not allowed");
    }
    return seeds;
}

// ---------------------------------------------------------------------------
// Voronoi wrapper
// ---------------------------------------------------------------------------

class PyVoronoi {
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

// A negative border_padding means "use the library default" -- the int-typed
// spelling of the reference's border_padding=None.
//
// This used to estimate one per call as sqrt(area / n_seeds), which is a global
// density argument for a quantity that depends on triangle shape: circumradius
// grows without limit as three points approach collinearity, at any seed count.
// Measured on a real frame the estimate was out by 3.5x at the tail. A fixed
// default is not a better estimate, it is an honest one -- the padding a seed
// set needs is bounded only when the caller controls how densely the hull
// boundary is sampled. See BORDER_PADDING_BOUND.md.
static int resolve_border_padding(int border_padding, int, int, int)
{
    if (border_padding >= 0) return border_padding;
    return DEFAULT_BORDER_PADDING;
}

//: Pack a triangulation into the (triangle_map, grid) pair Python sees.
//:
//: Triangles are held C++-side as a contiguous vector whose index IS the
//: triangle id. The dict form rebuilds that as one Python int plus one 5-tuple
//: per triangle; as_arrays hands back the (N_tri, 3) indices instead, skipping
//: the per-triangle Python objects.
//:
//: One definition for all three entry points. The two classes carried a copy
//: each -- semantically identical, differing only in how they wrote the vertex
//: array -- which is the same drift risk the detection kernel already had.
static py::tuple _build_output(const std::vector<TriangleEntry>& tri_map,
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

class PyGridTriangulation {
public:
    py::tuple compute(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = -1,
        bool as_arrays = false)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        int H = static_cast<int>(info.shape[0]);
        int W = static_cast<int>(info.shape[1]);

        auto seeds = sort_seeds(seeds_obj, W, H);
        int N_seeds = static_cast<int>(seeds.size());
        border_padding = resolve_border_padding(border_padding, W, H, N_seeds);

        std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
        for (int i = 0; i < N_seeds; ++i) {
            seed_xs[i] = seeds[i].x;
            seed_ys[i] = seeds[i].y;
        }

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;

        cuda_compute_triangulation(W, H,
            static_cast<const int32_t*>(info.ptr),
            seed_xs, seed_ys, tri_map, flat_out, nullptr, border_padding);

        return _build_output(tri_map, flat_out, H, W, as_arrays);
    }

    py::tuple compute_timed(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = -1)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        int H = static_cast<int>(info.shape[0]);
        int W = static_cast<int>(info.shape[1]);

        auto seeds = sort_seeds(seeds_obj, W, H);
        int N_seeds = static_cast<int>(seeds.size());
        border_padding = resolve_border_padding(border_padding, W, H, N_seeds);
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
            seed_xs, seed_ys, tri_map, flat_out, &timings, border_padding);

        py::dict py_timings;
        py_timings["detect_ms"] = timings.detect_ms;
        py_timings["dedup_ms"]  = timings.dedup_ms;
        py_timings["assign_ms"] = timings.assign_ms;

        // Timing/debug variants keep the dict form; not a hot path.
        auto result = _build_output(tri_map, flat_out, H, W, /*as_arrays=*/false);
        return py::make_tuple(result[0], result[1], py_timings);
    }

    py::tuple compute_debug(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& vgrid,
        const py::object& seeds_obj,
        int border_padding = -1)
    {
        auto info = vgrid.request();
        if (info.ndim != 3 || info.shape[2] != 2)
            throw std::invalid_argument("voronoi_grid must have shape (H, W, 2)");

        int H = static_cast<int>(info.shape[0]);
        int W = static_cast<int>(info.shape[1]);

        auto seeds = sort_seeds(seeds_obj, W, H);
        int N_seeds = static_cast<int>(seeds.size());
        border_padding = resolve_border_padding(border_padding, W, H, N_seeds);
        std::vector<int32_t> seed_xs(N_seeds), seed_ys(N_seeds);
        for (int i = 0; i < N_seeds; ++i) {
            seed_xs[i] = seeds[i].x;
            seed_ys[i] = seeds[i].y;
        }

        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> flat_out;
        std::vector<int32_t> padded_flat;

        cuda_compute_triangulation(W, H,
            static_cast<const int32_t*>(info.ptr),
            seed_xs, seed_ys, tri_map, flat_out,
            nullptr, border_padding, &padded_flat);

        int H_det = H + 2 * border_padding;
        int W_det = W + 2 * border_padding;
        py::array_t<int32_t> padded_arr({H_det, W_det, 2});
        if (!padded_flat.empty())
            std::memcpy(padded_arr.mutable_data(), padded_flat.data(),
                        padded_flat.size() * sizeof(int32_t));

        // Timing/debug variants keep the dict form; not a hot path.
        auto result = _build_output(tri_map, flat_out, H, W, /*as_arrays=*/false);
        return py::make_tuple(result[0], result[1], padded_arr);
    }

};

// ---------------------------------------------------------------------------
// Delaunay wrapper
// ---------------------------------------------------------------------------

class PyDelaunay {
public:
    PyDelaunay(int width, int height, int max_seeds,
                          int border_padding)
        : impl_(width, height, max_seeds, border_padding) {}

    py::tuple insert(const py::object& seeds_obj, bool as_arrays)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        impl_.insert(xs, ys, tri_map, tgrid);
        return _build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                             as_arrays);
    }

    py::tuple insert_timed(const py::object& seeds_obj, bool as_arrays)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        InsertTimings t;
        impl_.insert(xs, ys, tri_map, tgrid, &t);
        auto out = _build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                                 as_arrays);
        return py::make_tuple(out[0], out[1], _timings_dict(t));
    }

    void insert_deferred(const py::object& seeds_obj)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        impl_.insert_deferred(xs, ys);
    }

    py::dict insert_deferred_timed(const py::object& seeds_obj)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        InsertTimings t;
        impl_.insert_deferred(xs, ys, &t);
        return _timings_dict(t);
    }

    py::tuple finalise(bool as_arrays)
    {
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        impl_.finalise(tri_map, tgrid);
        return _build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                             as_arrays);
    }

    py::tuple finalise_timed(bool as_arrays)
    {
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        InsertTimings t;
        impl_.finalise(tri_map, tgrid, &t);
        auto out = _build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                                 as_arrays);
        return py::make_tuple(out[0], out[1], _timings_dict(t));
    }

    // (N_tri, 3) vertex ids only -- no raster, no canonical pixels.
    py::array_t<int32_t> get_triangles() const
    {
        std::vector<TriangleEntry> tris;
        impl_.get_triangles(tris);
        const int n = (int)tris.size();
        py::array_t<int32_t> arr({n, 3});
        auto* p = arr.mutable_data();
        for (int i = 0; i < n; ++i) {
            p[i*3]   = tris[i].id_a;
            p[i*3+1] = tris[i].id_b;
            p[i*3+2] = tris[i].id_c;
        }
        return arr;
    }

    py::array_t<int32_t> get_edges() const
    {
        std::vector<int32_t> flat;
        impl_.get_edges(flat);
        const int n = (int)(flat.size() / 2);
        py::array_t<int32_t> arr({n, 2});
        if (n > 0)
            std::memcpy(arr.mutable_data(), flat.data(),
                        flat.size() * sizeof(int32_t));
        return arr;
    }

    py::array_t<int32_t> sorted_rank() const
    {
        const auto& r = impl_.sorted_rank();
        py::array_t<int32_t> arr((py::ssize_t)r.size());
        if (!r.empty())
            std::memcpy(arr.mutable_data(), r.data(), r.size() * sizeof(int32_t));
        return arr;
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

    int  seed_count()     const { return impl_.seed_count(); }
    int  width()          const { return impl_.width(); }
    int  height()         const { return impl_.height(); }
    int  border_padding() const { return impl_.border_padding(); }
    bool has_pending()    const { return impl_.has_pending(); }

private:
    Delaunay impl_;

    static py::dict _timings_dict(const InsertTimings& t)
    {
        py::dict d;
        d["bfs_ms"]    = t.bfs_ms;
        d["bfs_iters"] = t.bfs_iters;
        d["detect_ms"] = t.detect_ms;
        d["dedup_ms"]  = t.dedup_ms;
        d["assign_ms"] = t.assign_ms;
        return d;
    }

    static void _parse_seeds(const py::object& seeds_obj,
                             std::vector<int32_t>& xs, std::vector<int32_t>& ys)
    {
        // Fast path for the (N, 2) integer arrays callers actually pass. The
        // generic path below casts two Python objects per seed, which is fine
        // for a handful and not for the thousands a refinement round inserts.
        if (py::isinstance<py::array>(seeds_obj)) {
            auto arr = py::array_t<int32_t, py::array::c_style |
                                            py::array::forcecast>::ensure(seeds_obj);
            if (arr && arr.ndim() == 2 && arr.shape(1) == 2) {
                const py::ssize_t n = arr.shape(0);
                const int32_t* p = arr.data();
                xs.resize(n); ys.resize(n);
                for (py::ssize_t i = 0; i < n; ++i) {
                    xs[i] = p[i*2];
                    ys[i] = p[i*2 + 1];
                }
                return;
            }
        }
        for (auto item : seeds_obj) {
            auto pair = item.cast<py::sequence>();
            xs.push_back(pair[0].cast<int32_t>());
            ys.push_back(pair[1].cast<int32_t>());
        }
    }

};

// ---------------------------------------------------------------------------
// triangulate: seeds in, triangulation out
// ---------------------------------------------------------------------------

//: The whole pipeline in one call, and the one callers should reach for.
//:
//: Voronoi().compute() followed by GridTriangulation().compute() was the usual
//: pairing, and at any padding above 0 it built the diagram twice: once in the
//: caller, and once inside the triangulator on the padded canvas, because
//: detection needs the padded one. The caller's copy then served only to fill
//: two output channels that the padded diagram already contains in its interior
//: window. Going through here computes it once.
static py::tuple triangulate(
    int width, int height,
    const py::object& seeds_obj,
    int border_padding = -1,
    bool as_arrays = false)
{
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("width and height must be positive");

    auto seeds = sort_seeds(seeds_obj, width, height);
    int N_seeds = static_cast<int>(seeds.size());
    border_padding = resolve_border_padding(border_padding, width, height, N_seeds);

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

    return _build_output(tri_map, flat_out, height, width, as_arrays);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_delauney_cuda, m)
{
    m.doc() = "CUDA-accelerated Voronoi + Delaunay triangulation";

    py::class_<PyVoronoi>(m, "Voronoi")
        .def(py::init<>())
        .def("compute", &PyVoronoi::compute,
             py::arg("width"), py::arg("height"), py::arg("seeds"),
             "Compute an L2-distance (Euclidean) Voronoi diagram.\n\n"
             "Returns int32 array of shape (height, width, 2): "
             "(seed_id, squared L2 distance) per cell.");

    m.def("triangulate", &triangulate,
          py::arg("width"), py::arg("height"), py::arg("seeds"),
          py::arg("border_padding") = -1,
          py::arg("as_arrays") = false,
          "Triangulate a seed set. The normal entry point.\n\n"
          "Equivalent to Voronoi().compute() followed by\n"
          "GridTriangulation().compute(), but builds the Voronoi diagram once\n"
          "instead of twice: at border_padding > 0 the triangulator needs a\n"
          "padded diagram for detection, and the interior window of that one is\n"
          "bit-identical to the unpadded diagram a caller would compute, so the\n"
          "caller's copy was pure duplicate work.\n\n"
          "border_padding: pixels of Voronoi canvas added on each side before\n"
          "detection, exposing border triangles whose circumcentre lies outside\n"
          "the image. Negative (the default) uses DEFAULT_BORDER_PADDING; 0\n"
          "disables padding. See BORDER_PADDING_BOUND.md.\n\n"
          "Returns (triangle_map, triangulation_grid), the same pair\n"
          "GridTriangulation.compute returns.");

    py::class_<PyGridTriangulation>(m, "GridTriangulation")
        .def(py::init<>())
        .def("compute", &PyGridTriangulation::compute,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             py::arg("border_padding") = -1,
             py::arg("as_arrays") = false,
             "Extract a Delaunay triangulation from a Voronoi grid.\n\n"
             "border_padding: pixels of Voronoi canvas to add on each side\n"
             "before triangle detection, exposing border triangles whose\n"
             "circumcenter lies outside the image. Negative (the default)\n"
             "uses delauney.DEFAULT_BORDER_PADDING, matching the reference's\n"
             "border_padding=None; 0 disables padding. BORDER_PADDING_BOUND.md\n"
             "gives the boundary spacing a given padding admits.\n"
             "The output grid is always the original (H, W, 3) resolution.\n\n"
             "as_arrays: return the vertex indices as an (N_tri, 3) int32 array\n"
             "instead of a {tid: (x, y, id_a, id_b, id_c)} dict, skipping the\n"
             "per-triangle Python objects. The (x, y) detection pixel is not\n"
             "included; ask for the dict if you need it.\n\n"
             "Returns (triangle_map, triangulation_grid). Pixels no triangle\n"
             "contains carry -1 in channel 2 of the grid.")
        .def("compute_timed", &PyGridTriangulation::compute_timed,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             py::arg("border_padding") = -1,
             "Same as compute() but also returns a timings dict.\n\n"
             "Returns (triangle_map, triangulation_grid, timings) where "
             "timings has keys detect_ms, dedup_ms, assign_ms (float, ms).")
        .def("compute_debug", &PyGridTriangulation::compute_debug,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             py::arg("border_padding") = -1,
             "Same as compute() but also returns the padded Voronoi grid.\n\n"
             "Returns (triangle_map, triangulation_grid, padded_voronoi_grid) where "
             "padded_voronoi_grid has shape (H+2*P, W+2*P, 2).");

    py::class_<PyDelaunay>(m, "Delaunay")
        .def(py::init<int, int, int, int>(),
             py::arg("width"), py::arg("height"), py::arg("max_seeds"),
             py::arg("border_padding") = -1,
             "Create an incremental Delaunay triangulator with device-resident state.\n\n"
             "max_seeds: upper bound on total seeds ever inserted.\n"
             "border_padding: width of the padded detection canvas; < 0 uses\n"
             "  delauney.DEFAULT_BORDER_PADDING.\n\n"
             "  A triangle is registered where three Voronoi regions meet, i.e. at\n"
             "  its circumcentre, and boundary triangles frequently have\n"
             "  circumcentres outside the image, so at 0 they are never detected.\n"
             "  Unlike the batch API this is fixed at construction, because the\n"
             "  padded canvas is the persistent device state -- and the auto rule\n"
             "  shrinks as seeds are added, so a state well below max_seeds is\n"
             "  under-padded relative to what the batch path would pick for it.\n"
             "  Pass a value explicitly when the eventual seed count is not close\n"
             "  to max_seeds. Padding costs ((W+2P)(H+2P))/(WH) in grid work.")
        .def("insert", &PyDelaunay::insert,
             py::arg("seeds"), py::arg("as_arrays") = false,
             "Insert a batch of (x, y) seeds and update the triangulation.\n\n"
             "Returns (triangle_map, triangulation_grid) where triangle_map is\n"
             "{int: (x,y,id_a,id_b,id_c)}, or an (N_tri, 3) vertex-id array when\n"
             "as_arrays=True, and triangulation_grid has shape (H,W,3).\n\n"
             "Equivalent to insert_deferred() followed by finalise().")
        .def("insert_timed", &PyDelaunay::insert_timed,
             py::arg("seeds"), py::arg("as_arrays") = false,
             "Same as insert() but also returns a timings dict with keys\n"
             "bfs_ms, detect_ms, dedup_ms, assign_ms.")
        .def("insert_deferred", &PyDelaunay::insert_deferred,
             py::arg("seeds"),
             "Insert a batch, updating the Voronoi diagram and triangle topology\n"
             "but not the per-pixel assignment, and returning nothing.\n\n"
             "Pixel assignment is the expensive stage and the only one whose\n"
             "dirty region saturates for large scattered batches, so a caller\n"
             "that inserts several times before it needs a raster should defer\n"
             "it and call finalise() once.  get_triangles() stays valid in\n"
             "between; the triangulation grid does not.")
        .def("insert_deferred_timed", &PyDelaunay::insert_deferred_timed,
             py::arg("seeds"),
             "Same as insert_deferred() but returns the timings dict.\n"
             "assign_ms is always 0 here; it is reported by finalise().")
        .def("finalise", &PyDelaunay::finalise,
             py::arg("as_arrays") = false,
             "Assign pixels for everything deferred since the last finalise and\n"
             "return (triangle_map, triangulation_grid).\n\n"
             "Picks a masked or a full assignment by whichever covers less work,\n"
             "so a small accumulated change stays cheap.  Calling this with\n"
             "nothing pending just rebuilds the outputs.")
        .def("finalise_timed", &PyDelaunay::finalise_timed,
             py::arg("as_arrays") = false,
             "Same as finalise() but also returns the timings dict.")
        .def("get_triangles", &PyDelaunay::get_triangles,
             "Triangle topology alone as an (N_tri, 3) int32 array, with no\n"
             "raster copied back.\n\n"
             "Seed ids are INSERTION-order, not the sorted numbering insert()\n"
             "and finalise() report, so that a caller appending seeds across\n"
             "several inserts keeps its own per-seed arrays aligned.  Use\n"
             "sorted_rank to translate.")
        .def("get_edges", &PyDelaunay::get_edges,
             "The distinct undirected edges as an (N_edges, 2) int32 array,\n"
             "each row (a, b) with a < b.\n\n"
             "Seed ids are INSERTION-order, matching get_triangles().\n\n"
             "Deduplicated on the device. A triangulation spells out 3T edges\n"
             "and every interior edge appears in two triangles, so a caller\n"
             "that wants edges would otherwise pull all 3T triangle indices\n"
             "across the bus and sort them on the host. Consumers score edges\n"
             "rather than triangles, so this is the shape the data is wanted\n"
             "in, and it moves less of it.")
        .def_property_readonly("sorted_rank", &PyDelaunay::sorted_rank,
             "int32 array mapping insertion-order seed id -> sorted (x asc,\n"
             "y asc) id, which is the numbering insert()/finalise() report.")
        .def("get_voronoi_grid", &PyDelaunay::get_voronoi_grid,
             "Return the current Voronoi grid as int32 array of shape (H, W, 2).")
        .def_property_readonly("seed_count", &PyDelaunay::seed_count)
        .def_property_readonly("width",  &PyDelaunay::width)
        .def_property_readonly("height", &PyDelaunay::height)
        .def_property_readonly("border_padding",
             &PyDelaunay::border_padding,
             "Resolved width of the padded detection canvas.")
        .def_property_readonly("has_pending", &PyDelaunay::has_pending,
             "True when deferred inserts are awaiting a finalise().");
}
