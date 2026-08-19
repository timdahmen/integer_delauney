// pybind11 bindings for the CUDA Voronoi + triangulation kernels.
//
// Exposed module: _delauney_cuda -- Voronoi, GridTriangulation and
// Delaunay.  Per-method signatures live in the docstrings below;
// API_REFERENCE.md documents the Python-facing contract.
//
// `seeds` / `seed_positions` must be a sequence of (x, y) pairs.
// Seed ordering is (ascending x, tiebreak ascending y).

#include "bindings/binding_helpers.hpp"
#include "bindings/py_voronoi.hpp"
#include "bindings/py_grid_triangulation.hpp"
#include "bindings/py_device_array_view.hpp"
#include "bindings/py_delaunay.hpp"
#include "bindings/triangulate.hpp"

PYBIND11_MODULE(_delauney_cuda, m)
{
    m.doc() = "CUDA-accelerated Voronoi + Delaunay triangulation on bounded integer coordinates";

    py::class_<PyVoronoi>(m, "Voronoi")
        .def(py::init<>())
        .def("compute", &PyVoronoi::compute,
             py::arg("width"), py::arg("height"), py::arg("seeds"),
             "Compute an L2-distance (Euclidean) Voronoi diagram.\n\n"
             "Returns int32 array of shape (height, width, 2): "
             "(seed_id, squared L2 distance) per cell.");

    m.def("triangulate", &triangulate,
          py::arg("width"), py::arg("height"), py::arg("seeds"),
          py::arg("border_padding") = AUTO_BORDER_PADDING,
          py::arg("as_arrays") = false,
          "Triangulate a seed set. The normal entry point.\n\n"
          "Equivalent to Voronoi().compute() followed by\n"
          "GridTriangulation().compute(), but builds the Voronoi diagram only once.\n"
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
             py::arg("border_padding") = AUTO_BORDER_PADDING,
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
             py::arg("border_padding") = AUTO_BORDER_PADDING,
             "Same as compute() but also returns a timings dict.\n\n"
             "Returns (triangle_map, triangulation_grid, timings) where "
             "timings has keys detect_ms, dedup_ms, assign_ms (float, ms).")
        .def("compute_debug", &PyGridTriangulation::compute_debug,
             py::arg("voronoi_grid"), py::arg("seed_positions"),
             py::arg("border_padding") = AUTO_BORDER_PADDING,
             "Same as compute() but also returns the padded Voronoi grid.\n\n"
             "Returns (triangle_map, triangulation_grid, padded_voronoi_grid) where "
             "padded_voronoi_grid has shape (H+2*P, W+2*P, 2).");

    py::class_<PyDeviceArrayView>(m, "DeviceArrayView")
        .def_property_readonly("__cuda_array_interface__",
             &PyDeviceArrayView::cuda_array_interface,
             "CUDA Array Interface (v3), read-only. Raises RuntimeError if the\n"
             "owning Delaunay object has been mutated since this view was made.")
        .def("to_host", &PyDeviceArrayView::to_host,
             "Download this view into a plain host numpy array. Raises\n"
             "RuntimeError under the same staleness condition as\n"
             "__cuda_array_interface__.");

    py::class_<PyDelaunay>(m, "Delaunay")
        .def(py::init<int, int, int, int>(),
             py::arg("width"), py::arg("height"), py::arg("max_seeds"),
             py::arg("border_padding") = AUTO_BORDER_PADDING,
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
             "Equivalent to insert_deferred() followed by finalise(). Use this\n"
             "when you need the triangulation grid back after every batch; use\n"
             "insert_deferred() instead when you are about to insert several\n"
             "batches in a row and only need the grid after the last one.")
        .def("insert_timed", &PyDelaunay::insert_timed,
             py::arg("seeds"), py::arg("as_arrays") = false,
             "Same as insert() but also returns a timings dict with keys\n"
             "bfs_ms, detect_ms, dedup_ms, assign_ms.")
        .def("insert_deferred", &PyDelaunay::insert_deferred,
             py::arg("seeds"), py::arg("values") = py::none(),
             "Insert a batch, updating the Voronoi diagram and triangle topology\n"
             "but not the per-pixel assignment, and returning nothing.\n\n"
             "values, when given, is one float per seed in the batch: the\n"
             "scalar field sampled there, appended in step with the seeds so\n"
             "it cannot fall out of alignment. edge_scores() needs it.\n\n"
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
             "nothing pending just rebuilds the outputs. Call this once after a\n"
             "run of insert_deferred() calls.")
        .def("finalise_timed", &PyDelaunay::finalise_timed,
             py::arg("as_arrays") = false,
             "Same as finalise() but also returns the timings dict.")
        .def("finalise_device", &PyDelaunay::finalise_device,
             "Same as finalise(as_arrays=True), except pixel_tids,\n"
             "pixel_seed_ids and outside_hull_mask stay on the device: each\n"
             "comes back as a __cuda_array_interface__ view onto this object's\n"
             "own memory instead of a NumPy array.\n\n"
             "Returns (triangle_verts, pixel_tids, pixel_seed_ids,\n"
             "outside_hull_mask). The three views are valid until this object's\n"
             "next mutating call (insert/insert_deferred/finalise/\n"
             "finalise_device) or destruction; reading one afterward raises.")
        .def("get_triangles", &PyDelaunay::get_triangles,
             "Triangle topology alone as an (N_tri, 3) int32 array, with no\n"
             "raster copied back.\n\n"
             "Seed ids are INSERTION-order, not the sorted numbering insert()\n"
             "and finalise() report, so that a caller appending seeds across\n"
             "several inserts keeps its own per-seed arrays aligned.  Use\n"
             "sorted_rank to translate.")
        .def("edge_scores", &PyDelaunay::edge_scores,
             py::arg("min_length") = 0.0,
             "Per-edge |dv| * |ab| over the scalar field, as an (N_edges,) "
             "float32 array in get_edges() order.\n\n"
             "Edges shorter than min_length score 0, their midpoint rounding "
             "onto an endpoint. Needs values to have been supplied to "
             "insert_deferred.")
        .def("select_midpoints", &PyDelaunay::select_midpoints,
             py::arg("min_length"), py::arg("count"), py::arg("threshold") = 0.0f,
             "Midpoints of the best `count` edges, as an (M, 2) int32 array.\n\n"
             "Edges are ordered by score descending, edge index ascending -- a "
             "total order, so `count` is exact rather than a threshold with "
             "ties to settle, and the whole selection happens on the device. "
             "threshold excludes edges scoring at or below it, so fewer than "
             "`count` may come back.\n\n"
             "Midpoints already occupied by a seed are dropped -- a seed is a "
             "cell at distance zero, so no record of taken positions is "
             "needed -- and midpoints colliding with one another collapse to "
             "the one from the better-scoring edge.")
        .def("in_circumsphere", &PyDelaunay::in_circumsphere,
             py::arg("points"), py::arg("t"),
             "The in-circle predicate lifted out of the plane by t.\n\n"
             "For each point, is it inside the circumsphere of the triangle "
             "containing it, with the triangle at t = 0 and the point at t:\n\n"
             "    dx^2 + dy^2 + t^2 < R^2\n\n"
             "A caller using t for elapsed time reads this as close enough in "
             "space and recent enough in time; a triangle of circumradius R "
             "admits nothing beyond t = R.\n\n"
             "Returns (mask, triangle_ids). The triangle is reported because a "
             "caller asking this usually wants it and it is found on the way; "
             "-1 where no triangle contains the point, where the mask is False.")
        .def("locate", &PyDelaunay::locate, py::arg("points"),
             "Triangle containing each point of an (N, 2) int32 array, in "
             "image coordinates, as an (N,) int32 array. -1 where no triangle "
             "contains it.\n\n"
             "Ids index get_triangles(), which finalise()'s triangle map "
             "agrees with.\n\n"
             "For containment at a list of positions rather than a raster. "
             "finalise() answers the same question for every pixel; asking it "
             "about a few thousand points and reading the rest back out is "
             "about twenty times the work at a real canvas size.")
        .def("get_seeds", &PyDelaunay::get_seeds,
             "Seed positions as an (N, 2) int32 array, in insertion order.")
        .def("get_values", &PyDelaunay::get_values,
             "The scalar field as an (N,) float32 array, in insertion order.")
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
