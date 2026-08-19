// Python wrapper for the incremental Delaunay triangulator. See bindings.cpp
// for docs.
#pragma once

#include "binding_helpers.hpp"
#include "py_device_array_view.hpp"
#include "../delaunay.cuh"

#include <cstring>

class PyDelaunay {
public:
    PyDelaunay(int width, int height, int max_seeds, int border_padding)
        : impl_(width, height, max_seeds, border_padding) {}

    py::tuple insert(const py::object& seeds_obj, bool as_arrays)
    {
        return _insert_impl(seeds_obj, as_arrays, nullptr);
    }

    py::tuple insert_timed(const py::object& seeds_obj, bool as_arrays)
    {
        InsertTimings t;
        auto out = _insert_impl(seeds_obj, as_arrays, &t);
        return py::make_tuple(out[0], out[1], _timings_dict(t));
    }

    void insert_deferred(const py::object& seeds_obj,
                         const py::object& values_obj)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        if (values_obj.is_none()) { impl_.insert_deferred(xs, ys); return; }
        auto v = values_obj.cast<py::array_t<float,
                     py::array::c_style | py::array::forcecast>>();
        std::vector<float> vals(v.data(), v.data() + v.size());
        impl_.insert_deferred(xs, ys, nullptr, &vals);
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
        return _finalise_impl(as_arrays, nullptr);
    }

    py::tuple finalise_timed(bool as_arrays)
    {
        InsertTimings t;
        auto out = _finalise_impl(as_arrays, &t);
        return py::make_tuple(out[0], out[1], _timings_dict(t));
    }

    // As finalise(as_arrays=True), except the (H,W,3) raster never comes back
    // to the host: pixel_tids, pixel_seed_ids and outside_hull_mask are
    // returned as __cuda_array_interface__ views straight onto the Delaunay
    // object's own device memory instead. Returns
    // (triangle_verts, pixel_tids, pixel_seed_ids, outside_hull_mask).
    py::tuple finalise_device()
    {
        std::vector<TriangleEntry> tri_map;
        impl_.finalise_device(tri_map);

        py::array_t<int32_t> verts({(int)tri_map.size(), 3});
        auto* p = verts.mutable_data();
        for (size_t tid = 0; tid < tri_map.size(); ++tid) {
            p[tid * 3]     = tri_map[tid].id_a;
            p[tid * 3 + 1] = tri_map[tid].id_b;
            p[tid * 3 + 2] = tri_map[tid].id_c;
        }

        // What each returned view keeps alive; see PyDeviceArrayView.
        py::object self = py::cast(this, py::return_value_policy::reference);
        const py::ssize_t n = (py::ssize_t)impl_.width() * impl_.height();
        const uint64_t gen = impl_.generation();

        py::object pixel_tids = py::cast(PyDeviceArrayView(
            self, &impl_, impl_.device_pixel_tids(), n, "<i4", gen));
        py::object pixel_seed_ids = py::cast(PyDeviceArrayView(
            self, &impl_, impl_.device_pixel_seed_ids(), n, "<i4", gen));
        py::object outside_mask = py::cast(PyDeviceArrayView(
            self, &impl_, impl_.device_outside_mask(), n, "|u1", gen));

        return py::make_tuple(verts, pixel_tids, pixel_seed_ids, outside_mask);
    }

    // Returns (N_tri, 3) vertex ids only.
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

    py::array_t<int32_t> get_seeds() const
    {
        std::vector<int32_t> flat;
        impl_.get_seeds(flat);
        const int n = (int)(flat.size() / 2);
        py::array_t<int32_t> arr({n, 2});
        if (n > 0)
            std::memcpy(arr.mutable_data(), flat.data(),
                        flat.size() * sizeof(int32_t));
        return arr;
    }

    py::array_t<float> get_values() const
    {
        std::vector<float> v;
        impl_.get_values(v);
        py::array_t<float> arr((py::ssize_t)v.size());
        if (!v.empty())
            std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
        return arr;
    }

    py::array_t<int32_t> locate(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& pts)
    {
        auto info = pts.request();
        if (info.ndim != 2 || info.shape[1] != 2)
            throw std::invalid_argument("points must have shape (N, 2)");
        const int n = (int)info.shape[0];
        const auto* p = static_cast<const int32_t*>(info.ptr);
        std::vector<int32_t> xs(n), ys(n);
        for (int i = 0; i < n; ++i) { xs[i] = p[i * 2]; ys[i] = p[i * 2 + 1]; }

        std::vector<int32_t> out;
        impl_.locate(xs, ys, out);
        py::array_t<int32_t> arr((py::ssize_t)out.size());
        if (!out.empty())
            std::memcpy(arr.mutable_data(), out.data(),
                        out.size() * sizeof(int32_t));
        return arr;
    }

    py::tuple in_circumsphere(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast>& pts,
        const py::array_t<double, py::array::c_style | py::array::forcecast>& t)
    {
        auto info = pts.request();
        if (info.ndim != 2 || info.shape[1] != 2)
            throw std::invalid_argument("points must have shape (N, 2)");
        const int n = (int)info.shape[0];
        if ((int)t.size() != n)
            throw std::invalid_argument("t must have one entry per point");

        const auto* p = static_cast<const int32_t*>(info.ptr);
        std::vector<int32_t> xs(n), ys(n);
        for (int i = 0; i < n; ++i) { xs[i] = p[i*2]; ys[i] = p[i*2+1]; }
        std::vector<double> ts(t.data(), t.data() + n);

        std::vector<uint8_t> mask;
        std::vector<int32_t> tids;
        impl_.in_circumsphere(xs, ys, ts, mask, tids);

        py::array_t<bool> m((py::ssize_t)mask.size());
        py::array_t<int32_t> ti((py::ssize_t)tids.size());
        auto* mp = m.mutable_data();
        for (size_t i = 0; i < mask.size(); ++i) mp[i] = mask[i] != 0;
        if (!tids.empty())
            std::memcpy(ti.mutable_data(), tids.data(),
                        tids.size() * sizeof(int32_t));
        return py::make_tuple(m, ti);
    }

    py::array_t<float> edge_scores(double min_length) const
    {
        std::vector<float> sc;
        impl_.edge_scores(min_length, sc);
        py::array_t<float> arr((py::ssize_t)sc.size());
        if (!sc.empty())
            std::memcpy(arr.mutable_data(), sc.data(), sc.size() * sizeof(float));
        return arr;
    }

    py::array_t<int32_t> select_midpoints(double min_length, int count,
                                          float threshold) const
    {
        std::vector<int32_t> flat;
        impl_.select_midpoints(min_length, count, threshold, flat);
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

    // Shared by insert()/insert_timed(): parse, insert, pack the output.
    py::tuple _insert_impl(const py::object& seeds_obj, bool as_arrays,
                           InsertTimings* t)
    {
        std::vector<int32_t> xs, ys;
        _parse_seeds(seeds_obj, xs, ys);
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        impl_.insert(xs, ys, tri_map, tgrid, t);
        return build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                            as_arrays);
    }

    // Shared by finalise()/finalise_timed(): finalise, pack the output.
    py::tuple _finalise_impl(bool as_arrays, InsertTimings* t)
    {
        std::vector<TriangleEntry> tri_map;
        std::vector<int32_t> tgrid;
        impl_.finalise(tri_map, tgrid, t);
        return build_output(tri_map, tgrid, impl_.height(), impl_.width(),
                            as_arrays);
    }

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
