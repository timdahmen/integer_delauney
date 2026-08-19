// A read-only window onto one of a Delaunay object's device buffers, exposed
// to Python as __cuda_array_interface__ only -- no other way to reach the
// pointer. Deliberately not a registry: one view, one owner, one buffer,
// found only via finalise_device()'s return value.
//
// Two independent things keep this safe to hand to another CUDA extension:
// `owner_` is a reference to the parent Delaunay's Python object, so ordinary
// refcounting keeps the device memory allocated for as long as any view over
// it is reachable, even if the caller's own reference to the mesh has already
// been reassigned. `mesh_` lets __cuda_array_interface__ compare the mesh's
// *current* generation() against the value snapshotted when this view was
// made, so a view read after the mesh was mutated again raises instead of
// silently exposing memory that has moved on.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../delaunay.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

class PyDeviceArrayView {
public:
    PyDeviceArrayView(py::object owner, const Delaunay* mesh,
                      const void* ptr, py::ssize_t count,
                      std::string typestr, uint64_t generation)
        : owner_(std::move(owner)), mesh_(mesh), ptr_(ptr), count_(count),
          typestr_(std::move(typestr)), generation_(generation) {}

    py::dict cuda_array_interface() const
    {
        _check_fresh();
        py::dict d;
        d["shape"]   = py::make_tuple(count_);
        d["typestr"] = typestr_;
        d["data"]    = py::make_tuple((uintptr_t)ptr_, /*read_only=*/true);
        d["strides"] = py::none();
        d["version"] = 3;
        return d;
    }

    // Host numpy array with this view's content.
    py::array to_host() const
    {
        _check_fresh();
        if (typestr_ == "<i4") {
            py::array_t<int32_t> out(count_);
            cudaMemcpy(out.mutable_data(), ptr_,
                       (size_t)count_ * sizeof(int32_t), cudaMemcpyDeviceToHost);
            return out;
        }
        if (typestr_ == "|u1") {
            py::array_t<uint8_t> out(count_);
            cudaMemcpy(out.mutable_data(), ptr_,
                       (size_t)count_ * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            return out;
        }
        throw std::runtime_error("to_host(): unsupported typestr '" + typestr_ + "'");
    }

private:
    void _check_fresh() const
    {
        if (mesh_->generation() != generation_)
            throw std::runtime_error(
                "device view is stale: the mesh was mutated after "
                "finalise_device() produced this view");
    }

    py::object owner_;        // keeps the parent Delaunay (and its buffers) alive
    const Delaunay* mesh_;    // valid as long as owner_ is -- see above
    const void* ptr_;
    py::ssize_t count_;
    std::string typestr_;
    uint64_t generation_;     // snapshot at construction time
};
