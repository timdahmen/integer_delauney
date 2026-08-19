// Move-only RAII wrapper for a function-local device allocation.
//
// For the Delaunay class's own persistent, object-lifetime buffers, see
// delaunay.cu's constructor/destructor and free_device_buffers_() instead --
// converting those to DeviceBuffer would touch every one of their many
// existing raw-pointer call sites for no benefit, since their lifetime
// already matches a single owner with a matching destructor.
//
// This type exists for the opposite case: a function that mallocs several
// scratch buffers, does some work, and frees them all before returning.
// Without RAII, a CUDA_CHECK throw partway through such a function's
// allocations leaks whatever it already allocated. DeviceBuffer<T> closes
// that gap and lets the function's trailing manual cudaFree block be deleted.
#pragma once

#include "cuda_check.cuh"
#include <cstddef>

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    //: Allocates room for `count` elements of T. Throws (CUDA_CHECK) on
    //: failure, e.g. out of device memory.
    explicit DeviceBuffer(size_t count) : count_(count)
    {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceBuffer() { CUDA_CHECK_NOTHROW(cudaFree(ptr_)); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            CUDA_CHECK_NOTHROW(cudaFree(ptr_));
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    //: Implicit so existing call sites that pass the raw pointer into a
    //: kernel launch or cudaMemcpy keep compiling unchanged.
    operator T*() const { return ptr_; }
    T* get() const { return ptr_; }
    size_t count() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};
