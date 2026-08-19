// Optional CUDA-event stopwatch, shared by every call site that times a
// sequence of kernel phases only when a caller asks for it: detect/dedup in
// delaunay_topology.cu and triangulation.cu, the single-phase BFS and
// assignment timers in delaunay_voronoi.cu and delaunay_assign.cu.
#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

//: N-mark event timer, inert when constructed inactive.
//:
//: mark(i) records the i-th of N points in program order; elapsed_ms(a, b) is
//: the time between two of them. Both are no-ops/return 0 when inactive, so a
//: caller marks unconditionally instead of guarding every call site on
//: whether timing was requested.
template <int N>
class PhaseTimer {
public:
    explicit PhaseTimer(bool active) : active_(active)
    {
        if (!active_) return;
        try {
            for (created_ = 0; created_ < N; ++created_)
                CUDA_CHECK(cudaEventCreate(&marks_[created_]));
        } catch (...) {
            destroy_created_();
            throw;
        }
    }
    ~PhaseTimer()
    {
        destroy_created_();
    }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;

    void mark(int i) { if (active_) CUDA_CHECK(cudaEventRecord(marks_[i])); }

    float elapsed_ms(int from, int to) const
    {
        if (!active_) return 0.f;
        CUDA_CHECK(cudaEventSynchronize(marks_[to]));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, marks_[from], marks_[to]));
        return ms;
    }

private:
    // created_ tracks how many of marks_ actually got a cudaEventCreate, so a
    // create failure partway through construction destroys only those and a
    // normal destructor (created_ == N by then) destroys all of them.
    void destroy_created_() noexcept
    {
        for (int i = 0; i < created_; ++i) CUDA_CHECK_NOTHROW(cudaEventDestroy(marks_[i]));
        created_ = 0;
    }

    cudaEvent_t marks_[N]{};
    bool active_;
    int created_ = 0;
};
