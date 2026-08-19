// Optional CUDA-event stopwatch, shared by every call site that times a
// sequence of kernel phases only when a caller asks for it: detect/dedup in
// delaunay_topology.cu and triangulation.cu, the single-phase BFS and
// assignment timers in delaunay_voronoi.cu and delaunay_assign.cu.
#pragma once

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
        if (active_) for (auto& e : marks_) cudaEventCreate(&e);
    }
    ~PhaseTimer()
    {
        if (active_) for (auto& e : marks_) cudaEventDestroy(e);
    }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;

    void mark(int i) { if (active_) cudaEventRecord(marks_[i]); }

    float elapsed_ms(int from, int to) const
    {
        if (!active_) return 0.f;
        cudaEventSynchronize(marks_[to]);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, marks_[from], marks_[to]);
        return ms;
    }

private:
    cudaEvent_t marks_[N]{};
    bool active_;
};
