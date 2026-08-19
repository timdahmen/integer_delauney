// Error-checking primitives for the raw CUDA runtime API calls scattered
// across this package. Every cudaMalloc/cudaMemcpy/cudaMemset/cudaEvent*/
// cudaDeviceSynchronize call, and every kernel launch, goes through one of
// the three macros below instead of being called bare.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

inline void cuda_check_throw(cudaError_t err, const char* expr,
                              const char* file, int line)
{
    if (err != cudaSuccess) {
        // A failing call leaves the runtime's internal "last error" set to
        // this error until something calls cudaGetLastError() -- unlike what
        // the CUDA docs' phrasing suggests, later *successful* calls do not
        // reliably clear it first. Left alone, an unrelated cudaGetLastError()
        // call much later (e.g. CUDA_CHECK_LAST_ERROR() after some later,
        // perfectly fine kernel launch, possibly on a different Delaunay
        // object) reports this same stale error. Draining it here scopes the
        // failure to the call that actually caused it.
        cudaGetLastError();
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expr +
            " failed: " + cudaGetErrorString(err));
    }
}

inline void cuda_check_nothrow(cudaError_t err, const char* expr,
                                const char* file, int line) noexcept
{
    if (err != cudaSuccess) {
        cudaGetLastError();  // see cuda_check_throw's comment on why
        std::fprintf(stderr, "%s:%d: %s failed: %s\n",
                      file, line, expr, cudaGetErrorString(err));
    }
}

//: Throws std::runtime_error (surfaces as a Python RuntimeError through
//: pybind11's built-in translation, the same path std::invalid_argument
//: already uses at this package's API boundary) on a non-success cudaError_t.
#define CUDA_CHECK(expr) cuda_check_throw((expr), #expr, __FILE__, __LINE__)

//: Same failure info as CUDA_CHECK, but logs to stderr instead of throwing.
//: Destructors run during unwinding and are implicitly noexcept, so a throw
//: out of one calls std::terminate instead of reaching Python -- use this
//: for every cudaFree (and cudaEventDestroy) in a destructor or a cleanup
//: helper a destructor calls.
#define CUDA_CHECK_NOTHROW(expr) \
    cuda_check_nothrow((expr), #expr, __FILE__, __LINE__)

//: Checks the last error left by an asynchronous kernel launch. Call this on
//: the line immediately after every `kernel<<<...>>>(...)` -- cudaGetLastError
//: is a cheap, non-synchronizing host-side check (it does not force a device
//: sync), so this catches launch-configuration errors right at the launch
//: site without adding new synchronization. An in-kernel runtime fault (e.g.
//: an out-of-bounds write) still only surfaces at the next real sync point
//: (cudaDeviceSynchronize, or a cudaMemcpy back to host) -- those are checked
//: too, via CUDA_CHECK, for exactly this reason.
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())
