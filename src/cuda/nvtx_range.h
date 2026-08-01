// Scoped NVTX ranges for Nsight Systems timelines.
//
// Everything compiles to nothing unless DELAUNEY_WITH_NVTX is defined, so the
// pybind11 extension / wheel build is unaffected.  The standalone profiling
// executable defines it (CMake option WITH_NVTX, on by default there).
//
// Usage:
//     DELAUNEY_NVTX_RANGE("phase name");        // ends at end of scope
//     DELAUNEY_NVTX_MARK("something happened"); // instantaneous marker
//
// Colours group related phases in the timeline: host-side work, device
// kernels and memory transfers each get their own hue.

#pragma once

#ifdef DELAUNEY_WITH_NVTX

// nvToolsExt.h pulls in windows.h; keep its min/max macros out of the way.
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#endif

#include <nvtx3/nvToolsExt.h>

#include <cstdint>

namespace delauney_nvtx {

// Palette (ARGB).  Pick one via DELAUNEY_NVTX_RANGE_C.
enum Color : uint32_t {
    kHost   = 0xFF4E79A7,   // blue   -- host-side CPU work
    kKernel = 0xFF59A14F,   // green  -- kernel launch + sync
    kMemcpy = 0xFFE15759,   // red    -- H2D / D2H transfers
    kPhase  = 0xFFF28E2B,   // orange -- top-level phase
};

class ScopedRange {
public:
    ScopedRange(const char* name, uint32_t color)
    {
        nvtxEventAttributes_t a = {};
        a.version       = NVTX_VERSION;
        a.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        a.colorType     = NVTX_COLOR_ARGB;
        a.color         = color;
        a.messageType   = NVTX_MESSAGE_TYPE_ASCII;
        a.message.ascii = name;
        nvtxRangePushEx(&a);
    }
    ~ScopedRange() { nvtxRangePop(); }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

}  // namespace delauney_nvtx

#define DELAUNEY_NVTX_CONCAT_(a, b) a##b
#define DELAUNEY_NVTX_CONCAT(a, b)  DELAUNEY_NVTX_CONCAT_(a, b)

#define DELAUNEY_NVTX_RANGE_C(name, color)                                     \
    ::delauney_nvtx::ScopedRange DELAUNEY_NVTX_CONCAT(_nvtx_range_, __LINE__)  \
        (name, color)

#define DELAUNEY_NVTX_RANGE(name) \
    DELAUNEY_NVTX_RANGE_C(name, ::delauney_nvtx::kHost)

#define DELAUNEY_NVTX_MARK(name) nvtxMarkA(name)

#else  // !DELAUNEY_WITH_NVTX

#define DELAUNEY_NVTX_RANGE_C(name, color) ((void)0)
#define DELAUNEY_NVTX_RANGE(name)          ((void)0)
#define DELAUNEY_NVTX_MARK(name)           ((void)0)

#endif  // DELAUNEY_WITH_NVTX
