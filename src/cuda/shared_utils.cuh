#pragma once

#include <cstdint>

// Normally, union memory space sharing is bad
// except of course if this allows us to pack something closer
// this is more a proof-of-concept    
union PackedCoordinate {
    struct {
        uint16_t x;
        uint16_t y;
    };
    uint32_t packed;

    __host__ __device__ constexpr PackedCoordinate(uint16_t _x, uint16_t _y) : x(_x), y(_y) {}
    __host__ __device__ constexpr PackedCoordinate(uint32_t _packed) : packed(_packed) {}
};

union UnpackedCoordinate {
    struct {
        uint32_t x;
        uint32_t y;
    };
    uint64_t packed;

    __host__ __device__ constexpr UnpackedCoordinate(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}
    __host__ __device__ constexpr UnpackedCoordinate(uint64_t _packed) : packed(_packed) {}
};

// Cache struct that holds vertex cache for compressed sparse rows
// The compiler should flatten this so no extra copies get made
//
// Indexed by CSR *slot*, not by triangle id 
struct CsrEntryVertexCache {
    #ifndef EXTEND_COORDINATE_LIMIT

    PackedCoordinate a;
    PackedCoordinate b;
    PackedCoordinate c;

    __host__ __device__ constexpr CsrEntryVertexCache(
        uint32_t _x_a, uint32_t _y_a,
        uint32_t _x_b, uint32_t _y_b,
        uint32_t _x_c, uint32_t _y_c) : 
            a(PackedCoordinate(uint16_t(_x_a), uint16_t(_y_a))), 
            b(PackedCoordinate(uint16_t(_x_b), uint16_t(_y_b))), 
            c(PackedCoordinate(uint16_t(_x_c), uint16_t(_y_c))) {}
    
    #else
    
    UnpackedCoordinate a;
    UnpackedCoordinate b;
    UnpackedCoordinate c;

    __host__ __device__ constexpr CsrEntryVertexCache(
        uint32_t _x_a, uint32_t _y_a,
        uint32_t _x_b, uint32_t _y_b,
        uint32_t _x_c, uint32_t _y_c) : 
            a(UnpackedCoordinate(_x_a, _y_a)), 
            b(UnpackedCoordinate(_x_b, _y_b)), 
            c(UnpackedCoordinate(_x_c, _y_c)) {}
    
    #endif
};

struct RawTriangle {
    int32_t x, y;
    int32_t a, b, c;           // sorted dedup key  (a <= b <= c)
    int32_t orig_a, orig_b, orig_c;
};

__global__
void build_csr_verts_kernel(
    CsrEntryVertexCache* __restrict__   csr_verts_cache,
    const int32_t* __restrict__         csr_idx,
    const RawTriangle* __restrict__     triangles,
    const int32_t* __restrict__         seed_xs,
    const int32_t* __restrict__         seed_ys,
    int csr_size
);