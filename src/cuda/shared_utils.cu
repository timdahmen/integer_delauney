
#include "shared_utils.cuh"

// ---------------------------------------------------------------------------
// Kernel: rebuild CSR on GPU, saves a copy sequence
// ---------------------------------------------------------------------------

__global__
void build_csr_verts_kernel(
    CsrEntryVertexCache* __restrict__   csr_verts_cache,
    const int32_t* __restrict__         csr_idx,
    const RawTriangle* __restrict__     triangles,
    const int32_t* __restrict__         seed_xs,
    const int32_t* __restrict__         seed_ys,
    int csr_size
) {
    int csr_slot = blockIdx.x * blockDim.x + threadIdx.x; // Same calculation as for the loops
    if (csr_slot < csr_size) {
        const RawTriangle& which = triangles[csr_idx[csr_slot]];
        csr_verts_cache[csr_slot] = CsrEntryVertexCache(
            seed_xs[which.orig_a], seed_ys[which.orig_a],
            seed_xs[which.orig_b], seed_ys[which.orig_b],
            seed_xs[which.orig_c], seed_ys[which.orig_c]
        );
    }
}