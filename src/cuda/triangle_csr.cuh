// Seed -> triangle adjacency, shared by the incremental registry
// (delaunay_topology.cu's rebuild_csr_and_upload_) and the batch pipeline
// (triangulation.cu): both turn a flat triangle list into the same CSR that
// assign_triangles_kernel/locate_at then walk per seed.
#pragma once

#include <cstdint>
#include <vector>

//: Build the seed -> triangle CSR from a list of triangles exposing
//: orig_a/orig_b/orig_c. `is_dead(tid)` skips a slot -- pass a functor that
//: always returns false where every slot is live, as in the batch pipeline.
template <typename Triangles, typename IsDead>
inline void build_seed_triangle_csr(
    const Triangles& triangles, int n_seeds, IsDead is_dead,
    std::vector<int32_t>& csr_ptr_out, std::vector<int32_t>& csr_idx_out)
{
    const int n_tri = (int)triangles.size();
    csr_ptr_out.assign(n_seeds + 1, 0);
    for (int tid = 0; tid < n_tri; ++tid) {
        if (is_dead(tid)) continue;
        const auto& t = triangles[tid];
        csr_ptr_out[t.orig_a + 1]++;
        csr_ptr_out[t.orig_b + 1]++;
        csr_ptr_out[t.orig_c + 1]++;
    }
    for (int s = 1; s <= n_seeds; ++s) csr_ptr_out[s] += csr_ptr_out[s - 1];

    csr_idx_out.resize(csr_ptr_out[n_seeds]);
    std::vector<int32_t> fill(n_seeds, 0);
    for (int tid = 0; tid < n_tri; ++tid) {
        if (is_dead(tid)) continue;
        const auto& t = triangles[tid];
        for (int32_t s : {t.orig_a, t.orig_b, t.orig_c}) {
            csr_idx_out[csr_ptr_out[s] + fill[s]] = tid;
            fill[s]++;
        }
    }
}
