// Standalone C++/CUDA port of profiling.py -- no Python, no pybind11.
//
// Build:
//     cmake -S . -B build/native -DWITH_CUDA=ON -DBUILD_PROFILING=ON \
//           -DBUILD_PYTHON_MODULE=OFF
//     cmake --build build/native --config Release --target delauney_profiling
//
// Run:
//     build/native/Release/delauney_profiling.exe [options]
//
// Options
// -------
//   -w, --width N        grid width                       (default 1024)
//   -h, --height N       grid height                      (default 1024)
//   -n, --seeds N        seed count                       (default 100000)
//       --seed N         RNG seed                         (default 42)
//       --warm N         warm single-seed inserts to time (default 9)
//       --skip-ref       skip the CPU reference sections (2 and 4)
//       --only LIST      comma-separated section numbers to run, e.g. --only 3,6
//       --help
//
// Sections
// --------
// 1. Seed generation
// 2. RegularDelaunay   -- CPU reference  (BFS only)
// 3. RegularDelaunay   -- CUDA
// 4. GridTriangulation -- CPU reference  (detection + dedup; assignment skipped
//                         because ~237 k triangles x 1 M pixels = ~248 G
//                         containment tests)
// 5. GridTriangulation -- CUDA           (full pipeline)
// 6. IncrementalDelaunay -- CUDA         (cold insert + warm single-seed insert)
// 7. Summary table

#include "native_api.h"
#include "nvtx_range.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t UNDEFINED = -1;

// ---------------------------------------------------------------------------
// Helpers  (mirrors of the timer / section / hr helpers in profiling.py)
// ---------------------------------------------------------------------------

std::map<std::string, double> g_results;   // label -> seconds

// Scoped timer: prints (and optionally stores) on destruction, like the
// `with timer(...)` context manager.  Doubles as an NVTX range, so every
// line of the printed table is also a labelled range in the Nsight timeline.
class Timer {
public:
    explicit Timer(std::string label, bool store = true)
        : label_(std::move(label)), store_(store)
    {
#ifdef DELAUNEY_WITH_NVTX
        nvtx_.emplace(label_.c_str(), delauney_nvtx::kPhase);
#endif
        t0_ = Clock::now();
    }

    ~Timer()
    {
        double elapsed = std::chrono::duration<double>(Clock::now() - t0_).count();
#ifdef DELAUNEY_WITH_NVTX
        nvtx_.reset();
#endif
        if (store_)
            g_results[label_] = elapsed;
        std::printf("    %-52s %10.1f ms\n", label_.c_str(), elapsed * 1000.0);
    }

private:
    std::string label_;
    bool store_;
    Clock::time_point t0_;
#ifdef DELAUNEY_WITH_NVTX
    std::optional<delauney_nvtx::ScopedRange> nvtx_;
#endif
};

void section(const std::string& title)
{
    DELAUNEY_NVTX_MARK(title.c_str());
    std::printf("\n  %s\n", std::string(62, '-').c_str());
    std::printf("  %s\n", title.c_str());
    std::printf("  %s\n", std::string(62, '-').c_str());
}

void hr()
{
    std::printf("\n  %s\n", std::string(68, '=').c_str());
}

// "100000" -> "100,000"
std::string with_commas(long long v)
{
    std::string s = std::to_string(v);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3)
        s.insert(i, ",");
    return s;
}

void print_kv(const char* label, const std::string& value)
{
    std::printf("    %-52s %10s\n", label, value.c_str());
}

void print_ms(const char* label, float ms)
{
    std::printf("    %-52s %9.1f ms\n", label, ms);
}

// One row of a host wall-clock breakdown table.
struct HostRow { const char* label; float total; };

// Prints a host wall-clock breakdown with per-row percentages.  Every value is
// divided by `n`, so the same helper serves a single cold insert (n = 1) and a
// warm average (n = WARM_N).  `scratch` is the overlapping cudaMalloc/cudaFree
// sub-measure, reported separately because it is already counted in the rows.
void print_host_breakdown(const char* title, const std::vector<HostRow>& rows,
                          double n, float scratch)
{
    double sum = 0.0;
    for (const auto& r : rows) sum += r.total / n;

    std::printf("\n    %s\n", title);
    for (const auto& r : rows) {
        const double ms = r.total / n;
        std::printf("    %-46s %8.2f ms  %5.1f %%\n",
                    r.label, ms, sum > 0.0 ? 100.0 * ms / sum : 0.0);
    }
    std::printf("    %-46s %8.2f ms\n", "  accounted total", sum);
    std::printf("    %-46s %8.2f ms  (of which, counted above)\n",
                "  cudaMalloc/cudaFree scratch", scratch / n);
}

// ---------------------------------------------------------------------------
// CPU reference: Manhattan Voronoi BFS with an iteration counter.
// Port of _ref_voronoi_timed() -- the NumPy version snapshots the grid per
// direction pass (np.pad copies), so each of the four passes reads a frozen
// state and writes into the live grid.
// ---------------------------------------------------------------------------

struct RefVoronoi {
    delauney::VoronoiGrid grid;
    int iterations = 0;
};

RefVoronoi ref_voronoi_timed(int W, int H, const std::vector<Seed>& seeds_in)
{
    auto seeds = delauney::sort_seeds(seeds_in, W, H);

    const size_t n_cells = static_cast<size_t>(W) * H;
    std::vector<int32_t> sid(n_cells, UNDEFINED), dst(n_cells, UNDEFINED);

    for (size_t seed_id = 0; seed_id < seeds.size(); ++seed_id) {
        size_t idx = static_cast<size_t>(seeds[seed_id].y) * W + seeds[seed_id].x;
        sid[idx] = static_cast<int32_t>(seed_id);
        dst[idx] = 0;
    }

    // Neighbour source offsets, in the same order as the NumPy SHIFTS list.
    const int dxs[4] = {-1, +1, 0, 0};
    const int dys[4] = {0, 0, -1, +1};

    std::vector<int32_t> prev_sid(n_cells), prev_dst(n_cells);
    int iters = 0;

    while (true) {
        ++iters;
        bool changed = false;

        for (int d = 0; d < 4; ++d) {
            prev_sid = sid;
            prev_dst = dst;

            for (int y = 0; y < H; ++y) {
                const int ny = y + dys[d];
                if (ny < 0 || ny >= H)
                    continue;
                for (int x = 0; x < W; ++x) {
                    const int nx = x + dxs[d];
                    if (nx < 0 || nx >= W)
                        continue;

                    const size_t n_idx = static_cast<size_t>(ny) * W + nx;
                    const int32_t n_id = prev_sid[n_idx];
                    if (n_id < 0)                      // neighbour undefined
                        continue;

                    const size_t idx = static_cast<size_t>(y) * W + x;
                    const int32_t n_d   = prev_dst[n_idx] + 1;
                    const int32_t cur_id = prev_sid[idx];
                    const int32_t cur_d  = prev_dst[idx];

                    const bool update = (cur_id == UNDEFINED) ||
                                        (n_d < cur_d) ||
                                        (n_d == cur_d && n_id > cur_id);
                    if (update) {
                        changed = true;
                        sid[idx] = n_id;
                        dst[idx] = n_d;
                    }
                }
            }
        }

        if (!changed)
            break;
    }

    RefVoronoi out;
    out.iterations = iters;
    std::vector<int32_t> flat(n_cells * 2);
    for (size_t i = 0; i < n_cells; ++i) {
        flat[i * 2 + 0] = sid[i];
        flat[i * 2 + 1] = dst[i];
    }
    out.grid = delauney::VoronoiGrid(W, H, 2, std::move(flat));
    return out;
}

// ---------------------------------------------------------------------------
// CPU reference: triangle detection + dedup (GridTriangulation steps 2-3).
// Port of _ref_detect_triangles(); returns the number of unique triangles.
// ---------------------------------------------------------------------------

size_t ref_detect_triangles(const delauney::VoronoiGrid& vgrid)
{
    const int W = vgrid.width();
    const int H = vgrid.height();

    // The four L-shape orientations: (cell, neighbour, diagonal) offsets
    // relative to the (rx, ry) window origin.
    struct Orientation { int cell_dx, cell_dy, nb_dx, nb_dy, dn_dx, dn_dy; };
    const Orientation orients[4] = {
        {1, 0, 0, 0, 1, 1},   // n_grid[:-1, 1:],  [:-1, :-1], [1:, 1:]
        {0, 0, 1, 0, 0, 1},   // n_grid[:-1, :-1], [:-1, 1:],  [1:, :-1]
        {0, 1, 1, 1, 0, 0},   // n_grid[1:, :-1],  [1:, 1:],   [:-1, :-1]
        {1, 1, 0, 1, 1, 0},   // n_grid[1:, 1:],   [1:, :-1],  [:-1, 1:]
    };

    auto seed_at = [&](int x, int y) { return vgrid.at(x, y, 0); };

    std::unordered_set<uint64_t> seen;
    seen.reserve(1u << 20);

    for (const auto& o : orients) {
        for (int ry = 0; ry < H - 1; ++ry) {
            for (int rx = 0; rx < W - 1; ++rx) {
                const int32_t s_cell = seed_at(rx + o.cell_dx, ry + o.cell_dy);
                const int32_t s_nb   = seed_at(rx + o.nb_dx,   ry + o.nb_dy);
                const int32_t s_dn   = seed_at(rx + o.dn_dx,   ry + o.dn_dy);

                if (s_cell == s_nb || s_cell == s_dn || s_nb == s_dn)
                    continue;

                int32_t t[3] = {s_nb, s_cell, s_dn};
                std::sort(t, t + 3);
                seen.insert((static_cast<uint64_t>(t[0]) << 42) |
                            (static_cast<uint64_t>(t[1]) << 21) |
                            static_cast<uint64_t>(t[2]));
            }
        }
    }
    return seen.size();
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    int W = 1024;
    int H = 1024;
    int N = 100000;
    uint64_t rng_seed = 42;
    int warm_inserts = 9;
    bool skip_ref = false;
    bool verify = false;
    std::set<int> only;          // empty == run everything
};

bool want(const Config& cfg, int sec)
{
    return cfg.only.empty() || cfg.only.count(sec) > 0;
}

// ---------------------------------------------------------------------------
// Extra seeds for the warm-insert path: `count` positions not already in use.
// ---------------------------------------------------------------------------

uint64_t seed_key(int x, int y)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint32_t>(y);
}

std::vector<Seed> make_extra_seeds(int W, int H, const std::vector<Seed>& seeds,
                                   int count)
{
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int> dx(0, W - 1), dy(0, H - 1);

    std::unordered_set<uint64_t> used;
    used.reserve(seeds.size() * 2);
    for (const auto& s : seeds)
        used.insert(seed_key(s.x, s.y));

    std::vector<Seed> extra;
    while (static_cast<int>(extra.size()) < count) {
        int x = dx(rng), y = dy(rng);
        if (used.insert(seed_key(x, y)).second)
            extra.push_back({x, y});
    }
    return extra;
}

// ---------------------------------------------------------------------------
// Verification: the incremental warm path (partial_triangulate_) must produce
// the same triangulation as inserting every seed in one cold batch
// (full_triangulate_).  Both instances see the same seeds in the same
// insertion order, so seed IDs -- and therefore triplets -- are comparable.
// ---------------------------------------------------------------------------

using Triplet = std::array<int32_t, 3>;

Triplet canonical(const TriangleEntry& e)
{
    Triplet t{e.id_a, e.id_b, e.id_c};
    std::sort(t.begin(), t.end());
    return t;
}

void run_verification(const Config& cfg, const std::vector<Seed>& seeds)
{
    section("V. Verification -- warm path vs full rebuild");

    const int W = cfg.W, H = cfg.H;
    const int n_extra = std::max(1, cfg.warm_inserts);
    const auto extra = make_extra_seeds(W, H, seeds, n_extra);

    std::vector<Seed> all = seeds;
    all.insert(all.end(), extra.begin(), extra.end());
    const int cap = static_cast<int>(all.size()) + 20;

    // A: every seed in one cold batch -> full_triangulate_
    delauney::IncrementalDelaunay a(W, H, cap);
    auto ra = a.insert(all);

    // B: base seeds cold, then one warm insert per extra seed
    delauney::IncrementalDelaunay b(W, H, cap);
    auto rb = b.insert(seeds);
    for (const auto& s : extra)
        rb = b.insert({s});

    print_kv("Warm inserts applied", std::to_string(n_extra));
    print_kv("Triangles (full rebuild)",
             with_commas(static_cast<long long>(ra.triangle_map.size())));
    print_kv("Triangles (warm path)",
             with_commas(static_cast<long long>(rb.triangle_map.size())));

    // 1. Triangle sets must be identical.
    auto to_set = [](const std::vector<TriangleEntry>& m) {
        std::set<Triplet> s;
        for (const auto& e : m)
            s.insert(canonical(e));
        return s;
    };
    const auto sa = to_set(ra.triangle_map);
    const auto sb = to_set(rb.triangle_map);

    print_kv("Triplet sets identical", sa == sb ? "YES" : "NO");
    if (sa != sb) {
        std::vector<Triplet> only_a, only_b;
        std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(),
                            std::back_inserter(only_a));
        std::set_difference(sb.begin(), sb.end(), sa.begin(), sa.end(),
                            std::back_inserter(only_b));
        print_kv("  Missing from warm path",
                 with_commas(static_cast<long long>(only_a.size())));
        print_kv("  Extra in warm path",
                 with_commas(static_cast<long long>(only_b.size())));
    }

    // 2. Per-pixel assignment must resolve to the same triangle.  Triangle IDs
    //    are numbered differently between the two, so compare via triplets.
    auto id_table = [](const std::vector<TriangleEntry>& m) {
        std::vector<Triplet> t(m.size());
        for (size_t i = 0; i < m.size(); ++i)
            t[i] = canonical(m[i]);
        return t;
    };
    const auto ta = id_table(ra.triangle_map);
    const auto tb = id_table(rb.triangle_map);

    size_t pixel_mismatch = 0, out_of_range = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int32_t ia = ra.grid.at(x, y, 2);
            const int32_t ib = rb.grid.at(x, y, 2);

            if (ia >= (int32_t)ta.size() || ib >= (int32_t)tb.size()) {
                ++out_of_range;
                continue;
            }
            const bool va = ia >= 0, vb = ib >= 0;
            if (va != vb || (va && ta[ia] != tb[ib]))
                ++pixel_mismatch;
        }
    }

    const double pct = 100.0 * pixel_mismatch / (double(W) * H);
    print_kv("Pixels assigned to a different triangle",
             with_commas(static_cast<long long>(pixel_mismatch)));
    std::printf("    %-52s %9.3f %%\n", "  as a fraction of the grid", pct);
    print_kv("Triangle IDs out of range (must be 0)",
             with_commas(static_cast<long long>(out_of_range)));
}

void print_usage(const char* argv0)
{
    std::printf(
        "Usage: %s [options]\n"
        "  -w, --width N     grid width                          (default 1024)\n"
        "  -h, --height N    grid height                         (default 1024)\n"
        "  -n, --seeds N     seed count                          (default 100000)\n"
        "      --seed N      RNG seed                            (default 42)\n"
        "      --warm N      warm single-seed inserts to time    (default 9)\n"
        "      --skip-ref    skip CPU reference sections (2, 4)\n"
        "      --only LIST   comma-separated sections, e.g. --only 3,6\n"
        "      --verify      check the incremental warm path against a full rebuild\n"
        "      --help        this message\n",
        argv0);
}

bool parse_args(int argc, char** argv, Config& cfg)
{
    auto need_value = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s requires a value\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-?") {
            print_usage(argv[0]);
            return false;
        } else if (a == "-w" || a == "--width") {
            const char* v = need_value(i); if (!v) return false;
            cfg.W = std::atoi(v);
        } else if (a == "-h" || a == "--height") {
            const char* v = need_value(i); if (!v) return false;
            cfg.H = std::atoi(v);
        } else if (a == "-n" || a == "--seeds") {
            const char* v = need_value(i); if (!v) return false;
            cfg.N = std::atoi(v);
        } else if (a == "--seed") {
            const char* v = need_value(i); if (!v) return false;
            cfg.rng_seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--warm") {
            const char* v = need_value(i); if (!v) return false;
            cfg.warm_inserts = std::atoi(v);
        } else if (a == "--skip-ref") {
            cfg.skip_ref = true;
        } else if (a == "--verify") {
            cfg.verify = true;
        } else if (a == "--only") {
            const char* v = need_value(i); if (!v) return false;
            std::string list = v;
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                cfg.only.insert(std::atoi(list.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
        } else {
            std::fprintf(stderr, "error: unknown option '%s'\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }

    if (cfg.W <= 0 || cfg.H <= 0 || cfg.N <= 0) {
        std::fprintf(stderr, "error: width, height and seeds must be positive\n");
        return false;
    }
    if (static_cast<long long>(cfg.N) > static_cast<long long>(cfg.W) * cfg.H) {
        std::fprintf(stderr, "error: more seeds than grid cells\n");
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int run(int argc, char** argv)
{
    Config cfg;
    if (!parse_args(argc, argv, cfg))
        return 1;

    const int W = cfg.W, H = cfg.H, N = cfg.N;
    const bool have_cuda = delauney::cuda_available();

    hr();
    std::printf("\n  Delauney profiling  |  grid %dx%d  |  %s seed points\n",
                W, H, with_commas(N).c_str());
    std::printf("  CUDA device: %s\n",
                have_cuda ? delauney::cuda_device_name() : "[none detected]");
    hr();

    // -- 1. Seed generation --------------------------------------------------
    section("1. Seed generation");
    std::vector<Seed> seeds;
    {
        Timer t("Generate unique positions");
        std::mt19937_64 rng(cfg.rng_seed);
        std::uniform_int_distribution<int> dx(0, W - 1), dy(0, H - 1);

        std::unordered_set<uint64_t> coords;
        coords.reserve(static_cast<size_t>(N) * 2);
        seeds.reserve(N);
        while (static_cast<int>(seeds.size()) < N) {
            int x = dx(rng), y = dy(rng);
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
                           static_cast<uint32_t>(y);
            if (coords.insert(key).second)
                seeds.push_back({x, y});
        }
    }

    // -- 2. RegularDelaunay -- CPU reference ---------------------------------
    delauney::VoronoiGrid ref_vgrid;
    int n_iter = 0;
    const bool run_ref_bfs = want(cfg, 2) && !cfg.skip_ref;

    if (run_ref_bfs) {
        section("2. RegularDelaunay -- CPU reference");
        RefVoronoi ref;
        {
            Timer t("compute()  (full BFS)");
            ref = ref_voronoi_timed(W, H, seeds);
        }
        n_iter = ref.iterations;
        int32_t max_d = 0;
        for (size_t i = 1; i < ref.grid.data().size(); i += 2)
            max_d = std::max(max_d, ref.grid.data()[i]);
        print_kv("BFS convergence iterations", std::to_string(n_iter));
        print_kv("Max distance in grid (pixels)", std::to_string(max_d));
        ref_vgrid = std::move(ref.grid);
    }

    // -- 3. RegularDelaunay -- CUDA ------------------------------------------
    delauney::VoronoiGrid cuda_vgrid;
    if (want(cfg, 3)) {
        section("3. RegularDelaunay -- CUDA");
        if (have_cuda) {
            delauney::RegularDelaunay cuda_vd;
            cuda_vd.compute(32, 32, {{0, 0}, {31, 31}});          // warm-up
            {
                Timer t("compute()  (warm-up excluded)");
                cuda_vgrid = cuda_vd.compute(W, H, seeds);
            }
            if (!ref_vgrid.empty()) {
                bool match = (cuda_vgrid == ref_vgrid);
                print_kv("Result matches CPU reference", match ? "YES" : "NO");
            } else {
                print_kv("Result matches CPU reference", "n/a");
            }
        } else {
            std::printf("    [CUDA device not available -- skipped]\n");
        }
    }
    if (cuda_vgrid.empty())
        cuda_vgrid = ref_vgrid;

    // -- 4. GridTriangulation -- CPU reference (detection only) --------------
    size_t T = 0;
    double hours_est = 0.0;
    if (want(cfg, 4) && !cfg.skip_ref && !ref_vgrid.empty()) {
        section("4. GridTriangulation -- CPU reference (detection + dedup only)");
        {
            Timer t("Triangle detection + deduplication");
            T = ref_detect_triangles(ref_vgrid);
        }
        double ops = static_cast<double>(W) * H * static_cast<double>(T);
        hours_est = ops / 1e7 / 3600.0;
        print_kv("Unique triangles found", with_commas(static_cast<long long>(T)));
        std::printf("    %-52s %9.1f G\n", "Assignment ops  W x H x T  (skipped)", ops / 1e9);
        std::printf("    ('-> ~%.1f h in a Python loop; GPU required for this scale)\n",
                    hours_est);
    }

    // -- 5. GridTriangulation -- CUDA ----------------------------------------
    bool have_tri_timings = false;
    TriTimings gpu_timings;
    if (want(cfg, 5)) {
        section("5. GridTriangulation -- CUDA (full pipeline incl. assignment)");
        if (have_cuda && !cuda_vgrid.empty()) {
            delauney::GridTriangulation cuda_tri;
            delauney::TriangulationResult res;
            {
                Timer t("compute_timed()  (detection + dedup + assignment)");
                res = cuda_tri.compute_timed(cuda_vgrid, seeds, gpu_timings);
            }
            have_tri_timings = true;

            const size_t got = res.triangle_map.size();
            std::string verdict;
            if (T == 0)
                verdict = with_commas(static_cast<long long>(got));
            else if (got == T)
                verdict = "YES (" + std::to_string(got) + ")";
            else
                verdict = "NO (" + std::to_string(got) + " vs " + std::to_string(T) + ")";
            print_kv("Triangle count matches reference", verdict);

            std::printf("\n    GPU sub-phase breakdown:\n");
            print_ms("  detect (find_triangle_seeds kernel)", gpu_timings.detect_ms);
            print_ms("  dedup  (thrust sort + unique)",       gpu_timings.dedup_ms);
            print_ms("  assign (assign_triangles kernel)",    gpu_timings.assign_ms);
        } else {
            std::printf("    [%s -- skipped]\n",
                        have_cuda ? "no Voronoi grid available" : "CUDA device not available");
        }
    }

    // -- 6. IncrementalDelaunay -- CUDA --------------------------------------
    if (want(cfg, 6)) {
        section("6. IncrementalDelaunay -- CUDA");
        if (have_cuda) {
            // Capacity must cover the cold batch plus every warm insert below.
            const int cap = N + std::max(1, cfg.warm_inserts) + 20;
            delauney::IncrementalDelaunay inc(W, H, cap);

            IncrementalTimings inc_t0;
            delauney::TriangulationResult cold;
            {
                Timer t("insert_timed()  cold  (all seeds, full triangulate)");
                cold = inc.insert_timed(seeds, inc_t0);
            }
            print_kv("  Seeds inserted", with_commas(inc.seed_count()));
            print_kv("  Triangles found",
                     with_commas(static_cast<long long>(cold.triangle_map.size())));

            std::printf("\n    GPU sub-phase breakdown (cold):\n");
            print_ms("  bfs    (BFS until convergence)",      inc_t0.bfs_ms);
            print_ms("  detect (find_triangle_seeds kernel)", inc_t0.detect_ms);
            print_ms("  dedup  (thrust sort + unique)",       inc_t0.dedup_ms);
            print_ms("  assign (assign_triangles kernel)",    inc_t0.assign_ms);

            {
                const auto& C = inc_t0.host;
                const std::vector<HostRow> rows = {
                    { "  insert: validate seeds",         C.validate_ms },
                    { "  insert: register seeds",         C.seed_reg_ms },
                    { "  insert: seed H2D + clear",       C.seed_h2d_ms },
                    { "  insert: write_seeds kernel",     C.write_seeds_ms },
                    { "  insert: bfs (wall)",             C.bfs_ms },
                    { "  full: detect",                   C.detect_ms },
                    { "  full: dedup",                    C.dedup_ms },
                    { "  full: D2H dedup triangles",      C.d2h_new_ms },
                    { "  full: build registry",           C.build_registry_ms },
                    { "  full: rebuild_csr",              C.csr_ms },
                    { "  full: assign",                   C.assign_ms },
                    { "  outputs: fill tri_map",          C.out_trimap_ms },
                    { "  outputs: D2H grids",             C.out_d2h_ms },
                    { "  outputs: interleave",            C.out_interleave_ms },
                };
                print_host_breakdown("Host wall-clock breakdown (cold):",
                                     rows, 1.0, C.scratch_ms);
            }

            // Warm inserts: single extra seed each (exercises partial_triangulate_).
            const int WARM_N = std::max(1, cfg.warm_inserts);
            const auto extra = make_extra_seeds(W, H, seeds, WARM_N + 1);

            IncrementalTimings scratch;
            inc.insert_timed({extra[0]}, scratch);          // prime caches

            IncrementalTimings warm_tot;
            double warm_wall = 0.0;
            {
                DELAUNEY_NVTX_RANGE_C("warm insert loop", delauney_nvtx::kPhase);
                for (int i = 1; i <= WARM_N; ++i) {
                    DELAUNEY_NVTX_RANGE("warm insert");
                    IncrementalTimings wt;
                    auto t0 = Clock::now();
                    inc.insert_timed({extra[i]}, wt);
                    warm_wall += std::chrono::duration<double>(Clock::now() - t0).count();
                    warm_tot.bfs_ms    += wt.bfs_ms;
                    warm_tot.detect_ms += wt.detect_ms;
                    warm_tot.dedup_ms  += wt.dedup_ms;
                    warm_tot.assign_ms += wt.assign_ms;

                    const auto& h = wt.host;
                    auto& H = warm_tot.host;
                    H.validate_ms       += h.validate_ms;
                    H.seed_reg_ms       += h.seed_reg_ms;
                    H.seed_h2d_ms       += h.seed_h2d_ms;
                    H.write_seeds_ms    += h.write_seeds_ms;
                    H.bfs_ms            += h.bfs_ms;
                    H.d2h_changed_ms    += h.d2h_changed_ms;
                    H.expand_ms         += h.expand_ms;
                    H.mark_stale_ms     += h.mark_stale_ms;
                    H.h2d_border_ms     += h.h2d_border_ms;
                    H.detect_ms         += h.detect_ms;
                    H.dedup_ms          += h.dedup_ms;
                    H.d2h_new_ms        += h.d2h_new_ms;
                    H.build_registry_ms += h.build_registry_ms;
                    H.collect_ms        += h.collect_ms;
                    H.compact_ms        += h.compact_ms;
                    H.upload_tri_ms     += h.upload_tri_ms;
                    H.csr_ms            += h.csr_ms;
                    H.remap_ms          += h.remap_ms;
                    H.h2d_reassign_ms   += h.h2d_reassign_ms;
                    H.assign_ms         += h.assign_ms;
                    H.out_trimap_ms     += h.out_trimap_ms;
                    H.out_d2h_ms        += h.out_d2h_ms;
                    H.out_interleave_ms += h.out_interleave_ms;
                    H.scratch_ms        += h.scratch_ms;
                }
            }

            const double warm_wall_avg = warm_wall / WARM_N * 1000.0;
            g_results["insert_timed()  warm  (single seed, partial triangulate)"] =
                warm_wall / WARM_N;

            std::string warm_label =
                "insert_timed()  warm  (single seed, avg over " +
                std::to_string(WARM_N) + ")";
            std::printf("\n    %-52s %9.1f ms\n", warm_label.c_str(), warm_wall_avg);

            std::printf("\n    GPU sub-phase breakdown (warm, avg):\n");
            print_ms("  bfs",    warm_tot.bfs_ms    / WARM_N);
            print_ms("  detect", warm_tot.detect_ms / WARM_N);
            print_ms("  dedup",  warm_tot.dedup_ms  / WARM_N);
            print_ms("  assign", warm_tot.assign_ms / WARM_N);

            {
                const auto& H = warm_tot.host;
                const std::vector<HostRow> rows = {
                    { "  insert: validate seeds",        H.validate_ms },
                    { "  insert: register seeds",        H.seed_reg_ms },
                    { "  insert: seed H2D + clear",      H.seed_h2d_ms },
                    { "  insert: write_seeds kernel",    H.write_seeds_ms },
                    { "  insert: bfs (wall)",            H.bfs_ms },
                    { "  partial: D2H changed mask",     H.d2h_changed_ms },
                    { "  partial: expand masks",         H.expand_ms },
                    { "  partial: mark stale",           H.mark_stale_ms },
                    { "  partial: H2D border mask",      H.h2d_border_ms },
                    { "  partial: detect",               H.detect_ms },
                    { "  partial: dedup",                H.dedup_ms },
                    { "  partial: D2H new triangles",    H.d2h_new_ms },
                    { "  partial: collect new triplets", H.collect_ms },
                    { "  partial: compact registry",     H.compact_ms },
                    { "  partial: upload_triangles",     H.upload_tri_ms },
                    { "  partial: rebuild_csr",          H.csr_ms },
                    { "  partial: remap t_grid",         H.remap_ms },
                    { "  partial: H2D reassign mask",    H.h2d_reassign_ms },
                    { "  partial: assign",               H.assign_ms },
                    { "  outputs: fill tri_map",         H.out_trimap_ms },
                    { "  outputs: D2H grids",            H.out_d2h_ms },
                    { "  outputs: interleave",           H.out_interleave_ms },
                };
                print_host_breakdown("Host wall-clock breakdown (warm, avg):",
                                     rows, (double)WARM_N, H.scratch_ms);
            }
        } else {
            std::printf("    [CUDA device not available -- skipped]\n");
        }
    }

    // -- Verification (optional) ---------------------------------------------
    if (cfg.verify) {
        if (have_cuda)
            run_verification(cfg, seeds);
        else
            std::printf("\n    [--verify needs a CUDA device -- skipped]\n");
    }

    // -- 7. Summary ----------------------------------------------------------
    section("7. Summary");
    constexpr int COL = 46;
    const std::pair<const char*, const char*> labels[] = {
        {"Generate seeds",               "Generate unique positions"},
        {"RegularDelaunay  (CPU ref)",   "compute()  (full BFS)"},
        {"RegularDelaunay  (CUDA)",      "compute()  (warm-up excluded)"},
        {"GridTriang. detect (CPU ref)", "Triangle detection + deduplication"},
        {"GridTriang. full   (CUDA)",    "compute_timed()  (detection + dedup + assignment)"},
        {"Incremental cold   (CUDA)",    "insert_timed()  cold  (all seeds, full triangulate)"},
        {"Incremental warm   (CUDA)",    "insert_timed()  warm  (single seed, partial triangulate)"},
    };

    std::printf("\n    %-*s  %11s  %10s\n", COL, "Phase", "Time", "vs ref BFS");
    std::printf("    %s  %s  %s\n", std::string(COL, '-').c_str(),
                std::string(11, '-').c_str(), std::string(10, '-').c_str());

    const auto ref_it = g_results.find("compute()  (full BFS)");
    const double ref_bfs_s = (ref_it != g_results.end()) ? ref_it->second : 0.0;

    for (const auto& [display, key] : labels) {
        auto it = g_results.find(key);
        if (it == g_results.end()) {
            std::printf("    %-*s  %11s  %10s\n", COL, display, "n/a", "");
            continue;
        }
        std::string speedup;
        if (ref_bfs_s > 0.0 && std::strcmp(key, "compute()  (warm-up excluded)") == 0) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1fx", ref_bfs_s / it->second);
            speedup = buf;
        }
        std::printf("    %-*s  %10.1fms  %10s\n", COL, display,
                    it->second * 1000.0, speedup.c_str());
    }

    std::printf("\n    Complexity note:\n");
    if (n_iter)
        std::printf("      Voronoi BFS:      O(W x H x iters)  -- iters = %d here\n", n_iter);
    else
        std::printf("      Voronoi BFS:      O(W x H x iters)\n");
    std::printf("      Tri. detection:   O(W x H x 4)\n");
    if (T) {
        std::printf("      Tri. assignment:  O(W x H x T)      -- T = %s triangles\n",
                    with_commas(static_cast<long long>(T)).c_str());
        std::printf("        Reference Python: ~%.1f h\n", hours_est);
    } else {
        std::printf("      Tri. assignment:  O(W x H x T)\n");
    }
    if (have_tri_timings) {
        std::printf("        CUDA kernel breakdown:\n");
        std::printf("          detect: %.1f ms\n", gpu_timings.detect_ms);
        std::printf("          dedup:  %.1f ms  (Thrust sort+unique on device)\n",
                    gpu_timings.dedup_ms);
        std::printf("          assign: %.1f ms\n", gpu_timings.assign_ms);
    }
    hr();

    return 0;
}

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n  ERROR: %s\n", e.what());
        return 2;
    }
}
