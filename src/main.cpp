#include "graphlab/core.hpp"
#ifdef GRAPHLAB_CUDA
#include "graphlab/cuda.hpp"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace graphlab;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static std::string quote(std::string_view text) {
    std::ostringstream s;
    s << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            s << '\\' << c;
        } else if (c == '\n') {
            s << '\\' << 'n';
        } else if (c == '\r') {
            s << '\\' << 'r';
        } else if (c == '\t') {
            s << '\\' << 't';
        } else if (c < 32) {
            s << '\\' << 'u' << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        } else
            s << c;
    }
    s << '"';
    return s.str();
}
using Options = std::map<std::string, std::string>;
static Options options(int argc, char **argv) {
    Options o;
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--smoke") {
            o[k] = "true";
            continue;
        }
        if (k.rfind("--", 0) != 0 || i + 1 == argc)
            throw std::runtime_error("expected --option value");
        if (o.contains(k))
            throw std::runtime_error("duplicate option: " + k);
        o[k] = argv[++i];
    }
    const std::vector<std::string> known = {"--nodes", "--features", "--family",  "--seed",
                                            "--input", "--output",   "--backend", "--groups",
                                            "--calls", "--warmup",   "--smoke",   "--group-offset"};
    for (auto &[k, v] : o)
        if (std::find(known.begin(), known.end(), k) == known.end())
            throw std::runtime_error("unknown option: " + k);
    return o;
}
static std::string get(const Options &o, const std::string &k, const std::string &d) {
    auto it = o.find(k);
    return it == o.end() ? d : it->second;
}
static int64_t integer(const Options &o, const std::string &k, int64_t d) {
    auto s = get(o, k, std::to_string(d));
    size_t p = 0;
    auto value = std::stoll(s, &p);
    if (p != s.size())
        throw std::runtime_error("invalid integer for " + k);
    return value;
}
static std::vector<float> serial(const Graph &g) {
    validate(g);
    std::vector<float> y(g.x.size());
    for (int64_t v = 0; v < g.nodes; ++v)
        for (int64_t f = 0; f < g.features; ++f) {
            float sum = 0;
            auto begin = g.offsets[static_cast<size_t>(v)],
                 end = g.offsets[static_cast<size_t>(v) + 1];
            for (auto e = begin; e < end; ++e)
                sum += g.x[static_cast<size_t>(g.columns[e]) * static_cast<size_t>(g.features) +
                           static_cast<size_t>(f)];
            y[static_cast<size_t>(v * g.features + f)] =
                begin == end ? 0.0f : sum / static_cast<float>(end - begin);
        }
    return y;
}
static void save(std::ostream &s, const Graph &g) {
    validate(g);
    s << "graphlab-csr-v1\n" << g.nodes << ' ' << g.features << ' ' << g.columns.size() << '\n';
    for (auto x : g.offsets)
        s << x << ' ';
    s << '\n';
    for (auto x : g.columns)
        s << x << ' ';
    s << '\n';
    s << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (auto x : g.x)
        s << x << ' ';
    s << '\n';
    if (!s)
        throw std::runtime_error("fixture write failed");
}
static Graph load(std::istream &s) {
    std::string magic;
    int64_t n = 0, f = 0, e = 0;
    s >> magic >> n >> f >> e;
    if (!s || magic != "graphlab-csr-v1" || n < 0 || n > INT32_MAX || f < 1 || f > INT32_MAX ||
        e < 0 || e > INT32_MAX)
        throw Error(ErrorCode::invalid_input, "invalid fixture header");
    const uint64_t elements = static_cast<uint64_t>(n) * static_cast<uint64_t>(f);
    if (elements > 128ULL * 1024 * 1024)
        throw Error(ErrorCode::unsupported_domain,
                    "fixture feature allocation exceeds host budget");
    const uint64_t bytes = 4 * (static_cast<uint64_t>(n) + 1 + static_cast<uint64_t>(e) + elements);
    if (bytes > 512ULL * 1024 * 1024)
        throw Error(ErrorCode::unsupported_domain, "fixture exceeds 512 MiB host input budget");
    Graph g{n, f, {}, {}, {}};
    g.offsets.resize(static_cast<size_t>(n) + 1);
    g.columns.resize(static_cast<size_t>(e));
    g.x.resize(static_cast<size_t>(elements));
    for (auto &x : g.offsets)
        if (!(s >> x))
            throw Error(ErrorCode::invalid_input, "truncated/invalid offsets");
    for (auto &x : g.columns)
        if (!(s >> x))
            throw Error(ErrorCode::invalid_input, "truncated/invalid columns");
    for (auto &x : g.x)
        if (!(s >> x))
            throw Error(ErrorCode::invalid_input, "truncated/invalid features");
    s >> std::ws;
    if (s.peek() != std::char_traits<char>::eof())
        throw Error(ErrorCode::invalid_input, "trailing fixture data");
    validate(g);
    return g;
}
static Graph input(const Options &o) {
    auto file = get(o, "--input", "");
    if (!file.empty()) {
        std::ifstream s(file);
        if (!s)
            throw std::runtime_error("cannot open input fixture");
        return load(s);
    }
    auto seed = integer(o, "--seed", 7);
    if (seed < 0)
        throw std::runtime_error("seed must be nonnegative");
    return generate(integer(o, "--nodes", 4096), integer(o, "--features", 32),
                    get(o, "--family", "regular8"), static_cast<uint64_t>(seed));
}
static void check_output(const Graph &g, const Oracle &r, const std::vector<float> &y) {
    auto check = verify(g, r, y);
    if (!check.passed)
        throw std::runtime_error("numerical_mismatch: " + check.reason +
                                 " index=" + std::to_string(check.worst_index));
}
#ifdef GRAPHLAB_CUDA
static cuda::Variant variant(std::string_view name) {
    if (name == "row_split")
        return cuda::Variant::row_split;
    if (name == "naive")
        return cuda::Variant::naive;
    if (name == "cusparse")
        return cuda::Variant::cusparse;
    throw std::runtime_error("backend must be cpu, naive, row_split or cusparse");
}
#endif
static void info() {
    std::cout << R"({"type":"environment","contract":"csr-mean-v1","cxx":)" <<
#ifdef _MSVC_LANG
        _MSVC_LANG;
#else
        __cplusplus;
#endif
#ifdef _MSC_VER
    std::cout << R"(,"compiler":)" << quote("MSVC " + std::to_string(_MSC_FULL_VER));
#endif
#ifdef __VERSION__
    std::cout << R"(,"compiler":)" << quote(__VERSION__);
#endif
#ifdef GRAPHLAB_CUDA
    auto r = cuda::runtime_info();
    std::cout << R"(,"cuda_build":true,"gpu_available":)" << (r.available ? "true" : "false")
              << R"(,"device":)" << quote(r.name) << R"(,"error":)" << quote(r.error)
              << R"(,"driver_version":)" << r.driver_version << R"(,"runtime_version":)"
              << r.runtime_version << R"(,"cusparse_version":)" << r.cusparse_version
              << R"(,"compute_major":)" << r.compute_major << R"(,"compute_minor":)"
              << r.compute_minor << R"(,"total_bytes":)" << r.total_bytes << R"(,"free_bytes":)"
              << r.free_bytes;
#else
    std::cout << R"(,"cuda_build":false,"gpu_available":false)";
#endif
    std::cout << "}\n";
}
static int verify_suite(const Options &o) {
    auto backend = get(o, "--backend", "cpu");
    if (backend != "cpu" && backend != "all" && backend != "naive" && backend != "row_split" &&
        backend != "cusparse")
        throw std::runtime_error("invalid verification backend");
#ifndef GRAPHLAB_CUDA
    if (backend != "cpu") {
        std::cout
            << R"({"type":"verification","status":"not_run","reason":"CUDA disabled at build time"}
)";
        return 77;
    }
#else
    if (backend != "cpu" && !cuda::runtime_info().available) {
        std::cout
            << R"({"type":"verification","status":"not_run","reason":"CUDA device unavailable"}
)";
        return 77;
    }
#endif
    std::vector<std::pair<std::string, Graph>> cases;
    cases.push_back(
        {"directed",
         Graph{4, 2, {0, 2, 2, 3, 4}, {1, 3, 0, 2}, {1, 0, -1, 1, 0.5f, -0.5f, 0, 0.25f}}});
    cases.push_back({"empty", Graph{0, 7, {0}, {}, {}}});
    cases.push_back({"zero_edges", Graph{3, 1, {0, 0, 0, 0}, {}, {1, -1, 0.5f}}});
    cases.push_back({"subnormal", Graph{3,
                                        1,
                                        {0, 1, 2, 3},
                                        {0, 1, 2},
                                        {std::numeric_limits<float>::denorm_min(),
                                         -std::numeric_limits<float>::denorm_min(),
                                         std::numeric_limits<float>::min()}}});
    cases.push_back(
        {"tiny_cancellation", Graph{2,
                                    1,
                                    {0, 2, 2},
                                    {0, 1},
                                    {std::nextafter(std::numeric_limits<float>::min(), 1.0f),
                                     -std::numeric_limits<float>::min()}}});
    cases.push_back(
        {"cancellation",
         Graph{3, 1, {0, 3, 3, 3}, {0, 1, 2}, {1, std::numeric_limits<float>::epsilon() / 4, -1}}});
    for (int f : {1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129})
        for (int n : {1, 31, 32, 33, 255, 256, 257})
            cases.emplace_back("tails_" + std::to_string(n) + "_" + std::to_string(f),
                               generate(n, f, "skewed", 42));
    cases.emplace_back("high_degree", generate(4096, 7, "skewed", 2026));
    Graph maximum{4096, 1, std::vector<int32_t>(4097, 4096), {}, std::vector<float>(4096)};
    maximum.offsets[0] = 0;
    for (int32_t i = 0; i < 4096; ++i) {
        maximum.columns.push_back(i);
        maximum.x[i] = (i % 2) ? -0.5f : 1.0f;
    }
    cases.emplace_back("maximum_degree", std::move(maximum));
    std::vector<std::string> backends =
        backend == "all" ? std::vector<std::string>{"cpu", "naive", "row_split", "cusparse"}
                         : std::vector<std::string>{backend};
    size_t count = 0;
    for (const auto &[name, g] : cases) {
        auto r = oracle(g);
        for (const auto &b : backends) {
            std::vector<float> y;
            if (b == "cpu")
                y = serial(g);
#ifdef GRAPHLAB_CUDA
            else {
                cuda::Session session(g, variant(b));
                y = session.output();
                check_output(g, r, y);
                y = session.output();
            }
#endif
            auto check = verify(g, r, y);
            if (name == "subnormal" && y != g.x) {
                check.passed = false;
                check.reason = "degree-one subnormal identity failed";
            }
            std::cout << R"({"type":"verification","fixture":)" << quote(name)
                      << R"(,"fingerprint":)" << quote(fingerprint(g)) << R"(,"backend":)"
                      << quote(b) << R"(,"status":)" << quote(check.passed ? "passed" : "failed")
                      << R"(,"max_abs_error":)" << check.max_abs_error << R"(,"max_scaled_error":)"
                      << check.max_scaled_error << "}\n";
            if (!check.passed)
                return 1;
            ++count;
        }
    }
    std::cout << R"({"type":"verification_summary","status":"passed","cases":)" << count << "}\n";
    return 0;
}
static int bench(const Options &o) {
    auto requested = get(o, "--backend", "naive");
    if (requested != "cpu" && requested != "naive" && requested != "row_split" &&
        requested != "cusparse" && requested != "all")
        throw std::runtime_error("invalid benchmark backend");
    const auto groups = integer(o, "--groups", 5), calls = integer(o, "--calls", 100),
               warmup = integer(o, "--warmup", 20), offset = integer(o, "--group-offset", 0);
    if (groups < 1 || groups > 1000 || calls < 1 || calls > 1000000 || warmup < 0 ||
        warmup > 100000 || offset < 0 ||
        offset > std::numeric_limits<int64_t>::max() - (groups - 1))
        throw std::runtime_error("invalid repetition counts");
    const bool smoke = o.contains("--smoke");
    if (!smoke && (calls < 100 || warmup < 20))
        throw std::runtime_error(
            "campaign requires >=100 calls and >=20 warmups; use --smoke for diagnostics");
    const auto generation_start = Clock::now();
    auto g = input(o);
    auto generation_ms = elapsed(generation_start);
    auto reference_start = Clock::now();
    auto r = oracle(g);
    auto oracle_ms = elapsed(reference_start);
    auto id = fingerprint(g);
    auto stats = graph_stats(g);
    for (int64_t group = 0; group < groups; ++group) {
        auto backends = requested == "all"
                            ? std::vector<std::string>{"naive", "row_split", "cusparse"}
                            : std::vector<std::string>{requested};
        if ((group + offset) % 2)
            std::reverse(backends.begin(), backends.end());
        for (const auto &backend : backends) {
            double preparation_ms = 0, resident_ms = 0, prepared_wall_ms = 0, cold_wall_ms = 0,
                   h2d_ms = 0, aggregation_ms = 0, d2h_ms = 0;
            uint64_t memory = 0, workspace = 0, budget = 0, reserve = 0;
            std::string execution_path = "host_serial";
            std::vector<float> y;
            if (backend == "cpu") {
                y = serial(g);
                check_output(g, r, y);
                for (int64_t i = 0; i < warmup; ++i)
                    y = serial(g);
                auto begin = Clock::now();
                for (int64_t i = 0; i < calls; ++i)
                    y = serial(g);
                resident_ms = elapsed(begin);
                check_output(g, r, y);
                begin = Clock::now();
                y = serial(g);
                prepared_wall_ms = elapsed(begin);
                cold_wall_ms = prepared_wall_ms;
            } else {
#ifdef GRAPHLAB_CUDA
                {
                    cuda::Session session(g, variant(backend));
                    y = session.output();
                    check_output(g, r, y);
                    preparation_ms = session.preparation_ms();
                    memory = session.memory_bytes();
                    workspace = session.workspace_bytes();
                    budget = session.memory_budget_bytes();
                    reserve = session.overhead_reserve_bytes();
                    execution_path = session.path();
                    session.warmup(static_cast<size_t>(warmup));
                    resident_ms = session.resident_batch(static_cast<size_t>(calls));
                    y = session.download_output();
                    check_output(g, r, y);
                    auto prepared = session.prepared_call();
                    y = std::move(prepared.output);
                    prepared_wall_ms = prepared.wall_ms;
                    h2d_ms = prepared.h2d_ms;
                    aggregation_ms = prepared.aggregation_ms;
                    d2h_ms = prepared.d2h_ms;
                    check_output(g, r, y);
                }
                auto cold = Clock::now();
                {
                    cuda::Session session(g, variant(backend));
                    y = session.output();
                }
                cold_wall_ms = elapsed(cold);
#else
                throw std::runtime_error("CUDA disabled at build time");
#endif
            }
            auto check = verify(g, r, y);
            check_output(g, r, y);
            std::cout
                << std::setprecision(17)
                << R"({"type":"measurement","schema":1,"status":"passed","contract":"csr-mean-v1","smoke":)"
                << (smoke ? "true" : "false") << R"(,"backend":)" << quote(backend)
                << R"(,"family":)"
                << quote(o.contains("--input") ? "external_fixture"
                                               : get(o, "--family", "regular8"))
                << R"(,"seed":)"
                << (o.contains("--input") ? "null" : std::to_string(integer(o, "--seed", 7)))
                << R"(,"nodes":)" << g.nodes << R"(,"features":)" << g.features << R"(,"edges":)"
                << g.columns.size() << R"(,"max_degree":)" << stats.max_degree
                << R"(,"host_peak_estimate_bytes":)"
                << (2ULL * (g.offsets.size() + g.columns.size()) * sizeof(int32_t) +
                    36ULL * g.x.size())
                << R"(,"numerical_policy":"fp32_rn_gradual_underflow_fmad_false","cusparse_algorithm":"CSR_ALG2","preprocessing":"none")"
                << R"(,"fingerprint":)" << quote(id) << R"(,"group":)" << group + offset
                << R"(,"calls":)" << calls << R"(,"warmup":)" << warmup << R"(,"path":)"
                << quote(execution_path) << R"(,"resident_boundary":)"
                << quote(backend == "cpu" ? "host_validation_allocation_aggregation"
                                          : "device_normalized_aggregation")
                << R"(,"batch_total_ms":)" << resident_ms << R"(,"per_call_ms":)"
                << resident_ms / static_cast<double>(calls) << R"(,"preparation_ms":)"
                << preparation_ms << R"(,"prepared_wall_ms":)" << prepared_wall_ms
                << R"(,"h2d_ms":)" << h2d_ms << R"(,"aggregation_ms":)" << aggregation_ms
                << R"(,"d2h_ms":)" << d2h_ms << R"(,"cold_initialized_runtime_wall_ms":)"
                << cold_wall_ms << R"(,"generation_ms":)" << generation_ms << R"(,"oracle_ms":)"
                << oracle_ms << R"(,"device_memory_bytes":)" << memory << R"(,"workspace_bytes":)"
                << workspace << R"(,"memory_budget_bytes":)" << budget
                << R"(,"overhead_reserve_bytes":)" << reserve << R"(,"max_abs_error":)"
                << check.max_abs_error << R"(,"max_scaled_error":)" << check.max_scaled_error
                << "}\n"
                << std::flush;
        }
    }
    return 0;
}
int main(int argc, char **argv) {
    try {
        std::cout << std::setprecision(17);
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            std::cout
                << "graphlab info | verify | bench | generate | validate | roundtrip\n"
                << "Options: --backend cpu|naive|row_split|cusparse|all --nodes N --features F\n"
                << "         --family regular8|regular32|skewed --seed S --input FILE\n"
                << "         --output FILE --groups G --calls C --warmup W --smoke\n";
            return 0;
        }
        auto o = options(argc, argv);
        std::string command = argv[1];
        if (command == "info") {
            info();
            return 0;
        }
        if (command == "verify")
            return verify_suite(o);
        if (command == "bench")
            return bench(o);
        if (command == "roundtrip") {
            auto g = generate(33, 7, "skewed", 2026);
            std::stringstream s;
            save(s, g);
            auto h = load(s);
            if (fingerprint(g) != fingerprint(h))
                throw std::runtime_error("fixture roundtrip mismatch");
            std::cout << R"({"type":"roundtrip","status":"passed"}
)";
            return 0;
        }
        if (command == "generate" || command == "validate") {
            auto g = input(o);
            if (command == "generate") {
                auto out = get(o, "--output", "");
                if (out.empty())
                    throw std::runtime_error("generate requires --output");
                std::ofstream s(out);
                if (!s)
                    throw std::runtime_error("cannot open output fixture");
                save(s, g);
            }
            std::cout << R"({"type":"fixture","status":"passed","nodes":)" << g.nodes
                      << R"(,"features":)" << g.features << R"(,"edges":)" << g.columns.size()
                      << R"(,"fingerprint":)" << quote(fingerprint(g)) << "}\n";
            return 0;
        }
        throw std::runtime_error("unknown command: " + command);
    } catch (const graphlab::Error &e) {
        std::cout << R"({"type":"error","status":"failed","code":)"
                  << quote(error_code_name(e.code)) << R"(,"message":)" << quote(e.what()) << "}\n";
        return 2;
#ifdef GRAPHLAB_CUDA
    } catch (const graphlab::cuda::Error &e) {
        std::cout << R"({"type":"error","status":"failed","code":)" << quote(e.code())
                  << R"(,"message":)" << quote(e.what()) << "}\n";
        return 3;
#endif
    } catch (const std::exception &e) {
        std::cout << R"({"type":"error","status":"failed","code":"runtime_error","message":)"
                  << quote(e.what()) << "}\n";
        return 1;
    }
}
