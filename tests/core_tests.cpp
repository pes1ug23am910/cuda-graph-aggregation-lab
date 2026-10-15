#include "graphlab/core.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace graphlab;
int assertions = 0;
void require(bool condition, const std::string &message) {
    ++assertions;
    if (!condition)
        throw std::runtime_error(message);
}
void throws_code(const std::function<void()> &call, ErrorCode code, const std::string &message) {
    try {
        call();
    } catch (const Error &error) {
        require(error.code == code, message + ": wrong error class");
        return;
    }
    throw std::runtime_error(message + ": did not reject");
}
Graph semantic_graph() {
    return {5,
            3,
            {0, 2, 3, 3, 5, 6},
            {1, 3, 0, 2, 3, 1},
            {1.0f, 0.5f, -1.0f, 0.0f, -0.5f, 0.5f, -1.0f, 1.0f, 0.0f, 0.5f, 0.0f, 1.0f, -0.5f,
             -1.0f, -0.5f}};
}
std::vector<float> serial(const Graph &graph, bool normalize = true, bool wrong_stride = false,
                          int skip_row = -1, bool stale_empty = false) {
    std::vector<float> output(graph.x.size(), 0.875f);
    for (std::int64_t v = 0; v < graph.nodes; ++v) {
        if (v == skip_row)
            continue;
        const auto degree = graph.offsets[static_cast<std::size_t>(v + 1)] -
                            graph.offsets[static_cast<std::size_t>(v)];
        if (degree == 0 && stale_empty)
            continue;
        for (std::int64_t f = 0; f < graph.features; ++f) {
            float sum = 0.0f;
            for (auto e = graph.offsets[static_cast<std::size_t>(v)];
                 e < graph.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
                const auto u = graph.columns[static_cast<std::size_t>(e)];
                const auto index = wrong_stride ? f * graph.nodes + u : u * graph.features + f;
                sum += graph.x[static_cast<std::size_t>(index)];
            }
            output[static_cast<std::size_t>(v * graph.features + f)] =
                degree && normalize ? sum / static_cast<float>(degree) : sum;
        }
    }
    return output;
}
std::vector<float> transposed(const Graph &graph) {
    std::vector<float> output(graph.x.size(), 0.0f);
    std::vector<int> degrees(static_cast<std::size_t>(graph.nodes), 0);
    for (std::int64_t v = 0; v < graph.nodes; ++v) {
        for (auto e = graph.offsets[static_cast<std::size_t>(v)];
             e < graph.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            const auto u = graph.columns[static_cast<std::size_t>(e)];
            ++degrees[static_cast<std::size_t>(u)];
            for (std::int64_t f = 0; f < graph.features; ++f)
                output[static_cast<std::size_t>(u * graph.features + f)] +=
                    graph.x[static_cast<std::size_t>(v * graph.features + f)];
        }
    }
    for (std::int64_t u = 0; u < graph.nodes; ++u)
        if (degrees[static_cast<std::size_t>(u)])
            for (std::int64_t f = 0; f < graph.features; ++f)
                output[static_cast<std::size_t>(u * graph.features + f)] /=
                    static_cast<float>(degrees[static_cast<std::size_t>(u)]);
    return output;
}
void semantic_tests() {
    const auto graph = semantic_graph();
    const auto ref = oracle(graph);
    const std::vector<double> expected{0.25, -0.25, 0.75, 1,   0.5, -1,   0,  0,
                                       0,    -0.25, 0.5,  0.5, 0,   -0.5, 0.5};
    require(ref.mean == expected, "handwritten directed, self-loop and isolated fixture");
    require(verify(graph, ref, serial(graph)).passed, "serial mean passes");
    require(!verify(graph, ref, transposed(graph)).passed, "negative control: transposed graph");
    require(!verify(graph, ref, serial(graph, false)).passed,
            "negative control: omitted normalization");
    require(!verify(graph, ref, serial(graph, true, true)).passed,
            "negative control: feature stride");
    require(!verify(graph, ref, serial(graph, true, false, 1)).passed,
            "negative control: skipped nonempty row");
    require(!verify(graph, ref, serial(graph, true, false, -1, true)).passed,
            "negative control: stale empty row");
    auto output = serial(graph);
    output[0] = std::numeric_limits<float>::quiet_NaN();
    require(!verify(graph, ref, output).passed, "NaN output rejected");
    output[0] = std::numeric_limits<float>::infinity();
    require(!verify(graph, ref, output).passed, "infinite output rejected");
    output = serial(graph);
    output[6] = -0.0f;
    require(!verify(graph, ref, output).passed, "empty row negative zero rejected");
    output.pop_back();
    throws_code([&] { verify(graph, ref, output); }, ErrorCode::invalid_input,
                "wrong output length");
    throws_code([&] { verify(graph, ref, graph.x); }, ErrorCode::invalid_input,
                "input/output aliasing");
    Oracle malformed;
    malformed.mean = ref.mean;
    throws_code([&] { verify(graph, malformed, serial(graph)); }, ErrorCode::invalid_input,
                "wrong oracle length");
    malformed = ref;
    malformed.abs_mean[0] = -1;
    throws_code([&] { verify(graph, malformed, serial(graph)); }, ErrorCode::invalid_input,
                "invalid oracle diagnostics");
    const auto stats = graph_stats(graph);
    require(stats.edges == 6 && stats.min_degree == 0 && stats.max_degree == 2 &&
                stats.isolated == 1 && stats.mean_degree == 1.2,
            "graph statistics");
    const Graph empty{0, 7, {0}, {}, {}};
    require(oracle(empty).mean.empty() && verify(empty, oracle(empty), {}).passed,
            "zero-node graph");
    const Graph edgeless{3, 2, {0, 0, 0, 0}, {}, {1, 0, 0, 1, -1, -1}};
    const auto empty_ref = oracle(edgeless);
    require(empty_ref.mean == std::vector<double>(6, 0.0), "edgeless graph oracle");
    require(verify(edgeless, empty_ref, serial(edgeless)).passed, "edgeless graph result");
    const Graph one{1, 1, {0, 1}, {0}, {-1.0f}};
    require(oracle(one).mean == std::vector<double>{-1.0}, "one self edge");
}
void rejection_tests() {
    const auto reject = [](const std::function<void(Graph &)> &mutate, ErrorCode code,
                           const std::string &label) {
        auto graph = semantic_graph();
        mutate(graph);
        throws_code([&] { validate(graph); }, code, label);
    };
    reject([](Graph &g) { g.nodes = -1; }, ErrorCode::invalid_input, "negative nodes");
    reject([](Graph &g) { g.features = 0; }, ErrorCode::invalid_input, "zero features");
    reject([](Graph &g) { g.features = -1; }, ErrorCode::invalid_input, "negative features");
    reject([](Graph &g) { g.nodes = std::int64_t{1} << 31; }, ErrorCode::unsupported_domain,
           "index width");
    reject(
        [](Graph &g) {
            g.nodes = INT32_MAX;
            g.features = INT32_MAX;
        },
        ErrorCode::invalid_input, "shape overflow");
    reject([](Graph &g) { g.offsets.pop_back(); }, ErrorCode::invalid_input, "offset length");
    reject([](Graph &g) { g.offsets[0] = 1; }, ErrorCode::invalid_input, "first offset");
    reject([](Graph &g) { g.offsets[2] = 1; }, ErrorCode::invalid_input, "decreasing offsets");
    reject([](Graph &g) { g.offsets[1] = -1; }, ErrorCode::invalid_input, "negative offsets");
    reject([](Graph &g) { g.offsets[1] = 7; }, ErrorCode::invalid_input,
           "offset beyond edge count");
    reject([](Graph &g) { g.offsets.back() = 5; }, ErrorCode::invalid_input, "final offset");
    reject([](Graph &g) { g.columns[0] = -1; }, ErrorCode::invalid_input, "negative source");
    reject([](Graph &g) { g.columns[0] = 5; }, ErrorCode::invalid_input,
           "source beyond node count");
    reject([](Graph &g) { g.columns[1] = 1; }, ErrorCode::invalid_input, "duplicate source");
    reject(
        [](Graph &g) {
            g.columns[0] = 3;
            g.columns[1] = 1;
        },
        ErrorCode::invalid_input, "unsorted sources");
    reject([](Graph &g) { g.x.pop_back(); }, ErrorCode::invalid_input, "feature length");
    reject([](Graph &g) { g.x[0] = std::numeric_limits<float>::quiet_NaN(); },
           ErrorCode::invalid_input, "NaN input");
    reject([](Graph &g) { g.x[0] = std::numeric_limits<float>::infinity(); },
           ErrorCode::invalid_input, "positive infinite input");
    reject([](Graph &g) { g.x[0] = -std::numeric_limits<float>::infinity(); },
           ErrorCode::invalid_input, "negative infinite input");
    reject([](Graph &g) { g.x[0] = std::nextafter(1.0f, 2.0f); }, ErrorCode::unsupported_domain,
           "positive magnitude domain");
    reject([](Graph &g) { g.x[0] = std::nextafter(-1.0f, -2.0f); }, ErrorCode::unsupported_domain,
           "negative magnitude domain");
    Graph high{4097, 1, std::vector<std::int32_t>(4098, 4097), std::vector<std::int32_t>(4097),
               std::vector<float>(4097, 1.0f)};
    high.offsets[0] = 0;
    std::iota(high.columns.begin(), high.columns.end(), 0);
    throws_code([&] { validate(high); }, ErrorCode::unsupported_domain, "degree domain");
    throws_code([] { generate(2, 1, "unknown", 0); }, ErrorCode::invalid_input, "generator family");
}
void numerical_tests() {
    const float tiny = std::bit_cast<float>(std::uint32_t{1});
    Graph graph{4, 1, {0, 1, 3, 4, 4}, {0, 0, 1, 2}, {tiny, 2 * tiny, -0.0f, 1.0f}};
    auto ref = oracle(graph);
    const std::vector<float> expected{tiny, 2 * tiny, -0.0f, 0.0f};
    require(ref.mean[0] == 0x1p-149 && ref.mean[1] == 0x1.8p-149,
            "FP64 oracle preserves subnormal inputs");
    require(verify(graph, ref, expected).passed, "subnormal ties-to-even output accepted");
    auto flushed = expected;
    flushed[0] = 0;
    require(!verify(graph, ref, flushed).passed, "negative control: flushed subnormal input");
    const Graph underflow{3, 1, {0, 3, 3, 3}, {0, 1, 2}, {tiny, 0, 0}};
    require(oracle(underflow).mean[0] > 0.0, "FP64 tiny mean remains nonzero");
    require(verify(underflow, oracle(underflow), serial(underflow)).passed,
            "correct rounding to zero accepted");
    const Graph cancellation{4, 1, {0, 4, 4, 4, 4}, {0, 1, 2, 3}, {1, 0x1p-24f, -1, 0x1p-24f}};
    require(oracle(cancellation).mean[0] == 0x1p-25, "cancellation oracle");
    require(verify(cancellation, oracle(cancellation), serial(cancellation)).passed,
            "FP32 cancellation within forward budget");
    const Graph zeros{2, 1, {0, 2, 2}, {0, 1}, {-0.0f, 0.0f}};
    require(verify(zeros, oracle(zeros), std::vector<float>{-0.0f, 0.0f}).passed,
            "nonempty signed zero equivalent");
    require(!verify(zeros, oracle(zeros), std::vector<float>{tiny, 0.0f}).passed,
            "all-zero exactness independent of tolerance");
    Graph max_degree{4096, 1, std::vector<std::int32_t>(4097, 4096),
                     std::vector<std::int32_t>(4096), std::vector<float>(4096, 1)};
    max_degree.offsets[0] = 0;
    std::iota(max_degree.columns.begin(), max_degree.columns.end(), 0);
    require(verify(max_degree, oracle(max_degree), serial(max_degree)).passed,
            "maximum supported degree");
    require(oracle(max_degree).mean[0] == 1, "maximum degree normalization");
    if (std::fesetround(FE_DOWNWARD) == 0) {
        bool rejected = false;
        try {
            oracle(graph);
        } catch (const Error &error) {
            rejected = error.code == ErrorCode::unsupported_domain;
        }
        std::fesetround(FE_TONEAREST);
        require(rejected, "non-nearest host arithmetic rejected");
    }
}
void generator_tests() {
    const std::array<int, 8> nodes{0, 1, 31, 32, 33, 127, 128, 129};
    const std::array<int, 12> widths{1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 3};
    for (int n : nodes)
        for (int f : widths) {
            auto graph = generate(n, f, "regular8", 123456);
            require(verify(graph, oracle(graph), serial(graph)).passed,
                    "deterministic boundary fixture");
            require(fingerprint(graph) == fingerprint(generate(n, f, "regular8", 123456)),
                    "repeatable generated bytes");
        }
    auto first = generate(65, 7, "skewed", 43);
    auto second = generate(65, 7, "skewed", 44);
    require(fingerprint(first) != fingerprint(second), "seed changes fixture");
    for (std::size_t v = 0; v < 65; ++v) {
        const auto degree = first.offsets[v + 1] - first.offsets[v];
        require(v % 64 == 0 ? degree == 65 : degree >= 1 && degree <= 8,
                "frozen skewed proportions");
    }
    const auto regular = generate(40, 1, "regular32", 7);
    require(graph_stats(regular).min_degree == 32 && graph_stats(regular).max_degree == 32,
            "regular32 degree");
    auto fixture = generate(4, 3, "regular8", 42);
    const std::array<std::uint32_t, 12> golden{0xbf4af852u, 0xbc138000u, 0xbf502a42u, 0x3ec17b28u,
                                               0x3f6a2688u, 0xbf5a98b4u, 0x3e4c6c78u, 0x3e7563a8u,
                                               0xbf5a079au, 0xbee3c564u, 0x3ef7c96cu, 0x3f122cfcu};
    for (std::size_t i = 0; i < golden.size(); ++i)
        require(std::bit_cast<std::uint32_t>(fixture.x[i]) == golden[i],
                "generator portable golden bytes");
    const auto original = fingerprint(fixture);
    fixture.x[0] = std::nextafter(fixture.x[0], 0.0f);
    require(original != fingerprint(fixture), "fingerprint covers exact feature bytes");
    fixture = semantic_graph();
    const auto graph_hash = fingerprint(fixture);
    fixture.columns[5] = 2;
    require(graph_hash != fingerprint(fixture), "fingerprint covers directed topology");
}
} // namespace
int main() {
    try {
        semantic_tests();
        rejection_tests();
        numerical_tests();
        generator_tests();
        std::cout
            << "PASS: " << assertions
            << " CPU assertions, including five semantic negative controls and subnormal control\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL after " << assertions << " assertions: " << error.what() << '\n';
        return 1;
    }
}
