#include "graphlab/core.hpp"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace graphlab {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
constexpr std::int64_t index_max = std::numeric_limits<std::int32_t>::max();

[[noreturn]] void invalid(const std::string &reason) {
    throw Error(ErrorCode::invalid_input, reason);
}
[[noreturn]] void unsupported(const std::string &reason) {
    throw Error(ErrorCode::unsupported_domain, reason);
}

std::size_t checked_shape(std::int64_t nodes, std::int64_t features) {
    if (nodes < 0 || features < 1)
        invalid("nodes must be nonnegative and features positive");
    if (nodes > index_max || features > index_max)
        unsupported("dimensions exceed signed 32-bit representation");
    const auto n = static_cast<std::uint64_t>(nodes);
    const auto f = static_cast<std::uint64_t>(features);
    // The FP64 oracle is the widest per-element allocation in the host contract.
    const auto max_bytes = static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (n > max_bytes / sizeof(double) / f)
        invalid("feature shape or byte size overflows addressable storage");
    if (n + 1 > max_bytes / sizeof(std::int32_t))
        invalid("offset byte size overflows addressable storage");
    return static_cast<std::size_t>(n * f);
}

double upward(double value) {
    return value == 0.0 ? 0.0 : std::nextafter(value, std::numeric_limits<double>::infinity());
}

double gamma_upper(std::int64_t operations, double unit_roundoff) {
    // These products and differences are exact for the supported degrees.
    const double product = static_cast<double>(operations) * unit_roundoff;
    if (!(product < 1.0))
        unsupported("floating-point error bound is undefined");
    return upward(product / (1.0 - product));
}

void require_arithmetic() {
    if (std::fegetround() != FE_TONEAREST)
        unsupported("host arithmetic must round to nearest");
    volatile float tiny = std::bit_cast<float>(std::uint32_t{1});
    volatile double converted = static_cast<double>(tiny);
    volatile float twice_tiny = std::bit_cast<float>(std::uint32_t{2});
    volatile float two = 2.0f;
    volatile float divided = twice_tiny / two;
    const float observed = divided;
    if (converted != 0x1p-149 || std::bit_cast<std::uint32_t>(observed) != 1u)
        unsupported("host arithmetic flushes subnormal inputs or results");
}

class SplitMix64 {
  public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    std::uint64_t bounded(std::uint64_t limit) {
        const std::uint64_t threshold = (std::uint64_t{0} - limit) % limit;
        for (;;) {
            const auto value = next();
            if (value >= threshold)
                return value % limit;
        }
    }

  private:
    std::uint64_t state_;
};
} // namespace

Error::Error(ErrorCode value, std::string message)
    : std::runtime_error(std::move(message)), code(value) {}
const char *error_code_name(ErrorCode code) noexcept {
    return code == ErrorCode::invalid_input ? "invalid_input" : "unsupported_domain";
}

void validate(const Graph &graph) {
    const auto elements = checked_shape(graph.nodes, graph.features);
    if (graph.offsets.size() != static_cast<std::size_t>(graph.nodes) + 1)
        invalid("wrong offset length");
    if (graph.offsets.front() != 0)
        invalid("first offset must be zero");
    if (graph.columns.size() > static_cast<std::size_t>(index_max))
        unsupported("edge count exceeds signed 32-bit representation");
    if (graph.x.size() != elements)
        invalid("wrong feature length");
    const auto edges = static_cast<std::int64_t>(graph.columns.size());
    if (graph.offsets.back() != edges)
        invalid("final offset differs from edge count");
    for (std::size_t v = 0; v < static_cast<std::size_t>(graph.nodes); ++v) {
        const std::int64_t begin = graph.offsets[v];
        const std::int64_t end = graph.offsets[v + 1];
        if (begin < 0 || end < begin || end > edges)
            invalid("offsets are negative, decreasing or out of range");
        if (end - begin > 4096)
            unsupported("degree exceeds 4096");
        std::int32_t previous = -1;
        for (auto e = begin; e < end; ++e) {
            const auto source = graph.columns[static_cast<std::size_t>(e)];
            if (source < 0 || source >= graph.nodes)
                invalid("source index out of range");
            if (source <= previous)
                invalid("row sources must be sorted and unique");
            previous = source;
        }
    }
    for (float value : graph.x) {
        if (!std::isfinite(value))
            invalid("non-finite feature value");
        if (value < -1.0f || value > 1.0f)
            unsupported("feature value outside [-1,1]");
    }
}

Oracle oracle(const Graph &graph) {
    validate(graph);
    require_arithmetic();
    Oracle result;
    result.mean.resize(graph.x.size(), 0.0);
    result.abs_mean.resize(graph.x.size(), 0.0);
    const auto width = static_cast<std::size_t>(graph.features);
    for (std::size_t v = 0; v < static_cast<std::size_t>(graph.nodes); ++v) {
        const auto begin = graph.offsets[v];
        const auto end = graph.offsets[v + 1];
        const auto degree = end - begin;
        if (degree == 0)
            continue;
        for (std::size_t f = 0; f < width; ++f) {
            double sum = 0.0;
            double absolute_sum_upper = 0.0;
            for (auto e = begin; e < end; ++e) {
                const auto source =
                    static_cast<std::size_t>(graph.columns[static_cast<std::size_t>(e)]);
                const double value = static_cast<double>(graph.x[source * width + f]);
                sum += value;
                absolute_sum_upper = upward(absolute_sum_upper + std::abs(value));
            }
            result.mean[v * width + f] = sum / static_cast<double>(degree);
            result.abs_mean[v * width + f] =
                upward(absolute_sum_upper / static_cast<double>(degree));
        }
    }
    return result;
}

CheckResult verify(const Graph &graph, const Oracle &reference, std::span<const float> output) {
    validate(graph);
    require_arithmetic();
    if (reference.mean.size() != graph.x.size() || reference.abs_mean.size() != graph.x.size())
        invalid("oracle shape mismatch");
    if (output.size() != graph.x.size())
        invalid("output shape mismatch");
    if (!output.empty()) {
        const std::less<const float *> less;
        if (less(output.data(), graph.x.data() + graph.x.size()) &&
            less(graph.x.data(), output.data() + output.size()))
            invalid("input/output aliasing is unsupported");
    }
    CheckResult result;
    const auto width = static_cast<std::size_t>(graph.features);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto row = i / width;
        const std::int64_t degree = graph.offsets[row + 1] - graph.offsets[row];
        const double reference_value = reference.mean[i];
        const double scale = reference.abs_mean[i];
        if (!std::isfinite(reference_value) || !std::isfinite(scale) || scale < 0.0)
            invalid("oracle contains invalid diagnostics");
        const double actual = static_cast<double>(output[i]);
        if (!std::isfinite(actual)) {
            result.passed = false;
            result.max_abs_error = std::numeric_limits<double>::infinity();
            result.max_scaled_error = std::numeric_limits<double>::infinity();
            result.worst_index = i;
            result.bound = 0.0;
            result.reason = "non-finite output";
            return result;
        }
        double budget = 0.0;
        bool accepted = false;
        std::string reason = "forward-error budget exceeded";
        if (degree == 0) {
            accepted = output[i] == 0.0f && !std::signbit(output[i]);
            reason = "empty row requires positive zero";
        } else if (scale == 0.0) {
            accepted = output[i] == 0.0f;
            reason = "all-zero inputs require zero output";
        } else {
            // An intentionally conservative operation count for sum-then-divide.
            // Additions of FP32 operands cannot round in the subnormal region:
            // their exact sums are integer multiples of 2^-149. Only division
            // requires the half-minimum-subnormal (2^-150) absolute allowance.
            // This bound does not authorize pre-normalized adjacency or FTZ.
            const double gpu_term = upward(gamma_upper(2 * degree + 2, 0x1p-24) * scale);
            const double oracle_term = upward(gamma_upper(degree + 2, 0x1p-53) * scale);
            budget = upward(upward(gpu_term + oracle_term) + 0x1p-150);
            accepted = upward(std::abs(actual - reference_value)) <= budget;
        }
        const double error = std::abs(actual - reference_value);
        const double scaled = budget > 0.0
                                  ? error / budget
                                  : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        result.max_abs_error = std::max(result.max_abs_error, error);
        if (i == 0 || scaled > result.max_scaled_error || (!accepted && result.passed)) {
            result.max_scaled_error = std::max(result.max_scaled_error, scaled);
            result.worst_index = i;
            result.bound = budget;
        }
        if (!accepted) {
            result.passed = false;
            result.reason = std::move(reason);
        }
    }
    return result;
}

Graph generate(std::int64_t nodes, std::int64_t features, std::string_view family,
               std::uint64_t seed) {
    const auto elements = checked_shape(nodes, features);
    if (family != "regular8" && family != "regular32" && family != "skewed")
        invalid("unknown graph family");
    Graph graph;
    graph.nodes = nodes;
    graph.features = features;
    graph.offsets.reserve(static_cast<std::size_t>(nodes) + 1);
    graph.offsets.push_back(0);
    SplitMix64 random(seed);
    for (std::int64_t v = 0; v < nodes; ++v) {
        std::int64_t degree;
        if (family == "regular8")
            degree = std::min<std::int64_t>(8, nodes);
        else if (family == "regular32")
            degree = std::min<std::int64_t>(32, nodes);
        else if (v % 64 == 0)
            degree = std::min<std::int64_t>(2048, nodes);
        else
            degree =
                std::min<std::int64_t>(1 + static_cast<std::int64_t>(random.bounded(8)), nodes);
        if (graph.columns.size() > static_cast<std::size_t>(index_max - degree))
            unsupported("generated edge count exceeds signed 32-bit representation");
        // Floyd sampling without replacement; std::set fixes canonical source order.
        std::set<std::int32_t> selected;
        for (std::int64_t j = nodes - degree; j < nodes; ++j) {
            const auto candidate =
                static_cast<std::int32_t>(random.bounded(static_cast<std::uint64_t>(j + 1)));
            if (!selected.insert(candidate).second)
                selected.insert(static_cast<std::int32_t>(j));
        }
        graph.columns.insert(graph.columns.end(), selected.begin(), selected.end());
        graph.offsets.push_back(static_cast<std::int32_t>(graph.columns.size()));
    }
    graph.x.reserve(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        const auto integer = static_cast<std::int32_t>(random.next() >> 40) - 8388608;
        graph.x.push_back(static_cast<float>(integer) * 0x1p-23f);
    }
    validate(graph);
    return graph;
}

std::string fingerprint(const Graph &graph) {
    validate(graph);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto byte = [&hash](std::uint8_t value) {
        hash = (hash ^ value) * UINT64_C(1099511628211);
    };
    const auto word = [&byte](std::uint64_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) {
            byte(static_cast<std::uint8_t>(value & 255u));
            value >>= 8;
        }
    };
    for (char value : std::string_view("graphlab-csr-mean-v1"))
        byte(static_cast<std::uint8_t>(value));
    word(static_cast<std::uint64_t>(graph.nodes), 8);
    word(static_cast<std::uint64_t>(graph.features), 8);
    word(graph.offsets.size(), 8);
    for (auto value : graph.offsets)
        word(static_cast<std::uint32_t>(value), 4);
    word(graph.columns.size(), 8);
    for (auto value : graph.columns)
        word(static_cast<std::uint32_t>(value), 4);
    word(graph.x.size(), 8);
    for (float value : graph.x)
        word(std::bit_cast<std::uint32_t>(value), 4);
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

GraphStats graph_stats(const Graph &graph) {
    validate(graph);
    GraphStats result;
    result.edges = graph.columns.size();
    if (graph.nodes == 0)
        return result;
    result.min_degree = std::numeric_limits<std::int32_t>::max();
    for (std::size_t row = 0; row < static_cast<std::size_t>(graph.nodes); ++row) {
        const auto degree = graph.offsets[row + 1] - graph.offsets[row];
        result.min_degree = std::min(result.min_degree, degree);
        result.max_degree = std::max(result.max_degree, degree);
        result.isolated += degree == 0 ? 1u : 0u;
    }
    result.mean_degree = static_cast<double>(result.edges) / static_cast<double>(graph.nodes);
    return result;
}
} // namespace graphlab
