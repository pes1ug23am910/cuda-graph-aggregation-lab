#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace graphlab {

enum class ErrorCode { invalid_input, unsupported_domain };
const char *error_code_name(ErrorCode code) noexcept;
class Error : public std::runtime_error {
  public:
    ErrorCode code;
    Error(ErrorCode code, std::string message);
};

struct Graph {
    std::int64_t nodes = 0;
    std::int64_t features = 1;
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> columns;
    std::vector<float> x;
};

void validate(const Graph &graph);
struct Oracle {
    std::vector<double> mean;
    // An outward-rounded upper bound on sum(abs(x))/degree, zero for empty rows.
    std::vector<double> abs_mean;
};
Oracle oracle(const Graph &graph);

struct CheckResult {
    bool passed = true;
    double max_abs_error = 0;
    double max_scaled_error = 0;
    std::size_t worst_index = 0;
    double bound = 0;
    std::string reason = "accepted";
};
CheckResult verify(const Graph &graph, const Oracle &reference, std::span<const float> output);

// The fingerprint hashes the graphlab-csr-mean-v1 domain separator before fixture bytes.
// Generator changes require updating the versioned contract and golden fingerprints.
Graph generate(std::int64_t nodes, std::int64_t features, std::string_view family,
               std::uint64_t seed);
// FNV-1a-64 over a canonical little-endian encoding. Identity aid, not a cryptographic hash.
std::string fingerprint(const Graph &graph);

struct GraphStats {
    std::size_t edges = 0;
    std::int32_t min_degree = 0;
    std::int32_t max_degree = 0;
    double mean_degree = 0;
    std::size_t isolated = 0;
};
GraphStats graph_stats(const Graph &graph);

} // namespace graphlab
