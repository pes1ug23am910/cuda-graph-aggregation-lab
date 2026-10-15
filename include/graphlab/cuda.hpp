#pragma once

#include "graphlab/core.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace graphlab::cuda {

enum class Variant { naive, cusparse, row_split };
const char *variant_name(Variant variant) noexcept;

class Error : public std::runtime_error {
  public:
    Error(std::string code, std::string message, int native_code = 0);
    const std::string &code() const noexcept { return code_; }
    int native_code() const noexcept { return native_code_; }

  private:
    std::string code_;
    int native_code_;
};

struct RuntimeInfo {
    bool available = false;
    std::string error;
    std::string name;
    int device = 0;
    int driver_version = 0;
    int runtime_version = 0;
    int cusparse_version = 0;
    int compute_major = 0;
    int compute_minor = 0;
    int multiprocessors = 0;
    std::uint64_t free_bytes = 0;
    std::uint64_t total_bytes = 0;
};
RuntimeInfo runtime_info();

struct PreparedCall {
    std::vector<float> output;
    double h2d_ms = 0;
    double aggregation_ms = 0;
    double d2h_ms = 0;
    double wall_ms = 0;
};

// One explicit stream, immutable canonical CSR, and resident FP32 features.
// Construction validates input before initializing or allocating device resources.
// Each session owns its host input copy; callers may release the original Graph.
class Session {
  public:
    Session(const Graph &graph, Variant variant);
    ~Session();
    Session(Session &&) noexcept;
    Session &operator=(Session &&) noexcept;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    double preparation_ms() const noexcept;
    std::uint64_t memory_bytes() const noexcept;
    std::uint64_t workspace_bytes() const noexcept;
    std::uint64_t memory_budget_bytes() const noexcept;
    // Conservative allowance for unqueryable stream/event/library bookkeeping.
    std::uint64_t overhead_reserve_bytes() const noexcept;
    const char *path() const noexcept;

    // Includes normalized aggregation and download; does not re-upload features.
    std::vector<float> output();
    // Download the last computed result without submitting another aggregation.
    // Requires at least one successful output(), warmup(n>0), resident_batch(),
    // or prepared_call() on this session.
    std::vector<float> download_output();
    void warmup(std::size_t calls);
    // Returns total CUDA-event elapsed milliseconds for all calls in this batch.
    // Benchmark policy (warmup/repetition counts) is enforced by the caller.
    double resident_batch(std::size_t calls);
    // Re-uploads the same features, computes, downloads; wall time includes all.
    PreparedCall prepared_call();
    void synchronize();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace graphlab::cuda
