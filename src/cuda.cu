#include "graphlab/cuda.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace graphlab::cuda {
namespace {
using Clock = std::chrono::steady_clock;
constexpr unsigned threads_per_block = 256;
constexpr std::uint64_t bookkeeping_reserve = 1ULL << 20;
constexpr std::uint64_t maximum_budget = 1ULL << 30;

[[noreturn]] void fail_cuda(cudaError_t status, const char *operation,
                            const char *category = "device_failure") {
    if (status == cudaErrorMemoryAllocation)
        category = "allocation_failure";
    throw Error(category, std::string(operation) + ": " + cudaGetErrorString(status),
                static_cast<int>(status));
}
void check(cudaError_t status, const char *operation, const char *category = "device_failure") {
    if (status != cudaSuccess)
        fail_cuda(status, operation, category);
}
void check_sparse(cusparseStatus_t status, const char *operation) {
    if (status == CUSPARSE_STATUS_SUCCESS)
        return;
    const char *category = status == CUSPARSE_STATUS_ALLOC_FAILED    ? "allocation_failure"
                           : status == CUSPARSE_STATUS_NOT_SUPPORTED ? "unsupported"
                                                                     : "device_failure";
    std::ostringstream out;
    out << operation << ": cuSPARSE status " << static_cast<int>(status);
    throw Error(category, out.str(), static_cast<int>(status));
}
std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw Error("invalid_input", "Device memory estimate overflows uint64");
    return a + b;
}
std::uint64_t byte_count(std::size_t elements, std::size_t element_size) {
    if (elements > std::numeric_limits<std::size_t>::max() / element_size)
        throw Error("invalid_input", "Device buffer byte size overflows size_t");
    return static_cast<std::uint64_t>(elements * element_size);
}
double elapsed_wall(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() = default;
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
    Stream(Stream &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] Stream &operator=(Stream &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void create() {
        check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream");
    }
    void reset() noexcept {
        if (value) {
            cudaStreamDestroy(value);
            value = nullptr;
        }
    }
    ~Stream() { reset(); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() = default;
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    Event(Event &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] Event &operator=(Event &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void create() { check(cudaEventCreate(&value), "create timing event"); }
    void reset() noexcept {
        if (value) {
            cudaEventDestroy(value);
            value = nullptr;
        }
    }
    ~Event() { reset(); }
};
struct Buffer {
    void *value = nullptr;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void allocate(std::uint64_t bytes) {
        if (bytes > std::numeric_limits<std::size_t>::max())
            throw Error("invalid_input", "Allocation size exceeds size_t");
        if (bytes)
            check(cudaMalloc(&value, static_cast<std::size_t>(bytes)), "allocate device buffer",
                  "allocation_failure");
    }
    template <typename T> T *as() const { return static_cast<T *>(value); }
    void reset() noexcept {
        if (value) {
            cudaFree(value);
            value = nullptr;
        }
    }
    ~Buffer() { reset(); }
};
struct SparseHandle {
    cusparseHandle_t value = nullptr;
    SparseHandle() = default;
    SparseHandle(const SparseHandle &) = delete;
    SparseHandle &operator=(const SparseHandle &) = delete;
    SparseHandle(SparseHandle &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] SparseHandle &operator=(SparseHandle &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void reset() noexcept {
        if (value) {
            cusparseDestroy(value);
            value = nullptr;
        }
    }
    ~SparseHandle() { reset(); }
};
struct SparseMatrix {
    cusparseSpMatDescr_t value = nullptr;
    SparseMatrix() = default;
    SparseMatrix(const SparseMatrix &) = delete;
    SparseMatrix &operator=(const SparseMatrix &) = delete;
    SparseMatrix(SparseMatrix &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] SparseMatrix &operator=(SparseMatrix &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void reset() noexcept {
        if (value) {
            cusparseDestroySpMat(value);
            value = nullptr;
        }
    }
    ~SparseMatrix() { reset(); }
};
struct DenseMatrix {
    cusparseDnMatDescr_t value = nullptr;
    DenseMatrix() = default;
    DenseMatrix(const DenseMatrix &) = delete;
    DenseMatrix &operator=(const DenseMatrix &) = delete;
    DenseMatrix(DenseMatrix &&other) noexcept : value(std::exchange(other.value, nullptr)) {}
    [[maybe_unused]] DenseMatrix &operator=(DenseMatrix &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, nullptr);
        }
        return *this;
    }
    void reset() noexcept {
        if (value) {
            cusparseDestroyDnMat(value);
            value = nullptr;
        }
    }
    ~DenseMatrix() { reset(); }
};

__global__ void set_ones(float *values, std::uint64_t count) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
        values[i] = 1.0f;
}
__global__ void mean_naive(const std::int32_t *offsets, const std::int32_t *columns, const float *x,
                           float *y, std::uint64_t count, std::int64_t features) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const auto row = i / static_cast<std::uint64_t>(features);
        const auto feature = i % static_cast<std::uint64_t>(features);
        const auto begin = offsets[row];
        const auto end = offsets[row + 1];
        float sum = 0.0f;
        for (auto edge = begin; edge < end; ++edge)
            sum = __fadd_rn(sum, x[static_cast<std::uint64_t>(columns[edge]) * features + feature]);
        y[i] = begin == end ? 0.0f : __fdiv_rn(sum, static_cast<float>(end - begin));
    }
}
// Each block owns a row. Eight warps split its edge chain; lanes retain
// contiguous feature loads. All lanes reach both barriers, including F tails.
__global__ void mean_row_split(const std::int32_t *offsets, const std::int32_t *columns,
                               const float *x, float *y, std::int64_t nodes,
                               std::int64_t features) {
    __shared__ float partial[8][32];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x >> 5U;
    for (std::int64_t row = blockIdx.x; row < nodes; row += gridDim.x) {
        const auto begin = offsets[row];
        const auto end = offsets[row + 1];
        for (std::int64_t base = 0; base < features; base += 32) {
            const auto feature = base + lane;
            float sum = 0.0f;
            if (feature < features) {
                for (std::int64_t edge = static_cast<std::int64_t>(begin) + warp; edge < end;
                     edge += 8)
                    sum = __fadd_rn(
                        sum, x[static_cast<std::uint64_t>(columns[edge]) * features + feature]);
            }
            partial[warp][lane] = sum;
            __syncthreads();
            if (warp == 0 && feature < features) {
                sum = partial[0][lane];
#pragma unroll
                for (unsigned part = 1; part < 8; ++part)
                    sum = __fadd_rn(sum, partial[part][lane]);
                y[static_cast<std::uint64_t>(row) * features + feature] =
                    begin == end ? 0.0f : __fdiv_rn(sum, static_cast<float>(end - begin));
            }
            __syncthreads();
        }
    }
}
__global__ void scale_features(const float *x, float *scaled, std::uint64_t count) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
        scaled[i] = __fmul_rn(x[i], 0x1p100f);
}
template <bool Scaled>
__global__ void normalize(const std::int32_t *offsets, float *y, std::uint64_t count,
                          std::int64_t features) {
    for (std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const auto row = i / static_cast<std::uint64_t>(features);
        const auto degree = offsets[row + 1] - offsets[row];
        float mean = degree == 0 ? 0.0f : __fdiv_rn(y[i], static_cast<float>(degree));
        if constexpr (Scaled)
            mean = __fmul_rn(mean, 0x1p-100f);
        y[i] = mean;
    }
}
} // namespace

Error::Error(std::string code, std::string message, int native_code)
    : std::runtime_error(std::move(message)), code_(std::move(code)), native_code_(native_code) {}

const char *variant_name(Variant variant) noexcept {
    switch (variant) {
    case Variant::naive:
        return "naive";
    case Variant::cusparse:
        return "cusparse";
    case Variant::row_split:
        return "row_split";
    }
    return "unknown";
}

RuntimeInfo runtime_info() {
    RuntimeInfo info;
    try {
        int count = 0;
        check(cudaGetDeviceCount(&count), "query CUDA device count");
        if (!count) {
            info.error = "No CUDA device is available";
            return info;
        }
        check(cudaGetDevice(&info.device), "query current device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, info.device), "query device properties");
        info.name = properties.name;
        info.compute_major = properties.major;
        info.compute_minor = properties.minor;
        info.multiprocessors = properties.multiProcessorCount;
        check(cudaDriverGetVersion(&info.driver_version), "query driver version");
        check(cudaRuntimeGetVersion(&info.runtime_version), "query runtime version");
        std::size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "query device memory");
        info.free_bytes = free_bytes;
        info.total_bytes = total_bytes;
        SparseHandle handle;
        check_sparse(cusparseCreate(&handle.value), "create metadata handle");
        check_sparse(cusparseGetVersion(handle.value, &info.cusparse_version),
                     "query cuSPARSE version");
        info.available = true;
    } catch (const Error &error) {
        info.error = error.what();
    }
    return info;
}

struct Session::Impl {
    Graph graph;
    Variant variant;
    Stream stream;
    Event events[4];
    Buffer offsets, columns, x, scaled_x, y, values, workspace;
    SparseHandle handle;
    SparseMatrix adjacency;
    DenseMatrix input, result;
    std::uint64_t elements = 0;
    std::uint64_t feature_bytes = 0;
    std::uint64_t allocation_bytes = 0;
    std::uint64_t workspace_size = 0;
    std::uint64_t budget = 0;
    std::uint64_t reserve = 0;
    unsigned maximum_blocks = 0;
    double prep_ms = 0;
    bool failed = false;
    bool result_submitted = false;
    bool scale_cusparse_inputs = false;

    Impl(const Graph &source, Variant selected) : variant(selected) {
        const auto start = Clock::now();
        if (variant != Variant::naive && variant != Variant::cusparse &&
            variant != Variant::row_split)
            throw Error("invalid_input", "Unknown CUDA variant");
        graph = source;
        // With |x| >= 2^-90, FP32 inputs/sums lie on a 2^-113 lattice;
        // division by at most 4096 stays normal. Smaller inputs need exact
        // power-of-two conditioning to avoid library-internal FTZ behavior.
        scale_cusparse_inputs = variant == Variant::cusparse && !graph.columns.empty() &&
                                std::any_of(graph.x.begin(), graph.x.end(), [](float value) {
                                    return value != 0.0f && std::fabs(value) < 0x1p-90f;
                                });
        elements = graph.x.size();
        feature_bytes = byte_count(graph.x.size(), sizeof(float));
        if (!graph.nodes) {
            prep_ms = elapsed_wall(start);
            return;
        }
        try {
            prepare();
            prep_ms = elapsed_wall(start);
        } catch (...) {
            // Construction can fail with work still queued. Complete it before
            // member destructors release buffers referenced by that work.
            if (stream.value)
                cudaStreamSynchronize(stream.value);
            throw;
        }
    }
    ~Impl() {
        // Public operations observe completion/errors. This final barrier also
        // protects lifetime when a preceding operation threw during submission.
        if (stream.value)
            cudaStreamSynchronize(stream.value);
    }
    unsigned blocks(std::uint64_t count) const {
        const auto required = (count + threads_per_block - 1) / threads_per_block;
        return static_cast<unsigned>(std::min<std::uint64_t>(required, maximum_blocks));
    }
    void admit(std::uint64_t bytes) const {
        if (checked_add(bytes, reserve) > budget) {
            std::ostringstream out;
            out << "Device memory budget exceeded: " << bytes << " explicit bytes + " << reserve
                << " bookkeeping reserve > " << budget << " bytes";
            throw Error("unsupported", out.str());
        }
    }
    void prepare() {
        // Admission is per session, immediately before its device allocations.
        std::size_t free_bytes = 0, total_bytes = 0;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "query admission memory");
        budget = std::min<std::uint64_t>(maximum_budget, free_bytes / 4);
        reserve = bookkeeping_reserve;
        const auto offsets_bytes = byte_count(graph.offsets.size(), sizeof(std::int32_t));
        const auto columns_bytes = byte_count(graph.columns.size(), sizeof(std::int32_t));
        const auto values_bytes =
            variant == Variant::cusparse ? byte_count(graph.columns.size(), sizeof(float)) : 0;
        allocation_bytes =
            checked_add(checked_add(offsets_bytes, columns_bytes),
                        checked_add(checked_add(feature_bytes, feature_bytes), values_bytes));
        if (scale_cusparse_inputs)
            allocation_bytes = checked_add(allocation_bytes, feature_bytes);
        admit(allocation_bytes);
        int device = 0;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&device), "query current device");
        check(cudaGetDeviceProperties(&properties, device), "query launch limits");
        if (properties.maxThreadsPerBlock < static_cast<int>(threads_per_block) ||
            properties.maxGridSize[0] < 1)
            throw Error("unsupported", "Device does not support the required launch geometry");
        maximum_blocks = static_cast<unsigned>(std::min(properties.maxGridSize[0], 65535));
        stream.create();
        for (auto &event : events)
            event.create();
        offsets.allocate(offsets_bytes);
        columns.allocate(columns_bytes);
        x.allocate(feature_bytes);
        if (scale_cusparse_inputs)
            scaled_x.allocate(feature_bytes);
        y.allocate(feature_bytes);
        values.allocate(values_bytes);
        check(cudaMemcpyAsync(offsets.value, graph.offsets.data(),
                              static_cast<std::size_t>(offsets_bytes), cudaMemcpyHostToDevice,
                              stream.value),
              "upload CSR offsets");
        if (columns_bytes)
            check(cudaMemcpyAsync(columns.value, graph.columns.data(),
                                  static_cast<std::size_t>(columns_bytes), cudaMemcpyHostToDevice,
                                  stream.value),
                  "upload CSR columns");
        check(cudaMemcpyAsync(x.value, graph.x.data(), static_cast<std::size_t>(feature_bytes),
                              cudaMemcpyHostToDevice, stream.value),
              "upload features");
        if (variant == Variant::cusparse && !graph.columns.empty()) {
            set_ones<<<blocks(graph.columns.size()), threads_per_block, 0, stream.value>>>(
                values.as<float>(), graph.columns.size());
            check(cudaGetLastError(), "submit edge-value initialization", "launch_failure");
            check_sparse(cusparseCreate(&handle.value), "create cuSPARSE handle");
            check_sparse(cusparseSetStream(handle.value, stream.value), "set cuSPARSE stream");
            check_sparse(cusparseSetPointerMode(handle.value, CUSPARSE_POINTER_MODE_HOST),
                         "set cuSPARSE pointer mode");
            check_sparse(cusparseCreateCsr(&adjacency.value, graph.nodes, graph.nodes,
                                           static_cast<std::int64_t>(graph.columns.size()),
                                           offsets.value, columns.value, values.value,
                                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                         "create CSR descriptor");
            check_sparse(cusparseCreateDnMat(&input.value, graph.nodes, graph.features,
                                             graph.features,
                                             scale_cusparse_inputs ? scaled_x.value : x.value,
                                             CUDA_R_32F, CUSPARSE_ORDER_ROW),
                         "create input descriptor");
            check_sparse(cusparseCreateDnMat(&result.value, graph.nodes, graph.features,
                                             graph.features, y.value, CUDA_R_32F,
                                             CUSPARSE_ORDER_ROW),
                         "create output descriptor");
            const float alpha = 1.0f, beta = 0.0f;
            std::size_t queried_workspace = 0;
            check_sparse(cusparseSpMM_bufferSize(handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                                 adjacency.value, input.value, &beta, result.value,
                                                 CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2,
                                                 &queried_workspace),
                         "query SpMM workspace");
            workspace_size = queried_workspace;
            allocation_bytes = checked_add(allocation_bytes, workspace_size);
            admit(allocation_bytes);
            workspace.allocate(workspace_size);
        }
        check(cudaStreamSynchronize(stream.value), "complete graph preparation");
    }
    void require_healthy() const {
        if (failed)
            throw Error("device_failure",
                        "Session cannot be reused after a device submission/completion failure");
    }
    void checked(cudaError_t status, const char *operation,
                 const char *category = "device_failure") {
        if (status != cudaSuccess) {
            failed = true;
            fail_cuda(status, operation, category);
        }
    }
    void sparse_checked(cusparseStatus_t status, const char *operation) {
        if (status != CUSPARSE_STATUS_SUCCESS) {
            failed = true;
            check_sparse(status, operation);
        }
    }
    void record(unsigned index) {
        checked(cudaEventRecord(events[index].value, stream.value), "record timing event");
    }
    void wait_event(unsigned index) {
        checked(cudaEventSynchronize(events[index].value), "complete timed device work");
    }
    double elapsed(unsigned begin, unsigned end) {
        float ms = 0;
        checked(cudaEventElapsedTime(&ms, events[begin].value, events[end].value),
                "read event duration");
        return ms;
    }
    void run() {
        require_healthy();
        if (!graph.nodes) {
            result_submitted = true;
            return;
        }
        if (graph.columns.empty()) {
            checked(
                cudaMemsetAsync(y.value, 0, static_cast<std::size_t>(feature_bytes), stream.value),
                "zero empty-edge output", "launch_failure");
            result_submitted = true;
            return;
        }
        if (variant == Variant::naive) {
            mean_naive<<<blocks(elements), threads_per_block, 0, stream.value>>>(
                offsets.as<std::int32_t>(), columns.as<std::int32_t>(), x.as<float>(),
                y.as<float>(), elements, graph.features);
            checked(cudaGetLastError(), "submit naive aggregation", "launch_failure");
        } else if (variant == Variant::row_split) {
            const auto row_blocks =
                static_cast<unsigned>(std::min<std::uint64_t>(graph.nodes, maximum_blocks));
            mean_row_split<<<row_blocks, threads_per_block, 0, stream.value>>>(
                offsets.as<std::int32_t>(), columns.as<std::int32_t>(), x.as<float>(),
                y.as<float>(), graph.nodes, graph.features);
            checked(cudaGetLastError(), "submit row-split aggregation", "launch_failure");
        } else {
            if (scale_cusparse_inputs) {
                scale_features<<<blocks(elements), threads_per_block, 0, stream.value>>>(
                    x.as<float>(), scaled_x.as<float>(), elements);
                checked(cudaGetLastError(), "submit exact feature scaling", "launch_failure");
            }
            const float alpha = 1.0f, beta = 0.0f;
            sparse_checked(cusparseSpMM(handle.value, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, adjacency.value,
                                        input.value, &beta, result.value, CUDA_R_32F,
                                        CUSPARSE_SPMM_CSR_ALG2, workspace.value),
                           "submit SpMM");
            if (scale_cusparse_inputs)
                normalize<true><<<blocks(elements), threads_per_block, 0, stream.value>>>(
                    offsets.as<std::int32_t>(), y.as<float>(), elements, graph.features);
            else
                normalize<false><<<blocks(elements), threads_per_block, 0, stream.value>>>(
                    offsets.as<std::int32_t>(), y.as<float>(), elements, graph.features);
            checked(cudaGetLastError(), "submit row normalization", "launch_failure");
        }
        result_submitted = true;
    }
    void synchronize() {
        require_healthy();
        if (graph.nodes)
            checked(cudaStreamSynchronize(stream.value), "complete device work");
    }
    void download(std::vector<float> &destination) {
        checked(cudaMemcpyAsync(destination.data(), y.value,
                                static_cast<std::size_t>(feature_bytes), cudaMemcpyDeviceToHost,
                                stream.value),
                "download output");
    }
};

Session::Session(const Graph &graph, Variant variant) {
    validate(graph);
    try {
        impl_ = std::make_unique<Impl>(graph, variant);
    } catch (const std::bad_alloc &) {
        throw Error("allocation_failure", "Host allocation failed during CUDA preparation");
    }
}
Session::~Session() = default;
Session::Session(Session &&) noexcept = default;
Session &Session::operator=(Session &&) noexcept = default;
double Session::preparation_ms() const noexcept { return impl_ ? impl_->prep_ms : 0; }
std::uint64_t Session::memory_bytes() const noexcept { return impl_ ? impl_->allocation_bytes : 0; }
std::uint64_t Session::workspace_bytes() const noexcept {
    return impl_ ? impl_->workspace_size : 0;
}
std::uint64_t Session::memory_budget_bytes() const noexcept { return impl_ ? impl_->budget : 0; }
std::uint64_t Session::overhead_reserve_bytes() const noexcept {
    return impl_ ? impl_->reserve : 0;
}
const char *Session::path() const noexcept {
    if (!impl_)
        return "moved_from";
    if (!impl_->graph.nodes)
        return "empty_host_return";
    if (impl_->graph.columns.empty())
        return "zero_edge_fill";
    if (impl_->variant == Variant::naive)
        return "naive_kernel";
    if (impl_->variant == Variant::row_split)
        return "row_split_kernel";
    return impl_->scale_cusparse_inputs ? "cusparse_scaled_spmm_normalize"
                                        : "cusparse_spmm_normalize";
}
std::vector<float> Session::output() {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->require_healthy();
    std::vector<float> result;
    try {
        result.resize(impl_->graph.x.size());
    } catch (const std::bad_alloc &) {
        throw Error("allocation_failure", "Host output allocation failed");
    }
    impl_->run();
    if (impl_->graph.nodes)
        impl_->download(result);
    impl_->synchronize();
    return result;
}
std::vector<float> Session::download_output() {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->require_healthy();
    if (!impl_->result_submitted && impl_->graph.nodes)
        throw Error("invalid_input", "No aggregation result has been produced by this session");
    std::vector<float> result;
    try {
        result.resize(impl_->graph.x.size());
    } catch (const std::bad_alloc &) {
        throw Error("allocation_failure", "Host output allocation failed");
    }
    if (impl_->graph.nodes)
        impl_->download(result);
    impl_->synchronize();
    return result;
}
void Session::warmup(std::size_t calls) {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->require_healthy();
    for (std::size_t i = 0; i < calls; ++i)
        impl_->run();
    impl_->synchronize();
}
double Session::resident_batch(std::size_t calls) {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->require_healthy();
    if (!calls)
        throw Error("invalid_input", "Resident batch must contain at least one call");
    if (!impl_->graph.nodes)
        return 0;
    impl_->record(0);
    for (std::size_t i = 0; i < calls; ++i)
        impl_->run();
    impl_->record(1);
    impl_->wait_event(1);
    return impl_->elapsed(0, 1);
}
PreparedCall Session::prepared_call() {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->require_healthy();
    const auto start = Clock::now();
    PreparedCall call;
    try {
        call.output.resize(impl_->graph.x.size());
    } catch (const std::bad_alloc &) {
        throw Error("allocation_failure", "Host output allocation failed");
    }
    if (!impl_->graph.nodes) {
        call.wall_ms = elapsed_wall(start);
        return call;
    }
    impl_->record(0);
    impl_->checked(cudaMemcpyAsync(impl_->x.value, impl_->graph.x.data(),
                                   static_cast<std::size_t>(impl_->feature_bytes),
                                   cudaMemcpyHostToDevice, impl_->stream.value),
                   "upload call features");
    impl_->record(1);
    impl_->run();
    impl_->record(2);
    impl_->download(call.output);
    impl_->record(3);
    impl_->wait_event(3);
    call.wall_ms = elapsed_wall(start);
    call.h2d_ms = impl_->elapsed(0, 1);
    call.aggregation_ms = impl_->elapsed(1, 2);
    call.d2h_ms = impl_->elapsed(2, 3);
    return call;
}
void Session::synchronize() {
    if (!impl_)
        throw Error("invalid_input", "Cannot use a moved-from CUDA session");
    impl_->synchronize();
}

} // namespace graphlab::cuda
