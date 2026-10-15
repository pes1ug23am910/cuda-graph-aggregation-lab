# CUDA Graph Aggregation Lab

A C++20/CUDA study of incoming-neighbour mean aggregation on sparse directed graphs. It compares a simple CUDA kernel, one row-splitting variant, and an equivalent cuSPARSE SpMM path, using an independent FP64 oracle and reproducible synthetic workloads.

    Y[v,f] = sum(X[u,f] for u in incoming(v)) / degree(v)

Empty neighbourhoods produce zero. This is a standalone primitive; it does not implement training, a backward pass or an entire graph neural network.


Two complete 54-input campaigns ran on an RTX 4060 Laptop GPU, including a clean rebuild and repeat. Row splitting helped a narrow group of small skewed inputs and usually lost to the simple kernel; the [results](docs/results.md) retain all regressions and timing variation. The [evidence bundle](evidence/rtx4060-laptop/README.md) includes raw measurements, hashes and validation limits.

## Build and verify

The CPU build requires CMake 3.25+ and a C++20 compiler, with no CUDA dependency:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
python -m unittest discover -s tests -p "test_*.py" -v
```

For CUDA, use a supported NVIDIA Toolkit/host compiler and set the architecture for your GPU. This example targets compute capability 8.9:

```sh
cmake -S . -B build-cuda -DGRAPHLAB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release --parallel
ctest --test-dir build-cuda -C Release --output-on-failure
```

On Windows, run in an x64 Visual Studio developer shell. Add the Toolkit's runtime DLL directory to PATH. With a single-configuration generator the executable is build-cuda/graphlab.exe; Visual Studio generators place it in build-cuda/Release/. CMake presets are available for Ninja. A missing CUDA device makes the device test explicitly skipped, not passed.

## Run

```sh
build-cuda/graphlab info
build-cuda/graphlab verify --backend all
build-cuda/graphlab bench --backend naive --nodes 4096 --features 64 --family skewed --seed 42
python scripts/run_campaign.py --exe build-cuda/graphlab --output results/run-001
python scripts/summarize.py results/run-001/raw.jsonl --output results/run-001/report.md
```

Use the .exe suffix on Windows. The complete campaign covers 54 inputs with five groups of 100 measured calls after 20 warmups; backend order alternates between groups. Add --smoke to the campaign command for a short diagnostic run. CPU-only pipeline checks use --cpu-only and do not establish GPU performance.

## Design and evidence

- [Architecture and fixture format](docs/architecture.md)
- [Numerical error model and negative controls](docs/numerics.md)
- [Timing boundaries and reproduction protocol](docs/benchmarking.md)
- [Measured results and limitations](docs/results.md)

Inputs use canonical signed 32-bit CSR, row-major FP32 features in [-1,1], and degrees at most 4096. Validation rejects malformed structure, nonfinite values, unsupported magnitudes and overflow. The library adapter includes normalization in the timer and explicitly handles empty inputs.

The repository includes portable CI, fixture round-trip checks, host contract tests, device verification cases, campaign scripts, and profiling/sanitizer commands. Reported results distinguish executed checks from unavailable tooling. No speedup is claimed for a whole application.

MIT licensed. CUDA and cuSPARSE are separately installed NVIDIA dependencies; no vendor binaries are included.
