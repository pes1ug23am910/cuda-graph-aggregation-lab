# Evidence bundle: RTX 4060 Laptop GPU

The two ordinary timing campaigns use the same pinned source, `15602afe0cbb358c1f76e2fcf2d20ba84fda4909`. Run 02 was rebuilt into an empty build directory from a clean Git export; every exported file matched its Git blob. The executable hashes differ and are recorded separately. Bit-for-bit binary reproducibility is not claimed.

[UTC and IST timestamps](timestamps.json) identify both runs. Each run contains 54 synthetic inputs, three implementations, five groups, 100 measured calls per group, and 20 warmups. The full bundle retains all raw samples and failed telemetry attempts. [Results](../../docs/results.md) interpret the data.

## Contents

- `run-01/` and `run-02/`: unchanged campaign artifacts, including manifests, raw JSONL, preflight verification, exact invocation output, telemetry attempts and complete-group summaries.
- `build/`: clean configure/build/CTest evidence and portable CPU CTest output. Local repository prefixes are replaced with `<repository>`; original and published hashes are in `redactions.json`.
- `profiles/`: separately collected Nsight Systems summaries and diagnostic/version records. These are not ordinary timing samples.
- `calibration/`: exploratory subset measured before the full matrix. Source/binary hashes are retained; these records do not establish the final result.
- `regressions/initial-subnormal-failure.txt`: retained failure from the initial library adapter. The final adapter uses exact power-of-two range protection and passes the corresponding cases in both campaign preflights.

`SHA256SUMS` covers the public evidence files, excluding itself. Each original campaign manifest also seals its five primary raw artifacts. The generated reports were created after campaign completion and therefore are covered by the bundle checksum list rather than the original campaign manifest.

## Environment and build

Windows 11, MSVC 19.44.35227, CMake 3.31.6, Ninja, NVIDIA CUDA Toolkit components from the 12.9.1 redistribution (NVCC 12.9.86), Python 3.13.7. Device: NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, reported total memory 8,585,216,000 bytes. The runtime reports CUDA 12090 and cuSPARSE 12510. The `driver_version=13030` field is the CUDA driver API compatibility version, not the installed display-driver package version.

The tested release configuration uses C++20, architecture 89, the host oracle/validation core compiled with /fp:strict, and device `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo`. Vendor dependencies are separately installed; none are included in this repository.

From an x64 Visual Studio developer shell with CUDA and runtime DLLs available:

```sh
git checkout 15602afe0cbb358c1f76e2fcf2d20ba84fda4909
cmake -S . -B build-cuda -G Ninja -DGRAPHLAB_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
python scripts/run_campaign.py --exe build-cuda/graphlab.exe --output results/reproduction
python scripts/summarize.py results/reproduction/raw.jsonl --output results/reproduction/report.md
```

The clean rebuild passed all four CTest entries without skips, including 340 verification cases across CPU, naive CUDA, row-split CUDA and cuSPARSE. The CPU contract executable checked 327 assertions. Current repository changes after the measurement revision concern documentation, evidence and CLI/test usability; they do not alter the measured aggregation implementation.

## Interpretation limits

Power, temperatures, clocks and background activity were not controlled. NVML failed on every recorded telemetry attempt. Successful CUDA execution does not establish stable clocks. Report all variation without assigning a cause.

Compute Sanitizer 2025.2.1.0 could not initialize the WDDM debugger interface (exit 99). Its apparent zero-leak line is not successful instrumentation. Subsequent sanitizer modes were not run. No instrumented memory or race safety claim is made; static review and numerical tests are separate evidence.

Continuous integration is configured for portable Linux and Windows builds. Local tests have executed; remote GitHub Actions results are not asserted.

## Claim-to-evidence map

| Repository claim | Evidence | Scope |
|---|---|---|
| Validated directed CSR mean aggregation | [Core contract tests](../../tests/core_tests.cpp), [GPU/CPU cases](run-02/verification.jsonl) | Bounded documented FP32 domain |
| Equivalent custom and library comparison | [CUDA source](../../src/cuda.cu), [timing protocol](../../docs/benchmarking.md), raw records in both runs | Normalization included; no whole-application claim |
| One evaluated row-splitting variant | [Architecture](../../docs/architecture.md), [results](../../docs/results.md) | Wins and losses retained |
| Clean-build repetition | [Build output](build/clean-build.txt), [CTest](build/clean-ctest.txt), [repeat manifest](run-02/manifest.json) | Same source, separately hashed executable |
| Profiled kernel durations | [Nsight Systems exports](profiles/metadata.json) | Separate diagnostic capture; no automatic bottleneck-causality claim |
| Tiny-value library regression resolved | [Original failure](regressions/initial-subnormal-failure.txt), [numerical derivation](../../docs/numerics.md), final verification | Protected library path is tested; synthetic performance matrix uses the ordinary path |

Nsight Compute 2025.2.1.0 was attempted separately after timing completed, with clock control disabled. Hardware-counter access was denied (ERR_NVGPUCTRPERM, exit 1). [Metadata](profiles/ncu-metadata.json) records the verified download, command and redaction; [diagnostic output](profiles/ncu-naive-counter-permission.txt) is retained. No hardware-counter results were collected.

The final post-measurement usability fixes pass [36 Python tests](build/final-python-tests.txt) and [three CPU CTest entries](build/final-cpu-ctest.txt).
