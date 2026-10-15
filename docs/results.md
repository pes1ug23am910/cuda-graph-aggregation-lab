# Measured results

Row splitting was not a general speedup on this GPU. The simple CUDA kernel had the fastest median on 36 of 54 inputs in run 01 and 39 of 54 in the clean-build repeat. The candidate was fastest on three inputs in each run. It remains an explicit evaluated variant rather than replacing the baseline.

The repeat confirmed a narrow benefit over the simple kernel on small skewed graphs: N=4096 with F=32 or 64, for all three seeds. Across both runs, ratios were 2.09–2.25× at F=32 and 1.33–1.37× at F=64. cuSPARSE was still faster than the candidate at F=64. No whole-application speedup is claimed.

## Evidence and setup

[Run 01 manifest](../evidence/rtx4060-laptop/run-01/manifest.json) and [run 02 manifest](../evidence/rtx4060-laptop/run-02/manifest.json) record the same clean source revision, `15602afe0cbb358c1f76e2fcf2d20ba84fda4909`. Run 02 used a separate clean export and rebuild. Each run contains 810 measurement groups: 54 inputs × three implementations × five groups. Each group measures 100 calls after 20 warmups. Both runs completed with no failed or missing batches.

Hardware was an RTX 4060 Laptop GPU, compute capability 8.9, with approximately 8 GiB reported memory. Builds used MSVC 19.44 and NVCC 12.9.86 on Windows 11. Full build flags, environment versions, timestamps, hashes, test logs and reproduction commands are in the [evidence index](../evidence/rtx4060-laptop/README.md).

Power, temperatures, clocks and background activity were not controlled. All 216 telemetry attempts across the two campaigns failed to initialize NVML. The observations describe this machine under these conditions.

## Complete-matrix comparison

For each input, latency is the median of five per-call batch averages. A ratio is **baseline median / row_split median**: above 1 favors row_split. Geometric averages weight all 54 inputs equally. Runs are kept separate.

| Statistic | Run 01 | Clean-build run 02 |
|---|---:|---:|
| Row-split median wins against naive | 7 / 54 | 8 / 54 |
| Row-split median wins against cuSPARSE | 22 / 54 | 21 / 54 |
| Geometric mean ratio against naive | 0.586× | 0.595× |
| Geometric mean ratio against cuSPARSE | 0.847× | 0.853× |
| Fastest median: naive / row_split / cuSPARSE | 36 / 3 / 15 | 39 / 3 / 12 |

Only the six small skewed inputs described above favored row_split over naive in both runs. Three other inputs changed comparison direction between runs. A tiny 1.004× apparent benefit in run 01 did not repeat.

The table below shows resident aggregation latency in **microseconds**, as **run 01 / run 02**. Each entry is the median across the three seed-specific medians; individual seed results and full ranges remain in the linked reports.

| N | F | Family | Naive µs | Row-split µs | cuSPARSE µs |
|---:|---:|---|---:|---:|---:|
| 4096 | 32 | regular8 | 15.50 / 15.95 | 30.20 / 30.26 | 45.02 / 45.05 |
| 4096 | 32 | regular32 | 22.84 / 21.72 | 37.94 / 38.00 | 60.43 / 60.60 |
| 4096 | 32 | skewed | 122.71 / 122.70 | 56.59 / 56.55 | 66.38 / 66.57 |
| 4096 | 64 | regular8 | 16.02 / 15.44 | 44.84 / 44.88 | 52.89 / 53.78 |
| 4096 | 64 | regular32 | 33.84 / 34.10 | 61.60 / 61.76 | 67.19 / 67.66 |
| 4096 | 64 | skewed | 129.08 / 129.50 | 96.28 / 96.37 | 72.37 / 71.55 |
| 4096 | 128 | regular8 | 25.95 / 25.88 | 72.44 / 72.48 | 61.15 / 55.08 |
| 4096 | 128 | regular32 | 64.45 / 64.51 | 110.40 / 110.39 | 83.82 / 83.32 |
| 4096 | 128 | skewed | 159.83 / 160.76 | 178.34 / 178.54 | 91.14 / 90.43 |
| 65536 | 32 | regular8 | 91.89 / 91.83 | 429.18 / 429.16 | 247.15 / 246.92 |
| 65536 | 32 | regular32 | 248.82 / 248.71 | 553.91 / 553.98 | 700.85 / 700.68 |
| 65536 | 32 | skewed | 454.89 / 456.38 | 614.22 / 613.79 | 795.11 / 794.95 |
| 65536 | 64 | regular8 | 204.85 / 205.51 | 773.67 / 773.13 | 331.59 / 332.86 |
| 65536 | 64 | regular32 | 524.26 / 524.33 | 1002.23 / 1355.67 | 801.02 / 722.62 |
| 65536 | 64 | skewed | 769.16 / 769.31 | 1179.97 / 1175.43 | 726.11 / 731.12 |
| 65536 | 128 | regular8 | 1687.45 / 1924.82 | 2894.27 / 2815.17 | 1709.92 / 1722.11 |
| 65536 | 128 | regular32 | 2579.14 / 2068.01 | 2832.55 / 2682.93 | 2383.12 / 2668.06 |
| 65536 | 128 | skewed | 2639.76 / 2722.29 | 2947.67 / 2622.71 | 2363.41 / 2521.02 |

[All run 01 input summaries](../evidence/rtx4060-laptop/run-01/report.md), [all run 02 input summaries](../evidence/rtx4060-laptop/run-02/report.md), and [machine-readable comparison](../evidence/rtx4060-laptop/comparison.json) preserve every input. The original JSONL contains every batch and all timing boundaries.

## Variation and timing boundaries

Within an input/backend combination, maximum batch latency exceeded twice the minimum in 16 naive, 10 row-split and nine cuSPARSE groups in run 01; the repeat had 15, 10 and nine respectively. No groups were removed. The approximately 17.8× row-split slowdown observed for one first-run input occurred in a highly variable case and is not a stable algorithmic conclusion. Five batches and quartiles do not establish statistical significance. Missing telemetry prevents attributing the variation to a specific cause.

Resident event timing includes every device operation needed for normalized output. Prepared calls also include host result allocation and pageable feature/output transfers. Cold-call wall time includes session creation and cleanup, with runtime initialization excluded. Input generation and the oracle are separate. See the [timing protocol](benchmarking.md).

For the disclosed calibration input N=4096, F=64, skewed, seed=42, the clean-build repeat recorded:

| Implementation | Resident median ms | Prepared-call wall median ms | Initialized-runtime cold-call wall median ms |
|---|---:|---:|---:|
| naive | 0.1324 | 1.0812 | 2.8089 |
| row_split | 0.0964 | 0.9668 | 2.9463 |
| cuSPARSE | 0.0720 | 0.9605 | 3.0256 |

The resident benefit does not translate directly into an equal end-to-end benefit. Prepared and cold medians summarize five separately checked calls; they are not 100-call event batches.

## Optimization and separate profiling

The candidate changes the mapping from one thread per scalar output to one 256-thread block per row, with eight warp partial sums per feature tile. The hypothesis was that splitting a long hub traversal would reduce the critical path on skewed rows. This adds reduction and synchronization work on short rows.

The exploratory Nsight Systems capture for N=4096, F=64, skewed, seed=42 observed mean custom-kernel durations of 130.10 µs for naive and 96.40 µs for row_split. The library trace separately exposes SpMM, normalization and auxiliary kernels. [Unchanged CSV exports and capture hashes](../evidence/rtx4060-laptop/profiles/metadata.json) are retained.

These profiler values are separate from ordinary timings. The small-skewed improvement and low-degree regressions are consistent with the mapping tradeoff; duration traces alone do not prove a bandwidth, occupancy or stall explanation. The [exploratory calibration records](../evidence/rtx4060-laptop/calibration/calibration-baselines-metadata.json) are disclosed separately from the complete matrix.

Nsight Compute 2025.2.1.0 was installed into an isolated local tool directory from a checksum-verified official archive. Its first one-launch profile failed with ERR_NVGPUCTRPERM (exit 1): this user lacks GPU hardware-counter access. Further attempts stopped because the restriction applies to the device. Clock control was disabled for the attempt and no machine settings were changed. [Command metadata](../evidence/rtx4060-laptop/profiles/ncu-metadata.json) and the [permission diagnostic](../evidence/rtx4060-laptop/profiles/ncu-naive-counter-permission.txt) are retained; no hardware-counter findings are claimed.

## Correctness and remaining validation limits

Both full campaigns passed all 340 preflight verification cases across CPU and the three GPU implementations. The final output of every measured batch, and each separately timed prepared-call and cold-call output, was checked against the independent FP64 oracle. Intermediate outputs within a 100-call batch were not downloaded. Across both campaigns, maximum absolute error was 1.490116119×10^-7 and maximum error/budget ratio was 0.276708316, below the acceptance boundary of 1.

The initial unprotected cuSPARSE path failed a subnormal-input case. The [retained failure](../evidence/rtx4060-laptop/regressions/initial-subnormal-failure.txt) led to exact power-of-two range protection, whose derivation is in the [numerical contract](numerics.md). Both final verification sets pass those cases. All 540 cuSPARSE timing groups across the full matrices used the ordinary unscaled path.

The portable core checked 327 assertions. The clean CUDA build passed all four CTest entries without skips; the portable CPU build passed all three. The final Python suite passes 36 tests covering CLI parsing, fixture round trips and reporting/provenance failure cases.

Compute Sanitizer could not initialize the WDDM debugger interface and exited 99. Its memcheck output does not constitute successful instrumentation; subsequent modes were not run. Instrumented memory, initialization, synchronization and race coverage remains unavailable on this setup. The [diagnostic logs](../evidence/rtx4060-laptop/profiles/naive-memcheck.txt) are retained. Numerical tests and source review do not replace that coverage.

This is a bounded synthetic primitive study on one laptop GPU. It does not establish performance on real graph datasets, other devices, training/backward passes, graph neural network inference or production workloads.
