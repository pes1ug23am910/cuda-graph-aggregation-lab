# Reproduction and measurement

The frozen matrix has 54 inputs: N in {4096,65536}, F in {32,64,128}, family in {regular8,regular32,skewed}, and seed in {7,42,2026}. Regular degree is capped at N. Skewed rows with index divisible by 64 have degree min(2048,N); remaining rows draw degree min(1+bounded(8),N). the exact PRNG and consumption order live in src/core.cpp and are protected by golden-byte tests. Edges are unique, sorted and directed. Fixtures are synthetic; results do not establish whole-model or application acceleration.

Each implementation receives identical fixture bytes. Campaigns use 20 warmups, five groups and 100 calls per timed batch, alternating backend order by group. All batch totals are retained. A per-call average is the total divided by the disclosed batch size; it is not a distribution of individually timed calls. Summaries show median, Q1/Q3 and the complete observed range. Smoke runs are excluded.

| Boundary | Included |
|---|---|
| Resident GPU aggregation | Every device operation needed to produce normalized output; library normalization included |
| Prepared graph call | Host result allocation, pageable feature upload, aggregation, download and completion; event components also recorded |
| Cold call with initialized runtime | Host validation/copy, session creation, allocations, graph/features upload, descriptors, execution, download and cleanup |

Input generation and independent oracle time are separate. Runtime initialization is excluded from the cold-call number and is stated explicitly. Preparation is separately recorded. CPU diagnostic timings include validation and result allocation; they are not equivalent to resident GPU timings.

The harness checks correctness before timing, downloads the final measured output without recomputing it, and checks prepared/cold outputs as well. Failed executions stay in raw records and stop the campaign. Empty or zero-edge paths are wrapper measurements, not SpMM throughput.

Run:

    python scripts/run_campaign.py --exe build-cuda/graphlab --output results/run-001
    python scripts/summarize.py results/run-001/raw.jsonl --output results/run-001/report.md

On Windows use the .exe suffix. The output directory must be new, protecting prior records. For a short pipeline check add --smoke; --cpu-only exercises the portable host path.

The manifest records executable SHA-256, source revision, matrix, timestamps, runtime metadata and hashes of raw artifacts. Telemetry attempts to capture temperature, clocks, power and utilization; unavailable telemetry is retained as a limitation. Do not infer controlled clocks or thermal stability from successful CUDA execution.

Profiler captures are separate from ordinary timing. scripts/profile.ps1 and scripts/profile.sh provide Compute Sanitizer and Nsight commands. Profiler replay time is not benchmark latency. Counter restrictions are reported explicitly rather than treated as a successful profile.

Sources: [CUDA event timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#using-cuda-gpu-timers), [cuSPARSE SpMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

The PRNG is SplitMix64. Canonical neighbours use Floyd sampling without replacement and sorted row order. Features consume the high 24 bits of each subsequent PRNG word, subtract 8388608 and multiply by exactly 2^-23. Thus the generated range is [-1,1-2^-23]. Rejection sampling defines bounded integers without implementation-specific distributions.
