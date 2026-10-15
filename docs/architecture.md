# Operation and implementation

The library computes incoming-neighbour feature means. CSR row v is the destination; column u means the directed edge u -> v. X and Y are row-major N by F matrices. For nonempty rows, Y[v,f] = sum(X[u,f]) / degree(v). An empty row produces positive zero.

## Supported domain

- Signed 32-bit CSR indices; N >= 0 and F >= 1.
- Canonical rows: strictly increasing source indices, no duplicates.
- Finite FP32 features in [-1,1]; maximum degree 4096.
- Offsets begin at zero, are nondecreasing and end at the column count.
- N=0 has offsets [0] and empty feature/output arrays.
- Input/output aliasing is rejected by the verifier. Session owns its input copy.

Validation precedes device allocation. Shape products and byte counts are checked in wider host arithmetic. The text fixture loader limits host input to 512 MiB. Unsupported numerical domains are distinct from malformed structure.

## Components

| Component | Responsibility |
|---|---|
| graphlab_core | Validation, deterministic fixtures, FP64 oracle and numerical comparison |
| graphlab_cuda | Explicit-stream CUDA execution, device ownership, library adapter and event timing |
| graphlab | Fixture IO, exact-case verification and command-line benchmark records |
| run_campaign.py | Fixed matrix, alternating implementation order, failures and environment records |
| summarize.py | Complete-group medians and quartiles, retaining failed evidence |

The custom baseline maps a scalar output to each thread, with a grid-stride traversal. Adjacent threads access adjacent feature columns. Each output sums neighbours in CSR order with FP32 accumulation, then divides by degree.

The cuSPARSE path uses FP32 adjacency values of one, non-transposed SpMM, row-major dense matrices and CSR_ALG2. A separate normalization kernel divides sums by row degree. Both library work and normalization are inside the resident timing boundary. For tiny features the adapter uses exact power-of-two range protection, including a scaling kernel and additional buffer; see the numerical contract. No optional library preprocessing is used. N=0 returns through an empty wrapper; E=0 uses a zero-fill wrapper and does not call SpMM.

Sessions use move-only ownership and nonthrowing cleanup. An explicit stream orders copies, kernels and events. Submission errors are checked immediately and completion is checked at synchronization boundaries. A failed session refuses further work.

Device admission counts explicit buffers, the queried library workspace and a 1 MiB reserve for bookkeeping. The budget is min(1 GiB, 25% of current free VRAM); it is an admission estimate, not a reservation or an exact measurement of driver-internal storage. Workspaces are released between implementations. Host vectors used by prepared calls are pageable.

## Fixture format

Text files begin with graphlab-csr-v1, followed by N F E, then N+1 offsets, E columns and N*F decimal FP32 values. Whitespace separates values. Exactly round-trippable decimal precision is written. The reader rejects missing or trailing data. Source indices are zero-based.

Use graphlab generate --nodes 33 --features 7 --family skewed --seed 7 --output example.csr, then graphlab validate --input example.csr. The embedded fingerprint is an FNV-1a identity aid, not a cryptographic integrity claim. Campaign manifests use SHA-256 for artifacts.

## Evaluated row-splitting variant

The row_split kernel assigns one block of 256 threads to a row, traversing feature tiles of 32 columns. Eight warps accumulate disjoint neighbour subsequences. The first warp reduces eight shared-memory partials and writes the normalized result. Every thread reaches both barriers, including feature-tail lanes. Rows beyond the grid limit are handled by a block-level grid-stride loop.

This mapping was chosen after a Nsight Systems trace showed a long-running custom kernel on the declared skewed calibration case. It shortens each hub's serial traversal at the cost of more threads, reductions and synchronization for low-degree rows. It is an evaluated candidate, not a universal replacement for the baseline. The measured report includes regressions. Systems supports the kernel-duration observation; no hardware-counter claim about bandwidth, occupancy or stalls is made.
