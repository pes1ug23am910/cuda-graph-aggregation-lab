# Numerical contract and validation

The independent CPU oracle traverses the same FP32 bytes in FP64 and divides by the integer row degree. It also computes an outward-rounded upper bound A on sum(abs(X))/degree. The oracle is a floating-point reference, not exact real arithmetic.

For d>0, the comparison budget is the outward-rounded sum

    gamma(2*d+2, 2^-24) * A
  + gamma(d+2, 2^-53) * A
  + 2^-150

where gamma(k,u) = k*u/(1-k*u). Every positive bound calculation rounds outward. Degree <=4096 keeps the denominators positive. The FP32 operation count is intentionally conservative.

This derivation covers FP32 sums followed by division under round-to-nearest and gradual underflow. Adding FP32 numbers cannot round an exact subnormal sum: every operand is an integer multiple of 2^-149. Multiplication by the adjacency value one is exact. The final division can incur at most half a subnormal spacing, 2^-150. Nonzero FP64 reference means are at least 2^-161, far above FP64 underflow. Pre-normalized adjacency, reduced precision and flush-to-zero require separate analyses.

Custom kernels compile with ftz=false, prec-div=true and fmad=false. These switches do not control cuSPARSE internals; the library must independently pass subnormal and cancellation fixtures. The host checks round-to-nearest and subnormal handling. No global FP32 atomic reduction is used in custom kernels.

Empty rows must produce positive zero; a nonempty all-zero row may return either sign of zero. NaN or infinity always fails. Diagnostics include maximum absolute error, maximum error scaled by the local budget, the worst coordinate and its budget. Relative error against a near-zero answer is not the acceptance rule.

A conservative bound alone is insufficient. Tests include hand-calculated asymmetric directed graphs, self-loops, isolated vertices, maximum degree, feature-width/block tails, cancellation and subnormals. Negative controls intentionally transpose edges, omit normalization, skip a row, use the wrong feature stride, retain stale data and flush tiny inputs. Structural tests reach malformed offsets, duplicates, unsorted rows, bad indices, nonfinite features, domain violations and shape overflow.

References: [NVIDIA floating-point guidance](https://docs.nvidia.com/cuda/floating-point/index.html), [PTX floating-point atomic semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## Library range protection

Device tests found that the unmodified cuSPARSE path flushes the smallest subnormal input to zero. The adapter therefore detects nonzero features below 2^-90. Those cases use exact scaling by 2^100 before SpMM, followed by division and an explicit round-to-nearest multiplication by 2^-100 in the normalization kernel. Scaling, extra storage and normalization are included in the measured work.

Without tiny features, all inputs are multiples of at least 2^-113, so a nonzero sum divided by degree<=4096 is at least 2^-125 and remains normal. With scaling, the smallest lattice spacing becomes 2^-49 and the largest possible absolute sum is 2^112. Neither SpMM accumulation nor division underflows or overflows. Only the final downscale can underflow, so the existing half-subnormal allowance remains applicable. Zero is handled directly. The full synthetic matrix takes the ordinary unscaled library path; tiny-value tests exercise the protected path.
