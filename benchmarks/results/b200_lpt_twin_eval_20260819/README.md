# B200 checked-vs-unchecked LPT validation

Qwen3-32B was measured on B200 at default clocks using grout `3b07fd2` and
cutile-rs `e90f9b8`. Three paired, order-alternating rounds were run at each
length. Every process discarded JIT compilation and one additional warmup,
then averaged three measured requests with 36 generated tokens.

The checked arm used the default `fmha_prefill_gqa_lpt_checked`. The twin arm
set `GROUT_FMHA_PREFILL_LPT_UNSAFE_TWIN=1`, selecting an exact-body version
with unchecked lowering. All other settings matched the shipping sm_100 long
profile: `BM=16`, `BN=128`, group 4, swizzle 8, schedule 1, latency 2, and
mask splitting enabled.

| pp | checked / twin prefill (ms) | twin delta | checked / twin e2e (ms) | twin delta | output |
|---:|---:|---:|---:|---:|---|
| 16384 | **1261.838** / 1263.682 | +0.146% | **1737.731** / 1738.680 | +0.055% | byte-identical |
| 32768 | 3494.702 / **3477.748** | -0.485% | 3996.190 / **3982.200** | -0.350% | byte-identical |

The twin does not meet the greater-than-2% shipping threshold at either
length, so the checked kernel remains the default.

Fresh cubins show a large static resource contrast despite parity runtime:

| kernel | registers | stack (bytes) | static LDL/STL |
|---|---:|---:|---:|
| checked | 128 | 1232 | 904 |
| unchecked twin | 128 | 0 | 0 |

`results.csv` contains each round mean and generated-text hash.
`resources.txt` contains the complete `cuobjdump -res-usage` output. Raw logs,
generated text, and JIT caches are intentionally not committed.
