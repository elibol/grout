# Canonical B200 Qwen3-32B long-prefill ladder

- Engine source: grout `d963df1ecc92885507db19ce8b6ebac1e78d9183`
- Shim build fix: `ef83a5b`
- Compiler source: cutile-rs `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Sweep: pp 16384, 32768; tg 36
- Repetitions: 3 measured requests after 3 warmups
- Cache policy: vLLM prefix cache off; SGLang radix cache off

The grout-trtllm arm used the page-16 context kernel with zero-copy dense K/V
cache binding (`GROUT_TRTLLM_REPACK` unset). The current-tree checked cuTile
LPT arm was run immediately afterward and merged into `run.jsonl` as
`cutile-lpt`. `run_cutile.jsonl` and `summary_cutile_20260820_172331.txt`
retain that arm independently. SGLang selected `trtllm_mha`; vLLM
auto-selected TRTLLM prefill attention.

| engine / arm | 16K e2e (ms) | 32K e2e (ms) |
|---|---:|---:|
| grout, checked cuTile LPT | 1631.87 | 3487.45 |
| grout, trtllm page-16 zero-copy | 1569.97 | 3157.16 |
| SGLang | 1596.67 | 3178.43 |
| vLLM | **1560.35** | **3109.04** |

Relative to checked cuTile, grout-trtllm reduces full-request latency by 3.79%
at 16K and 9.47% at 32K. Its remaining gaps to vLLM are 0.62% and 1.55%; it is
1.67% and 0.67% faster than SGLang.
