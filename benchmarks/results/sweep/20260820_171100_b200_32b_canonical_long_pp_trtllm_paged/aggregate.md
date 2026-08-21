# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | cutile-lpt | 16384 | 36 | 3 | 1631.87 ± 3.31 | 22.1 | 10062.1 |
| grout | cutile-lpt | 32768 | 36 | 3 | 3487.45 ± 5.72 | 10.3 | 9406.3 |
| grout | trtllm-paged-zero-copy | 16384 | 36 | 3 | 1569.97 ± 1.15 | 22.9 | 10458.8 |
| grout | trtllm-paged-zero-copy | 32768 | 36 | 3 | 3157.16 ± 3.99 | 11.4 | 10390.4 |
| sglang | no-radix | 16384 | 36 | 3 | 1596.67 ± 1.18 | 22.5 | 10283.9 |
| sglang | no-radix | 32768 | 36 | 3 | 3178.43 ± 2.17 | 11.3 | 10320.8 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1560.35 ± 3.40 | 23.1 | 10523.3 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3109.04 ± 5.13 | 11.6 | 10551.2 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | cutile-lpt | 16384 | 1 | n/a | n/a | n/a |
| grout | cutile-lpt | 32768 | 1 | n/a | n/a | n/a |
| grout | trtllm-paged-zero-copy | 16384 | 1 | n/a | n/a | n/a |
| grout | trtllm-paged-zero-copy | 32768 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 16384 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 32768 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 16384 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 32768 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | cutile-lpt | 16384 | 36 | 1155.86 | 14174.8 | 477.42 | 75.4 |
| grout | cutile-lpt | 32768 | 36 | 2991.13 | 10955.1 | 496.31 | 72.5 |
| grout | trtllm-paged-zero-copy | 16384 | 36 | 1096.45 | 14942.7 | 473.08 | 76.1 |
| grout | trtllm-paged-zero-copy | 32768 | 36 | 2658.36 | 12326.4 | 498.79 | 72.2 |
| sglang | no-radix | 16384 | 36 | 1111.99 | 14733.9 | — | — |
| sglang | no-radix | 32768 | 36 | 2671.25 | 12266.9 | — | — |
| vllm | cuda-graph | 16384 | 36 | 1075.93 | 15227.7 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2612.51 | 12542.7 | — | — |
