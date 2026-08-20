# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 3 | 1627.65 ± 1.89 | 22.1 | 10088.2 |
| grout | default | 32768 | 36 | 3 | 3499.47 ± 7.39 | 10.3 | 9374.0 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1558.70 ± 1.64 | 23.1 | 10534.4 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3102.36 ± 2.77 | 11.6 | 10573.9 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 16384 | 1 | n/a | n/a | n/a |
| grout | default | 32768 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 16384 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 32768 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 1149.76 | 14250.0 | 475.22 | 75.8 |
| grout | default | 32768 | 36 | 2993.85 | 10945.1 | 500.55 | 71.9 |
| vllm | cuda-graph | 16384 | 36 | 1068.95 | 15327.2 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2596.78 | 12618.7 | — | — |
