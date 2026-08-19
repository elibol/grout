# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 3 | 1873.66 ± 4.65 | 19.2 | 8763.6 |
| grout | default | 32768 | 36 | 3 | 4176.77 ± 4.75 | 8.6 | 7853.9 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1789.52 ± 0.35 | 20.1 | 9175.6 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3700.81 ± 4.75 | 9.7 | 8864.0 |

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
| grout | default | 16384 | 36 | 1401.86 | 11687.4 | 471.79 | 76.3 |
| grout | default | 32768 | 36 | 3674.68 | 8917.2 | 497.99 | 72.3 |
| vllm | cuda-graph | 16384 | 36 | 1313.57 | 12472.8 | — | — |
| vllm | cuda-graph | 32768 | 36 | 3198.42 | 10245.1 | — | — |
