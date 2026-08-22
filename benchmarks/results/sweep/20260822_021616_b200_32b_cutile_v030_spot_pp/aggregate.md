# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 2048 | 36 | 3 | 568.02 ± 1.94 | 63.4 | 3668.9 |
| grout | default | 32768 | 36 | 3 | 3266.14 ± 2.74 | 11.0 | 10043.6 |
| sglang | no-radix | 2048 | 36 | 3 | 584.24 ± 2.09 | 61.6 | 3567.0 |
| sglang | no-radix | 32768 | 36 | 3 | 3010.73 ± 2.30 | 12.0 | 10895.7 |
| vllm | cuda-graph | 2048 | 36 | 3 | 567.49 ± 1.01 | 63.4 | 3672.3 |
| vllm | cuda-graph | 32768 | 36 | 3 | 2933.26 ± 1.11 | 12.3 | 11183.5 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| grout | default | 32768 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 2048 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 32768 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 32768 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 2048 | 36 | 105.25 | 19459.0 | 462.30 | 77.9 |
| grout | default | 32768 | 36 | 2768.53 | 11835.9 | 497.59 | 72.3 |
| sglang | no-radix | 2048 | 36 | 114.91 | 17822.2 | — | — |
| sglang | no-radix | 32768 | 36 | 2505.58 | 13078.0 | — | — |
| vllm | cuda-graph | 2048 | 36 | 104.67 | 19565.9 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2444.45 | 13405.1 | — | — |
