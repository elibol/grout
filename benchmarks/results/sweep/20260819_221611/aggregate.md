# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 3 | 1630.70 ± 2.09 | 22.1 | 10069.3 |
| grout | default | 32768 | 36 | 3 | 3488.40 ± 3.71 | 10.3 | 9403.7 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1566.66 ± 0.39 | 23.0 | 10480.9 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3131.21 ± 2.07 | 11.5 | 10476.5 |

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
| grout | default | 16384 | 36 | 1159.45 | 14130.9 | 471.64 | 76.3 |
| grout | default | 32768 | 36 | 2988.49 | 10964.7 | 499.89 | 72.0 |
| vllm | cuda-graph | 16384 | 36 | 1084.60 | 15106.1 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2631.07 | 12454.3 | — | — |
