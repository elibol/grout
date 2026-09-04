# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 476.74 ± 0.06 | 75.5 | 113.3 |
| grout | default | 128 | 36 | 10 | 475.31 ± 0.06 | 75.7 | 345.0 |
| grout | default | 512 | 36 | 10 | 503.07 ± 0.43 | 71.6 | 1089.3 |
| grout | default | 2048 | 36 | 3 | 604.84 ± 0.08 | 59.5 | 3445.5 |
| grout | default | 8192 | 36 | 3 | 1100.77 ± 1.19 | 32.7 | 7474.8 |
| vllm | cuda-graph | 18 | 36 | 10 | 453.71 ± 0.06 | 79.3 | 119.0 |
| vllm | cuda-graph | 128 | 36 | 10 | 454.78 ± 0.07 | 79.2 | 360.6 |
| vllm | cuda-graph | 512 | 36 | 10 | 475.65 ± 0.48 | 75.7 | 1152.1 |
| vllm | cuda-graph | 2048 | 36 | 3 | 562.64 ± 2.08 | 64.0 | 3704.0 |
| vllm | cuda-graph | 8192 | 36 | 3 | 944.47 ± 2.83 | 38.1 | 8711.8 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| grout | default | 8192 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 128 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 512 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 8192 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 15.81 | 1138.8 | 460.92 | 78.1 |
| grout | default | 128 | 36 | 16.51 | 7755.2 | 458.76 | 78.5 |
| grout | default | 512 | 36 | 29.07 | 17610.5 | 473.94 | 76.0 |
| grout | default | 2048 | 36 | 116.45 | 17587.6 | 488.36 | 73.7 |
| grout | default | 8192 | 36 | 582.06 | 14074.2 | 518.70 | 69.4 |
| vllm | cuda-graph | 18 | 36 | 16.18 | 1112.8 | — | — |
| vllm | cuda-graph | 128 | 36 | 17.32 | 7389.0 | — | — |
| vllm | cuda-graph | 512 | 36 | 30.96 | 16537.7 | — | — |
| vllm | cuda-graph | 2048 | 36 | 111.92 | 18298.2 | — | — |
| vllm | cuda-graph | 8192 | 36 | 476.21 | 17202.6 | — | — |
