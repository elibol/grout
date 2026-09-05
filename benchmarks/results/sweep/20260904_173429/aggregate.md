# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 456.76 ± 0.12 | 78.8 | 118.2 |
| grout | default | 128 | 36 | 10 | 456.39 ± 0.04 | 78.9 | 359.3 |
| grout | default | 512 | 36 | 10 | 483.82 ± 3.06 | 74.4 | 1132.7 |
| grout | default | 2048 | 36 | 3 | 584.27 ± 0.81 | 61.6 | 3566.8 |
| grout | default | 8192 | 36 | 3 | 979.87 ± 3.93 | 36.7 | 8397.0 |
| sglang | no-radix | 18 | 36 | 10 | 481.25 ± 0.51 | 74.8 | 112.2 |
| sglang | no-radix | 128 | 36 | 10 | 482.07 ± 0.31 | 74.7 | 340.2 |
| sglang | no-radix | 512 | 36 | 10 | 498.92 ± 0.74 | 72.2 | 1098.4 |
| sglang | no-radix | 2048 | 36 | 3 | 599.64 ± 1.25 | 60.0 | 3475.4 |
| sglang | no-radix | 8192 | 36 | 3 | 985.44 ± 1.06 | 36.5 | 8349.6 |
| vllm | cuda-graph | 18 | 36 | 10 | 461.88 ± 0.07 | 77.9 | 116.9 |
| vllm | cuda-graph | 128 | 36 | 10 | 463.31 ± 0.05 | 77.7 | 354.0 |
| vllm | cuda-graph | 512 | 36 | 10 | 485.58 ± 0.43 | 74.1 | 1128.6 |
| vllm | cuda-graph | 2048 | 36 | 3 | 581.20 ± 1.99 | 61.9 | 3585.7 |
| vllm | cuda-graph | 8192 | 36 | 3 | 965.51 ± 2.37 | 37.3 | 8521.9 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| grout | default | 8192 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 18 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 128 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 512 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 2048 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 8192 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 128 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 512 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 8192 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 15.85 | 1135.3 | 440.88 | 81.7 |
| grout | default | 128 | 36 | 16.25 | 7877.4 | 440.12 | 81.8 |
| grout | default | 512 | 36 | 29.56 | 17318.9 | 454.28 | 79.2 |
| grout | default | 2048 | 36 | 112.22 | 18250.0 | 472.05 | 76.3 |
| grout | default | 8192 | 36 | 511.75 | 16008.0 | 465.74 | 77.3 |
| sglang | no-radix | 18 | 36 | 26.34 | 683.4 | — | — |
| sglang | no-radix | 128 | 36 | 27.01 | 4738.9 | — | — |
| sglang | no-radix | 512 | 36 | 40.96 | 12500.7 | — | — |
| sglang | no-radix | 2048 | 36 | 128.22 | 15972.0 | — | — |
| sglang | no-radix | 8192 | 36 | 505.58 | 16203.1 | — | — |
| vllm | cuda-graph | 18 | 36 | 16.37 | 1099.6 | — | — |
| vllm | cuda-graph | 128 | 36 | 17.71 | 7228.8 | — | — |
| vllm | cuda-graph | 512 | 36 | 32.07 | 15964.2 | — | — |
| vllm | cuda-graph | 2048 | 36 | 113.27 | 18081.4 | — | — |
| vllm | cuda-graph | 8192 | 36 | 486.47 | 16839.6 | — | — |
