# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 458.92 ± 0.04 | 78.4 | 117.7 |
| grout | default | 128 | 36 | 10 | 456.14 ± 0.03 | 78.9 | 359.5 |
| grout | default | 512 | 36 | 10 | 480.70 ± 0.69 | 74.9 | 1140.0 |
| grout | default | 2048 | 36 | 3 | 586.76 ± 0.28 | 61.4 | 3551.7 |
| sglang | no-radix | 18 | 36 | 10 | 480.45 ± 0.47 | 74.9 | 112.4 |
| sglang | no-radix | 128 | 36 | 10 | 480.87 ± 0.28 | 74.9 | 341.0 |
| sglang | no-radix | 512 | 36 | 10 | 498.03 ± 0.35 | 72.3 | 1100.3 |
| sglang | no-radix | 2048 | 36 | 3 | 588.13 ± 1.01 | 61.2 | 3543.4 |
| vllm | cuda-graph | 18 | 36 | 10 | 461.75 ± 0.08 | 78.0 | 116.9 |
| vllm | cuda-graph | 128 | 36 | 10 | 463.44 ± 0.09 | 77.7 | 353.9 |
| vllm | cuda-graph | 512 | 36 | 10 | 484.82 ± 1.65 | 74.3 | 1130.3 |
| vllm | cuda-graph | 2048 | 36 | 3 | 577.34 ± 1.98 | 62.4 | 3609.6 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 18 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 128 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 512 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 128 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 512 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 16.45 | 1094.5 | 442.47 | 81.4 |
| grout | default | 128 | 36 | 16.47 | 7770.8 | 439.64 | 81.9 |
| grout | default | 512 | 36 | 29.19 | 17540.6 | 451.48 | 79.7 |
| grout | default | 2048 | 36 | 113.60 | 18027.7 | 472.17 | 76.2 |
| sglang | no-radix | 18 | 36 | 27.51 | 654.2 | — | — |
| sglang | no-radix | 128 | 36 | 27.91 | 4586.3 | — | — |
| sglang | no-radix | 512 | 36 | 41.22 | 12421.4 | — | — |
| sglang | no-radix | 2048 | 36 | 120.21 | 17036.3 | — | — |
| vllm | cuda-graph | 18 | 36 | 16.29 | 1105.2 | — | — |
| vllm | cuda-graph | 128 | 36 | 17.78 | 7198.7 | — | — |
| vllm | cuda-graph | 512 | 36 | 32.11 | 15943.2 | — | — |
| vllm | cuda-graph | 2048 | 36 | 112.23 | 18247.6 | — | — |
