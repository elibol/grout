# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 462.09 ± 0.03 | 77.9 | 116.9 |
| grout | default | 128 | 36 | 10 | 460.98 ± 0.10 | 78.1 | 355.8 |
| grout | default | 512 | 36 | 10 | 484.24 ± 0.77 | 74.3 | 1131.7 |
| grout | default | 2048 | 36 | 3 | 582.01 ± 0.44 | 61.9 | 3580.7 |
| vllm | cuda-graph | 18 | 36 | 10 | 468.26 ± 0.11 | 76.9 | 115.3 |
| vllm | cuda-graph | 128 | 36 | 10 | 469.83 ± 0.19 | 76.6 | 349.1 |
| vllm | cuda-graph | 512 | 36 | 10 | 493.19 ± 0.98 | 73.0 | 1111.1 |
| vllm | cuda-graph | 2048 | 36 | 3 | 592.26 ± 2.20 | 60.8 | 3518.7 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 128 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 512 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 16.33 | 1102.5 | 445.77 | 80.8 |
| grout | default | 128 | 36 | 16.39 | 7810.4 | 444.59 | 81.0 |
| grout | default | 512 | 36 | 29.71 | 17230.4 | 454.47 | 79.2 |
| grout | default | 2048 | 36 | 117.65 | 17407.7 | 465.57 | 77.3 |
| vllm | cuda-graph | 18 | 36 | 16.53 | 1089.2 | — | — |
| vllm | cuda-graph | 128 | 36 | 17.62 | 7266.0 | — | — |
| vllm | cuda-graph | 512 | 36 | 31.93 | 16034.1 | — | — |
| vllm | cuda-graph | 2048 | 36 | 114.61 | 17869.7 | — | — |
