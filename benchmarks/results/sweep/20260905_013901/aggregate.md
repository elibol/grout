# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 459.79 ± 0.06 | 78.3 | 117.4 |
| grout | default | 128 | 36 | 10 | 457.05 ± 0.05 | 78.8 | 358.8 |
| grout | default | 512 | 36 | 10 | 476.65 ± 0.10 | 75.5 | 1149.7 |
| grout | default | 2048 | 36 | 3 | 576.55 ± 1.25 | 62.4 | 3614.6 |
| grout | default | 8192 | 36 | 3 | 966.72 ± 1.00 | 37.2 | 8511.3 |
| sglang | no-radix | 18 | 36 | 10 | 468.51 ± 0.23 | 76.8 | 115.3 |
| sglang | no-radix | 128 | 36 | 10 | 469.36 ± 0.24 | 76.7 | 349.4 |
| sglang | no-radix | 512 | 36 | 10 | 485.36 ± 1.13 | 74.2 | 1129.1 |
| sglang | no-radix | 2048 | 36 | 3 | 574.82 ± 1.24 | 62.6 | 3625.5 |
| sglang | no-radix | 8192 | 36 | 3 | 961.61 ± 0.93 | 37.4 | 8556.5 |
| vllm | cuda-graph | 18 | 36 | 10 | 456.17 ± 0.08 | 78.9 | 118.4 |
| vllm | cuda-graph | 128 | 36 | 10 | 457.84 ± 0.03 | 78.6 | 358.2 |
| vllm | cuda-graph | 512 | 36 | 10 | 476.73 ± 1.79 | 75.5 | 1149.5 |
| vllm | cuda-graph | 2048 | 36 | 3 | 571.93 ± 0.07 | 62.9 | 3643.8 |
| vllm | cuda-graph | 8192 | 36 | 3 | 946.26 ± 2.39 | 38.0 | 8695.3 |

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
| grout | default | 18 | 36 | 15.68 | 1147.6 | 444.05 | 81.1 |
| grout | default | 128 | 36 | 16.01 | 7995.0 | 440.98 | 81.6 |
| grout | default | 512 | 36 | 28.23 | 18138.7 | 448.44 | 80.3 |
| grout | default | 2048 | 36 | 113.48 | 18047.7 | 464.01 | 77.6 |
| grout | default | 8192 | 36 | 501.23 | 16343.7 | 464.02 | 77.6 |
| sglang | no-radix | 18 | 36 | 26.39 | 682.0 | — | — |
| sglang | no-radix | 128 | 36 | 27.09 | 4725.4 | — | — |
| sglang | no-radix | 512 | 36 | 39.75 | 12881.7 | — | — |
| sglang | no-radix | 2048 | 36 | 120.38 | 17013.0 | — | — |
| sglang | no-radix | 8192 | 36 | 496.15 | 16511.3 | — | — |
| vllm | cuda-graph | 18 | 36 | 16.06 | 1120.9 | — | — |
| vllm | cuda-graph | 128 | 36 | 17.43 | 7344.2 | — | — |
| vllm | cuda-graph | 512 | 36 | 30.69 | 16681.5 | — | — |
| vllm | cuda-graph | 2048 | 36 | 111.21 | 18414.8 | — | — |
| vllm | cuda-graph | 8192 | 36 | 474.67 | 17258.2 | — | — |
