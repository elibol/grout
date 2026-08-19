# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 3 | 1679.11 ± 1.37 | 21.4 | 9779.0 |
| grout | default | 32768 | 36 | 3 | 3625.25 ± 3.89 | 9.9 | 9048.8 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1586.57 ± 4.06 | 22.7 | 10349.4 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3169.33 ± 0.59 | 11.4 | 10350.4 |

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
| grout | default | 16384 | 36 | 1202.98 | 13619.5 | 478.00 | 75.3 |
| grout | default | 32768 | 36 | 3123.57 | 10490.6 | 509.74 | 70.6 |
| vllm | cuda-graph | 16384 | 36 | 1104.67 | 14831.6 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2671.20 | 12267.1 | — | — |
