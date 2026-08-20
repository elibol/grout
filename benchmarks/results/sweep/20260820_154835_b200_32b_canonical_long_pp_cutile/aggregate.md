# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 3 | 1641.31 ± 2.34 | 21.9 | 10004.2 |
| grout | default | 32768 | 36 | 3 | 3479.52 ± 10.10 | 10.3 | 9427.7 |
| sglang | no-radix | 16384 | 36 | 3 | 1596.40 ± 0.94 | 22.6 | 10285.7 |
| sglang | no-radix | 32768 | 36 | 3 | 3181.57 ± 3.48 | 11.3 | 10310.6 |
| vllm | cuda-graph | 16384 | 36 | 3 | 1565.88 ± 2.55 | 23.0 | 10486.1 |
| vllm | cuda-graph | 32768 | 36 | 3 | 3112.56 ± 1.17 | 11.6 | 10539.2 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 16384 | 1 | n/a | n/a | n/a |
| grout | default | 32768 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 16384 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 32768 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 16384 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 32768 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 16384 | 36 | 1165.14 | 14061.8 | 477.27 | 75.4 |
| grout | default | 32768 | 36 | 2985.51 | 10975.7 | 494.00 | 72.9 |
| sglang | no-radix | 16384 | 36 | 1109.14 | 14771.8 | — | — |
| sglang | no-radix | 32768 | 36 | 2676.21 | 12244.2 | — | — |
| vllm | cuda-graph | 16384 | 36 | 1086.51 | 15079.4 | — | — |
| vllm | cuda-graph | 32768 | 36 | 2632.84 | 12445.9 | — | — |
