# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | pp2048-v030 | 2048 | 36 | 3 | 563.72 ± 0.80 | 63.9 | 3696.9 |
| grout | pp32768-cutile-v030 | 32768 | 36 | 3 | 3258.41 ± 3.29 | 11.0 | 10067.5 |
| grout | tg128-v030 | 18 | 128 | 3 | 1638.33 ± 0.03 | 78.1 | 89.1 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | pp2048-v030 | 2048 | 1 | n/a | n/a | n/a |
| grout | pp32768-cutile-v030 | 32768 | 1 | n/a | n/a | n/a |
| grout | tg128-v030 | 18 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | pp2048-v030 | 2048 | 36 | 103.28 | 19830.5 | 460.42 | 78.2 |
| grout | pp32768-cutile-v030 | 32768 | 36 | 2756.90 | 11885.8 | 501.13 | 71.8 |
| grout | tg128-v030 | 18 | 128 | 16.40 | 1097.7 | 1621.92 | 78.9 |
