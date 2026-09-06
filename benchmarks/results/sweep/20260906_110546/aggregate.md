# Paper Sweep — Median + IQR

n reps = 10 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 462.75 ± 0.03 | 77.8 | 116.7 |
| grout | default | 128 | 36 | 10 | 461.38 ± 0.02 | 78.0 | 355.5 |
| grout | default | 512 | 36 | 10 | 476.56 ± 0.20 | 75.5 | 1149.9 |
| grout | default | 2048 | 36 | 10 | 566.95 ± 2.11 | 63.5 | 3675.8 |
| grout | default | 8192 | 36 | 10 | 922.55 ± 2.95 | 39.0 | 8918.8 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| grout | default | 8192 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 18.20 | 989.0 | 444.54 | 81.0 |
| grout | default | 128 | 36 | 18.39 | 6959.5 | 442.98 | 81.3 |
| grout | default | 512 | 36 | 29.32 | 17462.2 | 447.19 | 80.5 |
| grout | default | 2048 | 36 | 106.07 | 19307.6 | 460.92 | 78.1 |
| grout | default | 8192 | 36 | 457.68 | 17898.9 | 463.97 | 77.6 |
