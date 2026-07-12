# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 450.23 ± 0.23 | 80.0 | 119.9 |
| grout | default | 128 | 36 | 10 | 452.34 ± 0.17 | 79.6 | 362.6 |
| grout | default | 512 | 36 | 10 | 473.14 ± 0.43 | 76.1 | 1158.2 |
| grout | default | 2048 | 36 | 3 | 575.05 ± 1.39 | 62.6 | 3624.0 |
| grout | default | 8192 | 36 | 3 | 997.72 ± 0.46 | 36.1 | 8246.8 |

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
| grout | default | 18 | 36 | 15.25 | 1180.1 | 434.96 | 82.8 |
| grout | default | 128 | 36 | 16.69 | 7670.4 | 435.60 | 82.6 |
| grout | default | 512 | 36 | 31.09 | 16470.7 | 442.06 | 81.4 |
| grout | default | 2048 | 36 | 121.91 | 16799.1 | 453.91 | 79.3 |
| grout | default | 8192 | 36 | 544.12 | 15055.6 | 453.39 | 79.4 |
