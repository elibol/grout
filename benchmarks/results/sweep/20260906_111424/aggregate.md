# Paper Sweep — Median + IQR

n reps = 10 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 462.78 ± 0.03 | 77.8 | 116.7 |
| grout | default | 128 | 36 | 10 | 459.72 ± 0.03 | 78.3 | 356.7 |
| grout | default | 512 | 36 | 10 | 477.73 ± 0.27 | 75.4 | 1147.1 |
| grout | default | 2048 | 36 | 10 | 567.07 ± 2.06 | 63.5 | 3675.0 |
| grout | default | 8192 | 36 | 10 | 936.88 ± 0.96 | 38.4 | 8782.3 |

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
| grout | default | 18 | 36 | 15.78 | 1141.0 | 446.99 | 80.5 |
| grout | default | 128 | 36 | 16.13 | 7934.8 | 443.58 | 81.2 |
| grout | default | 512 | 36 | 27.48 | 18634.8 | 450.27 | 80.0 |
| grout | default | 2048 | 36 | 104.28 | 19639.1 | 462.41 | 77.9 |
| grout | default | 8192 | 36 | 474.35 | 17269.9 | 462.85 | 77.8 |
