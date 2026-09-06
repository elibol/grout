# Paper Sweep — Median + IQR

n reps = 10 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 462.83 ± 0.01 | 77.8 | 116.7 |
| grout | default | 128 | 36 | 10 | 459.71 ± 0.03 | 78.3 | 356.7 |
| grout | default | 512 | 36 | 10 | 477.31 ± 0.21 | 75.4 | 1148.1 |
| grout | default | 2048 | 36 | 10 | 567.72 ± 1.94 | 63.4 | 3670.8 |
| grout | default | 8192 | 36 | 10 | 938.57 ± 3.38 | 38.4 | 8766.5 |

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
| grout | default | 18 | 36 | 15.78 | 1140.7 | 447.04 | 80.5 |
| grout | default | 128 | 36 | 16.13 | 7935.3 | 443.56 | 81.2 |
| grout | default | 512 | 36 | 27.44 | 18656.5 | 449.85 | 80.0 |
| grout | default | 2048 | 36 | 104.00 | 19691.4 | 462.92 | 77.8 |
| grout | default | 8192 | 36 | 473.13 | 17314.4 | 464.28 | 77.5 |
