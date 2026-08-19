# Paper Sweep — Median + IQR

n reps = 10 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request-Level Upper Bound

The roofline columns use an idealized batch-1 request-level upper bound with zero software overhead: `tg / (T_prefill_ideal + T_decode_roof)`. Decode uses FP16 weight bytes plus average KV-cache bytes. The table reports both nominal-bandwidth and effective-bandwidth roof fractions; nominal is the hardware-spec comparison, while effective is just the configured bandwidth fraction or explicit bandwidth override. Prefill uses `2 * P * pp / FLOPs_peak` only to match the request-level metric boundary.

Parameters: `BW_nominal=8000.0 GB/s`, `BW_eff=6800.0 GB/s` (85% of 8000 GB/s), `W=65.52 GB`, `KV_step=0.262 MB/context-token`, `prefill_peak=417.8 TFLOP/s`, `alpha=0`, model=<model>.

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 462.44 ± 0.05 | 77.8 | 120.9 | 64.4% | 102.9 | 75.6% | 116.8 |
| grout | default | 18 | 128 | 10 | 1633.21 ± 0.14 | 78.4 | 121.7 | 64.4% | 103.5 | 75.7% | 89.4 |
| grout | default | 18 | 512 | 10 | 6488.72 ± 0.18 | 78.9 | 121.9 | 64.7% | 103.6 | 76.2% | 81.7 |
| vllm | cuda-graph | 18 | 36 | 10 | 468.13 ± 0.12 | 76.9 | 120.9 | 63.6% | 102.9 | 74.7% | 115.4 |
| vllm | cuda-graph | 18 | 128 | 10 | 1656.12 ± 0.18 | 77.3 | 121.7 | 63.5% | 103.5 | 74.7% | 88.2 |
| vllm | cuda-graph | 18 | 512 | 10 | 6618.23 ± 0.82 | 77.4 | 121.9 | 63.5% | 103.6 | 74.7% | 80.1 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 3 | 9.72 | 12.6556 | 79.0 |
| vllm | cuda-graph | 18 | 3 | 2.65 | 12.9209 | 77.4 |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 16.53 | 1088.9 | 445.89 | 80.7 |
| grout | default | 18 | 128 | 16.69 | 1078.5 | 1616.43 | 79.2 |
| grout | default | 18 | 512 | 16.64 | 1081.9 | 6471.99 | 79.1 |
| vllm | cuda-graph | 18 | 36 | 16.39 | 1098.0 | — | — |
| vllm | cuda-graph | 18 | 128 | 16.53 | 1089.2 | — | — |
| vllm | cuda-graph | 18 | 512 | 16.56 | 1087.1 | — | — |
