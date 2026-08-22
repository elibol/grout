# Paper Sweep — Median + IQR

n reps = 3 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request-Level Upper Bound

The roofline columns use an idealized batch-1 request-level upper bound with zero software overhead: `tg / (T_prefill_ideal + T_decode_roof)`. Decode uses FP16 weight bytes plus average KV-cache bytes. The table reports both nominal-bandwidth and effective-bandwidth roof fractions; nominal is the hardware-spec comparison, while effective is just the configured bandwidth fraction or explicit bandwidth override. Prefill uses `2 * P * pp / FLOPs_peak` only to match the request-level metric boundary.

Parameters: `BW_nominal=8000.0 GB/s`, `BW_eff=6800.0 GB/s` (85% of 8000 GB/s), `W=65.52 GB`, `KV_step=0.262 MB/context-token`, `prefill_peak=417.8 TFLOP/s`, `alpha=0`, model=Qwen3-32B.

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 128 | 3 | 1638.54 ± 0.04 | 78.1 | 121.7 | 64.2% | 103.5 | 75.5% | 89.1 |
| sglang | no-radix | 18 | 128 | 3 | 1670.31 ± 0.85 | 76.6 | 121.7 | 63.0% | 103.5 | 74.0% | 87.4 |
| vllm | cuda-graph | 18 | 128 | 3 | 1640.92 ± 0.03 | 78.0 | 121.7 | 64.1% | 103.5 | 75.4% | 89.0 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 128 | 16.39 | 1098.2 | 1622.11 | 78.9 |
| sglang | no-radix | 18 | 128 | 35.34 | 509.3 | — | — |
| vllm | cuda-graph | 18 | 128 | 15.99 | 1125.8 | — | — |
