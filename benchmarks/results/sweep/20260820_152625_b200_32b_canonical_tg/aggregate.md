# Paper Sweep — Median + IQR

n reps = 10 per cell; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request-Level Upper Bound

The roofline columns use an idealized batch-1 request-level upper bound with zero software overhead: `tg / (T_prefill_ideal + T_decode_roof)`. Decode uses FP16 weight bytes plus average KV-cache bytes. The table reports both nominal-bandwidth and effective-bandwidth roof fractions; nominal is the hardware-spec comparison, while effective is just the configured bandwidth fraction or explicit bandwidth override. Prefill uses `2 * P * pp / FLOPs_peak` only to match the request-level metric boundary.

Parameters: `BW_nominal=8000.0 GB/s`, `BW_eff=6800.0 GB/s` (85% of 8000 GB/s), `W=65.52 GB`, `KV_step=0.262 MB/context-token`, `prefill_peak=417.8 TFLOP/s`, `alpha=0`, model=<models>/qwen3_32b.

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 459.13 ± 0.08 | 78.4 | 120.9 | 64.8% | 102.9 | 76.2% | 117.6 |
| grout | default | 18 | 128 | 10 | 1620.72 ± 0.08 | 79.0 | 121.7 | 64.9% | 103.5 | 76.3% | 90.1 |
| grout | default | 18 | 512 | 10 | 6445.89 ± 0.19 | 79.4 | 121.9 | 65.2% | 103.6 | 76.7% | 82.2 |
| sglang | no-radix | 18 | 36 | 10 | 480.34 ± 0.47 | 74.9 | 120.9 | 62.0% | 102.9 | 72.8% | 112.4 |
| sglang | no-radix | 18 | 128 | 10 | 1670.63 ± 0.48 | 76.6 | 121.7 | 62.9% | 103.5 | 74.0% | 87.4 |
| sglang | no-radix | 18 | 512 | 10 | 6645.93 ± 1.45 | 77.0 | 121.9 | 63.2% | 103.6 | 74.4% | 79.7 |
| vllm | cuda-graph | 18 | 36 | 10 | 462.02 ± 0.09 | 77.9 | 120.9 | 64.4% | 102.9 | 75.7% | 116.9 |
| vllm | cuda-graph | 18 | 128 | 10 | 1647.62 ± 1.72 | 77.7 | 121.7 | 63.8% | 103.5 | 75.1% | 88.6 |
| vllm | cuda-graph | 18 | 512 | 10 | 6589.55 ± 1.34 | 77.7 | 121.9 | 63.8% | 103.6 | 75.0% | 80.4 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 3 | 8.61 | 12.5738 | 79.5 |
| sglang | no-radix | 18 | 3 | 13.34 | 12.9540 | 77.2 |
| vllm | cuda-graph | 18 | 3 | -0.76 | 12.8720 | 77.7 |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 16.48 | 1092.4 | 442.65 | 81.3 |
| grout | default | 18 | 128 | 16.59 | 1085.3 | 1604.16 | 79.8 |
| grout | default | 18 | 512 | 16.77 | 1073.2 | 6429.05 | 79.6 |
| sglang | no-radix | 18 | 36 | 27.39 | 657.3 | — | — |
| sglang | no-radix | 18 | 128 | 26.72 | 673.6 | — | — |
| sglang | no-radix | 18 | 512 | 26.80 | 671.5 | — | — |
| vllm | cuda-graph | 18 | 36 | 16.29 | 1104.8 | — | — |
| vllm | cuda-graph | 18 | 128 | 16.45 | 1094.4 | — | — |
| vllm | cuda-graph | 18 | 512 | 16.39 | 1098.0 | — | — |
