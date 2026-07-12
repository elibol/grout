# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request-Level Upper Bound

The roofline columns use an idealized batch-1 request-level upper bound with zero software overhead: `tg / (T_prefill_ideal + T_decode_roof)`. Decode uses FP16 weight bytes plus average KV-cache bytes. The table reports both nominal-bandwidth and effective-bandwidth roof fractions; nominal is the hardware-spec comparison, while effective is just the configured bandwidth fraction or explicit bandwidth override. Prefill uses `2 * P * pp / FLOPs_peak` only to match the request-level metric boundary.

Parameters: `BW_nominal=8000.0 GB/s`, `BW_eff=6800.0 GB/s` (85% of 8000 GB/s), `W=65.52 GB`, `KV_step=0.262 MB/context-token`, `prefill_peak=417.8 TFLOP/s`, `alpha=0`, model=Qwen3-32B.

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 450.64 ± 0.17 | 79.9 | 120.9 | 66.1% | 102.9 | 77.6% | 119.8 |
| grout | default | 18 | 128 | 10 | 1589.21 ± 0.10 | 80.5 | 121.7 | 66.2% | 103.5 | 77.8% | 91.9 |
| grout | default | 18 | 512 | 10 | 6323.86 ± 0.20 | 81.0 | 121.9 | 66.4% | 103.6 | 78.1% | 83.8 |
| grout | default | 18 | 2048 | 3 | 25394.19 ± 0.88 | 80.6 | 121.6 | 66.3% | 103.3 | 78.0% | 81.4 |
| grout | default | 18 | 8192 | 3 | 103290.79 ± 4.17 | 79.3 | 120.1 | 66.0% | 102.1 | 77.7% | 79.5 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 5 | -133.03 | 12.6158 | 79.3 |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 15.63 | 1151.3 | 434.98 | 82.8 |
| grout | default | 18 | 128 | 15.47 | 1163.3 | 1573.74 | 81.3 |
| grout | default | 18 | 512 | 15.47 | 1163.6 | 6308.39 | 81.2 |
| grout | default | 18 | 2048 | 15.68 | 1148.2 | 25378.41 | 80.7 |
| grout | default | 18 | 8192 | 15.72 | 1144.7 | 103275.02 | 79.3 |
