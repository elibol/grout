# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request-Level Upper Bound

The roofline columns use an idealized batch-1 request-level upper bound with zero software overhead: `tg / (T_prefill_ideal + T_decode_roof)`. Decode uses FP16 weight bytes plus average KV-cache bytes. The table reports both nominal-bandwidth and effective-bandwidth roof fractions; nominal is the hardware-spec comparison, while effective is just the configured bandwidth fraction or explicit bandwidth override. Prefill uses `2 * P * pp / FLOPs_peak` only to match the request-level metric boundary.

Parameters: `BW_nominal=1792.0 GB/s`, `BW_eff=1523.2 GB/s` (85% of 1792 GB/s), `W=8.05 GB`, `KV_step=0.147 MB/context-token`, `prefill_peak=417.8 TFLOP/s`, `alpha=0`, model=/home/elibol/dev/grout/../hf_models/qwen3_4b.

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | nominal roof | % nominal | effective roof | % effective | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 210.02 ± 0.10 | 171.4 | 222.1 | 77.2% | 188.9 | 90.8% | 257.1 |
| grout | default | 18 | 128 | 10 | 735.20 ± 0.30 | 174.1 | 222.3 | 78.3% | 188.9 | 92.1% | 198.6 |
| grout | default | 18 | 512 | 10 | 2953.24 ± 0.50 | 173.4 | 221.6 | 78.2% | 188.4 | 92.0% | 179.5 |
| grout | default | 18 | 2048 | 3 | 11960.59 ± 2.07 | 171.2 | 218.6 | 78.3% | 185.8 | 92.2% | 172.7 |
| grout | default | 18 | 8192 | 3 | 50459.73 ± 82.68 | 162.3 | 207.1 | 78.4% | 176.0 | 92.2% | 162.7 |
| sglang | no-radix | 18 | 36 | 10 | 221.73 ± 0.26 | 162.4 | 222.1 | 73.1% | 188.9 | 86.0% | 243.5 |
| sglang | no-radix | 18 | 128 | 10 | 781.08 ± 2.12 | 163.9 | 222.3 | 73.7% | 188.9 | 86.7% | 186.9 |
| sglang | no-radix | 18 | 512 | 10 | 3017.14 ± 1.27 | 169.7 | 221.6 | 76.6% | 188.4 | 90.1% | 175.7 |
| sglang | no-radix | 18 | 2048 | 3 | 12324.11 ± 1.82 | 166.2 | 218.6 | 76.0% | 185.8 | 89.5% | 167.6 |
| sglang | no-radix | 18 | 8192 | 3 | 51710.35 ± 2.24 | 158.4 | 207.1 | 76.5% | 176.0 | 90.0% | 158.8 |
| vllm | cuda-graph | 18 | 36 | 10 | 220.04 ± 0.05 | 163.6 | 222.1 | 73.7% | 188.9 | 86.6% | 245.4 |
| vllm | cuda-graph | 18 | 128 | 10 | 779.87 ± 0.08 | 164.1 | 222.3 | 73.8% | 188.9 | 86.9% | 187.2 |
| vllm | cuda-graph | 18 | 512 | 10 | 3120.66 ± 1.01 | 164.1 | 221.6 | 74.0% | 188.4 | 87.1% | 169.8 |
| vllm | cuda-graph | 18 | 2048 | 3 | 12588.76 ± 1.07 | 162.7 | 218.6 | 74.4% | 185.8 | 87.6% | 164.1 |
| vllm | cuda-graph | 18 | 8192 | 3 | 52913.72 ± 1.84 | 154.8 | 207.1 | 74.8% | 176.0 | 87.9% | 155.2 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 5 | -209.48 | 6.1713 | 162.0 |
| sglang | no-radix | 18 | 5 | -193.34 | 6.3229 | 158.2 |
| vllm | cuda-graph | 18 | 5 | -200.94 | 6.4701 | 154.6 |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 6.76 | 2664.5 | 203.24 | 177.1 |
| grout | default | 18 | 128 | 6.82 | 2641.0 | 728.39 | 175.7 |
| grout | default | 18 | 512 | 6.86 | 2624.9 | 2946.38 | 173.8 |
| grout | default | 18 | 2048 | 6.86 | 2623.1 | 11953.30 | 171.3 |
| grout | default | 18 | 8192 | 6.86 | 2624.7 | 50452.87 | 162.4 |
| sglang | no-radix | 18 | 36 | 13.69 | 1315.1 | — | — |
| sglang | no-radix | 18 | 128 | 14.00 | 1285.8 | — | — |
| sglang | no-radix | 18 | 512 | 13.64 | 1319.8 | — | — |
| sglang | no-radix | 18 | 2048 | 13.88 | 1297.2 | — | — |
| sglang | no-radix | 18 | 8192 | 14.23 | 1264.8 | — | — |
| vllm | cuda-graph | 18 | 36 | 7.32 | 2458.7 | — | — |
| vllm | cuda-graph | 18 | 128 | 7.50 | 2401.3 | — | — |
| vllm | cuda-graph | 18 | 512 | 7.72 | 2330.8 | — | — |
| vllm | cuda-graph | 18 | 2048 | 7.82 | 2302.9 | — | — |
| vllm | cuda-graph | 18 | 8192 | 7.88 | 2283.3 | — | — |
