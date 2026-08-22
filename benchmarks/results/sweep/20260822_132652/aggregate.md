# Paper Sweep — Median + IQR

n reps vary by cell; see the n column; per-cell tables report medians unless labeled otherwise. The benchmark stdout summaries are means, so they are not expected to match these medians exactly. `decode_fit_*` columns come from a linear fit of e2e_median_ms vs tg for each (engine, variant, pp).

## Request Generation Throughput

| engine | variant | pp | tg | n | e2e_ms (median ± IQR/2) | request_gen_tps (median) | total_tps over e2e (median) |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 10 | 209.84 ± 0.04 | 171.6 | 257.3 |
| grout | default | 128 | 36 | 10 | 208.62 ± 0.04 | 172.6 | 786.1 |
| grout | default | 512 | 36 | 10 | 217.62 ± 0.11 | 165.4 | 2518.2 |
| grout | default | 2048 | 36 | 3 | 259.07 ± 0.11 | 139.0 | 8044.1 |
| grout | default | 8192 | 36 | 3 | 538.56 ± 0.14 | 66.8 | 15277.7 |
| sglang | no-radix | 18 | 36 | 10 | 216.13 ± 0.14 | 166.6 | 249.9 |
| sglang | no-radix | 128 | 36 | 10 | 219.25 ± 0.13 | 164.2 | 748.0 |
| sglang | no-radix | 512 | 36 | 10 | 238.19 ± 0.34 | 151.1 | 2300.7 |
| sglang | no-radix | 2048 | 36 | 3 | 304.87 ± 0.18 | 118.1 | 6835.7 |
| sglang | no-radix | 8192 | 36 | 3 | 668.36 ± 0.44 | 53.9 | 12310.7 |
| vllm | cuda-graph | 18 | 36 | 10 | 220.06 ± 0.17 | 163.6 | 245.4 |
| vllm | cuda-graph | 128 | 36 | 10 | 221.79 ± 0.09 | 162.3 | 739.4 |
| vllm | cuda-graph | 512 | 36 | 10 | 236.36 ± 0.10 | 152.3 | 2318.5 |
| vllm | cuda-graph | 2048 | 36 | 3 | 302.57 ± 0.14 | 119.0 | 6887.6 |
| vllm | cuda-graph | 8192 | 36 | 3 | 650.19 ± 0.50 | 55.4 | 12654.7 |

## Derived Decode From Cross-TG Fit

| engine | variant | pp | n_tg_points | prefill_ms (fit intercept) | decode_ms_per_tok (fit) | decode_fit_tps |
|---|---|---:|---:|---:|---:|---:|
| grout | default | 18 | 1 | n/a | n/a | n/a |
| grout | default | 128 | 1 | n/a | n/a | n/a |
| grout | default | 512 | 1 | n/a | n/a | n/a |
| grout | default | 2048 | 1 | n/a | n/a | n/a |
| grout | default | 8192 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 18 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 128 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 512 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 2048 | 1 | n/a | n/a | n/a |
| sglang | no-radix | 8192 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 18 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 128 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 512 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 2048 | 1 | n/a | n/a | n/a |
| vllm | cuda-graph | 8192 | 1 | n/a | n/a | n/a |

## Direct Phase Timings

Cells with `—` indicate the engine did not emit that same-request phase timer in run.jsonl. prefill_tps is prompt_tokens / prefill_ms; decode_direct_tps is generated_tokens / decode_ms. Note prefill_ms is pure prefill for grout/transformers but TTFT (prefill + 1 decode step) for vllm/sglang.

| engine | variant | pp | tg | prefill_ms (direct/TTFT) | prefill_tps | decode_ms (direct) | decode_direct_tps |
|---|---|---:|---:|---:|---:|---:|---:|
| grout | default | 18 | 36 | 6.71 | 2681.8 | 203.13 | 177.2 |
| grout | default | 128 | 36 | 7.61 | 16829.9 | 201.00 | 179.1 |
| grout | default | 512 | 36 | 14.27 | 35873.2 | 203.30 | 177.1 |
| grout | default | 2048 | 36 | 51.01 | 40148.2 | 208.06 | 173.0 |
| grout | default | 8192 | 36 | 277.82 | 29486.5 | 260.46 | 138.2 |
| sglang | no-radix | 18 | 36 | 13.37 | 1346.4 | — | — |
| sglang | no-radix | 128 | 36 | 14.40 | 8888.1 | — | — |
| sglang | no-radix | 512 | 36 | 29.17 | 17553.4 | — | — |
| sglang | no-radix | 2048 | 36 | 89.98 | 22761.6 | — | — |
| sglang | no-radix | 8192 | 36 | 433.93 | 18878.6 | — | — |
| vllm | cuda-graph | 18 | 36 | 7.33 | 2456.8 | — | — |
| vllm | cuda-graph | 128 | 36 | 8.42 | 15198.3 | — | — |
| vllm | cuda-graph | 512 | 36 | 22.77 | 22487.6 | — | — |
| vllm | cuda-graph | 2048 | 36 | 84.98 | 24098.7 | — | — |
| vllm | cuda-graph | 8192 | 36 | 405.23 | 20215.7 | — | — |
