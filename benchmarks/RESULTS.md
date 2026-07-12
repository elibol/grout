# Benchmark Results

Committed B200 Qwen3-32B result bundle:

```text
benchmarks/results/final/b200_qwen3_32b/
```

`benchmarks/results/` is ignored by default; this final bundle is the
paper handoff data. Timestamped sweep directories remain local scratch unless
they are explicitly added.

## Bundles

| Bundle | Sweep |
|---|---|
| `tg_sweep_pp18_8k_safe/` | `pp=18`, `tg={36,128,512,2048,8192}` |
| `pp_sweep_tg36_8k_safe/` | `pp={18,128,512,2048,8192}`, `tg=36` |

Each bundle contains `run.jsonl`, `aggregate.csv`, and `aggregate.md`.
Machine-specific console summaries are intentionally excluded.

## Headline Numbers

Decode sweep, `pp=18`:

| engine | tg=36 | tg=128 | tg=512 | tg=2048 | tg=8192 |
|---|---:|---:|---:|---:|---:|
| grout | 79.9 | 80.5 | 81.0 | 80.6 | 79.3 |

Values are median `request_gen_tps`. Direct phase-only decode throughput is:

| tg | 36 | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|
| decode tok/s | 82.8 | 81.3 | 81.2 | 80.7 | 79.3 |

Prefill sweep, `tg=36`:

| engine | pp=18 | pp=128 | pp=512 | pp=2048 | pp=8192 |
|---|---:|---:|---:|---:|---:|
| grout | 80.0 | 79.6 | 76.1 | 62.6 | 36.1 |

Direct prefill timings:

| pp | 18 | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|
| prefill ms | 15.25 | 16.69 | 31.09 | 121.91 | 544.12 |

These are Grout-only safe-kernel results. Use the bundle `aggregate.csv`
files for full median, IQR, phase-timing, and roofline columns.
