# B200 Qwen3-32B Safe-Kernel Results

Grout-only batch-1 inference results on one NVIDIA B200 (`sm_100`) using
FP16 Qwen3-32B weights and driver-controlled GPU clocks.

## Provenance

- Grout revision: `2451b78`
- cutile-rs revision: `baa61d8`
- Sampling: greedy
- Prefix cache: disabled
- EOS: ignored to enforce exact generation lengths
- Aggregation: median and IQR, with no outlier filtering
- Repetitions: 10 for lengths through 512; 3 for 2048 and 8192
- Warmups: 3 per canonical cell, excluding JIT and graph capture
- Baselines: intentionally omitted

The prefill profile explicitly disables the raw-pointer LPT kernel and uses
the safe mapped attention kernels at every reported prompt length.

## Bundles

- `pp_sweep_tg36_8k_safe/`: prompt lengths 18, 128, 512, 2048, and 8192 at
  a fixed generation length of 36.
- `tg_sweep_pp18_8k_safe/`: generation lengths 36, 128, 512, 2048, and 8192
  at a fixed prompt length of 18.

Each bundle contains the raw per-repetition `run.jsonl` source of truth and
the generated `aggregate.csv` and `aggregate.md` tables. Console summaries
are intentionally excluded because local runs include machine-specific paths
and environment diagnostics.
