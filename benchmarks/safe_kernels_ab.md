# Safe-kernel migration — A/B validation protocol

Branch `safe-kernels` ports grout kernels from the unsafe partition API
(`unsafe fn`, `unchecked_accesses=true`, `partition_mut` + raw-index stores)
to the safe mapped-partition API (`MappedPartitionMut` + `iter_indices()`,
bounds-checked loads, `unchecked_accesses=false`). The cutile paper's §5.1
safety-overhead experiment measured this API at parity with raw-pointer
kernels (GEMM within 0.3%, elementwise at parity or better), so the
acceptance bar here is **no regression, identical output**.

## Toggle

`GROUT_UNSAFE_KERNELS=1` selects the legacy kernels; unset/0 selects the
safe ports (default). The legacy kernels stay in-tree until each port is
validated on GPU.

## Ported so far

| kernel | safe variant | paths switched | still legacy |
|---|---|---|---|
| `rms_norm_f16` | `rms_norm_mapped_f16` | eager / StepGraph prefill (all per-layer input norms + final norm), warmup | decode-graph prime/capture (layer-0 norm) |
| `qk_norm_f16` | `qk_norm_mapped_f16` | warmup | unfused decode-graph path (non-default) |

Design note: both ports use one mapped index per row on a single
`[1, next_pow2(N)]` tile with tile-IR masking of the overhang (the
OOB sum-of-squares contribution is zero) — the same schedule the
`add_rms_norm` BS=2048/4096 sweeps already validated as fastest. This
means the hidden-size RMS tile changes from the legacy BS=512 loop to a
single BS=4096 masked tile; the A/B below is what confirms that on
sm_120.

## Not portable on cutile 0.2.0 (API gaps, verified in compiler source)

- `iter_indices()` / `MappedPartitionMut` are **rank-2 only**
  (`cutile-compiler .. compile_expression.rs`: "currently supports rank-2"),
  which blocks the rank-3-output kernels: `rope_seq*`, `kv_cache_update_seq*`,
  all `flash_attn_*` / `fmha_*`, `splitk_reduce_merge`, `qk_rope_dynpos_f16`.
- Mapped stores reject indices minted by a different partition's map
  (`compile_intrinsic.rs`: "store requires an index produced by this
  partition's iter_indices()"), which blocks dual-output kernels:
  `add_rms_norm_f16` (out + residual_out).

Both are concrete cutile-rs feature requests (rank-3 index schedules;
shared/broadcast partition maps), not grout limitations.

## A/B protocol (RTX 5090)

1. Correctness — identical output text, safe vs legacy:

   ```bash
   cargo build --release
   GROUT_UNSAFE_KERNELS=1 ./target/release/grout --model ../hf_models/qwen3_4b \
     --prompt "Hello, how are you?" --max-new-tokens 64 > /tmp/legacy.txt
   ./target/release/grout --model ../hf_models/qwen3_4b \
     --prompt "Hello, how are you?" --max-new-tokens 64 > /tmp/safe.txt
   diff <(grep -v 't/s' /tmp/legacy.txt) <(grep -v 't/s' /tmp/safe.txt) && echo "OUTPUT MATCH"
   ```

2. Prefill perf (RmsNorm runs 37×/step in prefill — the sensitive path):

   ```bash
   GROUT_UNSAFE_KERNELS=1 ./benchmarks/sweep_pp_sm120.sh   # baseline
   ./benchmarks/sweep_pp_sm120.sh                          # safe kernels
   ```

   Compare `prefill_ms` per pp in the two `aggregate.md`s; bar: within noise.

3. Per-op attribution if (2) shows a delta:

   ```bash
   GROUT_PROFILE_OPS=1 GROUT_PROFILE_SYNC_OPS=1 ./target/release/grout_bench \
     --model ../hf_models/qwen3_4b --prompt-file <pp_2048.txt> --raw-prompt \
     --max-new-tokens 0 --reps 3 --warmup-reps 1 --profile
   ```

   run with and without `GROUT_UNSAFE_KERNELS=1`; compare the `RmsNorm` row.

Once validated: switch the decode-graph sites (TODO markers in
`src/model.rs`), delete the legacy kernels, and drop the toggle.
