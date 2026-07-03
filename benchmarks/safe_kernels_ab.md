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

Built against cutile-rs branch `feat/mapped-partition-bounded-pipelined`
(via `[patch.crates-io]` path deps; see Cargo.toml). That branch added
the two APIs the previous blockers called for: sub-range mapped
iteration (`iter_indices_within[_with]`, runtime starts/lengths) and
bounds-checked pipelined loads (`Partition::load_pipelined::<LATENCY>`).

| kernel | safe variant | paths switched | still legacy |
|---|---|---|---|
| `rms_norm_f16` | `rms_norm_mapped_f16` | eager / StepGraph prefill (all per-layer input norms + final norm), warmup | decode-graph prime/capture (layer-0 norm) |
| `qk_norm_f16` | `qk_norm_mapped_f16` | warmup | unfused decode-graph path (non-default) |
| `add_rms_norm_f16` | `add_rms_norm_mapped_f16` (dual outputs share one index stream via `iter_indices_with`) | eager / StepGraph prefill (36 calls/step), warmup | — |
| `kv_cache_update_seq_f16` | `kv_cache_update_seq_mapped_f16` (per-token [1, 1, chunk] tiles; `iter_indices_within_with` bounds the seq axis to the `seq_len` written tokens and brands one index stream for both cache stores) | prefill KV update (`PositionInput::Host`) | — |
| `kv_cache_update_seq_dynpos_f16` | `kv_cache_update_seq_dynpos_mapped_f16` (sub-range start = position scalar read from device memory) | decode KV update (`PositionInput::Device`), decode-graph prime/record unfused sites | — |
| `fmha_decode_gqa_split` | `fmha_decode_gqa_split_mapped` — **partially safe**: mapped att store, `load_pipelined` K/V loads, `unchecked_accesses=false`, but the fn stays `unsafe` for the lse output (see blockers) | decode-graph prime/record (default split-KV attention) | — |
| `splitk_reduce_merge` | `splitk_reduce_merge_mapped` (fully safe; pipelined bounds-checked scratch loads) | decode-graph prime/record | — |
| `fmha_prefill_causal` | `fmha_prefill_causal_mapped` | default prefill attention (`GROUT_FMHA_PREFILL`) | — |
| `fmha_prefill_gqa` | `fmha_prefill_gqa_mapped` | GQA prefill attention arm (`GROUT_FMHA_PREFILL_GQA`) | — |
| `flash_attn_causal_seq_f16` | `flash_attn_causal_seq_mapped_f16` | attend fallback arm (prefill kernels disabled) | — |
| `flash_attn_causal_seq_dynpos_f16` | `flash_attn_causal_seq_dynpos_mapped_f16` | decode-graph prime/record non-split attention fallback | — |
| `fmha_causal` | `fmha_causal_mapped` | attend `PositionInput::Device` arm | — |

(`rope_seq*`, `add_2d`, `silu_mul`, `embedding*`, `argmax*`, `gather_row`
were already safe `fn`s and need no port.)

Design notes:

- The norm ports use one mapped index per row on a single
  `[1, next_pow2(N)]` tile with tile-IR masking of the overhang (the
  OOB sum-of-squares contribution is zero) — the same schedule the
  `add_rms_norm` BS=2048/4096 sweeps already validated as fastest. This
  means the hidden-size norm tiles change from the legacy BS=512/2048
  loops to a single BS=4096 masked tile; the A/B below is what confirms
  that on sm_120.
- The attention ports keep the legacy schedule exactly: map `[1, 1, 1]`
  with `num_tile_blocks` = the legacy grid size, so each CTA gets one
  mapped index; tile-block ids become `index.coords()` and the
  `load_view_tko(..., Some(LATENCY), ...)` pipelined loads become
  `load_pipelined::<LATENCY>` with identical hints. A/B bar is again
  no-regression.
- The KV-cache ports change the store granularity from per-CTA token
  loops to per-token mapped tiles bounded by `iter_indices_within_with`;
  `num_tile_blocks` mirrors the legacy CTA counts so occupancy is
  unchanged.
- In safe mode the `fmha_lse_partial` scratch is allocated one-row-per-CTA
  (`[kv_heads * splits, GROUP]`, same memory layout) so the split
  kernel's legacy per-CTA lse tile view lines up with the mapped 1-D
  grid; the merge kernel reads it through a `[kv_heads, splits * GROUP]`
  view.

## Remaining blockers (cutile API follow-ups)

- **Mixed-shape shared maps**: `iter_indices_with` requires both outputs
  to share tile and map shapes. `fmha_decode_gqa_split_mapped`'s second
  output (lse, f32 `[1, GROUP]` rank-2) cannot share the att output's
  `[1, GROUP, D]` rank-3 index stream, so the kernel keeps an
  `unsafe fn` signature with a legacy per-CTA tile view for the lse
  store (everything else — mapped att store, checked pipelined loads,
  `unchecked_accesses=false` — is on the safe API). A mixed-shape
  shared-map API would make it fully safe.
- The LPT/swizzled prefill kernels (`fmha_prefill_gqa_lpt*`) use raw
  device pointers and a hand-swizzled persistent schedule; they stay
  legacy until mapped partitions support custom swizzle schedules.
- The raw-pointer fused kernels (`qk_norm_rope_kv_*_raw`,
  `add_rms_norm_decode_raw`, `flash_decode.rs`) are Tier 2:
  A/B against typed equivalents first, then port or delete.

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
