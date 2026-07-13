# Safe-kernel migration — completed-migration record

**Status: migration complete.** The safe mapped-partition kernels
(`MappedPartitionMut` + `iter_indices()`, bounds-checked loads,
`unchecked_accesses=false`) are the **only** implementation of the ported
ops; the legacy `unsafe fn` variants and the `GROUT_UNSAFE_KERNELS`
toggle have been removed. The A/B gate opened on prefill best-vs-best
parity and decode parity after declared preconditions (details below).

Built against cutile-rs `feat/mapped-partition-bounded-pipelined`
(via `[patch.crates-io]` path deps; see Cargo.toml), which supplied
sub-range mapped iteration (`iter_indices_within[_with]`, runtime
starts/lengths) and bounds-checked pipelined loads
(`Partition::load_pipelined::<LATENCY>`).

## Migrated kernels

| legacy kernel (deleted) | safe replacement |
|---|---|
| `rms_norm_f16` | `rms_norm_mapped_f16` |
| `qk_norm_f16` | `qk_norm_mapped_f16` |
| `add_rms_norm_f16` | `add_rms_norm_mapped_f16` (dual outputs share one index stream via `iter_indices_with`) |
| `kv_cache_update_seq_f16` | `kv_cache_update_seq_mapped_f16` (`iter_indices_within_with` bounds the seq axis to the written tokens; one branded index stream for both cache stores) |
| `kv_cache_update_seq_dynpos_f16` | `kv_cache_update_seq_dynpos_mapped_f16` (sub-range start = position scalar read from device memory) |
| `fmha_decode_gqa_split` | `fmha_decode_gqa_split_mapped` (partially safe: mapped att store, checked pipelined K/V loads; `unsafe fn` remains for the lse store — mixed tile shapes cannot share a map yet) |
| `splitk_reduce_merge` | `splitk_reduce_merge_mapped` |
| `fmha_prefill_causal` | `fmha_prefill_causal_mapped` |
| `fmha_prefill_gqa` | `fmha_prefill_gqa_mapped` |
| `flash_attn_causal_seq_f16` | `flash_attn_causal_seq_mapped_f16` |
| `flash_attn_causal_seq_dynpos_f16` | `flash_attn_causal_seq_dynpos_mapped_f16` |
| `fmha_causal` | `fmha_causal_mapped` |

`rms_norm_persistent_f16` (documented dead code — its env dispatch was
never wired) was deleted alongside them. `rope_seq*`, `add_2d`,
`silu_mul`, `embedding*`, `argmax*`, `gather_row` were already safe
`fn`s and needed no port.

## Final results (RTX 5090, sm_120)

Correctness: identical output text, safe vs legacy, across the eager,
StepGraph-prefill, and decode-CUDA-graph paths.

**Prefill — best-vs-best parity.** Each form at its own swept tile
optimum (paired, CPU-pinned):

| pp | legacy (ms) | safe (ms) |
|---|---|---|
| 512 | 14.36 | 14.51 |
| 2048 | 52.64 | **52.36** (safe faster) |

The dispatch retune landed BM=64/BN=32 for pp >= 512; after the
compiler-side bounds-check hoisting fix, both forms share the same
optima.

**Decode — parity at full check discharge.** -0.20% paired at tg=2048
(BN=32 / NUM_KV_SPLITS=16), i.e. within noise, measured with the merge
kernel's three residual checks discharged at JIT time — which the tree
preserves via the minimal declared equalities below.

## Paper-relevant findings

1. **Tile/hint configs are per-kernel-form, not per-op.** Carrying one
   form's tuned shapes onto the other silently pessimizes it: a
   fixed-shape A/B measures the tuning mismatch, not the form's cost.
   The safety-overhead parity bar must be **best-vs-best** (each form
   at its own swept optimum). Hedge: the *magnitude* of the
   form-sensitivity is likely transient compiler maturity (For-region
   register allocation); the durable claim is the evaluation protocol,
   not the specific shifted optima.

2. **Bounds checks have a register-footprint cost channel beyond
   placement.** The merge kernel's 3 runtime bounds checks caused
   48 spill ops / STACK:424 under its REG:64 occupancy cap — the cost
   was not where the checks executed but the registers they pinned.
   An experiment that discharged all three at JIT time took the kernel
   to 0 spills / STACK:0 and closed the decode residual, isolating the
   mechanism. The durable statement is mechanism-neutral: JIT-time
   discharge eliminates the register channel; runtime checks carry it
   until codegen stops spilling around them.

## Check-placement policy

The kernels give the compiler enough information to place checks at
JIT time rather than runtime:

1. **Prefer JIT-time checks.** By construction first: the norm kernels
   tie their inputs' index spaces to the output's grid dims
   (`num_tiles(&out, ..)` + `with_bounds`) and load through
   proof-carrying coords minted from the mapped index stream — the
   persistent-gemm pattern — which discharges their cross-tensor
   coordinates with zero declared facts (measured 1/0/0
   discharged/hoisted/in-place on device, at perf parity in paired
   runs). Static shapes, literal coordinates, and same-view origins
   discharge likewise. Where construction is out of reach — currently
   the rank-3 loads in the splitk merge, since `with_bounds`/`coord`
   are rank-2 — a minimal declared dim-equality (launch-validated by
   the generated host launcher) provides the JIT-time discharge
   instead. Extending `with_bounds` past rank 2 would retire those
   facts and bring the attention KV loops under construction too.
2. **Runtime checks are a last resort**, aggressively optimized and
   hoisted out of hot loops (loop-invariant and affine-induction
   indices are checked once in the loop preheader).

Current declared-equality footprint, kept minimal by rule 1's
preference order: rms_norm 0, add_rms_norm 0 (both fully
by-construction), splitk merge 3 (rank-3 loads, beyond
`with_bounds`' current rank-2 reach).

## Verification boundary

Every bounds check in the safe kernel family is one of:

- **discharged at JIT time** — static shapes, literal coordinates,
  same-view branded indices, or a minimal declared (launch-validated)
  equality for cross-tensor coordinates;
- **hoisted** — a runtime check moved out of the hot loop; or
- **a named value-lattice runtime check** — arithmetic-derived
  coordinates the current lattice cannot relate: `row - num_q_rows`
  in qk_norm, `s - pos` in kv_cache, `head / GROUP` in attention.

That is the precise claim; the family is **not** "fully statically
verified".

## Remaining unsafe kernels (Tier 2, unchanged plan)

Raw-pointer / hand-scheduled kernels, to be A/B'd against typed
equivalents and then ported or deleted:

- `qk_norm_rope_kv_prefill_raw`, `qk_norm_rope_kv_decode_raw`
- `add_rms_norm_decode_raw`
- `fmha_prefill_gqa_lpt`
- `flash_decode.rs` (grouped decode attention, opt-in)
- `lm_head_argmax_blocks` (opt-in)

(`qk_rope_dynpos` has since been ported to the safe API;
`flash_attn_f16`/`flash_attn_causal_f16` were deleted as unused.
`fmha_prefill_gqa_lpt_split`, `prefill_splitk_reduce_merge`, and
`group_gemm_nt_desc` are not referenced by the engine — the first two
are dead code, the third is microbench-only.)

## Qwen3-engine kernel census

Counting only kernels the engine actually invokes (model.rs references):
**20 safe / 6 unsafe** (+1 opt-in unsafe in flash_decode.rs). The six:
the three raw fused kernels (default paths; portable once mixed-shape
shared maps land), `fmha_decode_gqa_split_mapped` (unsafe only for its
lse store — same gap), `fmha_prefill_gqa_lpt` (long-prefill profile;
needs custom swizzle schedules), and `lm_head_argmax_blocks` (opt-in
flag; dual f32/u32 outputs — same shared-map gap). With mixed-shape
shared maps, the engine's default paths reach 0 unsafe kernels except
LPT.

Plus the partial case above: `fmha_decode_gqa_split_mapped` keeps an
`unsafe fn` signature for its lse store until mixed-shape shared maps
land in cutile.

## Tracked follow-ups

- tileiras For-region register pressure (SASS/cubin artifacts saved on
  the cutile side).
- The occupancy-hint-not-raising-REG-cap oddity (the entry-level
  occupancy hint capped REG at 64 even when a lower occupancy was
  requested).
- The value-relational lattice as the research follow-up for
  arithmetic-derived coordinates (the three named cases above).
- B200/sm_100 retune with the same per-arm — now single-arm — protocol
  (`sweep_pp_tile.sh` / `sweep_tg_tile.sh`).
- Safe nested subpartitioning (cutile-rs ask). Chunked mapped traversal
  of wide rows fails both ways today: `map([1, num_chunks], rows)` is
  not row-local, and the row-local `map([rows, 1], rows)` persistent
  walk produces catastrophic codegen (RmsNorm REG up to 211 / STACK 224;
  AddRmsNorm STACK 80; 9.5 ms → 42.4 ms combined vs the flat historical
  form at REG 94 / STACK 0). Needed shape: a CTA takes a coarse mapped
  `[1, N]` row-ownership token, then derives `[1, BS]` mutable child
  views (`subpartition`) iterated by an ordinary inner loop, with child
  stores inheriting the parent row's ownership/disjointness proof.
  Acceptance bar: codegen parity with the flat form (AddRmsNorm ~REG:94,
  STACK:0, no local loads/stores). Experiment reverted from the grout
  tree; another instance of the register/stack cost channel in
  finding 2.
