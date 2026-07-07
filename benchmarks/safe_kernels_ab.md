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

**Decode — parity after declared preconditions.** -0.20% paired at
tg=2048 (BN=32 / NUM_KV_SPLITS=16), i.e. within noise. The residual
that had separated the arms was closed by the merge-kernel precondition
work described below.

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
   placement.** The merge kernel's 3 in-place bounds checks caused
   48 spill ops / STACK:424 under its REG:64 occupancy cap — the cost
   was not where the checks executed but the registers they pinned.
   Discharging them via declared dim-equality preconditions took the
   kernel to 0 spills / STACK:0 and closed the decode residual.

On the precondition mechanism, an important framing note: this was
**not** a missing discharge rule that had to be added to the checker.
The checker's brand -> grid -> declared-equality chaining already
existed; the fix was on the kernel side — the kernels declaring their
host contracts as dim-equality preconditions
(e.g. `dim(out, 0) == dim(x, 0)`), which the generated host launcher
validates at launch time.

## Verification boundary

Every bounds check in the safe kernel family is one of:

- **discharged** — proved by the checker from the partition map, grid
  brand, or a declared (launch-validated) precondition;
- **hoisted** — moved out of the hot loop by the compiler; or
- **a named value-lattice case** — arithmetic-derived coordinates the
  current lattice cannot relate: `row - num_q_rows` in qk_norm,
  `s - pos` in kv_cache, `head / GROUP` in attention.

That is the precise claim; the family is **not** "fully statically
verified".

## Remaining unsafe kernels (Tier 2, unchanged plan)

Raw-pointer / hand-scheduled kernels, to be A/B'd against typed
equivalents and then ported or deleted:

- `qk_norm_rope_kv_prefill_raw`, `qk_norm_rope_kv_decode_raw`
- `add_rms_norm_decode_raw`
- `fmha_prefill_gqa_lpt`, `fmha_prefill_gqa_lpt_split`
- `prefill_splitk_reduce_merge`
- `flash_decode.rs` (grouped decode attention)
- `qk_rope_dynpos`
- `lm_head_argmax_blocks`
- `group_gemm_nt_desc`

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
