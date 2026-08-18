# B200 Safe-Kernel Evaluation

Date: 2026-08-17

## Revisions

- grout: `4fc4d0fc74c0b579a3884b92c0e9f358a250247b` (`safe-kernels`)
- cutile-rs: `61012a71ca6837f0af02d7516578a99258724a11`
- cutile-rs derived-fact guide revision: `a902939ea83a280ce676e28a7f30f0cf9708a8ff` (ancestor of the evaluated merge)

Revision gate: PASS. `cutile-examples/examples/persistent_gemm.rs` contains no
`with_bounds` or `Dim::new` calls and uses `deny_in_kernel_checks = true`.

## Scope

- GPU: NVIDIA B200 (`sm_100`)
- Model: Qwen3-32B
- P0: release builds, coherent generation, and checked-LPT correctness smoke
- P1: bounds-check placement and spill evaluation at current shapes
- P2: paired mapped-vs-checked-LPT long-prefill experiment
- P3: decode tile sweep only if more than 45 minutes remain

Full retuning and canonical sweep refresh are out of scope for this allocation.

## P0: Build and correctness gate

Status: **PASS**

- `cargo build --release`: PASS
- `cargo build --release --features benchmarks --bin grout_bench`: PASS
- Short Qwen3-32B generation: coherent two-sentence explanation of KV caching
- LPT correctness prompt: 2048 exact raw tokens, 24 generated tokens
- Default checked-LPT generated-text SHA-256: `407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`
- `GROUT_FMHA_PREFILL_GQA_LPT=0` generated-text SHA-256: `407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`
- Exact generated-text comparison: identical

Gate-only single-run timing was 220 ms prefill with checked-LPT and 268 ms
with mapped attention. These are not used for the P2 decision; P2 uses paired,
order-alternating same-session measurements.

## P1: Check placement and spill evaluation

Status: **PASS, with sm_100 spills recorded below**

`CUTILE_JIT_TIMING=1` was collected from a 2048-token prefill, a short decode,
and an explicit `--device-argmax` run with
`GROUT_FUSED_LM_HEAD_ARGMAX=1`. Counts are discharged/hoisted/in-place. Rows
with a documented sm_120 reference match it exactly; no placement divergence
was observed on sm_100.

| kernel | sm_100 D/H/I | documented sm_120 D/H/I | comparison |
|---|---:|---:|---|
| `rms_norm_mapped_f16` | 3/0/0 | 3/0/0 | exact |
| `add_rms_norm_mapped_f16` | 7/0/0 | 7/0/0 | exact |
| `splitk_reduce_merge_mapped` | 5/0/0 | 5/0/0 | exact |
| `add_rms_norm_decode_bounded_f16` | 13/0/0 | 13/0/0 | exact |
| `fmha_decode_gqa_split_mapped` | 8/2/0 | 8/2/0 | exact |
| `lm_head_argmax_blocks_f16` | 3/1/0 | 3/1/0 | exact |
| `fmha_prefill_gqa_lpt_checked` | 4/4/4 | 4/4/4 | exact |
| `embedding_batch_f16` | 0/0/3 | not tabulated | observed |
| `rope_seq_f16` | 3/0/4 | not tabulated | observed |
| `kv_cache_update_seq_mapped_f16` | 7/0/2 | not tabulated | observed |
| `fmha_prefill_causal_mapped` | 5/4/0 | not tabulated | observed |
| `add_2d_f16` | 0/0/4 | not tabulated | observed |
| `silu_mul_2d_f16` | 0/0/4 | not tabulated | observed |
| `gather_row_f16` | 0/0/2 | not tabulated | observed |
| `argmax_blocks_f16` | 0/0/1 | not tabulated | observed |
| `qk_norm_mapped_f16` | 4/0/2 | not tabulated | observed |
| `qk_rope_dynpos_mapped_f16` | 10/0/4 | not tabulated | observed |
| `qk_norm_rope_kv_prefill_f16` | 17/0/24 | not tabulated | observed |
| `qk_norm_rope_kv_decode_f16` | 8/0/11 | not tabulated | observed |
| `argmax_reduce_blocks_to_u32` | 2/0/0 | not tabulated | observed |
| `rope_seq_dynpos_f16` | 4/0/4 | not tabulated | observed |
| `kv_cache_update_seq_dynpos_mapped_f16` | 8/0/2 | not tabulated | observed |
| `fmha_causal_mapped` | 6/4/0 | not tabulated | observed |
| creation `full` kernels | 0/0/0 | not tabulated | observed |

The fused prefill QK/RoPE/KV kernel retains its straight-line, once-per-CTA
checks (`17/0/24`) as designed. All attention loop checks are either discharged
or hoisted; checked-LPT's four in-place checks are schedule-derived one-shots,
also matching the sm_120 reference.

Current-shape cubins were retained during JIT compilation and audited with
`cuobjdump -res-usage` and `nvdisasm -c`. `LDL/STL` is the number of matching
local load/store instructions in the disassembly.

| kernel and current specialization | REG | STACK (bytes) | LOCAL | LDL/STL |
|---|---:|---:|---:|---:|
| merge, `NKS=16` | 64 | 0 | 0 | 0 |
| decode split, `BM=1 BN=32 NKS=16 LAT=4` | 128 | 32 | 0 | 4 |
| checked-LPT, `BM=16 BN=32 SWIZZLE=8 SCHED=1 LAT=2` | 128 | 136 | 0 | 19 |
| mapped prefill warmup, `BM=1 BN=32 LAT=2` | 128 | 64 | 0 | 8 |
| mapped prefill, `BM=16 BN=32 LAT=2` | 128 | 1248 | 0 | 467 |

The merge matches the sm_120 resource reference (`REG 64`, `STACK 0`, zero
`LDL/STL`). Placement is architecture-independent and matches on sm_100, but
register allocation is not: the current sm_100 decode-split and prefill cubins
spill. P2 audits the requested `BM=16/BN=64` checked-LPT shape separately; the
inherited `BM=16/BN=32` spill above is not assumed to describe that candidate.
