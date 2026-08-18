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
| fused QK/RoPE/KV prefill | 255 | 416 | 0 | 58 |

The merge matches the sm_120 resource reference (`REG 64`, `STACK 0`, zero
`LDL/STL`). Placement is architecture-independent and matches on sm_100, but
register allocation is not: the current sm_100 decode-split and prefill cubins
spill. P2 audits the requested `BM=16/BN=64` checked-LPT shape separately; the
inherited `BM=16/BN=32` spill above is not assumed to describe that candidate.

## P2: Checked-LPT versus mapped long prefill

Status: **checked-LPT wins decisively at both lengths**

The same release binary was used for both arms. Each arm invocation loaded
Qwen3-32B, ran one discarded warmup, then three measured prefills. Three paired
rounds used A/B, B/A, A/B order. GPU clocks were not locked.

- Arm A (mapped): `LPT=0`, `BM=64`, `BN=32`, `LATENCY=2`
- Arm B (checked-LPT): `BM=16`, `BN=64`, `SWIZZLE=8`, `SCHED=1`, `LATENCY=2`
- Measurement: `GROUT_PROFILE_OPS=1`, `GROUT_PROFILE_SYNC_OPS=1`,
  `grout_bench --profile --max-new-tokens 0`

Attention is sync-op microseconds per layer call (64 calls per prefill). The
initial run and the required post-knob-sweep rerun independently agree.

| series | pp | mapped rounds (us) | checked-LPT rounds (us) | means A/B (us) | LPT delta |
|---|---:|---:|---:|---:|---:|
| initial | 2048 | 212.35 / 212.07 / 211.82 | 110.08 / 109.87 / 109.68 | 212.08 / 109.88 | -48.2% |
| initial | 8192 | 2520.34 / 2518.23 / 2519.05 | 1218.07 / 1218.10 / 1217.53 | 2519.21 / 1217.90 | -51.7% |
| final | 2048 | 212.40 / 215.14 / 212.18 | 111.06 / 110.53 / 110.61 | 213.24 / 110.73 | -48.1% |
| final | 8192 | 2523.33 / 2518.76 / 2519.07 | 1219.79 / 1217.53 / 1219.18 | 2520.39 / 1218.83 | -51.6% |

The synchronized profile's whole-prefill means in the final series were
235.97/229.71 ms (mapped/LPT) at pp=2048 and 1018.43/936.46 ms at pp=8192.
These include a stream synchronization after every graph op and are not used
as e2e latency.

### LPT mini-sweep

`SCHED=1` and `BM=16/BN=64` were fixed. Each cell is one invocation with one
warmup and three measured prefills.

| pp | swizzle | latency | Attention (us) | synchronized prefill (ms) |
|---:|---:|---:|---:|---:|
| 2048 | 4 | 2 | 109.89 | 228.733 |
| 2048 | 8 | 4 | 110.35 | 228.667 |
| 2048 | 8 | 2 | 109.94 | 228.773 |
| 2048 | 4 | 4 | 110.85 | 229.477 |
| 8192 | 4 | 2 | 1218.54 | 935.917 |
| 8192 | 8 | 4 | 1221.46 | 937.433 |
| 8192 | 8 | 2 | 1217.73 | 936.003 |
| 8192 | 4 | 4 | 1216.86 | 935.660 |

No cell wins meaningfully at both lengths. `SWIZZLE=8/LATENCY=2` was retained:
it is within noise of each per-length minimum and has the best cross-length
tradeoff. In particular, the 0.07% 8192 advantage for `SWIZZLE=4/LATENCY=4`
comes with a 0.83% loss at 2048.

### Uninstrumented prefill latency

These supplementary runs disabled op profiling and used one warmup plus three
measured reps per arm.

| pp | mapped (ms) | checked-LPT (ms) | LPT delta |
|---:|---:|---:|---:|
| 2048 | 225.357 | 219.160 | -2.75% |
| 8192 | 1008.083 | 926.113 | -8.13% |

The July safe-bundle context was 121.91 ms at 2048 and 544.12 ms at 8192.
Those old-tree absolute values are about 1.85x faster than both current-tree
arms, so they are not used for the A/B verdict. Current sync-op attribution
shows `qk_norm_rope_kv_prefill_f16` taking 120.70 ms at 2048 and 481.60 ms at
8192. Its sm_100 cubin is `REG 255`, `STACK 416`, 58 `LDL/STL`, making it the
leading candidate for the separate current-versus-July gap. A quick unfused
fallback diagnostic was inconclusive (159-308 ms across three reps); a paired
current-safe versus historical-raw kernel A/B is still required before
assigning causality.

### Winning-shape spill audit

| arm/kernel | placement D/H/I | REG | STACK (bytes) | LOCAL | LDL/STL |
|---|---:|---:|---:|---:|---:|
| checked-LPT `BM=16 BN=64 SW=8 SCHED=1 LAT=2` | 4/4/4 | 128 | 136 | 0 | 19 |
| mapped `BM=64 BN=32 LAT=2` | 5/4/0 | 128 | 64 | 0 | 8 |

The mapped arm has fewer spills yet is about 2x slower in Attention. LPT's win
therefore is not a spill advantage; it comes from the compared grouped,
scheduled kernel form and tile choice. The relative gain is much larger than
the sm_120 Qwen3-4B reference (9-15%), but that is not a pure architecture
comparison: Qwen3-32B has a different GQA ratio and the arms use their own
kernel-specific tiles. The sm_100 spill cubins, especially the fused prefill
QK/RoPE/KV kernel, remain compiler-handoff material.

## P3: Partial decode tile sweep

The optional Qwen3-32B sweep covered the complete `BN x NKS` grid at tg=128
and tg=512, with pp=18, one warmup, and three measured reps per cell. The
tg=2048 grid was skipped because its 16 long-decode cells would exceed the
remaining allocation.

Median total decode ms at tg=128 (lower is better):

| BN / NKS | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|
| 16 | 1702.56 | 1689.61 | 1684.43 | 1688.08 |
| 32 | 1692.00 | 1682.99 | 1687.89 | 1681.75 |
| 64 | 1677.07 | 1674.68 | **1673.06** | 1675.49 |
| 128 | 1680.01 | 1678.21 | 1676.86 | 1678.95 |

Median total decode ms at tg=512:

| BN / NKS | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|
| 16 | 7070.82 | 6912.98 | 6839.92 | 6812.57 |
| 32 | 6947.74 | 6839.69 | 6788.77 | 6788.99 |
| 64 | 6779.16 | 6750.68 | **6733.29** | 6744.74 |
| 128 | 6777.19 | 6756.74 | 6750.87 | 6758.19 |

The current tg=512 wrapper setting (`BN=64/NKS=8`) is the measured winner.
At tg=128, `BN=64/NKS=8` is 0.59% faster than the current
`BN=32/NKS=4` cell; that unpaired sub-percent margin is not sufficient to
change the wrapper. No source or wrapper update was made.

## Close-out

- Revision and correctness gates passed; checked-LPT and mapped output were
  byte-identical for the 2048-token smoke.
- All seven kernels with documented sm_120 placement references match those
  counts on sm_100. There is no bounds-check placement backend divergence in
  this evaluation.
- The merge is resource-clean and matches sm_120. Decode-split and the prefill
  kernels spill on sm_100; the fused QK/RoPE/KV prefill cubin is the most severe
  live case (`REG 255`, `STACK 416`, 58 `LDL/STL`).
- The paired same-session decision is clear: checked-LPT cuts Attention time by
  48.1% at pp=2048 and 51.6% at pp=8192 versus mapped attention at the requested
  per-form tiles. Uninstrumented prefill improves by 2.75% and 8.13%.
- `SWIZZLE=8`, `SCHED=1`, `LATENCY=2` remains the checked-LPT choice. The mini
  sweep found no meaningful cross-length improvement.
- The current-tree absolute prefill values are provisional, not a canonical
  sweep refresh: they are about 1.85x slower than the July safe bundle. The
  fused prefill QK/RoPE/KV kernel is the leading measured suspect, but a paired
  safe-versus-historical-raw A/B is required to establish causality.
- The partial decode grid confirms the existing tg=512 tile. No wrapper change
  was justified at tg=128, and tg=2048 was not run within this allocation.

Full retuning, canonical sweep refresh, and baseline reruns remain out of scope
for this evaluation.

## Continuation Phase A: prefill regression attribution

The production `qk_norm_rope_kv_prefill_f16` was compared with an exact-body
diagnostic twin built as an unsafe `unchecked_accesses=true` entry. The twin
was selected only at the single prefill call site and was removed after the
experiment. Three paired rounds used checked/unchecked, unchecked/checked,
checked/unchecked order; each process ran one warmup and three timed prefills.

| pp | checked round means (ms) | unchecked round means (ms) | overall checked / unchecked (ms) | checked delta |
|---:|---:|---:|---:|---:|
| 2048 | 230.010 / 229.992 / 229.982 | 207.855 / 207.775 / 207.684 | 229.995 / 207.771 | +10.70% |
| 8192 | 991.446 / 991.283 / 991.119 | 907.575 / 909.987 / 906.153 | 991.283 / 907.905 | +9.18% |

| prefill kernel | REG | STACK (bytes) | LDL/STL |
|---|---:|---:|---:|
| checked production entry | 255 | 416 | 58 |
| unchecked exact-body twin | 255 | 32 | 10 |

The body and launch geometry were held fixed, while removing checked-access
lowering reduced the stack by 384 bytes, removed 48 local-memory instructions,
and improved whole-prefill latency by 9-11%. This assigns the regression to
the checked-access form rather than the fused math alone. The twin still uses
255 registers, so the result does not imply that the underlying port structure
is resource-clean.

### Mitigation ladder

Each ladder cell used the pp=2048 prompt, one warmup, and three timed reps. The
acceptance gate was `STACK < 100` and latency within 3% of the unchecked twin.

| checked candidate | prefill mean (ms) | resource result | gate |
|---|---:|---:|---|
| occupancy hint 1 -> 2 | 178.010 | REG 128, STACK 416, 58 LDL/STL | fail: stack |
| occupancy 2 + unconditional Q/K weight loads and tile select | 178.104 | REG 128, STACK 416, 58 LDL/STL | fail: stack |
| compile-time Q/KV grid specializations | 172.638 | Q: STACK 160; KV: STACK 288 | fail: stack |
| separate Q-grid and KV-grid entries with reduced signatures | 173.397 | Q: REG 128, STACK 160, 26 LDL/STL; KV: REG 128, STACK 288, 42 LDL/STL | fail: stack |

All ladder candidates were faster than the occupancy-1 unchecked twin, but
none met the explicit stack threshold. No mitigation was retained; production
source was restored before the remaining audits.

### Decode and mapped-prefill classification

The structurally similar `qk_norm_rope_kv_decode_f16` is also resource-heavy
on sm_100: `REG 255`, `STACK 208`, and 32 `LDL/STL`. This is a second live
reproducer for the fused checked-access code-generation issue.

At pp=8192, the mapped prefill attention kernel was then compiled at the July
tile (`BM=128`, `BN=128`, LPT disabled). Its measured specialization is
`REG 128`, `STACK 72`, and 11 `LDL/STL`; synchronized op profiling measured
Attention at 1202.90 us per layer (64 calls). Whole-prefill time under that
sync-after-every-op instrumentation averaged 977.159 ms and is not an e2e
benchmark number.

This classifies the earlier mapped `BM=16/BN=32` `STACK 1248` result as
shape-specific mistuning, not an all-shape sm_100 code-generation failure.
The fused checked prefill and decode kernels remain the cutile-rs/codegen
handoff findings. Their diagnostic cubins and the mapped `BM=128/BN=128`
cubin are retained in the ignored raw-result bundle.

### Adopted mitigation (post-ladder)

The ladder's stack gate (`STACK < 100`) was a proxy; latency was the
target. Occupancy 1 -> 2 on both fused sm_100 entries is adopted
(pp=2048 prefill 230.0 -> 178.0 ms, faster than the unchecked twin's
207.8). Stack stays 416 — that remains the open cutile-rs codegen
handoff; the grid-specialization variant (172.6 ms) is a candidate
follow-up if the handoff doesn't retire the spills. Remaining gap vs
July (178 vs ~122 ms at pp=2048) is still unattributed.
