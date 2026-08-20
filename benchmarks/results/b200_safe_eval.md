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

## Continuation Phase B: persistent GEMM evaluation

Phase B parent revisions were grout
`80d65686142a73a802583f6b452014179844c0d9` and cutile-rs
`61012a71ca6837f0af02d7516578a99258724a11`. GPU clocks were left at their
defaults. The engine's cuBLAS call sites were not changed.

The unused in-tree `gemm_f16` test kernel was replaced by
`gemm_persistent_f16`, following the derived-facts form in the cutile-rs
persistent-GEMM example: mapped mutable output, persistent `iter_indices`, no
bounds annotations, `deny_in_kernel_checks=true`, and two CTAs per CGA. A
separate benchmark binary compares this kernel with the same cuBLAS wrapper
used by the engine.

### Method

The sweep covered the four Qwen3-32B projection shapes at
`M={1,512,2048,8192}`, with `BM={16,32,64,128}`,
`BN={64,128,256,512}`, and `BK={32,64,128}`: 768 candidates in total. For
`M=1`, the cuTile input/output M axis was padded to BM and masked by tensor
bounds; cuBLAS retained logical M=1. Each candidate was JIT-compiled and
launch-tested, correctness-gated against cuBLAS at representative first/last
elements with relative tolerance `1e-2`, and timed with alternating-order CUDA
event samples. All 768 candidates passed. The winner table below comes from a
fresh nine-sample paired rerun of each broad-sweep winner.

`ratio` is persistent/cuBLAS, so values below 1 favor the persistent kernel.

| M | projection (`N x K`) | best `BM x BN x BK` | cuBLAS (us) | persistent (us) | ratio |
|---:|---|---:|---:|---:|---:|
| 1 | qkv (`10240 x 5120`) | `64 x 256 x 64` | 17.220 | 19.393 | 1.1262 |
| 1 | o (`5120 x 8192`) | `128 x 128 x 128` | 13.822 | 18.822 | 1.3617 |
| 1 | gate_up (`51200 x 5120`) | `64 x 256 x 64` | 85.708 | 81.329 | **0.9489** |
| 1 | down (`5120 x 25600`) | `64 x 128 x 128` | 46.467 | 51.655 | 1.1116 |
| 512 | qkv | `128 x 128 x 128` | 38.230 | 52.544 | 1.3744 |
| 512 | o | `128 x 128 x 128` | 28.266 | 49.667 | 1.7572 |
| 512 | gate_up | `128 x 512 x 128` | 150.122 | 219.261 | 1.4606 |
| 512 | down | `128 x 128 x 128` | 103.283 | 158.650 | 1.5361 |
| 2048 | qkv | `128 x 128 x 128` | 124.896 | 172.864 | 1.3841 |
| 2048 | o | `128 x 128 x 128` | 91.462 | 142.438 | 1.5573 |
| 2048 | gate_up | `128 x 512 x 128` | 529.549 | 758.522 | 1.4324 |
| 2048 | down | `128 x 128 x 128` | 314.522 | 474.432 | 1.5084 |
| 8192 | qkv | `128 x 512 x 128` | 433.781 | 619.456 | 1.4280 |
| 8192 | o | `128 x 512 x 128` | 347.840 | 498.507 | 1.4331 |
| 8192 | gate_up | `128 x 256 x 128` | 2067.200 | 3219.424 | 1.5574 |
| 8192 | down | `128 x 512 x 128` | 1196.235 | 1518.571 | 1.2695 |

The decode/GEMV result is mixed rather than an engine-wide win. Persistent
gate-up is 5.11% faster, but qkv, o, and down are 12.62%, 36.17%, and 11.16%
slower. At M=512 the persistent winners are 37.44-75.72% slower; at M=2048
they are 38.41-55.73% slower; and at M=8192 they are 26.95-55.74% slower.
The verdict is therefore no engine integration in the GEMV, small-M, or
large-M regimes from these measurements.

### Winner resource audit

All distinct winning specializations compile with `REG 255`, `STACK 0`, and
zero `LDL/STL`. Their static shared-memory requirements are:

| tile | shared memory (bytes) |
|---:|---:|
| `64 x 256 x 64` | 206140 |
| `64 x 128 x 128` | 189836 |
| `128 x 128 x 128` | 165172 |
| `128 x 256 x 128` | 173300 |
| `128 x 512 x 128` | 230636 |

The derived-facts kernel compiled with the deny gate for every sweep cell and
the winners do not spill. Phase B therefore did not produce a new cutile-rs
check-placement or spill finding.

### Fusion opportunity sizing

The four measured M=1 cuBLAS projections move 956.3 MB of weights in
160.2 us, an aggregate effective bandwidth of about 5.97 TB/s. The estimates
below divide eliminated activation traffic by that measured bandwidth; they
are bandwidth floors, not end-to-end speedup predictions.

| fusion boundary | traffic avoided per token per layer | time at 5.97 TB/s | persistent-GEMM gate |
|---|---:|---:|---|
| qkv GEMM -> QK norm/RoPE/KV | 40,960 bytes | 0.0069 us | closed: qkv GEMM is 12.6% slower at M=1 and 37-43% slower at prefill M |
| gate_up GEMM -> silu_mul -> down GEMM, two `[M,2*inter]` roundtrips | 409,600 bytes | 0.0686 us | closed: down GEMM is 11.2% slower at M=1 and both GEMMs lose at prefill M |
| lm_head GEMV -> argmax | 607,744 bytes | 0.1017 us | existing fused implementation measured separately below |

Across all 64 layers, the first two decode figures correspond to 2.62 MB and
26.21 MB per token, or bandwidth floors of 0.44 us and 4.39 us. At M=8192,
the per-layer figures are 335.5 MB/56.2 us for qkv and 3.36 GB/561.6 us for
the requested gate-up accounting. The current decode graph also materializes
gate/up slices and the SiLU output; a complete gate-up-to-down fusion would
remove 512,000 bytes per token per layer in that graph. These opportunities
remain hypothetical because the corresponding persistent GEMMs do not meet
the within-10%-of-cuBLAS gate.

The existing fused LM-head/argmax path was checked rather than assumed to be
an existence-proof win. Three paired rounds used separate/fused,
fused/separate, separate/fused order, with one warmup and three measured
128-token decodes per process.

| path | round means (ms) | overall mean (ms) | delta |
|---|---:|---:|---:|
| separate cuBLAS LM-head + argmax | 1708.574 / 1708.671 / 1708.654 | 1708.633 | reference |
| fused LM-head/argmax | 1783.547 / 1783.638 / 1783.560 | 1783.582 | +4.39% |

On this B200/Qwen3-32B configuration the existing fused path is an
implementation precedent, not a performance win. It structurally removes the
logits roundtrip, but its kernel implementation more than offsets the small
bandwidth floor measured here.

### Integration policy

No engine GEMM path was modified. Any future per-site persistent-GEMM dispatch
must use `GROUT_CUTILE_GEMM` through `env_bool_or`, with default `false`
retaining cuBLAS. The in-tree persistent kernel and microbenchmark are the only
Phase B source changes.

## Wide-tile fused-prefill family validation

Date: 2026-08-18

### Revisions and correctness

- New grout: `21f21355b626ae2644b33483f828c966fdfd8da9`
- Historical grout: `b924b882aee66b8a70a544a7e4824d8b565ea8a1`
- cutile-rs: `61012a71ca6837f0af02d7516578a99258724a11`
- GPU/model: NVIDIA B200 (`sm_100`), Qwen3-32B

Both `cargo build --release` and the separately rebuilt
`cargo build --release --features benchmarks --bin grout_bench` passed. The
six GPU kernel tests passed after rebuilding the complete cutile-rs dependency
graph. A 2048-token raw prompt followed by 24 generated tokens was byte-identical
to the prior ancestor run. Both generated outputs have SHA-256
`407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`.

### Paired commit A/B

Each commit used its own freshly built benchmark binary. Three rounds used
old/new, new/old, old/new order at each prompt length. Every process ran one
discarded warmup and three measured prefills. GPU clocks were left at their
defaults, generated tokens were zero, and no op profiling was enabled.

| pp | `b924b88` round means (ms) | `21f2135` round means (ms) | overall old / new (ms) | new delta |
|---:|---:|---:|---:|---:|
| 2048 | 176.566 / 177.702 / 178.121 | 137.531 / 139.382 / 140.930 | 177.463 / 139.281 | **-21.52%** |
| 8192 | 781.367 / 782.966 / 784.648 | 629.526 / 630.565 / 630.516 | 782.994 / 630.202 | **-19.51%** |

Against the July pp=2048 context of 121.91 ms, the old kernel-family tree was
1.456x slower and the new tree is 1.142x slower. The wide-tile rebuild removes
68.7% of the old tree's excess latency over that reference; 17.37 ms, or 14.25%,
remains. The July value is context rather than a paired comparison because it
came from an older tree and session.

### Resource audit

The wide kernels were compiled at the shipping `BM=32`; the per-row kernels do
not have a BM specialization. `LDL/STL` is the number of matching local-memory
instructions in `nvdisasm -c` output.

| kernel | REG | STACK (bytes) | SHARED (bytes) | LDL/STL |
|---|---:|---:|---:|---:|
| old `qk_norm_rope_kv_prefill_f16` baseline | 255 | 416 | not retained here | 58 |
| `q_norm_rope_prefill_wide_f16`, `BM=32` | 128 | 160 | 19764 | 176 |
| `k_norm_rope_v_prefill_wide_f16`, `BM=32` | 128 | 288 | 27972 | 192 |
| `q_norm_rope_prefill_f16` tail | 128 | 272 | 31156 | 90 |
| `k_norm_rope_v_prefill_f16` tail | 128 | 320 | 29124 | 86 |

All four entries reduce both register count and stack allocation versus the old
fused kernel, but none is spill-free on sm_100. The wide entries execute one CTA
over 32 rows, so their static local-instruction counts are not directly
comparable to the old one-row kernel's count. The tail entries run only for
remainder rows or unaligned cache starts.

Resource capture through a `CUTILE_TILEIRAS_PATH` wrapper required explicitly
setting `CUTILE_BYTECODE_VERSION=13.3`. Without that override, the wrapper-path
version probe selected a bytecode form that the installed 13.3 `tileiras`
accepted for its empty probe but rejected for a `cuda_tile.for` region. Normal
production discovery through `CUDA_TOOLKIT_PATH` selected 13.3 correctly; fresh
production builds and all tests passed. This is a capture-tooling issue to hand
off to cutile-rs, not a production-kernel compile failure.

### Synchronized op profile

At pp=2048 with `GROUT_PROFILE_OPS=1` and
`GROUT_PROFILE_SYNC_OPS=1`, the split family retained the graph-level
`QkNormRopeKvPrefill` label:

| op | calls | total (ms) | average (us/layer) |
|---|---:|---:|---:|
| `QkNormRopeKvPrefill` | 64 | 8.952 | **139.88** |
| `Attention` | 64 | 11.841 | 185.02 |

The 5090/Qwen3-4B reference is 49.4 us/layer. It is not shape-equivalent:
Qwen3-32B has 64 Q heads while Qwen3-4B has 32; both use eight KV heads and
head dimension 128.

### BM sweep

Each cell used one warmup and three measured prefills. The sweep was not paired,
so sub-percent differences are treated as noise.

| pp | BM=16 (ms) | BM=32 (ms) | BM=64 (ms) | observed best |
|---:|---:|---:|---:|---:|
| 2048 | 139.406 | **138.376** | 139.631 | 32 |
| 8192 | 627.490 | 632.287 | **624.505** | 64 |

BM32 remains the best pp=2048 cell. BM64 is 1.23% faster than BM32 at pp=8192
in this single sweep, while losing 0.91% at pp=2048. The cross-length evidence
does not justify changing the default from BM32 without a paired confirmation.

### Verdict

The wide-tile family is correct on sm_100 and recovers most of the B200
prefill regression: 21.5% at pp=2048 and 19.5% at pp=8192 versus `b924b88`.
The old `REG 255 / STACK 416` kernel is gone from the live prefill path, though
all four replacements still spill and remain sm_100 code-generation follow-up
material. No source or tuning default was changed by this validation.

## Deny-checked wide-Q compiler validation

Date: 2026-08-18

### Revisions and correctness

- grout: `8f3c36248919a75f02f98010ccfd858cbf58696a`
- cutile-rs: `0608dd72440cce68f22df0f2357fdb30a5520e00`
- GPU/model: NVIDIA B200 (`sm_100`), Qwen3-32B

Both release binaries were rebuilt, including the separately feature-gated
`grout_bench`. All six GPU kernel tests passed. A 2048-token raw prompt followed
by 24 generated tokens was byte-identical to the prior run, with SHA-256
`407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`.

The wide Q kernel compiled with `deny_in_kernel_checks = true`. Its JIT placement
counts at the shipping `BM=32` were 15 discharged, zero hoisted, and zero
in-place checks. Resource capture through a `CUTILE_TILEIRAS_PATH` wrapper also
succeeded with `CUTILE_BYTECODE_VERSION` unset, confirming the bytecode-probe
fix in this compiler revision.

### Resource audit

The before values are from the immediately preceding `21f2135` validation with
cutile-rs `61012a7`. The after values combine the grout and compiler revisions.

| kernel | REG before / after | STACK before / after (bytes) | LDL/STL before / after |
|---|---:|---:|---:|
| `q_norm_rope_prefill_wide_f16`, `BM=32` | 128 / 128 | 160 / **32** | 176 / **160** |
| `k_norm_rope_v_prefill_wide_f16`, `BM=32` | 128 / 128 | 288 / **96** | 192 / **168** |
| `q_norm_rope_prefill_f16` tail | 128 / 128 | 272 / 272 | 90 / 90 |
| `k_norm_rope_v_prefill_f16` tail | 128 / 128 | 320 / 320 | 86 / 86 |

The compiler update substantially reduces local storage for both wide kernels,
not only the newly deny-checked Q entry. Neither wide kernel is spill-free, and
the two tail kernels are unchanged.

### Prefill timing

Each round used a fresh process, one discarded warmup, and three measured
prefills. The round order was 2048/8192, 8192/2048, 2048/8192. GPU clocks were
left at their defaults, generated tokens were zero, and op profiling was off.

| pp | round means (ms) | overall (ms) | prior `21f2135` (ms) | delta |
|---:|---:|---:|---:|---:|
| 2048 | 116.316 / 117.349 / 122.598 | **118.754** | 139.281 | **-14.74%** |
| 8192 | 544.992 / 545.447 / 544.489 | **544.976** | 630.202 | **-13.52%** |

For historical context only, the July safe-bundle values were 121.91 ms at
pp=2048 and 544.12 ms at pp=8192. The new results are 2.59% lower and 0.16%
higher, respectively. These are not paired comparisons across those sessions.

### Synchronized op profile

At pp=2048, `QkNormRopeKvPrefill` measured 6.784 ms over 64 layer calls, or
**106.00 us/layer**, down 24.22% from the preceding 139.88 us/layer result.
This remains 7.07x the approximate 15 us/layer floor. `Attention` measured
157.70 us/layer in the same synchronized profile.

### Verdict

The zero-in-place-check wide Q kernel is correct on sm_100, and the updated tree
recovers essentially all of the remaining long-prefill gap to the July context.
The resource and timing gains are a combined grout-plus-compiler result, not a
paired single-variable attribution. Wide-kernel spills remain a cutile-rs
sm_100 code-generation follow-up; the tail-kernel resource use is unchanged.
No engine path or tuning default was changed by this validation.

## Prefix-coverage wide-Q validation

Date: 2026-08-18

### Revisions and correctness

- grout: `ef943f3876094438f93816afe3ec95b4babaee7a`
- cutile-rs: `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU/model: NVIDIA B200 (`sm_100`), Qwen3-32B

Both release binaries were rebuilt, including the feature-gated
`grout_bench`. All seven GPU kernel tests passed, including
`q_norm_rope_prefill_wide_prefix_coverage_semantics`. The established
2048-token prompt followed by 24 generated tokens remained byte-identical,
with SHA-256
`407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`.

An exact 2051-token raw prompt exercised the safe `BM=32` wide-Q prefix over
2048 rows followed by the three-row mapped tail. It completed without a CUDA or
launch-validation error and generated a coherent 24-token continuation.

### Paired commit A/B

`ac7ff84` and `ef943f3` used separate freshly built benchmark binaries against
the same cutile-rs `e90f9b8`. Three rounds used old/new, new/old, old/new order
at each prompt length. Each process ran one discarded warmup and three measured
prefills; generated tokens were zero and GPU clocks were left at their defaults.

| pp | `ac7ff84` round means (ms) | `ef943f3` round means (ms) | overall old / new (ms) | new delta |
|---:|---:|---:|---:|---:|
| 2048 | 119.047 / 120.545 / 117.628 | 117.996 / 118.900 / 119.803 | 119.073 / **118.899** | **-0.146%** |
| 8192 | 546.305 / 548.214 / 543.558 | 545.914 / 545.350 / 545.536 | 546.026 / **545.600** | **-0.078%** |

The new values are 0.122% and 0.115% above the prior-session `ac7ff84`
references of 118.754 and 544.976 ms. Both paired and cross-session results are
within noise; the prefix-coverage safety change has parity performance.

### Diagnostic cubins

Fresh cubins for the four prefill kernels and fused decode kernel are in
`benchmarks/results/b200_safe_eval_cubins_ef943f3/`. Resource capture used the
new compiler and `CUTILE_BYTECODE_VERSION` remained unset.

| kernel | generics | REG | STACK (bytes) | SHARED (bytes) | LDL/STL |
|---|---|---:|---:|---:|---:|
| `q_norm_rope_prefill_wide_f16` | `128,64,32` | 128 | 32 | 19764 | 160 |
| `k_norm_rope_v_prefill_wide_f16` | `128,64,32` | 128 | 96 | 27972 | 168 |
| `q_norm_rope_prefill_f16` tail | `128,64,1,1,2` | 128 | 272 | 31156 | 90 |
| `k_norm_rope_v_prefill_f16` tail | `128,64,1,1,2` | 128 | 320 | 29124 | 86 |
| `qk_norm_rope_kv_decode_f16` | `128,64,4096` | 128 | 176 | 13548 | 28 |

Wide Q exactly matches the prior `REG 128 / STACK 32 / 160 LDL-STL` result and
again compiles with 15 discharged, zero hoisted, and zero in-place checks. No
engine default or tuning profile changed in this phase.

## sm_100 retune after prefix coverage

Date: 2026-08-18

All cells used Qwen3-32B, at least one discarded warmup, and a freshly rebuilt
feature-gated benchmark binary. The stock tile scripts were used unchanged, but
their grids were bounded to the decision cells because every subprocess reloads
the 32B weights and the exhaustive 160-process matrix exceeds this allocation.

### Mapped prefill and decode screens

The long causal mapped-attention screen used three measured reps per cell. The
shipping `BM=128/BN=128` tile remained the observed winner at both lengths.

| pp | BM | BN=32 (ms) | BN=64 (ms) | BN=128 (ms) |
|---:|---:|---:|---:|---:|
| 2048 | 32 | 144.61 | 138.60 | 136.23 |
| 2048 | 64 | 121.44 | 117.73 | 117.31 |
| 2048 | 128 | 118.17 | 119.64 | **115.76** |
| 8192 | 32 | 955.09 | 854.65 | 812.09 |
| 8192 | 64 | 596.83 | 553.68 | 537.56 |
| 8192 | 128 | 557.53 | 541.67 | **521.42** |

The tg=36 decode endpoint screen was effectively tied: `BN=16/NKS=4`
measured 470.71 ms and `BN=32/NKS=4` measured 470.12 ms over 36 tokens. The
0.13% unpaired difference is noise, so no decode profile changed. The decode
kernel and its compiler lowering were unchanged by the wide-prefill work.

### Wide-Q BM paired sweep

Three balanced orderings were run for `BM={16,32,64}`. Each process contained
one warmup and three measured prefills.

| pp | BM=16 (ms) | BM=32 (ms) | BM=64 (ms) | decision |
|---:|---:|---:|---:|---|
| 2048 | 118.975 | **118.114** | 118.788 | retain 32 |
| 8192 | 547.996 | **544.124** | 544.361 | retain 32 |

BM64's 0.04% difference from BM32 at pp=8192 is noise and it loses by 0.57%
at pp=2048. The shipping wide-Q `BM=32` default therefore remains unchanged.

### Long LPT retune

The screen retained `BM=16`, `SWIZZLE=8`, `SCHED=1`, and `LATENCY=2`: BM8 was
substantially slower, while swizzle 4 and scheduler 0 did not improve either
long-context cell. Increasing BN from 64 to 128 won at both lengths, so it was
confirmed with two order-reversed paired rounds.

| pp | metric | BN=64 | BN=128 | delta |
|---:|---|---:|---:|---:|
| 16384 | Attention (us/layer) | 6090.24 | **5912.61** | **-2.92%** |
| 16384 | prefill (ms) | 1221.756 | **1206.347** | **-1.26%** |
| 32768 | Attention (us/layer) | 23805.67 | **23257.09** | **-2.30%** |
| 32768 | prefill (ms) | 3154.285 | **3111.935** | **-1.34%** |

`sweep_pp_sm100.sh` now explicitly selects checked LPT with
`BM=16/BN=128/SWIZZLE=8/SCHED=1/LATENCY=2` for pp=16384 and pp=32768. All
pp<=8192 profiles and `sweep_tg_sm100.sh` remain unchanged.

## Canonical Qwen3-32B sweeps after sm_100 retune

Date: 2026-08-19

GPU clocks were left at driver defaults. Each cell used three discarded
warmups; short cells used ten measured reps and long-prefill cells used three.
The public result bundles are:

- prefill: `benchmarks/results/sweep/20260819_003933`
- decode: `benchmarks/results/sweep/20260819_005600`
- long prefill: `benchmarks/results/sweep/20260819_011004`

### Prefill sweep

The table reports median request latency. The grout-vs-vLLM delta is negative
when grout is faster.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout delta | grout pure prefill (ms) |
|---:|---:|---:|---:|---:|
| 18 | **462.09** | 468.26 | **-1.32%** | 16.33 |
| 128 | **460.98** | 469.83 | **-1.88%** | 16.39 |
| 512 | **484.24** | 493.19 | **-1.82%** | 29.71 |
| 2048 | **582.01** | 592.26 | **-1.73%** | 117.65 |

### Decode sweep

| tg | grout request tok/s | vLLM request tok/s | grout delta | grout decode tok/s |
|---:|---:|---:|---:|---:|
| 36 | **77.8** | 76.9 | **+1.2%** | 80.7 |
| 128 | **78.4** | 77.3 | **+1.4%** | 79.2 |
| 512 | **78.9** | 77.4 | **+1.9%** | 79.1 |

The cross-tg latency fit gives 79.0 decode tok/s for grout and 77.4 for vLLM.

### Long prefill sweep

The checked-LPT winner from the retune was active in both grout cells.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout delta | grout pure prefill (ms) |
|---:|---:|---:|---:|---:|
| 16384 | 1679.11 | **1586.57** | **+5.83%** | 1202.98 |
| 32768 | 3625.25 | **3169.33** | **+14.39%** | 3123.57 |

The engine-specific prefill timers are not directly comparable: grout reports
pure prefill, while vLLM reports TTFT including one decode step. Request e2e is
therefore the cross-engine decision metric. SGLang was attempted by each
wrapper but its installed sm_100 extension could not load `libnuma.so.1`; no
SGLang rows are reported. The failure is retained in each sweep summary.

## Decode CUDA graph-node profile

Date: 2026-08-19

The profile used Qwen3-32B, default GPU clocks, an 18-token prompt, 512 fixed
generated tokens, `max_seq_len=4096`, one discarded warmup, and one measured
repetition. Nsight Systems 2026.2 used `--cuda-graph-trace=node`. The public
summary and kernel CSV are in
`benchmarks/results/profile_b200_32b_decode_20260819_012529`.

The export covers the graph-capture decode plus warmup and measured runs:
1,534 decode steps and 98,176 per-layer launches. Per-layer decode attention
averaged 5.854 us for `fmha_decode_gqa_split_mapped` plus 2.778 us for
`splitk_reduce_merge_mapped`, or **8.632 us combined**. The fused
`qk_norm_rope_kv_decode_f16` kernel averaged **2.862 us** and represented only
1.34% of total traced GPU kernel time, so restructuring it is low leverage.

| GPU kernel family | time (ms) | share |
|---|---:|---:|
| cuBLAS/cuBLASLt | 17447.712 | **83.03%** |
| cuTile | 3565.049 | **16.97%** |

The dominant cuBLAS projection kernels account for 40.7% and 35.7% of total
GPU kernel time; the associated split-K reduction adds 4.7%. The full trace is
kept locally rather than committed because the binary report contains
machine-specific metadata; the sanitized CSV is sufficient for these sums.

## Long-prefill gap follow-up

Date: 2026-08-19

This follow-up used grout `0b9da7f`, cutile-rs `e90f9b8`, Qwen3-32B, and
default B200 clocks. Detailed raw means and cubins are in
`benchmarks/results/b200_long_prefill_eval_20260819`.

### LPT spill and occupancy ladder

The shipping checked-LPT `BM=16/BN=128` even-length specialization uses
`REG=128`, `STACK=128`, 99,484 bytes shared memory, and 16 `LDL/STL`
instructions. Occupancy 3 increased this to `REG=168`, `STACK=136`, 165,052
bytes shared, and 24 `LDL/STL` instructions.

At pp=32768, three paired order-alternating rounds measured 3655.532 ms at
occupancy 2 and 3868.691 ms at occupancy 3. Occupancy 3 is **5.83% slower**, so
the shipping occupancy remains 2.

### Best-vs-best attention dispatch

Checked-LPT retained a small same-session lead over mapped
`BM=128/BN=128` attention at both lengths.

| pp | checked-LPT (ms) | mapped (ms) | LPT delta |
|---:|---:|---:|---:|
| 16384 | **1383.229** | 1389.006 | **-0.42%** |
| 32768 | **3653.174** | 3685.957 | **-0.89%** |

No long-prefill dispatch or tuning profile changed.

### Fresh grout-vLLM sweep

The fresh canonical bundle is `benchmarks/results/sweep/20260819_162701`.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout gap | prior gap |
|---:|---:|---:|---:|---:|
| 16384 | 1873.66 | **1789.52** | **+4.70%** | +5.83% |
| 32768 | 4176.77 | **3700.81** | **+12.86%** | +14.39% |

Both engines were 11-17% slower in absolute terms than the preceding session,
so the modest gap reduction is session drift, not a tuning win. Neither the
occupancy ladder nor mapped-attention fallback closes the gap. The honest
residual is **sm_100 long-context attention-kernel efficiency**, which requires
a separate kernel-engineering effort or a trtllm-gen comparison to quantify.

## LPT mask-split default adopted (2026-08-19, 5090 evidence)

SASS inspection of the sm_100 LPT cubin (which does use tcgen05:
16 UTCHMMA + LDTM/STTM — an earlier grep matched the HMMA substring and
misread it) showed 128 FSEL mask selects per loop body from the causal
mask running on every kv tile: GROUT_FMHA_PREFILL_LPT_MASK_SPLIT
defaulted to off and was never swept. 5090 paired A/B (LPT forced,
SWIZZLE=8/SCHED=1): pp=16384 755.6 vs 778.7 ms (-3.0%), pp=32768 2246.3
vs 2323.2 ms (-3.3%); bench text byte-identical (mask arithmetic is
exact: unmasked lanes add 0.0). Default flipped to on. B200
validation pending; the mask work is arch-independent register
elementwise, so a comparable win is expected at 16K/32K there
(prior gap vs vLLM: +5.8%/+14.4%).

## LPT mask-split validation on B200

Date: 2026-08-19

This validation used grout `b93bb0d`, cutile-rs `e90f9b8`, Qwen3-32B, and
default B200 clocks. Both release binaries were rebuilt, including the
feature-gated `grout_bench`, and all seven GPU kernel tests passed.

### Paired default A/B

Three rounds used old/new, new/old, old/new order at each length. Each process
discarded JIT compilation and one additional warmup, then averaged three
measured requests with 36 generated tokens. The old arm explicitly set
`GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=0`; the new arm left it unset, selecting the
new default. Generated output was byte-identical across every arm and round.

| pp | old / new prefill (ms) | prefill delta | old / new e2e (ms) | e2e delta |
|---:|---:|---:|---:|---:|
| 16384 | 1170.086 / **1154.499** | **-1.332%** | 1645.321 / **1631.896** | **-0.816%** |
| 32768 | 3051.327 / **2977.738** | **-2.412%** | 3548.429 / **3474.549** | **-2.082%** |

The B200 win is smaller than the 5090 result at 16K and approaches it at 32K.
The sm_100 long-context profile now enables mask splitting explicitly.

### Fresh grout-vLLM sweep

The fresh canonical bundle is `benchmarks/results/sweep/20260819_221611`.
Values below are same-session medians over three requests.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout gap | prior gap |
|---:|---:|---:|---:|---:|
| 16384 | 1630.70 | **1566.66** | **+4.09%** | +5.83% |
| 32768 | 3488.40 | **3131.21** | **+11.41%** | +14.39% |

The gap narrowed by 1.74 percentage points at 16K and 2.98 points at 32K.
SGLang was attempted by the standard wrapper but its sm_100 extension could
not load `libnuma.so.1`; this validation required only grout and vLLM. The
remaining 32K gap is the previously identified within-tcgen05 long-context
attention-efficiency limitation.

## Checked-vs-unchecked LPT on B200

Date: 2026-08-19

This phase used grout `3b07fd2`, cutile-rs `e90f9b8`, Qwen3-32B, and default
B200 clocks. Fresh release binaries were built, including the feature-gated
`grout_bench`, and all seven GPU kernel tests passed. The historical raw LPT
test again produced 32,768 mismatches out of 65,536 values on the padded cache
and zero on the contiguous cache.

Three rounds used checked/twin, twin/checked, checked/twin order at each
length. Each process discarded JIT compilation and one additional warmup,
then averaged three measured requests with 36 generated tokens. The twin was
selected only through `GROUT_FMHA_PREFILL_LPT_UNSAFE_TWIN=1`; all other knobs
matched the shipping sm_100 long profile. Generated output was byte-identical
across every arm and round.

| pp | checked / twin prefill (ms) | twin delta | checked / twin e2e (ms) | twin delta |
|---:|---:|---:|---:|---:|
| 16384 | **1261.838** / 1263.682 | +0.146% | **1737.731** / 1738.680 | +0.055% |
| 32768 | 3494.702 / **3477.748** | -0.485% | 3996.190 / **3982.200** | -0.350% |

**Decision: the checked kernel stays.** The exact-body unchecked twin does not
reach the greater-than-2% shipping threshold at either length.

Fresh cubins at `BM=16`, `BN=128`, group 4, mask split on show a large static
resource difference: checked is `REG=128`, `STACK=1232`, 904 static LDL/STL;
the twin is `REG=128`, `STACK=0`, zero LDL/STL. The spill-free twin's lack of a
material timing win means the checked lowering cost is amortized at these long
contexts; the resource result alone is not grounds to expand the unsafe
surface.

### Canonical group correction

The initial decision harness above forced group 4, but the canonical sm_100
profile exports group 0, which resolves to all eight query heads for Qwen3-32B.
The decision test was therefore repeated at group 8 with the same three-round
checked/twin, twin/checked, checked/twin order. All generated text remained
identical.

| pp | checked / twin prefill (ms) | twin delta | checked / twin e2e (ms) | twin delta |
|---:|---:|---:|---:|---:|
| 16384 | 1154.122 / **1152.532** | -0.138% | 1738.051 / **1736.621** | -0.082% |
| 32768 | 2994.462 / **2992.722** | -0.058% | 3689.999 / **3681.731** | -0.224% |

**The corrected decision is unchanged: checked stays.** At the canonical
group-8 shape, checked is `REG=128`, `STACK=200`, and 27 static LDL/STL; the
twin is `REG=128`, `STACK=0`, and zero LDL/STL. Runtime deltas remain far below
the greater-than-2% threshold.

## Long-prefill operation attribution

Date: 2026-08-20

Synchronized operation profiles used the canonical Qwen3-32B LPT profile:
`BM=16`, `BN=128`, group 8, latency 2, and swizzle 8. Each cell discarded JIT
compilation and one warmup. An initial group-4 capture was superseded after the
profile value `GROUP=0` was confirmed to resolve to all eight query heads.
The full operation table is preserved in
`benchmarks/results/b200_long_prefill_attribution_20260819/op_profiles.csv`.

| pp | prefill (ms) | Attention | MatMulSlice | MatMul | fused/other |
|---:|---:|---:|---:|---:|---:|
| 16384 | 1160.663 | 29.92% | 36.92% | 20.97% | 12.19% |
| 32768 | 2988.847 | 45.80% | 28.56% | 16.28% | 9.36% |

The 32K Nsight Systems trace isolated the measured request on its CUDA stream:
902 kernel launches over 2994.670 ms. LPT attention consumed 1381.399 ms
(46.13%, 64 launches averaging 21.584 ms), cuBLAS kernels consumed 1340.391 ms
(44.76%), and other grout kernels consumed 273.714 ms (9.14%). The union of
inter-kernel idle intervals was only 0.167 ms (0.006%). Launch or scheduling
overhead is therefore not a material source of the long-prefill deficit.

Non-attention compute is 53.9% of the 32K GPU timeline. Without a
matched vLLM operation trace, the current end-to-end gap cannot be assigned
entirely to attention. The backend ceiling comparison below is needed to bound
the attention-specific headroom.

## LPT retune with mask split enabled

Date: 2026-08-20

The canonical group-8 shape first received a one-request screen at both long
lengths. `BN=64/256` lost at both lengths. Latency 3 and swizzle 16 were the only
32K screen improvements and advanced to paired confirmation; latency 4 and
swizzle 4 did not. The requested group-4/group-2 screen was also run: group 2
was 123%/180% slower than group 4 at 16K/32K, while group 4 itself lost to the
canonical group 8 by 9.52%/17.03% in a three-round paired comparison.

| pp | default latency 2 (ms) | latency 3 (ms) | delta | default swizzle 8 (ms) | swizzle 16 (ms) | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 1158.220 | 1159.108 | +0.077% | 1157.537 | **1156.607** | -0.080% |
| 32768 | 2998.032 | **2991.374** | -0.222% | **2998.193** | 2998.881 | +0.023% |

These sub-quarter-percent mixed results are parity, not tuning wins. The
shipping long profile remains `BM=16`, `BN=128`, group 0/full-8, latency 2,
swizzle 8, schedule 1, mask split enabled, and occupancy 2.

## Long-prefill attention ceiling

Date: 2026-08-20

The comparison uses the exact Qwen3-32B layer shape (`Hq=64`, `Hkv=8`,
`D=128`, FP16, causal, `q_len=kv_len`). FlashInfer and trtllm-gen numbers are
CUDA-event medians after 10 warmups and 50 measured calls. Grout is the
synchronized Attention row; its 32K Nsight average is also shown to bound
method skew.

| backend | 16K us/layer | vs grout | 32K us/layer | vs grout |
|---|---:|---:|---:|---:|
| grout checked LPT | 5425.7 | - | 21390.7 | - |
| grout checked LPT (Nsight) | - | - | 21584.4 | +0.9% vs sync |
| FlashInfer FA2 | 10749.5 | +98.1% | 42029.1 | +96.5% |
| trtllm-gen, page 16 | **3885.7** | **-28.4%** | **15209.3** | **-28.9%** |

The installed trtllm-gen artifact contains page-size-16 context kernels only.
The requested one-page dense form fails before timing with a missing-kernel
diagnostic for `numTokensPerPage=16384`; page 16 is the production-supported
fallback. PDL enabled and disabled results agree within 0.1%.

Replacing only grout's attention time with the measured trtllm-gen ceiling
would save about 98.6 ms at 16K and 395.6 ms at 32K over 64 layers. Those
amounts exceed the current same-session grout-vLLM e2e gaps of 64.0 ms and
357.2 ms. Therefore the residual gap is fully explainable by attention-kernel
efficiency; launch overhead is not responsible, and tcgen05-class kernel work
has enough measured headroom to matter.

## Final canonical long-prefill sweep

Date: 2026-08-20

The fresh canonical bundle is `benchmarks/results/sweep/20260820_012214`.
Both engines used default clocks and three measured requests after three
warmups; vLLM reported that TRTLLM prefill attention was auto-selected. SGLang
was attempted by the standard wrapper but its extension still cannot load
`libnuma.so.1`, so only the requested grout and vLLM arms are reported.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout gap | grout prefill / vLLM TTFT (ms) |
|---:|---:|---:|---:|---:|
| 16384 | 1627.65 | **1558.70** | **+4.42%** | 1149.76 / 1068.95 |
| 32768 | 3499.47 | **3102.36** | **+12.80%** | 2993.85 / 2596.78 |

The e2e gaps are 69.0 ms and 397.1 ms. The 32K direct prefill difference is
also 397.1 ms, nearly identical to the 395.6 ms attention-only headroom from
the trtllm-gen ceiling. This same-session result confirms the attribution:
closing the long-prefill gap is an attention-kernel efficiency project, not a
launch-overhead, safety-check, or non-attention dispatch problem.

## Raw-LPT resurrection audit (2026-08-20, 5090)

Question: is the checked LPT kernel slower than the deleted unsafe one —
should we ship one less safe kernel? Evidence:

1. **The raw kernel's bug is real and reconfirmed on device.** The
   verbatim historical kernel (`fmha_prefill_gqa_lpt_raw_resurrected`,
   kept in-tree as a diagnostic pinned by the stride test) produces
   32768/65536 wrong outputs (exactly the kv-head-1 half) on the
   engine's padded cache layout, 0 wrong on a contiguous cache — the
   kv_len-vs-max_seq head stride. "Going back" as-was ships wrong
   results.
2. **The fixed unsafe variant is the checked kernel.** Same body; the
   exact-body unchecked twin (`fmha_prefill_gqa_lpt_unchecked_twin`,
   env GROUT_FMHA_PREFILL_LPT_UNSAFE_TWIN=1) exists for a definitive
   sm_100 A/B; sm_120 twin A/Bs measured parity twice, and sm_100
   cross-compilation shows the same UTCHMMA/TMA SASS class.
3. **Shape matches or beats the vLLM/SGLang Triton references.** Both
   use BLOCK_M rows x one q head (K/V reloaded per head); ours packs
   BM=16 x GROUP=4 at the same mma M=64 with 4x K/V reuse. Their
   BLOCK_M=128 shape (our BM=32, M_EFF=128) measured 5.8x slower at
   16K/32K on the 5090 — register cliff. Note vLLM's B200 long-context
   numbers come from trtllm-gen/FA3 CUDA backends, not these Triton
   kernels, so no Triton-shape change can close that gap.

## Opt-in trtllm-gen backend validation

Date: 2026-08-20

- grout: `69fb4ed5657dc4e239d0e7686aa7de6e4a89d06d`
- cutile-rs: `e90f9b8f1c24d0eb9528741395edda02bf441361`
- FlashInfer checkout: `d527621951244b8be4dd412b1a6729295697f3a7`
- trtllm-gen artifact family: `158f6fa11ef139a098cfddcdddce73ca99d164ad`
- GPU/model: NVIDIA B200 (`sm_100`), Qwen3-32B, default clocks

Both release binaries were rebuilt, including the feature-gated
`grout_bench`. The shim builds after adding FlashInfer's CUTLASS include paths
and deriving `TLLM_GEN_FMHA_METAINFO_HASH` from the artifact manifest. The
launcher body was not changed.

The pp=2048 smoke exposed an artifact coverage gap. The runner requested an
FP16 causal `SeparateQkv` context kernel (`qkvLayout=0`,
`numTokensPerPage=0`, `Hqk=Hv=128`, persistent scheduler), but the artifact
does not contain that kernel. It returned the explicit `Missing TRTLLM-GEN
kernel ragged attention` diagnostic and grout correctly fell back to checked
cuTile LPT on all 193 attempted prefill calls. The artifact contains paged FP16
context kernels and separate-QKV BF16 context kernels, but no separate-QKV
FP16 context kernel.

The default and fallback continuations were byte-identical, both with SHA-256
`407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`;
there were no launch errors after fallback, NaNs, or incoherent text. This does
not validate trtllm numerical parity because its kernel never launched.
Consequently the paired long-prefill trtllm A/B and canonical trtllm sweep arm
were skipped rather than reporting fallback timings as backend results. A
matching FP16 `SeparateQkv` causal context artifact is required to resume them.

## Canonical paper sweep: short and mid prefill

Date: 2026-08-20

The clean B200/Qwen3-32B bundle is
`benchmarks/results/sweep/20260820_145859_b200_32b_canonical_pp`. The engine
source is grout `69fb4ed` with cutile-rs `e90f9b8`; clocks were left at their
defaults. Cells through pp=512 used 10 measured requests and pp=2048 used 3,
all after 3 warmups. vLLM prefix caching and SGLang radix caching were disabled.
SGLang was backfilled successfully after supplying the missing system
`libnuma.so.1` at runtime; it selected `trtllm_mha`, while vLLM auto-selected
TRTLLM prefill attention.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout vs vLLM | SGLang e2e (ms) |
|---:|---:|---:|---:|---:|
| 18 | **458.92** | 461.75 | -0.61% | 480.45 |
| 128 | **456.14** | 463.44 | -1.58% | 480.87 |
| 512 | **480.70** | 484.82 | -0.85% | 498.03 |
| 2048 | 586.76 | **577.34** | +1.63% | 588.13 |

The corresponding direct grout prefill medians are 16.45, 16.47, 29.19, and
113.60 ms. Baseline `prefill_ms` is TTFT rather than a pure prefill timer, so
the table uses the cross-engine e2e metric for comparisons.

## Canonical paper sweep: decode

Date: 2026-08-20

The clean B200/Qwen3-32B decode bundle is
`benchmarks/results/sweep/20260820_152625_b200_32b_canonical_tg`. It used the
same engine/compiler revisions and default clocks as the canonical prefill
bundle. Each pp=18 cell contains 10 fixed-length measured requests after 3
warmups; EOS termination was disabled.

| tg | grout request tok/s | vLLM request tok/s | grout vs vLLM | SGLang request tok/s |
|---:|---:|---:|---:|---:|
| 36 | **78.4** | 77.9 | +0.63% | 74.9 |
| 128 | **79.0** | 77.7 | +1.66% | 76.6 |
| 512 | **79.4** | 77.7 | +2.23% | 77.0 |

The cross-tg linear fit gives decode rates of 79.5 tok/s for grout, 77.7 for
vLLM, and 77.2 for SGLang. Grout's direct phase timers report 81.3, 79.8, and
79.6 tok/s at tg 36, 128, and 512 respectively.
