# B200 Qwen3-32B decode kernel profile

Date: 2026-08-19

This profile used default GPU clocks, an 18-token prompt, 512 fixed generated
tokens, `max_seq_len=4096`, one discarded warmup, and one measured repetition.
Nsight Systems 2026.2 traced CUDA graph nodes. The exported summary includes
the graph-capture decode plus the warmup and measured repetitions, giving 1,534
profiled decode steps and 98,176 per-layer launches (1,534 x 64 layers).

## Per-launch results

| kernel | instances | average (us) | median (us) | share of GPU kernel time |
|---|---:|---:|---:|---:|
| `fmha_decode_gqa_split_mapped_entry` | 98,176 | 5.854 | 5.824 | 2.74% |
| `splitk_reduce_merge_mapped_entry` | 98,176 | 2.778 | 2.784 | 1.30% |
| attention pair | 98,176 | **8.632** | - | **4.03%** |
| `qk_norm_rope_kv_decode_f16_entry` | 98,177 | **2.862** | 2.816 | **1.34%** |
| `add_rms_norm_decode_bounded_f16_entry` | 196,352 | 11.045 | 11.072 | 10.32% |

## GPU kernel-time split

| family | total (ms) | share |
|---|---:|---:|
| cuBLAS/cuBLASLt (`cutlass3x`, `nvjet`, `cublasLt`) | 17,447.712 | **83.03%** |
| cuTile (`*_entry`) | 3,565.049 | **16.97%** |
| total | 21,012.761 | 100.00% |

The dominant cuBLAS kernels were the CUTLASS projection kernel at 87.213 us
per launch (40.7% of total kernel time), the main NVJet split-K kernel at
25.495 us per launch (35.7%), and its cuBLASLt reduction at 3.332 us per
launch (4.7%). The lm-head NVJet launch averaged 228.926 us and accounted for
1.7%.

`grout_kern.csv` is the direct `cuda_gpu_kern_sum` export. The `.nsys-rep` and
SQLite export remain local because they contain machine-specific metadata and
are not needed to reproduce the table.
