# B200 Qwen3-32B long-prefill follow-up

Date: 2026-08-19

- grout: `0b9da7f71ce56232121a9e0f51c2f1b514c90444`
- cutile-rs: `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- model: Qwen3-32B

## Shipping LPT resources

The shipping checked-LPT specialization is
`BM=16/BN=128/SWIZZLE=8/SCHED=1/LATENCY=2/OCCUPANCY=2`.

| occupancy | REG | STACK (bytes) | SHARED (bytes) | LDL/STL |
|---:|---:|---:|---:|---:|
| 2 | 128 | 128 | 99484 | 16 |
| 3 | 168 | 136 | 165052 | 24 |

Both even-length and overhang cubins are retained under `cubins/`. The measured
16k/32k cells use the `even1` specializations.

## Occupancy ladder

Three paired, order-alternating rounds at pp=32768 used one discarded warmup
and three measured prefills per process.

| occupancy | round means (ms) | overall mean (ms) | delta vs occupancy 2 |
|---:|---:|---:|---:|
| 2 | 3668.367 / 3640.283 / 3657.947 | **3655.532** | - |
| 3 | 3868.607 / 3879.350 / 3858.117 | 3868.691 | **+5.83%** |

Occupancy 3 increases registers, stack traffic, and shared memory while losing
performance, so occupancy 2 remains selected.

## Checked-LPT versus mapped attention

Arm A is checked-LPT at the shipping configuration. Arm B is mapped attention
at `BM=128/BN=128/OCCUPANCY=2`, with LPT disabled. Each cell used three paired,
order-alternating rounds, one discarded warmup, and three measured prefills per
process.

| pp | checked-LPT round means (ms) | mapped round means (ms) | overall LPT / mapped (ms) | LPT delta |
|---:|---:|---:|---:|---:|
| 16384 | 1375.647 / 1384.737 / 1389.303 | 1390.270 / 1387.427 / 1389.320 | **1383.229** / 1389.006 | **-0.42%** |
| 32768 | 3649.200 / 3650.883 / 3659.440 | 3682.030 / 3687.717 / 3688.123 | **3653.174** / 3685.957 | **-0.89%** |

Mapped attention does not win either length, so the existing checked-LPT
dispatch remains unchanged.

## Fresh long-prefill sweep

The canonical same-session result bundle is
`benchmarks/results/sweep/20260819_162701`. Request e2e is the comparable
cross-engine metric.

| pp | grout e2e (ms) | vLLM e2e (ms) | grout gap |
|---:|---:|---:|---:|
| 16384 | 1873.66 | **1789.52** | **+4.70%** |
| 32768 | 4176.77 | **3700.81** | **+12.86%** |

Absolute times were 11-17% slower than the preceding session for both engines,
so the smaller percentage gaps versus the prior 5.83%/14.39% are not credited
to a configuration change. Neither tested lever closes the gap. The remaining
limitation is sm_100 long-context attention-kernel efficiency.
