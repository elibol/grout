# cutile-rs v0.3.0 tuning screens

Date: 2026-08-22

- grout: `2ca4f07` (functional source `b3978d6`)
- cutile-rs: `v0.3.0` (`0839fe4`)
- GPU/model: NVIDIA B200 (`sm_100`), Qwen3-32B, default clocks
- Method: one measured request after one warmup; candidates had to clear a
  1% screen before a three-round paired adoption run.

## Prefill attention

Prefill times are milliseconds; lower is better. The shipping BM=128/BN=128
cell remains best at both lengths.

| pp | BM64/BN64 | BM64/BN128 | BM64/BN256 | BM128/BN64 | BM128/BN128 | BM128/BN256 |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 109.71 | 106.49 | 113.96 | 108.15 | **106.14** | 115.77 |
| 8192 | 504.30 | 498.76 | 592.87 | 498.34 | **471.14** | 583.28 |

## Decode attention

Decode times cover 128 tokens in milliseconds; lower is better.

| BN | NKS=2 | NKS=4 | NKS=8 |
|---:|---:|---:|---:|
| 16 | 1735.86 | 1724.22 | 1721.13 |
| 32 | 1726.20 | **1716.14 (shipping)** | 1716.22 |
| 64 | 1712.71 | 1708.36 | 1707.88 |

BN=64/NKS=8 screened 0.48% faster than shipping, below the 1% adoption bar.

## Long-prefill LPT

Prefill times are milliseconds. BM=16, swizzle=8, schedule=1, and fused-Q
BM=32 were fixed.

| pp | BN64/lat2/mask1 | BN128/lat2/mask1 | BN256/lat2/mask1 | BN128/lat3/mask1 | BN128/lat2/mask0 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 1091.03 | **1077.04** | 1324.95 | 1076.94 | 1086.19 |
| 32768 | 2814.48 | **2760.44** | 3838.50 | 2749.55 | 2829.73 |

Latency 3 screened within 0.4% of latency 2. Mask split remained beneficial,
especially at 32K where disabling it cost 2.51%.

## Wide fused prefill Q

| pp | BM=16 | BM=32 (shipping) | BM=64 |
|---:|---:|---:|---:|
| 2048 | 114.32 | **110.84** | 111.35 |
| 8192 | 478.89 | 477.05 | **476.81** |

BM=64's 8K advantage was 0.05% while it lost 0.46% at 2K. No candidate met
the adoption bar, so no paired finalist run was required and no sm_100 profile
changed.

**Phase 2 tuning changes adopted: none.**
