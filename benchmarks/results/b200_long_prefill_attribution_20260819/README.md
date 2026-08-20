# B200 Qwen3-32B long-prefill attribution

This bundle records grout `3b07fd2` with cutile-rs `e90f9b8` at default B200
clocks. The shipping long-context LPT profile was used: `BM=16`, `BN=128`,
group 4, swizzle 8, schedule 1, mask split enabled, latency 2, occupancy 2.

`run_profiles.sh` collected synchronized per-op profiles after one JIT run and
one warmup. `run_nsys.sh` collected one 32K prefill request after the JIT run;
the compact measured-request analysis is in `nsys_summary.csv`. Raw profiler
reports are intentionally not committed.

At 32K, attention occupies 52.0% of request GPU time, cuBLAS kernels 39.5%,
other grout kernels 8.0%, and inter-kernel idle gaps only 0.52%. Launch and
scheduling gaps are therefore not a material source of the current gap. The
non-attention half is substantial, so a backend ceiling comparison is required
before assigning the full vLLM gap to attention efficiency.
