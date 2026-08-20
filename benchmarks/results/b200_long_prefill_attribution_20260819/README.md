# B200 Qwen3-32B long-prefill attribution

This bundle records grout `3b07fd2` with cutile-rs `e90f9b8` at default B200
clocks. The shipping long-context LPT profile was used: `BM=16`, `BN=128`,
group 8 (the engine meaning of profile value 0), swizzle 8, schedule 1, mask
split enabled, latency 2, occupancy 2.

`run_profiles.sh` collected synchronized per-op profiles after one JIT run and
one warmup. `run_nsys.sh` collected one 32K prefill request after the JIT run;
the compact measured-request analysis is in `nsys_summary.csv`. Raw profiler
reports are intentionally not committed.

At 32K, attention occupies 46.1% of request GPU time, cuBLAS kernels 44.8%,
other grout kernels 9.1%, and inter-kernel idle gaps only 0.006%. Launch and
scheduling gaps are therefore not a material source of the current gap. The
non-attention majority is substantial, so a backend ceiling comparison is required
before assigning the full vLLM gap to attention efficiency.
