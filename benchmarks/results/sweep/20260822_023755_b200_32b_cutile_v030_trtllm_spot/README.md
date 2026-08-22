# B200 Qwen3-32B trtllm paged attention spot check

Date: 2026-08-22

- grout functional source: `b3978d6`
- cutile-rs: `v0.3.0` (`0839fe4`)
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Backend: opt-in page-16 trtllm context attention, zero-copy mode
- Method: 3 measured requests after 3 warmups, pp sweep with tg=36

| pp | current median e2e (ms) | prior trtllm e2e (ms) | delta |
|---:|---:|---:|---:|
| 16384 | 1491.74 | 1569.97 | -4.98% |
| 32768 | 2970.52 | 3157.16 | -5.91% |

No fallback warning or launch error appeared. The spot check passes the
compiler-revision non-regression expectation and was faster than the prior
trtllm result at both lengths. Because these sessions were not paired, the
cross-session improvements are not treated as speedup claims.
