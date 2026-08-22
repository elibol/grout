# B200 Qwen3-32B cutile-rs v0.3.0 decode spot refresh

Date: 2026-08-22

- grout functional source: `b3978d6`
- cutile-rs: `v0.3.0` (`0839fe4`)
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Cell: pp=18, tg=128
- Method: 3 measured requests after 3 warmups, prefix cache disabled
- Arms: grout, SGLang no-radix, and vLLM cuda-graph

| engine | e2e (ms) | request generation (tok/s) |
|---|---:|---:|
| grout | 1638.54 | 78.1 |
| SGLang | 1670.31 | 76.6 |
| vLLM | 1640.92 | 78.0 |

Grout's direct decode timer reports 78.9 tok/s. Request-level performance is
within 0.2% of vLLM and 1.9% faster than SGLang.
