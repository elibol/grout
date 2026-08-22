# B200 Qwen3-32B cutile-rs v0.3.0 prefill spot refresh

Date: 2026-08-22

- grout functional source: `b3978d6`
- cutile-rs: `v0.3.0` (`0839fe4`)
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Method: 3 measured requests after 3 warmups, prefix cache disabled
- Arms: grout checked cuTile, SGLang no-radix, and vLLM cuda-graph

| engine | pp=2048 e2e (ms) | pp=32768 e2e (ms) |
|---|---:|---:|
| grout | 568.02 | 3266.14 |
| SGLang | 584.24 | 3010.73 |
| vLLM | 567.49 | 2933.26 |

At pp=2048 grout is within 0.1% of vLLM and 2.8% faster than SGLang. At
pp=32768 grout is 11.35% slower than vLLM and 8.48% slower than SGLang. The
initial SGLang wrapper attempt lacked `libnuma.so.1`; the retained SGLang rows
are successful retries using a local runtime copy of that system library.
