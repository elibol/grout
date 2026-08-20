# Canonical B200 Qwen3-32B prefill sweep

- Engine source: grout `69fb4ed5657dc4e239d0e7686aa7de6e4a89d06d`
- Compiler source: cutile-rs `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Sweep: pp 18, 128, 512, 2048; tg 36
- Repetitions: 10 through pp=512 and 3 at pp=2048, after 3 warmups
- Cache policy: vLLM prefix cache off; SGLang radix cache off

The standard wrapper initially could not load SGLang because `libnuma.so.1`
was absent. A private runtime copy of the system package was supplied without
changing the environment, and the four SGLang cells were rerun with the same
wrapper arguments into `run.jsonl`. `aggregate.csv` and `aggregate.md` were
regenerated after those successful cells. SGLang selected `trtllm_mha`; vLLM
auto-selected TRTLLM prefill attention.
