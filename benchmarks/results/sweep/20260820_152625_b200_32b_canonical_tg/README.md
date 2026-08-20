# Canonical B200 Qwen3-32B decode sweep

- Engine source: grout `69fb4ed5657dc4e239d0e7686aa7de6e4a89d06d`
- Compiler source: cutile-rs `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Sweep: pp 18; tg 36, 128, 512
- Repetitions: 10 per cell after 3 warmups
- Decode policy: fixed generation length with EOS ignored
- Cache policy: vLLM prefix cache off; SGLang radix cache off

All three engine arms completed in the standard wrapper. SGLang used a
runtime-provided system `libnuma.so.1` and selected `trtllm_mha`; vLLM used
its CUDA-graph mode with prefix caching disabled.
