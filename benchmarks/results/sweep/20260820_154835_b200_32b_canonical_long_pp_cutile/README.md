# Canonical B200 Qwen3-32B long-prefill sweep (cuTile LPT)

- Engine source: grout `69fb4ed5657dc4e239d0e7686aa7de6e4a89d06d`
- Compiler source: cutile-rs `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Sweep: pp 16384, 32768; tg 36
- Repetitions: 3 measured requests after 3 warmups
- Cache policy: vLLM prefix cache off; SGLang radix cache off
- grout attention: checked cuTile LPT shipping profile

All three engines completed. SGLang selected `trtllm_mha`, while vLLM
auto-selected TRTLLM prefill attention.

The requested companion grout trtllm-gen arm was not run because the cached
artifact lacks the FP16 causal `SeparateQkv` context kernel required by the
shim. The smoke test fell back to cuTile, so reporting its timings as trtllm-gen
would be invalid. See `benchmarks/results/b200_safe_eval.md` for the diagnostic.
