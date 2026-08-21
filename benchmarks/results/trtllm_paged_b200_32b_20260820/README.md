# B200 Qwen3-32B paged trtllm paired validation

- Engine source: grout `d963df1ecc92885507db19ce8b6ebac1e78d9183`
- Shim build fix: `ef83a5b`
- Compiler source: cutile-rs `e90f9b8f1c24d0eb9528741395edda02bf441361`
- GPU: NVIDIA B200 (`sm_100`), default clocks
- Workload: pure prefill at pp 16384 and 32768
- Protocol: 3 paired rounds, order alternated, 1 measured request per arm
  after 1 warmup request; first JIT/capture excluded
- cuTile arm: checked LPT shipping profile
- trtllm arm: page-16 context kernel, `GROUT_TRTLLM_REPACK` unset

`paired.jsonl` retains the individual measurements and encodes round/order in
the variant name.

| pp | round | cuTile (ms) | trtllm (ms) | trtllm delta |
|---:|---:|---:|---:|---:|
| 16384 | 1 | 1151.563 | 1097.098 | -4.73% |
| 16384 | 2 | 1151.737 | 1080.499 | -6.19% |
| 16384 | 3 | 1159.513 | 1095.695 | -5.50% |
| 32768 | 1 | 2965.687 | 2643.262 | -10.87% |
| 32768 | 2 | 2966.215 | 2645.165 | -10.82% |
| 32768 | 3 | 2970.151 | 2644.818 | -10.95% |

The means are 1154.271 vs 1091.097 ms at 16K (-5.47%) and 2967.351
vs 2644.415 ms at 32K (-10.88%). No fallback warning occurred in any trtllm
arm.

The pp=2048 plus 24-token smoke also ran without fallback or repacking. Its
generated text was byte-identical to the default cuTile output. The dense K/V
cache was consumed in place; only page-table and sequence-length metadata were
updated.
