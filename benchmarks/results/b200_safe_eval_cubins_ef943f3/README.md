# B200 safe-kernel diagnostic cubins

Captured on NVIDIA B200 (`sm_100`) with CUDA 13.3 and Qwen3-32B launch
shapes.

- grout: `ef943f3876094438f93816afe3ec95b4babaee7a`
- cutile-rs: `e90f9b8f1c24d0eb9528741395edda02bf441361`

`LDL/STL` is the static count of matching local-memory instructions in
`nvdisasm -c` output.

| cubin | generics | REG | STACK | SHARED | LDL/STL | SHA-256 |
|---|---|---:|---:|---:|---:|---|
| `q_norm_rope_prefill_wide_f16_sm100_bm32.cubin` | `128,64,32` | 128 | 32 | 19764 | 160 | `bbfea644ebac35f49c53a63ef5bc49f12439d05264ef2343edf40062cfcad35d` |
| `k_norm_rope_v_prefill_wide_f16_sm100_bm32.cubin` | `128,64,32` | 128 | 96 | 27972 | 168 | `eeb3a5a5f473e3c179e51209bd849703f4f5dc1e77bbff7685e9bd16d41683f6` |
| `q_norm_rope_prefill_f16_sm100_tail.cubin` | `128,64,1,1,2` | 128 | 272 | 31156 | 90 | `eefeb8d160ba1a1984c2b08d2197b6113ac3122396b9b5ab43d923a2317abefa` |
| `k_norm_rope_v_prefill_f16_sm100_tail.cubin` | `128,64,1,1,2` | 128 | 320 | 29124 | 86 | `083874dcf84cbf65e1d2769913d18e5e38eda6cfff3e0a261c7339ea3a29bf69` |
| `qk_norm_rope_kv_decode_f16_sm100_maxseq4096.cubin` | `128,64,4096` | 128 | 176 | 13548 | 28 | `8b83a94a526893285ea8dd02a580c7d00bea366291f688920e7b0a280541260c` |

The wide Q entry compiled with 15 checks discharged, zero hoisted, and zero
in-place checks. The other JIT placement counts were 23/0/4 for wide KV,
7/0/2 for the Q tail, 12/0/3 for the KV tail, and 10/0/9 for decode.
