# B200 Safe-Kernel Evaluation

Date: 2026-08-17

## Revisions

- grout: `4fc4d0fc74c0b579a3884b92c0e9f358a250247b` (`safe-kernels`)
- cutile-rs: `61012a71ca6837f0af02d7516578a99258724a11`
- cutile-rs derived-fact guide revision: `a902939ea83a280ce676e28a7f30f0cf9708a8ff` (ancestor of the evaluated merge)

Revision gate: PASS. `cutile-examples/examples/persistent_gemm.rs` contains no
`with_bounds` or `Dim::new` calls and uses `deny_in_kernel_checks = true`.

## Scope

- GPU: NVIDIA B200 (`sm_100`)
- Model: Qwen3-32B
- P0: release builds, coherent generation, and checked-LPT correctness smoke
- P1: bounds-check placement and spill evaluation at current shapes
- P2: paired mapped-vs-checked-LPT long-prefill experiment
- P3: decode tile sweep only if more than 45 minutes remain

Full retuning and canonical sweep refresh are out of scope for this allocation.

## P0: Build and correctness gate

Status: **PASS**

- `cargo build --release`: PASS
- `cargo build --release --features benchmarks --bin grout_bench`: PASS
- Short Qwen3-32B generation: coherent two-sentence explanation of KV caching
- LPT correctness prompt: 2048 exact raw tokens, 24 generated tokens
- Default checked-LPT generated-text SHA-256: `407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`
- `GROUT_FMHA_PREFILL_GQA_LPT=0` generated-text SHA-256: `407673405ccd7be79b119811ced911cba08b12f1ea59758a3bc676fd8d68cf74`
- Exact generated-text comparison: identical

Gate-only single-run timing was 220 ms prefill with checked-LPT and 268 ms
with mapped attention. These are not used for the P2 decision; P2 uses paired,
order-alternating same-session measurements.
