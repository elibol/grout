# cutile-rs perf regression: v0.3.1 → main (0210b63), measured through grout

Date: 2026-09-21. GPU: RTX 5090 (sm_120), CUDA 13.3 host toolkit, default
clocks. Model: Qwen3-4B f16. Grout `safe-kernels` @ `119fb85` built twice from
the same source against two cutile-rs revisions taken from a read-only clone
of `../cutile-rs`; tuning records disabled on both arms (a record stamped for
one cutile version is refused by the other and would confound the pairing).
Runs alternate arm order every round; 4 rounds × 3 reps per cell. Numerics
were verified identical beforehand (all 28 engine kernels JIT with identical
check-placement counts; attention and GEMM output hashes and greedy text
byte-identical to 0.3.1).

Reproduce: `benchmarks/cutile_perf_regression.sh v0.3.1 HEAD` (grout
`safe-kernels`; `CUTILE_REPO` selects the cutile-rs checkout; exits 1 on a
regression > 1.5% with non-overlapping interquartile ranges).

## Result: FAIL

| cell | metric | v0.3.1 (2e4510f) | main (0210b63) | ratio | v0.3.1 IQR | main IQR | n |
|---|---|---:|---:|---:|---|---|---|
| pp18_tg128 | prefill_ms | 6.82 | 7.58 | 1.111 | 6.8–6.8 | 7.2–7.8 | 12/12 | **REGRESSION** |
| pp18_tg128 | decode_tok_s | 171.60 | 167.25 | 0.975 | 171.3–171.8 | 167.2–167.4 | 12/12 | **REGRESSION** |
| pp2048_tg128 | prefill_ms | 59.66 | 59.62 | 0.999 | 58.9–59.9 | 58.8–60.5 | 12/12 | |
| pp2048_tg128 | decode_tok_s | 164.45 | 160.30 | 0.975 | 164.3–164.6 | 160.1–160.4 | 12/12 | **REGRESSION** |
| pp8192_tg16 | prefill_ms | 395.36 | 397.72 | 1.006 | 394.5–395.9 | 396.9–398.4 | 12/12 | |
| pp8192_tg16 | decode_tok_s | 157.90 | 154.20 | 0.977 | 156.0–158.2 | 153.6–154.9 | 12/12 | **REGRESSION** |

Shape of the signature: decode −2.5% in every cell regardless of context
length, short prefill +5–11%, long prefill at parity. That is a fixed
per-step / per-launch host cost, not kernel speed (a cuBLAS-only GEMV
microbench is unchanged between the two builds).

## Per-launch host cost (prefill StepGraph ops, `--profile` + `GROUT_PROFILE_OPS=1`)

| op (host launch, avg µs) | calls/step | v0.3.1 | main | ratio |
|---|---:|---:|---:|---:|
| MatMulSlice (cuBLAS) | 180 | 4.34 | 4.61 | 1.06 |
| MatMul (cuBLAS) | 72 | 4.92 | 5.20 | 1.06 |
| QkNormRopeKvPrefill | 36 | 7.09 | 7.86 | 1.11 |
| Attention | 36 | 3.93 | 4.48 | 1.14 |
| AddRmsNorm | 36 | 3.37 | 4.24 | 1.26 |
| RmsNorm | 37 | 2.58 | 3.23 | 1.25 |
| SiluMul | 36 | 2.30 | 2.86 | 1.24 |
| Add | 36 | 2.29 | 2.71 | 1.18 |
| **sum over one prefill step** | | 1971 | 2204 | **1.12** |

Every cuTile kernel launch costs +0.4–0.9 µs more (18–26% on the small
kernels); ops routed through grout's own cuBLAS `DeviceOp` (which also run
through an `ExecutionContext`) +6%.

## Bisect (same grout source, paired, records off)

| cutile-rs revision | pp18 decode tok/s | pp18 prefill ms | note |
|---|---:|---:|---|
| d2c50c8 (0.4.0 version bump, before #275) | 173.4 | 6.83 | baseline behaviour |
| 17c1835 — #275 "f8 rounding, latency generics, async tensor safety" | 170.4 (−1.7%) | 6.90 | **decode regression lands here**; per-launch cost of cuTile ops +0.3 µs (AddRmsNorm 3.43→3.99, Attention 4.03→4.40, SiluMul 2.39→2.70) |
| 11f7665 — #285 | ≈ #275 | ≈ #275 | no further change resolvable above noise |
| 0210b63 — main (incl. #298) | ≈ #275 (1.003 vs #275) | 7.19–7.58 | short-prefill increment after #275 is real in aggregate (+5–11% vs v0.3.1) but too noisy to pin to #285 vs #298 on a desktop GPU |

## Mechanism (from the #275 diff)

`cuda-async/src/submission.rs` + `device_operation.rs`: every
`ExecutionContext::new` now allocates an `Arc<Submission>` holding two
`Mutex<Vec<…>>`; each launch `retain()`s its tensor arguments as
`Box<dyn Send>` under the lock and `complete()`s under the lock again. Graph
replay (`GraphLaunch::execute`) calls `context.retain(exec.clone())` and then
walks **every recorded resource** of the graph, calling `retain_for_launch`
for each, on **every** replay. Grout's decode step is one graph replay of a
36-layer step (hundreds of recorded tensors), so that loop runs per token —
the constant −2.5% across context lengths.

## Ask

Keep the lifetime guarantee, drop the per-launch cost:

1. Graph replay: retain the recorded resource set as a single shared owner
   (one `Arc` of the frozen `Vec<Arc<dyn ReplayResource>>`, cloned once per
   launch) instead of per-resource `retain_for_launch` per replay.
2. Per-launch: create the `Submission` lazily (many ops retain nothing) or
   pool it; avoid the `Box<dyn Send>` per retained argument (store the
   storage `Arc` directly, `SmallVec` inline); a lock-free or `Cell`-based
   owner list for the common single-thread case.
3. Add the paired test to the release gate: `cutile-examples`/grout
   `benchmarks/cutile_perf_regression.sh <prev-release> HEAD` must PASS
   (no metric > 1.5% worse with non-overlapping IQRs) before a version bump.

Grout's side needs no change; the numbers above are what 0.4.0 would ship
with as of 0210b63.

## Fix verification (same day): `perf/submission-overhead` @ 4b0f442

Fix branch (off main 3ac6fb9): per-launch access leases no longer allocate
(u64 ids, inline storage); graph replay reacquires a frozen, per-storage
deduplicated resource list instead of walking every recorded context.

### Run 1 — v0.3.1 (2e4510f) → fix (4b0f442): PASS

| cell | metric | v0.3.1 | fix | ratio | v0.3.1 IQR | fix IQR |
|---|---|---:|---:|---:|---|---|
| pp18_tg128 | prefill_ms | 7.20 | 7.13 | 0.991 | 6.8–7.3 | 7.1–7.1 |
| pp18_tg128 | decode_tok_s | 173.20 | 172.75 | 0.997 | 171.9–173.6 | 172.7–172.8 |
| pp2048_tg128 | prefill_ms | 58.62 | 59.10 | 1.008 | 58.1–59.0 | 58.6–59.4 |
| pp2048_tg128 | decode_tok_s | 166.30 | 165.75 | 0.997 | 165.9–166.5 | 165.5–165.9 |
| pp8192_tg16 | prefill_ms | 391.84 | 393.71 | 1.005 | 391.5–392.5 | 393.1–394.7 |
| pp8192_tg16 | decode_tok_s | 159.10 | 159.10 | 1.000 | 158.5–160.4 | 157.4–160.0 |

### Run 2 — pre-#275 (d2c50c8) → fix (4b0f442): PASS

| cell | metric | pre-#275 | fix | ratio | pre-#275 IQR | fix IQR |
|---|---|---:|---:|---:|---|---|
| pp18_tg128 | prefill_ms | 6.88 | 7.13 | 1.036 | 6.8–7.2 | 7.1–7.4 |
| pp18_tg128 | decode_tok_s | 173.40 | 172.75 | 0.996 | 173.2–173.5 | 172.4–172.9 |
| pp2048_tg128 | prefill_ms | 58.78 | 58.55 | 0.996 | 58.4–59.3 | 58.3–59.5 |
| pp2048_tg128 | decode_tok_s | 165.90 | 165.20 | 0.996 | 165.8–166.0 | 165.0–165.4 |
| pp8192_tg16 | prefill_ms | 392.13 | 393.79 | 1.004 | 391.7–392.6 | 393.2–394.1 |
| pp8192_tg16 | decode_tok_s | 158.45 | 158.75 | 1.002 | 157.6–158.7 | 158.3–159.1 |

Decode is back to pre-regression within 0.4% everywhere. The pp18 prefill
+3.6% has overlapping IQRs (and the same fix read 0.99× v0.3.1 in run 1), so
it is within the resolution of a desktop GPU, not a residual regression.

### Per-launch host cost, done properly

The single-run op tables printed by the script are too noisy to size
sub-microsecond residuals (the same binary read 1971 and 2141 µs per step in
two runs). Interleaving three binaries over five rounds with eleven prefill
steps each and taking medians:

| op (host launch, µs) | calls/step | pre-#275 | main unfixed | fix | fix − pre |
|---|---:|---:|---:|---:|---:|
| MatMulSlice (cuBLAS) | 180 | 4.34 | 4.32 | 4.34 | +0.00 |
| MatMul (cuBLAS) | 72 | 4.94 | 4.89 | 4.91 | −0.03 |
| QkNormRopeKvPrefill | 36 | 6.78 | 6.52 | 6.70 | −0.08 |
| Attention | 36 | 3.84 | 3.91 | 3.84 | +0.00 |
| AddRmsNorm | 36 | 3.31 | 3.67 | 3.34 | +0.03 |
| RmsNorm | 37 | 2.53 | 2.66 | 2.53 | +0.00 |
| SiluMul | 36 | 2.26 | 2.35 | 2.28 | +0.02 |
| Add | 36 | 2.25 | 2.39 | 2.26 | +0.01 |
| **sum per prefill step** | | 1944 | 1957 | 1944 | −1 |

Median residual per cuTile kernel launch, fix vs pre-#275: **+0.00 µs**
(range −0.08 … +0.03). The expected ~0.24 µs/launch of remaining mutex
traffic is not visible through grout's launch path at this resolution; the
unfixed main's per-launch cost on the small kernels is +0.1–0.36 µs here,
so the decode regression was dominated by the per-replay resource walk,
which the fix removes. No case for a lock-free access tracker from grout's
side at this point.
