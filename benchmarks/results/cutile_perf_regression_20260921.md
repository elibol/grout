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

## Open items (stopped 2026-09-21 evening; pick up here)

1. **pp18 prefill residual on the fix is probably real, not noise.** The fix
   binary read 7.13 ms in both runs; pre-#275 / v0.3.1 read 6.82, 6.83 and
   6.88 ms in three quiet runs. That is a consistent ~0.25–0.3 ms per prefill
   step, invisible in the per-launch op table (per-launch cost is +0.00 µs)
   and absent from decode — so a per-step, prefill-path-only cost (the
   prefill step executes DeviceOps through ExecutionContexts and awaits
   futures; decode replays a graph). The "0.99× v0.3.1" I cited relied on one
   noisy v0.3.1 arm (IQR 6.8–7.3). To settle: three arms interleaved
   (pre-#275 d2c50c8, v0.3.1 2e4510f, fix 4b0f442), pp18 / pp128 / pp512
   prefill, 6 rounds × 3 reps, records off. If pp128/pp512 show the same
   absolute ~0.25 ms, it is per-step; if it scales, it is per-launch after
   all. Qualify the "merge as is" verdict to the cutile-rs agent with the
   result.
2. **v0.3.0 baseline.** The 09-02 check of the 0.3.1 line against 0.3.0 saw a
   one-pass pp18 decode −4.3% that was dismissed when a second pass did not
   reproduce it. Run paired: `benchmarks/cutile_perf_regression.sh v0.3.0 v0.3.1`
   and `... v0.3.0 4b0f442` (`CUTILE_REPO=/tmp/claude-1000/wt-perf`). The grout
   lib should build against 0.3.0 through `src/driver_compat.rs`; untested.
3. **Paper cross-check.** This test runs records OFF, so its prefill numbers
   (58.6 ms @2048, ~392 @8192) are built-in-default numbers, not the paper's
   records-on cells (~52 / ~287 ms). Decode matches the paper (171–174 tok/s
   at pp18). Both d2c50c8 and 4b0f442 are 0.4.0-stamped, so the sm_120 records
   load on both: run pre-#275 vs fix with records ON at pp18_tg128 /
   pp2048_tg128 / pp8192_tg16 to compare against the paper cells directly.
4. Pre-built binaries for all three arms:
   `/tmp/claude-1000/cpr_fix/target_{d2c50c8,2e4510f,4b0f442}/release/grout_bench`
   (records dir must be `/nonexistent` for records-off arms).

## 2026-09-22: merged main (6b6351b, includes the #275 fix as #302), quiet system

All pairings 4 rounds × 3 reps unless noted; IQRs are a few hundredths wide.

| pairing | pp18 prefill | pp18 decode | pp2048 prefill / decode | pp8192 prefill / decode | gate |
|---|---:|---:|---:|---:|---|
| v0.3.0 → v0.3.1 | 0.995 | 1.000 | 1.000 / 1.000 | 1.001 / 0.999 | **PASS** (no 0.3.1 regression; 0.3.1 halves per-launch host cost, 2593 → 1881 µs/step) |
| v0.3.1 → main | **1.044** | 0.996 | 1.005 / 0.996 | 1.004 / 0.997 | **FAIL** (pp18 prefill) |
| v0.3.0 → main | **1.037** | 0.998 | 1.005 / 0.996 | 1.005 / 0.997 | **FAIL** (pp18 prefill) |
| pre-#275 d2c50c8 → main | **1.043** | 0.996 | 1.005 / 0.996 | 1.005 / 0.997 | **FAIL** (pp18 prefill) |

Decode is recovered to within 0.4% (a consistent −0.3…−0.4% with disjoint
IQRs remains in every cell, under the gate). Long prefill +0.4–0.5%. The
short-prefill residual is real and reproducible: +0.29–0.32 ms per prefill
step against every baseline.

### Where the residual lives (3 arms interleaved, 6 rounds × 3 reps, records off)

| cell | v0.3.1 | main | Δ |
|---|---:|---:|---:|
| pp18 | 6.86 ms | 7.18 ms | **+0.32 ms (+4.7%)** |
| pp128 | 7.82 | 7.87 | +0.05 ms |
| pp512 | 15.66 | 15.76 | +0.10 ms |

Same ~650 ops per prefill step at every length, yet the delta collapses as the
prompt grows: the host-bound → GPU-bound transition. At pp18 the step is
host-bound (GPU idle between launches), so per-op host cost lands on the wall
clock in full; from pp128 up the GPU is the bottleneck and hides it. The
per-launch `execute` cost is at parity (op table sum 1876 → 1917 µs vs
pre-#275), so the ~0.5 µs/op sits in the **await path** — `DeviceFuture::poll`
now runs `ctx.complete()` (a `Submission::complete()` under its lock) for every
awaited op. Decode replays one graph per token and awaits once, hence no
decode signature.

### Paper configuration (records ON, sm_120 records, pre-#275 vs main)

| cell | metric | pre-#275 | main | ratio | paper (5090/4B) |
|---|---|---:|---:|---:|---|
| pp18_tg128 | prefill ms | 6.82 | 7.12 | 1.043 | ~6.9–7.0 |
| pp18_tg128 | decode tok/s | 176.7 | 176.1 | 0.997 | 171–175 |
| pp2048_tg128 | prefill ms | 50.84 | 51.03 | 1.004 | 51.0–52.4 |
| pp2048_tg128 | decode tok/s | 169.4 | 168.7 | 0.996 | ~169 |
| pp8192_tg16 | prefill ms | 277.7 | 281.8 | 1.015 | 284–292 |
| pp8192_tg16 | decode tok/s | 162.4 | 161.8 | 0.997 | ~160 |

The records-on numbers reproduce the paper's cells (this was a quiet system,
hence slightly better than the published medians). Impact of the residual on
the paper's reported metrics: pp18 pure prefill would read 7.1 instead of
6.9 ms; request tok/s at pp18/tg36 changes by <0.1% (0.3 ms of a ~750 ms
request); every other cell within 1.5%.

### Verdict

The #275 decode regression is fixed by #302. A per-op host cost of ~0.5 µs
in the await path remains; it costs +4.3% on host-bound short prefill and is
hidden elsewhere. Gate: FAIL on pp18 prefill until that is addressed.

## 2026-09-22 (later): await-path fix `perf/await-path` @ e6bc238 — no measurable change

Gate results (quiet 5090, records off, 4 rounds × 3 reps):

| pairing | pp18 prefill | decode (18/2048/8192) | long prefill | gate |
|---|---:|---:|---:|---|
| v0.3.1 → e6bc238 | **+4.8%** (6.79 → 7.11) | 0.996 / 0.996 / 0.996 | +0.5% / +0.4% | **FAIL** |
| pre-#275 → e6bc238 | **+4.4%** (6.79 → 7.09) | 0.996 / 0.996 / 0.997 | +0.5% / +0.4% | **FAIL** |

Four arms interleaved (6 rounds × 3 reps, n=18 each), prefill only:

| cell | pre-#275 | v0.3.1 | main 6b6351b | fix e6bc238 |
|---|---:|---:|---:|---:|
| pp18 | 6.85 | 6.86 | 7.18 | 7.19 (**+0.34 ms**) |
| pp128 | 7.81 | 7.81 | 7.87 | 7.88 (+0.07) |
| pp512 | 15.59 | 15.58 | 15.70 | 15.74 (+0.15) |

Per-op host launch cost (`execute` window), four arms interleaved, 5 rounds ×
11 prefill steps, medians:

| op | calls | pre-#275 | v0.3.1 | main | fix | fix − pre |
|---|---:|---:|---:|---:|---:|---:|
| QkNormRopeKvPrefill | 36 | 5.93 | 5.63 | 6.11 | 6.21 | +0.28 |
| Attention | 36 | 3.55 | 3.56 | 3.76 | 3.79 | +0.24 |
| AddRmsNorm | 36 | 3.07 | 3.06 | 3.21 | 3.38 | +0.31 |
| RmsNorm | 37 | 2.35 | 2.36 | 2.52 | 2.64 | +0.29 |
| SiluMul | 36 | 2.10 | 2.09 | 2.20 | 2.31 | +0.21 |
| Add | 36 | 2.09 | 2.08 | 2.30 | 2.36 | +0.27 |
| MatMul / MatMulSlice (cuBLAS via ctx) | 72 / 180 | 4.82 / 4.23 | 4.84 / 4.27 | 4.94 / 4.31 | 4.89 / 4.29 | +0.07 / +0.06 |
| **sum per prefill step (µs)** | | 1842 | 1841 | 1908 | 1921 | **+79** |

Accounting for the pp18 step (+340 µs vs pre-#275): +79 µs is inside
`execute` (≈ +0.25 µs per cuTile launch, ≈ +0.06 µs per cuBLAS op — the
per-argument lease work scales with argument count), and ≈ +260 µs is outside
it, i.e. in the await/completion path, ≈ 0.4 µs per awaited op over ~650
ops. e6bc238 does not move either component relative to main (the
interleaved medians put it 13 µs/step *worse* inside `execute`, within
noise). Yesterday's "+0.00 µs per launch" was a noisy-day artefact; with a
quiet system the execute-side cost is resolvable and real.

Verdict: still FAIL on pp18 prefill. The remaining cost is what the fix's
author describes as inherent to tracking accesses at all (one submission
allocation per awaited op, Arc inc/dec + two CASes per argument, one lock per
launch) plus a completion step per await. Grout never needed this tracking —
it owns its tensors for the engine's lifetime and orders work on one stream —
so the ask from grout's side is an **opt-out**, not further shaving: an
`ExecutionContext` / launcher mode with access tracking disabled (unsafe if
it must be, like `async_on`), or a cargo feature, restoring the pre-#275
launch and await paths for callers that manage lifetimes themselves.

## 2026-09-22 (evening): the await-side residual bisects to #298, not #275

pp18 prefill only, records off, six builds interleaved (6 rounds × 3 reps,
n=18 each), with the per-op `execute` sum from four interleaved profile runs
(11 prefill steps each) so the step delta splits into inside/outside execute:

| rev | pp18 prefill ms [IQR] | Δ step vs pre (µs) | execute sum (µs) | Δ execute | Δ outside execute |
|---|---:|---:|---:|---:|---:|
| d2c50c8 pre-#275 | 6.855 [6.85–6.86] | +0 | 1854 | +0 | +0 |
| 17c1835 #275 | 6.930 [6.92–6.94] | +75 | 1921 | +66 | +9 |
| 11f7665 #285 | 6.935 [6.93–6.94] | +80 | 1914 | +59 | +21 |
| e04245b #295 | 6.930 [6.92–6.94] | +75 | 1916 | +62 | +13 |
| **3ac6fb9 #298** | **7.250 [7.24–7.25]** | **+395** | 1924 | +70 | **+325** |
| 6b6351b #302/main | 7.180 [7.17–7.19] | +325 | 1898 | +44 | +281 |

Two separate costs, cleanly separated:

- **#275 (access tracking): +66 µs per step, all inside `execute`** —
  ≈ +0.25 µs per cuTile launch, argument-count dependent. #302 trims it to
  +44. This is the part an opt-out would remove; on its own it is +1.1% at
  pp18, under the gate.
- **#298 ("Tile IR 13.4 raw APIs, target gates, and unsafe PDL"): +325 µs
  per step, all outside `execute`** — ≈ 0.5 µs per op somewhere between the
  op's `execute` returning and the next op being issued (future creation /
  poll / completion). This is the component that fails the gate, and it has
  nothing to do with access tracking; an opt-out from tracking would not
  clear it.

Grout does not enable programmatic dependent launch, so whatever #298 added
runs on the default path for every op. Candidates from the diff: per-launch
`ToolkitCapabilities` / target-gate lookups on the generated launcher's
non-execute path, or the launch-site specialization cache no longer hitting
(0.3.1's cache-hit path was what made launches cheap).
