# Plan: cutile-rs 0.3.0 autotuner + persistence + warmup API in grout

Status: planned (this doc); the num-warps env plumbing (phase 0) is done.
The benchmark currently autotunes via bash sweep scripts re-running
grout_bench per cell; warmup happens by launching dummy kernels through
the KernelKind prime path at model load.

## What 0.3.0 provides

- `cutile::tune::Autotuner`: declared `Config` spaces (named int/str
  params), pluggable `Searcher` (resumable `GridSearch` in-tree),
  device-event benching (`cutile::bench`), JSONL trial log with resume,
  `.budget()`, `.prune()`.
- `cutile::tune::TuningRecord`: persisted winners keyed per kernel/shape
  bucket, provenance-checked (source hash, toolchain fingerprint, arch);
  `load_verified` refuses records whose provenance no longer matches.
- Warmup: per-kernel generated `.compile()` terminal (compile-and-cache
  without launching) on a process-global single-flight kernel cache.
- `CompileOptions::num_worker_warps_per_cta` (phase-0 envs:
  GROUT_FMHA_PREFILL_WARPS / GROUT_FMHA_DECODE_WARPS /
  GROUT_QK_PREFILL_WARPS).

## Phase 1 — warmup (DONE, revised design)

Implemented as the **persistent on-disk JIT store**
(`cutile_compiler::jit_cache`, enabled at engine load; GROUT_JIT_CACHE=0
opts out, GROUT_JIT_CACHE_DIR overrides the default location): every
startup after the first serves all kernel compiles from disk
(stage2_source=disk for all 37 engine kernels; measured model-ready time
3.25 s -> 1.29 s on the 5090/4B).

Deliberate deviation from the original plan: the KernelKind prime path's
executor-driven launches are RETAINED rather than replaced with
meta-tensor `.compile()` calls. The executors are the dispatch-fidelity
anchor — they compute the exact generics/compile-options the real run
uses (including record-driven values), so a hand-mirrored compile list
would reintroduce the warm-list/dispatch drift class of bug. Once
compiles are disk-served, the prime launches cost microseconds; the
`.compile()` conversion would now save allocation of a handful of tiny
dummy tensors and nothing else. Revisit only if cutile-rs grows a
compile-only execute terminal on the same launcher builders the
executors use.

## Phase 2 — in-binary autotuning

Add `grout_bench autotune [--site <name>] [--arch-bucket <pp|tg list>]`:

- One declared `Config` space per tunable site x shape bucket:
  prefill attention (BM, BN, LATENCY, warps, occupancy; LPT adds
  SWIZZLE, SCHED, BN separately per long-pp bucket), decode attention
  (BN_DECODE, NUM_KV_SPLITS, warps), wide fused prefill (BM, warps),
  merge (occupancy).
- The `setup` closure allocates buffers once per config, returns the
  launch closure; candidates that fail to compile or launch are recorded
  as failed trials (the invalid-config path the searcher expects), not
  errors.
- Correctness gate: after timing, winner output is compared against the
  default config's output for the same inputs; a mismatched winner is
  discarded (records the finding — this caught real bugs before).
- Trial log at benchmarks/tuning/trials-<arch>-<site>.jsonl: interrupted
  sweeps resume instead of restarting (the B200-allocation-died-midway
  problem).

## Phase 3 — persistence via TuningRecord

Winners saved to benchmarks/tuning/<arch>.record.json. Engine startup
does `TuningRecord::load_verified` (path from GROUT_TUNING_RECORD, with
the in-repo file as default): a record whose source hash / toolchain /
arch no longer matches is refused with a warning and the engine falls
back to built-in defaults. This retires the entire stale-profile hazard
class (bit us twice: the legacy-tuned decode wrapper, the July sm_100
profiles). Env vars remain as per-run overrides on top of the record.

## Phase 4 — retire script tuning

sweep_pp_tile.sh / sweep_tg_tile.sh become thin wrappers around
`grout_bench autotune`. Canonical sweep scripts keep only the
cross-engine comparison role. The per-pp env tables in
sweep_pp_sm100.sh / sweep_tg_sm100.sh are replaced by the record file;
the scripts shrink to engine-arm orchestration.

Rollout: phase 1 standalone (pure win); phases 2+3 land together behind
the subcommand without touching engine dispatch, then dispatch switches
to the record after one B200 + one 5090 record is produced and validated
at parity with the current env tables; phase 4 last.
