# B200 (sm_100) bench bring-up runbook

Order of operations for benchmarking the safe-kernel tree on a B200 box.
The tree is arch-agnostic (kernels JIT per arch); what is arch-specific
is *tuning*, and the sm_100 profiles in this repo predate the safe-kernel
migration — re-derive them before quoting any number.

## 0. Box prerequisites

Directory layout is sibling-relative to the grout checkout:

```
dev/
├── grout/            # branch safe-kernels
├── cutile-rs/        # optional: only needed to build against an unreleased cutile-rs rev
├── hf_models/qwen3_4b/           # HF snapshot (config + safetensors + tokenizer)
├── bench_envs/                   # only for baseline arms
│   ├── vllm_env/                 # python venv with vllm
│   ├── sglang_env/               # python venv with sglang
│   └── .cache/                   # HF/vllm caches land here (must be writable)
└── llama.cpp/                    # optional; disabled by default
```

Grout depends on the crates.io release of cutile-rs (0.3.1; see
`Cargo.toml`) — no sibling checkout is required to build. To validate an
unreleased cutile-rs revision, add a `[patch.crates-io]` section pointing
the five crates (cuda-bindings, cuda-core, cuda-async, cutile-compiler,
cutile) at a local checkout; `src/driver_compat.rs` keeps the source
building against both the 0.3.0 and 0.3.1 driver-wrapper signatures.

## 1. Build + correctness smoke

```
cargo build --release
cargo run --release -- --model ../hf_models/qwen3_4b \
    --prompt "Explain KV caching in two sentences." --max-new-tokens 64
```

Read the output text for coherence (this catches wrong-result kernels
that perf runs won't). First run JIT-compiles every kernel for sm_100 —
slow once, cached after. `CUTILE_JIT_LOG=1` to watch kernels compile.

## 2. Check-placement audit (before any perf number)

The perf story assumes bounds checks are discharged at JIT time or
hoisted out of hot loops. Confirm that holds under sm_100 codegen:

1. `CUTILE_JIT_TIMING=1` on one prefill + one decode run; each kernel
   prints `checks_discharged/_hoisted/_in_place` counters. Norms and the
   splitk merge must show `checks_in_place=0`; the merge specifically
   must show `checks_discharged=5 checks_hoisted=0 checks_in_place=0`
   (2 lse + 3 att axes through the bounded `with_bounds`/`coord` path —
   these are JIT-time proofs, so the counters are arch-independent; if
   they differ, the *build* is stale, not the backend). Any kernel
   showing in-place checks here that shows none on sm_120 is a
   compiler-backend gap — hand it to the cutile-rs agent, don't tune
   around it.
2. `./benchmarks/attn_ab.sh` — paired ablation, base vs
   `CUTILE_DISABLE_CHECK_HOISTING=1`. A large positive nohoist delta is
   the expected/healthy result (hoisting is load-bearing); ~0% means the
   checks weren't in the hot path to begin with — verify with the JIT
   counters before concluding anything.

## 2b. Spill audit (the second cost channel)

Discharged checks are necessary but not sufficient: the decode residual
on the 5090 was *register/stack pressure* under occupancy caps, not
check instructions, and tileiras allocates per-arch — sm_100 can spill
where sm_120 does not. After one prefill + one decode run, JIT-compiled
cubins sit in `$TMPDIR` as `<uuid>.cubin`; find a kernel's cubin by
symbol and check resources:

```
for f in $(ls -t ${TMPDIR:-/tmp}/*.cubin | head -40); do
  cuobjdump -symbols "$f" | grep -q splitk_reduce_merge_mapped && echo "$f" && break
done
cuobjdump -res-usage "$f"        # expect REG:64 STACK:0 LOCAL:0 for the merge
nvdisasm -c "$f" | grep -cE "LDL|STL"   # expect 0
```

sm_120 reference (2026-07-08, bounded variant): merge REG:64 STACK:0,
zero LDL/STL. Repeat for `fmha_prefill` / `fmha_decode_gqa_split_mapped`
at the shapes the retune (step 3) selects — spill counts are
tile-shape-dependent, so audit the *winning* shapes, not the inherited
ones. Nonzero STACK or a three-digit LDL/STL count on a hot kernel is a
cutile-rs-agent handoff, with the cubin and the exact generics line from
`CUTILE_JIT_TIMING`.

## 3. Tile retune (do not trust inherited shapes)

Lesson from the migration, in `safe_kernels_ab.md`: **tile/hint configs
are per-kernel-form, not per-op** — and they are also per-arch. The
current `sweep_*_sm100.sh` profiles were tuned on B200 before the
migration (partly on Qwen3-32B, LPT prefill path); treat them as a
starting grid, not an answer.

```
./benchmarks/sweep_pp_tile.sh 18 128 512 2048 8192   # BM x BN per pp
./benchmarks/sweep_tg_tile.sh                        # BN_DECODE x NUM_KV_SPLITS per tg
```

Both scripts rebuild before sweeping and abort on build failure. Update
the winners into `benchmarks/sweep_pp_sm100.sh` /
`benchmarks/sweep_tg_sm100.sh` (per-pp/per-tg env overrides live there;
generic fallbacks in `sweep_pp.sh` are 5090-tuned).

## 4. Canonical sweeps

```
./benchmarks/sweep_pp_sm100.sh                              # prefill scan, all engines
./benchmarks/sweep_tg_sm100.sh                              # decode scan, all engines
SWEEP_PP_VALUES="16384 32768" ./benchmarks/sweep_pp_sm100.sh  # long prefill
```

Baseline arms (vLLM/SGLang) run automatically when `bench_envs/` venvs
exist. Results: `benchmarks/results/sweep/<timestamp>/{run.jsonl,summary_*.txt}`.

Comparable decode metric across engines: `gen_tokens / (e2e_ms - prefill_ms)`
(grout's `decode_ms` equals that span exactly; baselines don't emit
`decode_ms`).

## 5. Known gotchas (all hit in practice on the 5090)

- **vLLM startup OOM** ("warming up sampler with 256 dummy requests"):
  vLLM budgets `gpu_memory_utilization × total VRAM`; anything else
  resident (e.g. a display) pushes the 0.9 default over. Set
  `VLLM_GPU_MEM_UTIL=0.8`. Unlikely on a headless 180 GB B200.
- **nsys on decode**: default graph tracing hides in-graph kernels; use
  `--cuda-graph-trace=node` to itemize them.
- **Op-share profiling**: `--quiet` suppresses the profile report — drop
  it when grepping `GROUT_PROFILE_OPS=1` output. Values are padded
  (`avg_us=   67.26`); match across spaces, don't split on fields.
- **Never edit a sweep script while it is running** — bash reads the
  file incrementally and shifted content mid-run corrupts the sweep.
- **A/B claims need the paired, order-alternating protocol**
  (`attn_ab.sh` shape). Sequential arm sweeps drift; single-digit-%
  deltas from unpaired runs are not findings.
- One warmup rep is enough for steady-state (`--warmup-reps`, default 1
  in grout; sweeps use 3), but the *first-ever* run pays JIT cost —
  never let it into a measured cell.
- **`grout_bench` is feature-gated** (`required-features =
  ["benchmarks"]`): plain `cargo build --release` silently skips it and
  leaves a stale binary in target/release. ALWAYS
  `cargo build --release --features benchmarks --bin grout_bench`
  before benching (the sweep/tile scripts do this; ad-hoc runs must
  too). A stale bench binary produced a vacuous A/B on the 5090.
- **Long prefill on sm_100 auto-enables the LPT path** (q_len >= 2048).
  The raw LPT kernel had a KV head-stride bug (kv_len*D vs the cache's
  max_seq*D — wrong output for kv heads >= 1 whenever kv_len !=
  max_seq); the checked replacement (`fmha_prefill_gqa_lpt_checked`,
  the only LPT kernel now) fixes it by construction at parity perf.
  Any pre-existing B200 numbers taken with LPT on used the buggy
  kernel. Include LPT tile/swizzle/sched in the 3 retune and run the
  2b register audit at the winning shapes.

## 5b. Kernel-level attention comparison (FlashInfer / trtllm-gen)

`benchmarks/bench_flashinfer_attn.py` benches FlashInfer's dense
single-request kernels at grout's Qwen3 shapes (needs a venv with
flashinfer; JIT-compiles on first call). On sm_100 also add the
trtllm-gen arms via the batch wrappers with batch=1
(`trtllm_batch_context_with_kv_cache` / `trtllm_batch_decode_with_kv_cache`,
paged KV with page_size = kv_len as the degenerate dense case) — that
backend is Blackwell-only and is the comparison that matters there.

Grout's side: prefill from the `GROUT_PROFILE_SYNC_OPS=1` Attention row;
decode from nsys (`--cuda-graph-trace=node`) per-launch times for
`fmha_decode_gqa_split_mapped` + `splitk_reduce_merge_mapped` (sum the
pair — flashinfer's decode call merges internally).

Method caveat: sync-ops attribution carries per-op sync overhead that a
CUDA-event kernel loop does not. For externally quotable numbers, time
both sides with nsys. 5090 reference (2026-07-08, method-skewed as
above): FA2 prefill 39.7/258/2813 us at 512/2048/8192 vs grout sync-ops
47.9/377/- ; FA2 decode 12.3/10.4/16.4 us at kv=512/2048/8192.

## 6. Deliverables to bring back

- Retuned `sweep_pp_sm100.sh` / `sweep_tg_sm100.sh` profiles.
- JIT check counters (step 2) for the kernel family on sm_100.
- The three sweep result dirs (pp / tg / long-pp), all engines.
- Op-share profiles at pp = 512 / 8192 / 32768 and a decode nsys
  (`--cuda-graph-trace=node`) — the cuBLAS-vs-cuTile share split is a
  paper number and arch-dependent.
- Anything where sm_100 behaves differently from sm_120 in the check
  audit or the ablation — that's cutile-rs agent material.
