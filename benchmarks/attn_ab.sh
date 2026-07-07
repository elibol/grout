#!/usr/bin/env bash
# Paired ablation for bounds-check placement in the mapped attention kernels.
#
# The legacy unsafe kernels are deleted (safe_kernels_ab.md), so this no
# longer A/Bs legacy vs safe. It measures what check optimization is worth
# on the current tree by pairing:
#   base    — shipped configuration (checks discharged at JIT time or
#             hoisted out of hot loops)
#   nohoist — CUTILE_DISABLE_CHECK_HOISTING=1, same binary: every dynamic
#             bounds check stays in place in the loop body
# on the two surfaces where in-loop checks showed up as regressions during
# the migration:
#   1. prefill: Attention avg_us/call at pp=512, BM=32/BN=16 (sync-ops profile)
#   2. decode:  decode_ms at pp=18 for tg in DECODE_TGS (attention share grows
#      with kv_len, so an attention-check cost grows with tg)
#
# Runs ROUNDS alternating base->nohoist pairs so clock/thermal drift cancels
# (paired design; sequential-arm sweeps left a drift confound in 2026-07-03
# data). Reports per-round deltas. Useful on a new arch (e.g. B200/sm_100)
# to confirm check hoisting/discharge is doing its job under that backend's
# codegen before trusting perf numbers.
#
# Usage:
#   ./benchmarks/attn_ab.sh                 # after rebuilding grout_bench
#   ROUNDS=5 DECODE_TGS="512 2048" ./benchmarks/attn_ab.sh
#
# Reference (RTX 5090, 2026-07-05): nohoist reproduced the pre-hoisting-fix
# prefill-attention regression (~+64% at this pp/tile shape); base is the
# parity configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
ROUNDS="${ROUNDS:-3}"
DECODE_TGS="${DECODE_TGS:-512}"
BENCH="$GROUT_DIR/target/release/grout_bench"

echo "==> Building grout_bench (picks up ../cutile-rs path-dep changes)"
(cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench 2>&1 | tail -1)

# Prompt file at exactly 512 tokens: reuse the newest sweep's, else generate.
PP512="$(ls -t "$SCRIPT_DIR"/results/sweep/*/prompts/pp_512.txt 2>/dev/null | head -1 || true)"
if [[ -z "$PP512" ]]; then
    PY="$GROUT_DIR/../bench_envs/vllm_env/bin/python"; [[ -x "$PY" ]] || PY=python3
    TMP_PROMPTS="$(mktemp -d)"
    "$PY" "$SCRIPT_DIR/make_prompts.py" --model "$MODEL_HF" --out-dir "$TMP_PROMPTS" --pp 512 >/dev/null
    PP512="$TMP_PROMPTS/pp_512.txt"
fi
echo "==> prompt: $PP512"
OUT="$(mktemp -d)"

# --- measurement helpers ----------------------------------------------------

prefill_attn_us() {  # $1 = "base"|"nohoist"; prints Attention avg_us
    local envs=(GROUT_PROFILE_OPS=1 GROUT_PROFILE_SYNC_OPS=1
                GROUT_ATTN_BM_PREFILL=32 GROUT_ATTN_BN_PREFILL=16)
    [[ "$1" == nohoist ]] && envs+=(CUTILE_DISABLE_CHECK_HOISTING=1)
    # NB: grout pads the value after '=' (e.g. "avg_us=   67.26"), so match
    # across the spaces rather than splitting on fields.
    env "${envs[@]}" "$BENCH" --model "$MODEL_HF" \
        --prompt-file "$PP512" --raw-prompt --max-new-tokens 0 \
        --reps 3 --warmup-reps 1 --profile 2>&1 \
      | grep -E '^  Attention ' | grep -oE 'avg_us= *[0-9.]+' | grep -oE '[0-9.]+' | head -1
}

decode_ms() {  # $1 = "base"|"nohoist", $2 = tg; prints median decode_ms
    local envs=()
    [[ "$1" == nohoist ]] && envs+=(CUTILE_DISABLE_CHECK_HOISTING=1)
    env "${envs[@]:-_=_}" "$BENCH" --model "$MODEL_HF" \
        --prompt "Hello, how are you?" --max-new-tokens "$2" --ignore-eos \
        --reps 3 --warmup-reps 1 --quiet 2>&1 \
      | grep -oE '\[timed\].*decode_ms=[0-9.]+' | grep -oE 'decode_ms=[0-9.]+' \
      | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

pct() { awk -v l="$1" -v s="$2" 'BEGIN{printf "%+.2f%%", 100*(s-l)/l}'; }

# --- paired rounds ------------------------------------------------------------

echo
echo "== 1. prefill Attention avg_us/call (pp=512, BM=32/BN=16, sync-ops) =="
echo "   (nohoist = CUTILE_DISABLE_CHECK_HOISTING=1, same binary: in-loop checks)"
printf "  %-7s %10s %12s %14s\n" round base_us nohoist_us nohoist_delta
for r in $(seq 1 "$ROUNDS"); do
    b="$(prefill_attn_us base)"; nh="$(prefill_attn_us nohoist)"
    printf "  %-7s %10s %12s %14s\n" "$r" "$b" "$nh" "$(pct "$b" "$nh")"
done

for TG in $DECODE_TGS; do
    echo
    echo "== 2. decode median decode_ms (pp=18, tg=$TG) =="
    printf "  %-7s %10s %12s %14s\n" round base_ms nohoist_ms nohoist_delta
    for r in $(seq 1 "$ROUNDS"); do
        b="$(decode_ms base "$TG")"; nh="$(decode_ms nohoist "$TG")"
        printf "  %-7s %10s %12s %14s\n" "$r" "$b" "$nh" "$(pct "$b" "$nh")"
    done
done

echo
echo "Read: nohoist_delta >> 0%% and consistent in sign = hoisting/discharge is"
echo "load-bearing on this arch (expected). nohoist_delta ~0%% at a shape where"
echo "checks sit in the hot loop = investigate whether checks were emitted at"
echo "all (CUTILE_JIT_TIMING=1 prints discharged/hoisted/in-place counters)."
