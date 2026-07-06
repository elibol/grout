#!/usr/bin/env bash
# Paired A/B for the mapped-attention perf regression (safe_kernels_ab.md).
#
# Measures the two device-confirmed regression surfaces, legacy vs safe:
#   1. prefill: Attention avg_us/call at pp=512, BM=32/BN=16 (sync-ops profile)
#   2. decode:  decode_ms at pp=18 for tg in DECODE_TGS (attention share grows
#      with kv_len, so a mapped-attention regression grows with tg)
#
# Runs ROUNDS alternating legacy->safe pairs so clock/thermal drift cancels
# (paired design; the 2026-07-03 sweeps ran the modes 35 min apart, which left
# a drift confound). Reports per-round deltas and the median delta.
#
# Usage:
#   ./benchmarks/attn_ab.sh                 # after rebuilding grout_bench
#   ROUNDS=5 DECODE_TGS="512 2048" ./benchmarks/attn_ab.sh
#
# Baseline (pre-hoisting-fix, 2026-07-03): prefill Attention 41.1 -> 67.3
# us/call (+64%); decode +2%..+6.6% over tg=128..8192. Parity bar: ~0%.

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

prefill_attn_us() {  # $1 = "legacy"|"safe"|"safe-nohoist"; prints Attention avg_us
    local envs=(GROUT_PROFILE_OPS=1 GROUT_PROFILE_SYNC_OPS=1
                GROUT_ATTN_BM_PREFILL=32 GROUT_ATTN_BN_PREFILL=16)
    [[ "$1" == legacy ]] && envs+=(GROUT_UNSAFE_KERNELS=1)
    # Same-binary ablation: keeps every dynamic bounds check in place, so this
    # arm should reproduce the pre-fix regressed numbers (attribution sanity).
    [[ "$1" == safe-nohoist ]] && envs+=(CUTILE_DISABLE_CHECK_HOISTING=1)
    # NB: grout pads the value after '=' (e.g. "avg_us=   67.26"), so match
    # across the spaces rather than splitting on fields.
    env "${envs[@]}" "$BENCH" --model "$MODEL_HF" \
        --prompt-file "$PP512" --raw-prompt --max-new-tokens 0 \
        --reps 3 --warmup-reps 1 --profile 2>&1 \
      | grep -E '^  Attention ' | grep -oE 'avg_us= *[0-9.]+' | grep -oE '[0-9.]+' | head -1
}

decode_ms() {  # $1 = "legacy"|"safe", $2 = tg; prints median decode_ms
    local envs=()
    [[ "$1" == legacy ]] && envs+=(GROUT_UNSAFE_KERNELS=1)
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
echo "   (safe-nohoist = CUTILE_DISABLE_CHECK_HOISTING=1, same binary; should"
echo "    reproduce the pre-fix regression if hoisting is what fixed it)"
printf "  %-7s %10s %10s %12s %10s %14s\n" round legacy_us safe_us nohoist_us delta nohoist_delta
for r in $(seq 1 "$ROUNDS"); do
    l="$(prefill_attn_us legacy)"; s="$(prefill_attn_us safe)"; nh="$(prefill_attn_us safe-nohoist)"
    printf "  %-7s %10s %10s %12s %10s %14s\n" "$r" "$l" "$s" "$nh" "$(pct "$l" "$s")" "$(pct "$l" "$nh")"
done

for TG in $DECODE_TGS; do
    echo
    echo "== 2. decode median decode_ms (pp=18, tg=$TG) =="
    printf "  %-7s %10s %10s %10s\n" round legacy_ms safe_ms delta
    for r in $(seq 1 "$ROUNDS"); do
        l="$(decode_ms legacy "$TG")"; s="$(decode_ms safe "$TG")"
        printf "  %-7s %10s %10s %10s\n" "$r" "$l" "$s" "$(pct "$l" "$s")"
    done
done

echo
echo "Read: per-round deltas consistent in sign = kernel-real; alternating"
echo "sign / shrinking toward 0%% = drift or fixed. Parity bar ~0%%; the"
echo "pre-fix baseline was +64%% prefill-attention, +2..6.6%% decode."
