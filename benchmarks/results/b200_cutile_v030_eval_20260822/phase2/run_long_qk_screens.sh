#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCH="$ROOT/target/release/grout_bench"
MODEL_HF="${MODEL_HF:?set MODEL_HF to the Qwen3-32B model directory}"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON="$OUT/long_qk_screens.jsonl"

if [[ "${RESET_SCREENS:-0}" == 1 ]]; then
    : > "$JSON"
else
    touch "$JSON"
fi

common_env=(
    GROUT_TUNING_PROFILE=sm100_b200
    GROUT_KV_CACHE_BM_S=16
    GROUT_FUSED_QK_ROPE_KV_PREFILL=1
    GROUT_FMHA_PREFILL=1
    GROUT_FMHA_PREFILL_GQA=0
    GROUT_FMHA_PREFILL_GQA_GROUP=0
    GROUT_FMHA_PREFILL_LPT_SWIZZLE=8
    GROUT_FMHA_PREFILL_LPT_SCHED=1
    GROUT_FMHA_PREFILL_OCCUPANCY=2
    GROUT_RMS_BLOCK=8192
)

run_cell() {
    local pp="$1" variant="$2" bm="$3" bn="$4" lpt="$5" latency="$6" mask="$7" qk_bm="$8"
    local prompt="$ROOT/benchmarks/results/sweep/20260820_154835_b200_32b_canonical_long_pp_cutile/prompts/pp_${pp}.txt"
    case "$pp" in
        2048) prompt="$ROOT/benchmarks/results/sweep/20260820_145859_b200_32b_canonical_pp/prompts/pp_2048.txt" ;;
        8192) prompt="$ROOT/benchmarks/results/b200_safe_eval_current/prompts/pp_8192.txt" ;;
    esac

    if grep -q "\"variant\":\"$variant\",\"pp\":$pp," "$JSON"; then
        echo "SKIP pp=$pp variant=$variant (already recorded)"
        return
    fi

    echo "START pp=$pp variant=$variant"
    local output
    output="$(env "${common_env[@]}" \
        GROUT_ATTN_BM_PREFILL="$bm" \
        GROUT_ATTN_BN_PREFILL="$bn" \
        GROUT_FMHA_PREFILL_GQA_LPT="$lpt" \
        GROUT_FMHA_PREFILL_LATENCY="$latency" \
        GROUT_FMHA_PREFILL_LPT_MASK_SPLIT="$mask" \
        GROUT_QK_PREFILL_BM="$qk_bm" \
        "$BENCH" \
        --model "$MODEL_HF" \
        --prompt-file "$prompt" \
        --raw-prompt \
        --max-new-tokens 0 \
        --max-seq-len "$((pp + 36))" \
        --reps 1 \
        --warmup-reps 1 \
        --json "$JSON" \
        --variant "$variant" \
        --pp-label "$pp" \
        --quiet 2>&1)"
    grep -E '^  \[timed\]|^  \[grout\] mean' <<< "$output"
}

for pp in 16384 32768; do
    run_cell "$pp" "lpt-bn64-lat2-mask1" 16 64 1 2 1 32
    run_cell "$pp" "lpt-bn128-lat2-mask1" 16 128 1 2 1 32
    run_cell "$pp" "lpt-bn256-lat2-mask1" 16 256 1 2 1 32
    run_cell "$pp" "lpt-bn128-lat3-mask1" 16 128 1 3 1 32
    run_cell "$pp" "lpt-bn128-lat2-mask0" 16 128 1 2 0 32
done

for pp in 2048 8192; do
    for qk_bm in 16 32 64; do
        run_cell "$pp" "qk-bm${qk_bm}" 128 128 0 2 0 "$qk_bm"
    done
done
