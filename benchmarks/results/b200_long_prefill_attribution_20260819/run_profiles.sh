#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/benchmarks/results/b200_long_prefill_attribution_20260819"
MODEL="${MODEL_HF:-$ROOT/../hf_models/qwen3_32b}"
BENCH="$ROOT/target/release/grout_bench"
PROMPTS="$ROOT/benchmarks/results/sweep/20260819_221611/prompts"

mkdir -p "$OUT/logs"

common_env=(
    GROUT_CUDA_GRAPH_DECODE=1
    GROUT_FMHA_PREFILL_GQA_LPT=1
    GROUT_ATTN_BM_PREFILL=16
    GROUT_ATTN_BN_PREFILL=128
    GROUT_FMHA_PREFILL_GQA_GROUP=8
    GROUT_FMHA_PREFILL_LPT_SWIZZLE=8
    GROUT_FMHA_PREFILL_LPT_SCHED=1
    GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=1
    GROUT_FMHA_PREFILL_LATENCY=2
    GROUT_FMHA_PREFILL_OCCUPANCY=2
    GROUT_PROFILE_OPS=1
    GROUT_PROFILE_SYNC_OPS=1
)

for pp in 16384 32768; do
    echo "START sync-op profile pp=$pp"
    env "${common_env[@]}" \
        "$BENCH" \
        --model "$MODEL" \
        --prompt-file "$PROMPTS/pp_${pp}.txt" \
        --raw-prompt \
        --max-new-tokens 0 \
        --max-seq-len "$pp" \
        --reps 1 \
        --warmup-reps 1 \
        --profile \
        >"$OUT/logs/op_profile_pp${pp}.log" 2>&1
    sed -n '/perf profile/,$p' "$OUT/logs/op_profile_pp${pp}.log"
done
