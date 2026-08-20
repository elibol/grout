#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/benchmarks/results/b200_long_prefill_attribution_20260819"
MODEL="${MODEL_HF:-$ROOT/../hf_models/qwen3_32b}"
BENCH="$ROOT/target/release/grout_bench"
PROMPT="$ROOT/benchmarks/results/sweep/20260819_221611/prompts/pp_32768.txt"

mkdir -p "$OUT/nsys_tmp"
export TMPDIR="$OUT/nsys_tmp"

nsys profile \
    --trace=cuda,osrt \
    --cuda-trace-scope=process-tree \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --stats=false \
    --output="$OUT/pp32768_group8" \
    /usr/bin/env \
        GROUT_CUDA_GRAPH_DECODE=1 \
        GROUT_FMHA_PREFILL_GQA_LPT=1 \
        GROUT_ATTN_BM_PREFILL=16 \
        GROUT_ATTN_BN_PREFILL=128 \
        GROUT_FMHA_PREFILL_GQA_GROUP=8 \
        GROUT_FMHA_PREFILL_LPT_SWIZZLE=8 \
        GROUT_FMHA_PREFILL_LPT_SCHED=1 \
        GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=1 \
        GROUT_FMHA_PREFILL_LATENCY=2 \
        GROUT_FMHA_PREFILL_OCCUPANCY=2 \
        "$BENCH" \
        --model "$MODEL" \
        --prompt-file "$PROMPT" \
        --raw-prompt \
        --max-new-tokens 0 \
        --max-seq-len 32768 \
        --reps 1 \
        --warmup-reps 0 \
        --quiet \
        >"$OUT/logs/nsys_pp32768_group8.log" 2>&1

if [[ -f "$OUT/pp32768_group8.nsys-rep" ]]; then
    nsys stats \
        --force-export=true \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output "$OUT/pp32768_group8_kern" \
        "$OUT/pp32768_group8.nsys-rep" \
        >/dev/null
    mv -f "$OUT/pp32768_group8_kern_cuda_gpu_kern_sum.csv" \
        "$OUT/pp32768_group8_kern.csv"
else
    echo "Nsight importer unavailable; raw pp32768_group8.qdstrm retained" >&2
fi
