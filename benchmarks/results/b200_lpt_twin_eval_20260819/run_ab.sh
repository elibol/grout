#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/benchmarks/results/b200_lpt_twin_eval_20260819"
MODEL="${MODEL_HF:-$ROOT/../hf_models/qwen3_32b}"
BENCH="$ROOT/target/release/grout_bench"
PROMPTS="$ROOT/benchmarks/results/sweep/20260819_221611/prompts"
CAPTURE_WRAPPER="$ROOT/benchmarks/results/b200_safe_eval_current/capture_tileiras.sh"

mkdir -p "$OUT/logs" "$OUT/text"
rm -f "$OUT/results.csv" "$OUT/runs.jsonl"
echo 'pp,arm,round,prefill_ms,decode_ms,e2e_ms,text_sha256' > "$OUT/results.csv"

run_one() {
    local pp="$1" arm="$2" round="$3" variant log text_file
    local prefill_ms decode_ms e2e_ms text_sha
    local -a capture_env=()

    case "$arm" in
        checked) ;;
        twin) ;;
        *) echo "unknown arm: $arm" >&2; exit 2 ;;
    esac

    variant="${arm}_r${round}"
    log="$OUT/logs/pp${pp}_${arm}_r${round}.log"
    text_file="$OUT/text/pp${pp}_${arm}_r${round}.txt"
    if [[ "$pp:$round" == "16384:1" ]]; then
        capture_env=(
            XDG_CACHE_HOME="$OUT/jit_cache_${arm}"
            CUTILE_TILEIRAS_PATH="$CAPTURE_WRAPPER"
            CUTILE_CAPTURE_DIR="$OUT/capture_${arm}"
        )
    fi

    echo "START pp=$pp arm=$arm round=$round"
    local -a common_env=(
        GROUT_CUDA_GRAPH_DECODE=1
        GROUT_FLASH_DECODE=0
        GROUT_FMHA_SPLIT_KV=1
        GROUT_FMHA_DECODE_LATENCY=4
        GROUT_FMHA_DECODE_OCCUPANCY=2
        GROUT_FMHA_MERGE_CHUNK_D=16
        GROUT_FMHA_MERGE_LATENCY=2
        GROUT_FUSED_QK_ROPE_KV_DECODE=1
        GROUT_QK_ROPE_LATENCY=2
        GROUT_QK_ROPE_OCCUPANCY=1
        GROUT_QK_ROPE_CGA=0
        GROUT_KV_CACHE_DYN_CHUNK_D=32
        GROUT_EMBED_BLOCK=1024
        GROUT_RMS_BLOCK=8192
        GROUT_ARGMAX_BLOCK=128
        GROUT_KV_CACHE_BM_S=16
        GROUT_FUSED_QK_ROPE_KV_PREFILL=1
        GROUT_ATTN_BM_PREFILL=16
        GROUT_ATTN_BN_PREFILL=128
        GROUT_FMHA_PREFILL_GQA=0
        GROUT_FMHA_PREFILL_GQA_LPT=1
        GROUT_FMHA_PREFILL_GQA_GROUP=4
        GROUT_FMHA_PREFILL_LPT_SWIZZLE=8
        GROUT_FMHA_PREFILL_LPT_SCHED=1
        GROUT_FMHA_PREFILL_LPT_MASK_SPLIT=1
        GROUT_FMHA_PREFILL_LATENCY=2
        GROUT_FMHA_PREFILL_OCCUPANCY=2
        GROUT_ATTN_BN_DECODE=64
        GROUT_FMHA_NUM_KV_SPLITS=32
    )
    local -a cmd=(
        "$BENCH"
        --model "$MODEL"
        --prompt-file "$PROMPTS/pp_${pp}.txt"
        --raw-prompt
        --max-new-tokens 36
        --max-seq-len "$((pp + 36))"
        --reps 3
        --warmup-reps 1
        --variant "$variant"
        --pp-label "$pp"
        --json "$OUT/runs.jsonl"
        --ignore-eos
    )

    if [[ "$arm" == checked ]]; then
        env -u GROUT_FMHA_PREFILL_LPT_UNSAFE_TWIN \
            "${capture_env[@]}" "${common_env[@]}" "${cmd[@]}" >"$log" 2>&1
    else
        env "${capture_env[@]}" "${common_env[@]}" \
            GROUT_FMHA_PREFILL_LPT_UNSAFE_TWIN=1 \
            "${cmd[@]}" >"$log" 2>&1
    fi

    awk 'found { print } /  \[grout\] mean over/ { found=1 }' "$log" \
        | sed '1{/^$/d;}' > "$text_file"
    text_sha="$(sha256sum "$text_file" | awk '{print $1}')"
    read -r prefill_ms decode_ms e2e_ms < <(
        grep '\[timed\]' "$log" | awk '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^prefill_ms=/) { split($i, a, "="); p += a[2] }
                    if ($i ~ /^decode_ms=/)  { split($i, a, "="); d += a[2] }
                    if ($i ~ /^e2e_ms=/)     { split($i, a, "="); e += a[2] }
                }
                n++
            }
            END { printf "%.3f %.3f %.3f\n", p/n, d/n, e/n }
        '
    )
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$pp" "$arm" "$round" "$prefill_ms" "$decode_ms" "$e2e_ms" "$text_sha" \
        | tee -a "$OUT/results.csv"
}

paired() {
    local pp="$1"
    run_one "$pp" checked 1
    run_one "$pp" twin 1
    run_one "$pp" twin 2
    run_one "$pp" checked 2
    run_one "$pp" checked 3
    run_one "$pp" twin 3
}

paired 16384
paired 32768

for pp in 16384 32768; do
    baseline="$OUT/text/pp${pp}_checked_r1.txt"
    for candidate in "$OUT"/text/pp"${pp}"_*.txt; do
        cmp "$baseline" "$candidate"
    done
    echo "OUTPUT_IDENTICAL pp=$pp sha256=$(sha256sum "$baseline" | awk '{print $1}')"
done
