#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/benchmarks/results/b200_long_prefill_eval_20260819"
MODEL="$ROOT/../hf_models/qwen3_32b"
BENCH="$ROOT/target/release/grout_bench"
PROMPTS="$ROOT/benchmarks/results/sweep/20260819_011004/prompts"
CAPTURE_WRAPPER="$ROOT/benchmarks/results/b200_safe_eval_current/capture_tileiras.sh"

mkdir -p "$OUT/logs"

run_one() {
    local experiment="$1" pp="$2" arm="$3" round="$4"
    local bm bn lpt occupancy log capture_dir mean

    case "$experiment:$arm" in
        occupancy:A) bm=16;  bn=128; lpt=1; occupancy=2 ;;
        occupancy:B) bm=16;  bn=128; lpt=1; occupancy=3 ;;
        best:A)      bm=16;  bn=128; lpt=1; occupancy="${LPT_OCCUPANCY:?}" ;;
        best:B)      bm=128; bn=128; lpt=0; occupancy=2 ;;
        *) echo "unknown experiment/arm: $experiment/$arm" >&2; exit 2 ;;
    esac

    log="$OUT/logs/${experiment}_pp${pp}_${arm}_r${round}.log"
    capture_dir=""
    if [[ "$experiment:$arm:$round" == "occupancy:B:1" ]]; then
        capture_dir="${TMPDIR:?set TMPDIR to a writable temporary directory}/grout_lpt_capture_occ3_20260819/cubins"
        mkdir -p "$capture_dir"
    fi

    echo "START experiment=$experiment pp=$pp arm=$arm round=$round BM=$bm BN=$bn LPT=$lpt OCC=$occupancy"
    env \
        ${capture_dir:+CUTILE_TILEIRAS_PATH="$CAPTURE_WRAPPER"} \
        ${capture_dir:+CUTILE_CAPTURE_DIR="$capture_dir"} \
        GROUT_FMHA_PREFILL_GQA_LPT="$lpt" \
        GROUT_ATTN_BM_PREFILL="$bm" \
        GROUT_ATTN_BN_PREFILL="$bn" \
        GROUT_FMHA_PREFILL_LPT_SWIZZLE=8 \
        GROUT_FMHA_PREFILL_LPT_SCHED=1 \
        GROUT_FMHA_PREFILL_LATENCY=2 \
        GROUT_FMHA_PREFILL_OCCUPANCY="$occupancy" \
        "$BENCH" \
        --model "$MODEL" \
        --prompt-file "$PROMPTS/pp_${pp}.txt" \
        --raw-prompt \
        --max-new-tokens 0 \
        --max-seq-len "$pp" \
        --reps 3 \
        --warmup-reps 1 \
        --quiet >"$log" 2>&1

    mean="$(grep -oE '\[timed\].*prefill_ms=[0-9.]+' "$log" \
        | grep -oE 'prefill_ms=[0-9.]+' | cut -d= -f2 \
        | awk '{sum += $1; n++} END {if (n) printf "%.3f", sum / n}')"
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$experiment" "$pp" "$arm" "$round" "$bm" "$bn" "$occupancy" "$mean" \
        | tee -a "$OUT/results.csv"
}

paired() {
    local experiment="$1" pp="$2" round arm
    for round in 1 2 3; do
        if (( round % 2 )); then
            for arm in A B; do run_one "$experiment" "$pp" "$arm" "$round"; done
        else
            for arm in B A; do run_one "$experiment" "$pp" "$arm" "$round"; done
        fi
    done
}

if [[ ! -f "$OUT/results.csv" ]]; then
    echo 'experiment,pp,arm,round,bm,bn,occupancy,prefill_ms' >"$OUT/results.csv"
fi

case "${1:?usage: $0 occupancy|best}" in
    occupancy) paired occupancy 32768 ;;
    best)
        : "${LPT_OCCUPANCY:?set LPT_OCCUPANCY to the occupancy-ladder winner}"
        paired best 16384
        paired best 32768
        ;;
    *) echo "unknown experiment: $1" >&2; exit 2 ;;
esac
