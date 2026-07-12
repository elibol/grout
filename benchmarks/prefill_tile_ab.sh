#!/usr/bin/env bash
# Paired, order-alternating comparison of two safe mapped prefill tile forms.

set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "usage: $0 PP A_GQA A_BM A_BN B_GQA B_BM B_BN" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
ROUNDS="${ROUNDS:-3}"
REPS="${REPS:-5}"
WARMUP_REPS="${WARMUP_REPS:-2}"
PP="$1"
A_GQA="$2"; A_BM="$3"; A_BN="$4"
B_GQA="$5"; B_BM="$6"; B_BN="$7"
MAX_SEQ_LEN=$((PP + 36))
(( MAX_SEQ_LEN < 4096 )) && MAX_SEQ_LEN=4096

PROMPTS_DIR="$(mktemp -d)"
trap 'rm -rf "$PROMPTS_DIR"' EXIT
PY="${VLLM_PY:-$GROUT_DIR/../bench_envs/vllm_env/bin/python3}"
[[ -x "$PY" ]] || PY=python3
"$PY" "$SCRIPT_DIR/make_prompts.py" \
    --model "$MODEL_HF" --out-dir "$PROMPTS_DIR" --pp "$PP" >/dev/null

(cd "$GROUT_DIR" && cargo build --release --features benchmarks --bin grout_bench >/dev/null)

run_one() {
    local gqa="$1" bm="$2" bn="$3"
    GROUT_FMHA_PREFILL_GQA_LPT=0 \
    GROUT_FMHA_PREFILL_GQA="$gqa" \
    GROUT_ATTN_BM_PREFILL="$bm" \
    GROUT_ATTN_BN_PREFILL="$bn" \
        "$GROUT_DIR/target/release/grout_bench" \
        --model "$MODEL_HF" \
        --prompt-file "$PROMPTS_DIR/pp_${PP}.txt" --raw-prompt \
        --max-new-tokens 36 --max-seq-len "$MAX_SEQ_LEN" \
        --reps "$REPS" --warmup-reps "$WARMUP_REPS" \
        --ignore-eos --quiet 2>&1 \
      | awk -F'prefill_ms=' '/^  \[timed\]/{split($2,a,","); print a[1]}' \
      | sort -n \
      | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%+.2f%%", 100*(b-a)/a}'; }

echo "pp=$PP reps=$REPS warmup=$WARMUP_REPS rounds=$ROUNDS"
echo "A: GQA=$A_GQA BM=$A_BM BN=$A_BN"
echo "B: GQA=$B_GQA BM=$B_BM BN=$B_BN"
printf '%-6s %-8s %-10s %-10s %-10s\n' round order a_ms b_ms b_delta
for round in $(seq 1 "$ROUNDS"); do
    if (( round % 2 == 1 )); then
        a="$(run_one "$A_GQA" "$A_BM" "$A_BN")"
        b="$(run_one "$B_GQA" "$B_BM" "$B_BN")"
        order=AB
    else
        b="$(run_one "$B_GQA" "$B_BM" "$B_BN")"
        a="$(run_one "$A_GQA" "$A_BM" "$A_BN")"
        order=BA
    fi
    printf '%-6s %-8s %-10s %-10s %-10s\n' \
        "$round" "$order" "$a" "$b" "$(pct "$a" "$b")"
done
