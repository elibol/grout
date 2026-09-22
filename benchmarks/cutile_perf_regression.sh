#!/usr/bin/env bash
# Perf regression test for a cutile-rs revision: builds THIS grout source
# against two cutile-rs revisions and compares them paired on the same GPU.
#
# Why paired: session-to-session drift on one box is 5-8% (clocks, thermal,
# desktop load), far above the regressions we care about; alternating the
# two builds round by round cancels it. Tuning records are disabled on both
# arms — a record stamped for one cutile version is refused by the other
# and would turn the comparison into records-vs-defaults (seen 2026-09-21).
#
# Usage (from the grout checkout):
#   benchmarks/cutile_perf_regression.sh [BASE_REV] [CAND_REV]
# Env:
#   CUTILE_REPO   git repo to take both revisions from (default ../cutile-rs,
#                 cloned read-only into $OUT/cutile; never modified)
#   MODEL_HF      default ../hf_models/qwen3_4b
#   PROMPTS_DIR   dir with pp_<n>.txt (default: newest sweep bundle prompts)
#   ROUNDS        alternating rounds per cell (default 4; 3 reps each)
#   CELLS         default "pp18_tg128 pp2048_tg128 pp8192_tg16"
#   THRESH_PCT    fail threshold on the median, percent (default 1.5)
#   OUT           work dir (default $TMPDIR/cutile_perf_regression)
# Exit 1 when any metric regresses by more than THRESH_PCT with
# non-overlapping interquartile ranges; the table is printed either way.
set -euo pipefail
GROUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_REV="${1:-v0.3.1}"
CAND_REV="${2:-HEAD}"
CUTILE_REPO="${CUTILE_REPO:-$GROUT_DIR/../cutile-rs}"
MODEL_HF="${MODEL_HF:-$GROUT_DIR/../hf_models/qwen3_4b}"
ROUNDS="${ROUNDS:-4}"
CELLS="${CELLS:-pp18_tg128 pp2048_tg128 pp8192_tg16}"
THRESH_PCT="${THRESH_PCT:-1.5}"
OUT="${OUT:-${TMPDIR:-/tmp}/cutile_perf_regression}"
if [[ -z "${PROMPTS_DIR:-}" ]]; then
    PROMPTS_DIR="$(ls -d "$GROUT_DIR"/benchmarks/results/sweep/*/prompts 2>/dev/null | tail -1)"
fi
mkdir -p "$OUT"

# Read-only clone: checking out revisions happens here, never in CUTILE_REPO.
if [[ ! -d "$OUT/cutile/.git" ]]; then
    git clone -q "$CUTILE_REPO" "$OUT/cutile"
else
    git -C "$OUT/cutile" fetch -q origin
fi
# Resolve revs against the source repo so "HEAD" means its HEAD.
resolve() { git -C "$CUTILE_REPO" rev-parse --short "$1"; }
BASE_SHA="$(resolve "$BASE_REV")"; CAND_SHA="$(resolve "$CAND_REV")"
echo "baseline $BASE_REV=$BASE_SHA  candidate $CAND_REV=$CAND_SHA  grout=$(git -C "$GROUT_DIR" rev-parse --short HEAD)"

# Grout worktree with the crates.io patch pointed at the clone.
if [[ ! -d "$OUT/grout" ]]; then
    git -C "$GROUT_DIR" worktree add -q "$OUT/grout" HEAD
fi
python3 - "$OUT/grout/Cargo.toml" "$OUT/cutile" <<'EOF'
import re, sys
path, clone = sys.argv[1], sys.argv[2]
s = open(path).read()
crates = ["cuda-bindings", "cuda-core", "cuda-async", "cutile-compiler", "cutile"]
s = re.sub(r"\n\[patch\.crates-io\][^\[]*", "\n", s)
s = s.rstrip() + "\n\n[patch.crates-io]\n" + "".join(f'{c} = {{ path = "{clone}/{c}" }}\n' for c in crates)
open(path, "w").write(s)
EOF

build() { # sha -> binary path (on stdout); build log on stderr
    local sha="$1"
    git -C "$OUT/cutile" checkout -q "$sha"
    # The path patch only applies when the version requirement matches the
    # clone's workspace version (semver), so follow it per revision: a
    # 0.3.x baseline and a 0.4.x candidate both build from the same source.
    python3 - "$OUT/grout/Cargo.toml" "$OUT/cutile/Cargo.toml" <<'EOF'
import re, sys
manifest, ws = sys.argv[1], sys.argv[2]
ver = re.search(r'^\[workspace\.package\][^\[]*?^version\s*=\s*"([^"]+)"', open(ws).read(), re.M | re.S).group(1)
s = open(manifest).read()
for c in ["cuda-bindings", "cuda-core", "cuda-async", "cutile-compiler"]:
    s = re.sub(rf'^{c} = "[^"]+"', f'{c} = "{ver}"', s, flags=re.M)
s = re.sub(r'^cutile = \{ version = "[^"]+"', f'cutile = {{ version = "{ver}"', s, flags=re.M)
open(manifest, "w").write(s)
EOF
    (cd "$OUT/grout" && CARGO_TARGET_DIR="$OUT/target_$sha" cargo build --release --features benchmarks --bin grout_bench 2>&1 | grep -E "^error|Finished" | sed "s/^/  build $sha: /" >&2)
    [[ -x "$OUT/target_$sha/release/grout_bench" ]] || { echo "build of $sha produced no grout_bench" >&2; exit 1; }
    echo "$OUT/target_$sha/release/grout_bench"
}
BASE_BIN="$(build "$BASE_SHA")"
CAND_BIN="$(build "$CAND_SHA")"

cell_args() {
    case "$1" in
        pp18_tg128)   echo "--prompt 'Write a short poem about the sea.' --max-new-tokens 128" ;;
        pp2048_tg128) echo "--prompt-file $PROMPTS_DIR/pp_2048.txt --raw-prompt --max-new-tokens 128" ;;
        pp8192_tg16)  echo "--prompt-file $PROMPTS_DIR/pp_8192.txt --raw-prompt --max-new-tokens 16" ;;
        pp512_tg16)   echo "--prompt-file $PROMPTS_DIR/pp_512.txt --raw-prompt --max-new-tokens 16" ;;
        *) echo "unknown cell $1" >&2; exit 2 ;;
    esac
}
CSV="$OUT/rows_${BASE_SHA}_${CAND_SHA}.csv"; : > "$CSV"
bench() { # arm bin cell
    local arm="$1" bin="$2" cell="$3"
    eval set -- "$(cell_args "$cell")"
    GROUT_TUNING_RECORD_DIR=/nonexistent "$bin" --model "$MODEL_HF" --reps 3 --warmup-reps 1 \
        --ignore-eos --quiet --max-seq-len 16384 "$@" 2>/dev/null \
        | grep "\[timed\]" \
        | sed -E "s/.*prefill_ms=([0-9.]+), decode_ms=([0-9.]+).*decode_phase_tps=([0-9.]+).*/$arm,$cell,\1,\2,\3/" >> "$CSV"
}
for ((r = 1; r <= ROUNDS; r++)); do
    if (( r % 2 )); then order="base cand"; else order="cand base"; fi
    for cell in $CELLS; do
        for arm in $order; do
            if [[ $arm == base ]]; then bench base "$BASE_BIN" "$cell"; else bench cand "$CAND_BIN" "$cell"; fi
        done
    done
    echo "  round $r/$ROUNDS done"
done

# Host launch overhead per op (prefill StepGraph ops; decode replays a graph).
for arm in base cand; do
    bin="$BASE_BIN"; [[ $arm == cand ]] && bin="$CAND_BIN"
    GROUT_PROFILE_OPS=1 GROUT_TUNING_RECORD_DIR=/nonexistent "$bin" --model "$MODEL_HF" --reps 3 --warmup-reps 1 \
        --ignore-eos --profile --prompt "Write a short poem about the sea." --max-new-tokens 8 2>/dev/null \
        | sed -n '/op profile/,$p' | grep "total_ms=" \
        | sed -E "s/^\s*([A-Za-z]+)\s+total_ms=\s*([0-9.]+) avg_us=\s*([0-9.]+) calls=([0-9]+)/$arm,\1,\3,\4/" > "$OUT/ops_$arm.csv"
done

python3 - "$CSV" "$OUT/ops_base.csv" "$OUT/ops_cand.csv" "$THRESH_PCT" "$BASE_SHA" "$CAND_SHA" <<'EOF'
import sys, collections, statistics as st
csv, ops_b, ops_c, thresh, base, cand = sys.argv[1:]
thresh = float(thresh)
d = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(csv):
    arm, cell, pre, dec, tps = l.strip().split(",")
    d[cell][arm].append((float(pre), float(tps)))
q = lambda v, f: v[round((len(v) - 1) * f)]
fail = []
print(f"\n| cell | metric | {base} | {cand} | ratio | {base} IQR | {cand} IQR | n |")
print("|---|---|---:|---:|---:|---|---|---|")
for cell in d:
    for i, name, lower_better in [(0, "prefill_ms", True), (1, "decode_tok_s", False)]:
        a = sorted(x[i] for x in d[cell]["base"]); b = sorted(x[i] for x in d[cell]["cand"])
        if not a or not b: continue
        ma, mb = st.median(a), st.median(b)
        ratio = mb / ma
        worse_pct = (ratio - 1) * 100 if lower_better else (1 - ratio) * 100
        overlap = not (q(b, .25) > q(a, .75) or q(b, .75) < q(a, .25))
        verdict = ""
        if worse_pct > thresh and not overlap:
            verdict = " **REGRESSION**"; fail.append(f"{cell} {name} {worse_pct:+.1f}%")
        print(f"| {cell} | {name} | {ma:.2f} | {mb:.2f} | {ratio:.4f} | {q(a,.25):.1f}-{q(a,.75):.1f} | {q(b,.25):.1f}-{q(b,.75):.1f} | {len(a)}/{len(b)} |{verdict}")
ops = {}
for arm, path in [("base", ops_b), ("cand", ops_c)]:
    for l in open(path):
        _, op, us, calls = l.strip().split(",")
        ops.setdefault(op, {})[arm] = (float(us), int(calls))
print(f"\n| op (host launch, avg us) | calls | {base} | {cand} | ratio |")
print("|---|---:|---:|---:|---:|")
tb = tc = 0.0
for op, v in sorted(ops.items(), key=lambda kv: -kv[1].get("base", (0, 0))[0] * kv[1].get("base", (0, 0))[1]):
    if "base" in v and "cand" in v:
        tb += v["base"][0] * v["base"][1]; tc += v["cand"][0] * v["cand"][1]
        print(f"| {op} | {v['base'][1]} | {v['base'][0]:.2f} | {v['cand'][0]:.2f} | {v['cand'][0]/v['base'][0]:.3f} |")
print(f"| **sum over a prefill step** | | {tb:.0f} | {tc:.0f} | {tc/tb:.3f} |")
if fail:
    print("\nFAIL: " + "; ".join(fail)); sys.exit(1)
print(f"\nPASS: no metric regressed by more than {thresh}% with non-overlapping IQRs")
EOF
