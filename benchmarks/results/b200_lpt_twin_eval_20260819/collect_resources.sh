#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/benchmarks/results/b200_lpt_twin_eval_20260819"
mkdir -p "$OUT/cubins"
: > "$OUT/resources.txt"

collect() {
    local arm="$1" symbol="$2" source="" destination
    while IFS= read -r candidate; do
        if cuobjdump -symbols "$candidate" 2>/dev/null | grep -q "$symbol"; then
            source="$candidate"
            break
        fi
    done < <(find "$OUT/capture_${arm}" -type f -name '*.cubin' | sort)
    if [[ -z "$source" ]]; then
        echo "missing cubin for $symbol" >&2
        exit 1
    fi

    destination="$OUT/cubins/${symbol}_bm16_bn128_group4.cubin"
    cp "$source" "$destination"
    {
        echo "==== $(basename "$destination")"
        cuobjdump -res-usage "$destination"
        printf 'LDL_STL=%s\n' "$(nvdisasm -c "$destination" | grep -cE 'LDL|STL' || true)"
    } >> "$OUT/resources.txt"
}

collect checked fmha_prefill_gqa_lpt_checked
collect twin fmha_prefill_gqa_lpt_unchecked_twin
