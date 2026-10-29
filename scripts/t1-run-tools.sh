#!/usr/bin/env bash
# Run one fresh T1.1 experiment: prepare, train, predict, evaluate.
# Usage: bash scripts/t1-run-tools.sh [C]

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 't1-run-tools: do not source this script; run it with: bash %q\n' "${BASH_SOURCE[0]}" >&2
    return 1
fi

if (($# > 1)); then
    printf 'Usage: %s [C]\n' "$(basename -- "$0")" >&2
    exit 2
fi

C="${1-1.0}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || exit 1
PY="$ROOT/.venv/bin/python"
CLI="$ROOT/src/cli.py"

if [[ ! -f "$PY" || ! -x "$PY" ]]; then
    printf 't1-run-tools: project interpreter not executable: %s\n' "$PY" >&2
    exit 1
fi

cd -- "$ROOT" || exit 1

prep_out=$("$PY" "$CLI" prepare-t1-run "--C=$C")
prep_status=$?
[[ -n "$prep_out" ]] && printf '%s\n' "$prep_out"
if ((prep_status != 0)); then
    exit "$prep_status"
fi

run_dir=""
run_lines=0
while IFS= read -r line; do
    case "$line" in
        "Run directory: "*)
            path="${line#Run directory: }"
            if [[ -n "$path" ]]; then
                run_lines=$((run_lines + 1))
                run_dir="$path"
            fi
            ;;
    esac
done <<< "$prep_out"

if ((run_lines != 1)) || [[ -z "$run_dir" ]] || [[ "$run_dir" != /* ]]; then
    printf 't1-run-tools: preparation did not report exactly one absolute Run directory line; aborting.\n' >&2
    exit 1
fi

"$PY" "$CLI" train-t1 --run-dir "$run_dir"
rc=$?
if ((rc != 0)); then
    printf 't1-run-tools: stage train-t1 failed with exit %s; run directory: %s\n' "$rc" "$run_dir" >&2
    exit "$rc"
fi

"$PY" "$CLI" predict-t1 --run-dir "$run_dir"
rc=$?
if ((rc != 0)); then
    printf 't1-run-tools: stage predict-t1 failed with exit %s; run directory: %s\n' "$rc" "$run_dir" >&2
    exit "$rc"
fi

"$PY" "$CLI" evaluate-t1-run --run-dir "$run_dir"
rc=$?
if ((rc != 0)); then
    printf 't1-run-tools: stage evaluate-t1-run failed with exit %s; run directory: %s\n' "$rc" "$run_dir" >&2
    exit "$rc"
fi

printf 'T1.1 run complete\n'
printf 'Run directory: %s\n' "$run_dir"