#!/usr/bin/env bash
# Source this file in Bash. Sourcing starts no experiment and creates no run.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Load this helper with: source %q\n' "${BASH_SOURCE[0]}" >&2
    exit 2
fi

_T1_TOOLS_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P) || return 1

_t1_load_settings() {
    local repo="$_T1_TOOLS_ROOT"
    local T1_DATA_DIR="data/dictorpus-data"
    local T1_SPLIT_FILE="data/derived/splits_v2026.09.csv"
    local T1_RUNS_DIR="${repo}.addon/vepkar-runs"
    local T1_PYTHON=".venv/bin/python"
    local T1_C="1.0"
    local config="$repo/.env.t1.local"
    if [[ -e "$config" || -L "$config" ]]; then
        [[ -f "$config" && -r "$config" ]] || {
            printf 'STOP: configuration is not a readable file: %s\n' "$config" >&2
            return 1
        }
        source "$config" || return 1
    fi
    local requested_c="${1:-$T1_C}"
    local py="$T1_PYTHON"
    [[ "$py" == /* ]] || py="$repo/$py"
    [[ -x "$py" ]] || { printf 'STOP: Python not executable: %s\n' "$py" >&2; return 1; }
    local resolved
    resolved=$("$py" - "$repo" "$T1_DATA_DIR" "$T1_SPLIT_FILE" "$T1_RUNS_DIR" "$py" "$requested_c" <<'PY'
import math
import sys
from pathlib import Path
root = Path(sys.argv[1])
for value in sys.argv[2:6]:
    if not value or '\n' in value or '\r' in value:
        raise SystemExit('STOP: configured paths must be nonempty single-line strings')
    p = Path(value).expanduser()
    print((p if p.is_absolute() else root / p).absolute())
try:
    c = float(sys.argv[6])
except ValueError:
    raise SystemExit('STOP: C must be numeric')
if not math.isfinite(c) or c <= 0:
    raise SystemExit('STOP: C must be finite and positive')
print(c)
print(format(c, '.17g'))
PY
    ) || return 1
    local -a fields
    mapfile -t fields <<< "$resolved"
    local data="${fields[0]}" split="${fields[1]}" runs="${fields[2]}"
    [[ -d "$data/corpus" ]] || { printf 'STOP: corpus directory not found: %s\n' "$data/corpus" >&2; return 1; }
    [[ -f "$split" ]] || { printf 'STOP: split file not found: %s\n' "$split" >&2; return 1; }
    REPO="$repo" DATA="$data" SPLIT="$split" RUNS="$runs" PY="${fields[3]}" C="${fields[4]}"
    _T1_C_LABEL="${fields[5]}"
    export REPO DATA SPLIT RUNS PY C
}

_t1_preflight() {
    "$PY" -c "import platform, numpy, scipy, sklearn, joblib; print('python', platform.python_version()); print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('scikit-learn', sklearn.__version__); print('joblib', joblib.__version__)" || return 1
    git --version || return 1
}

t1_run_paths() {
    printf 'REPO=%s\nDATA=%s\nSPLIT=%s\nRUNS=%s\nRUN=%s\nC=%s\nPY=%s\n' \
        "${REPO:-<unset>}" "${DATA:-<unset>}" "${SPLIT:-<unset>}" \
        "${RUNS:-<unset>}" "${RUN:-<unset>}" "${C:-<unset>}" "${PY:-<unset>}"
}

setup_t1_run() {
    if (( $# > 1 )); then
        printf 'Usage: setup_t1_run [C]\n' >&2
        return 2
    fi
    if [[ -n "${RUN:-}" ]]; then
        printf 'STOP: RUN is already set to %s. Use it, or unset RUN intentionally for a new attempt.\n' "$RUN" >&2
        return 1
    fi
    _t1_load_settings "${1:-}" || return 1
    _t1_preflight || return 1
    cd -- "$REPO" || return 1
    mkdir -p -- "$RUNS" || return 1
    local new_run
    new_run=$(mktemp -d "$RUNS/t1-dev-C${_T1_C_LABEL}-XXXXXX") || return 1
    RUN="$new_run"
    export RUN
    t1_run_paths
}

use_t1_run() {
    if (( $# != 1 )) || [[ -z "$1" ]]; then
        printf 'Usage: use_t1_run EXISTING_RUN_DIRECTORY\n' >&2
        return 2
    fi
    local requested="$1" resolved_run
    [[ -d "$requested" ]] || { printf 'STOP: run directory not found: %s\n' "$requested" >&2; return 1; }
    resolved_run=$(cd -- "$requested" && pwd -P) || return 1
    if [[ -n "${RUN:-}" && "$RUN" != "$resolved_run" ]]; then
        printf 'STOP: another RUN is active: %s. Unset RUN intentionally before switching.\n' "$RUN" >&2
        return 1
    fi
    _t1_load_settings || return 1
    _t1_preflight || return 1
    cd -- "$REPO" || return 1
    RUN="$resolved_run"
    export RUN
    t1_run_paths
    printf 'Existing run selected; nothing was trained or overwritten. Check its passport before continuing.\n'
}

run_stage() (
    if (( $# < 2 )); then
        printf 'Usage: run_stage STAGE COMMAND [ARGUMENT ...]\nNo command was executed and no logs were created.\n' >&2
        return 2
    fi
    local stage="$1"
    shift
    if [[ ! "$stage" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ || -z "$1" ]]; then
        printf 'STOP: invalid stage name or empty command.\n' >&2
        return 2
    fi
    if [[ -z "${RUN:-}" || "$RUN" != /* || ! -d "$RUN" ]]; then
        printf 'STOP: RUN must be an absolute existing run directory.\n' >&2
        return 2
    fi
    local base="$RUN/$stage" suffix rc
    for suffix in command.txt stdout.log stderr.log exitcode; do
        if [[ -e "$base.$suffix" || -L "$base.$suffix" ]]; then
            printf 'STOP: stage %s already has logs in %s; not overwriting.\n' "$stage" "$RUN" >&2
            return 1
        fi
    done
    local command_fd stdout_fd stderr_fd status_fd
    set -o noclobber
    if ! exec {command_fd}>"$base.command.txt" {stdout_fd}>"$base.stdout.log" \
              {stderr_fd}>"$base.stderr.log" {status_fd}>"$base.exitcode"; then
        printf 'STOP: could not reserve all logs. No command executed; inspect partial logs in %s.\n' "$RUN" >&2
        return 1
    fi
    { printf '%q ' "$@"; printf '\n'; } >&"$command_fd" || return 1
    printf '\n=== %s ===\n' "$stage"
    if "$@" >&"$stdout_fd" 2>&"$stderr_fd"; then rc=0; else rc=$?; fi
    if ! printf '%s\n' "$rc" >&"$status_fd"; then
        printf 'STOP: command returned %s, but exitcode could not be recorded.\n' "$rc" >&2
        return 1
    fi
    exec {command_fd}>&- {stdout_fd}>&- {stderr_fd}>&- {status_fd}>&-
    local display_failed=0
    cat -- "$base.stdout.log" || display_failed=1
    if [[ -s "$base.stderr.log" ]]; then
        printf '\n--- stderr ---\n' >&2
        cat -- "$base.stderr.log" >&2 || display_failed=1
    fi
    printf '\nStage %s exit code: %s\n' "$stage" "$rc"
    if (( display_failed )); then
        printf 'WARNING: displaying logs failed; recorded command status is %s.\n' "$rc" >&2
        (( rc != 0 )) || return 1
    fi
    return "$rc"
)

check_t1_passport() {
    if [[ -z "${RUN:-}" || -z "${REPO:-}" || -z "${PY:-}" || ! -x "$PY" ]]; then
        printf 'STOP: select a run using setup_t1_run or use_t1_run first.\n' >&2
        return 2
    fi
    "$PY" - <<'PY'
import json
import os
import sys
from pathlib import Path
repo = Path(os.environ['REPO']).resolve()
sys.path.insert(0, str(repo / 'src'))
from benchmark_context import _preflight_benchmark
from t1_features.run_manifest import collect_run_manifest
path = Path(os.environ['RUN']) / 'run.json'
if not path.is_file():
    raise SystemExit(f'STOP: passport not found: {path}')
expected = json.loads(path.read_text(encoding='utf-8'))
data, tag = _preflight_benchmark(Path(os.environ['DATA']))
current = collect_run_manifest(
    data_dir=data, data_tag=tag,
    split_file=Path(os.environ['SPLIT']), C=float(os.environ['C']),
)
changed = [key for key in sorted(set(expected) | set(current))
           if expected.get(key) != current.get(key)]
if changed:
    raise SystemExit('STOP: passport differs from current metadata: ' + ', '.join(changed))
print('Run passport matches current metadata.')
PY
}
