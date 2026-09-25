#!/bin/sh
set -eu

# Snapshot the local working tree, including uncommitted source changes.
cd "$(dirname "$0")"

if ! command -v gitingest >/dev/null 2>&1; then
    printf '%s\n' 'Error: gitingest is not installed or is not on PATH.' >&2
    exit 1
fi

mkdir -p out_gitingest
stamp=$(date '+%Y-%m-%d_%H-%M-%S')
output="out_gitingest/vepkar-morph-disambig_${stamp}.md"

if [ -e "$output" ]; then
    printf 'Error: output already exists: %s\n' "$output" >&2
    exit 1
fi

gitingest . \
    --include-pattern 'README.md' \
    --include-pattern 'README.ru.md' \
    --include-pattern 'LICENSE' \
    --include-pattern 'pyproject.toml' \
    --include-pattern '.gitignore' \
    --include-pattern 'gitingest.sh' \
    --include-pattern 'src/*.py' \
    --include-pattern 'src/**/*.py' \
    --include-pattern 'src/*.md' \
    --include-pattern 'src/**/*.md' \
    --include-pattern 'tests/*.py' \
    --include-pattern 'tests/**/*.py' \
    --include-pattern 'docs/*.md' \
    --include-pattern 'docs/**/*.md' \
    --exclude-pattern 'data/*' \
    --exclude-pattern 'results/*' \
    --exclude-pattern 'out_gitingest/*' \
    --exclude-pattern '.venv/*' \
    --exclude-pattern '.git/*' \
    --exclude-pattern '*/__pycache__/*' \
    --exclude-pattern '*.pyc' \
    --exclude-pattern '*.ipynb' \
    --output "$output"

printf 'Repository snapshot: %s\n' "$output"
