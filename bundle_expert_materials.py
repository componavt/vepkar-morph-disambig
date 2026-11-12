#!/usr/bin/env python3
"""Combine the expert-materials folder into one reviewable Markdown file."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).absolute()
REPO_ROOT = SCRIPT_PATH.parent
EXPERTS_DIR = REPO_ROOT / "pclaude" / "experts"


def markdown_fence(content: str) -> str:
    """Return a code fence longer than any backtick run in the content."""
    longest_run = max(
        (run for line in content.splitlines() for run in _backtick_runs(line)),
        default=0,
    )
    return "`" * max(4, longest_run + 1)


def _backtick_runs(line: str):
    run_length = 0
    for character in line:
        if character == "`":
            run_length += 1
        elif run_length:
            yield run_length
            run_length = 0
    if run_length:
        yield run_length


def collect_files(output_path: Path) -> list[Path]:
    """List regular files recursively, excluding this script and the bundle."""
    excluded = {SCRIPT_PATH.resolve(), output_path.resolve()}
    return sorted(
        (
            path
            for path in EXPERTS_DIR.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.resolve() not in excluded
        ),
        key=lambda path: path.relative_to(EXPERTS_DIR).as_posix().casefold(),
    )


def render_bundle(files: list[Path]) -> str:
    """Render each file with its relative path and complete contents."""
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    sections = [
        "# Экспертные материалы для проверки наставником",
        "",
        f"Собрано: `{created_at}`",
        f"Файлов: **{len(files)}**",
        "",
        "Содержимое каждого файла приведено ниже; путь указан перед ним.",
        "",
    ]

    for path in files:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        raw_content = path.read_bytes()
        sections.extend([f"## Файл от корня репозитория: `{relative_path}`", ""])
        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError:
            encoded = base64.b64encode(raw_content).decode("ascii")
            sections.extend(
                [
                    f"Бинарный файл, размер: {len(raw_content)} байт; содержимое в Base64.",
                    "",
                    "```base64",
                    encoded,
                    "```",
                    "",
                ]
            )
        else:
            fence = markdown_fence(content)
            sections.extend([fence + "text", content.rstrip("\n"), fence, ""])

    return "\n".join(sections).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        REPO_ROOT
        / "work"
        / "gitingest"
        / f"experts-review-bundle-{datetime.now():%Y%m%d-%H%M%S}.md"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output Markdown path (default: timestamped file under work/gitingest/)",
    )
    args = parser.parse_args()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(output_path)
    output_path.write_text(render_bundle(files), encoding="utf-8")
    print(f"Wrote {len(files)} files to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
