#!/usr/bin/env python3
"""Combine numbered project reports and the work plan for review."""

from __future__ import annotations

import argparse
import base64
import re
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parent
REPORTS_DIR = REPO_ROOT / "pclaude"
NUMBERED_REPORT = re.compile(r"\d{3}-[a-z0-9]+(?:-[a-z0-9]+){1,3}\.md\Z")
WORK_PLAN_NAME = "t1-work-plan.md"


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


def markdown_fence(content: str) -> str:
    """Return a code fence longer than any backtick run in the content."""
    longest_run = max(
        (run for line in content.splitlines() for run in _backtick_runs(line)),
        default=0,
    )
    return "`" * max(4, longest_run + 1)


def collect_files(output_path: Path) -> list[Path]:
    """Select numbered Markdown reports and the current work plan."""
    excluded = {Path(__file__).resolve(), output_path.resolve()}
    files = [
        path
        for path in REPORTS_DIR.iterdir()
        if path.is_file()
        and path.resolve() not in excluded
        and (
            NUMBERED_REPORT.fullmatch(path.name)
            or path.name == WORK_PLAN_NAME
        )
    ]
    return sorted(files, key=lambda path: path.name.casefold())


def render_bundle(files: list[Path]) -> str:
    """Render all selected files with repository-relative path headings."""
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    sections = [
        "# Нумерованные отчёты и рабочий план проекта",
        "",
        f"Собрано: `{created_at}`",
        f"Файлов: **{len(files)}**",
        "",
        "Ниже приведены все нумерованные отчёты и актуальный рабочий план.",
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
        / f"project-reports-bundle-{datetime.now():%Y%m%d-%H%M%S}.md"
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
