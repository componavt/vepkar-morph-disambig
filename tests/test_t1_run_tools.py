"""Black-box tests for ``scripts/t1-run-tools.sh`` as a subprocess.

Each test copies the real script into an owned temporary fake project
tree with an executable fake ``.venv/bin/python`` interpreter and a
placeholder ``src/cli.py``.  The fake interpreter records its received
arguments and returns controlled outputs/statuses from a plan file.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REAL_SCRIPT = ROOT / "scripts" / "t1-run-tools.sh"

FAKE_PYTHON = r'''#!/usr/bin/env python3
import json
import os
import sys

record = os.environ.get("T1_FAKE_RECORD")
if record:
    with open(record, "a", encoding="utf-8") as fh:
        json.dump({"argv": sys.argv, "cwd": os.getcwd()}, fh)
        fh.write("\n")
plan_path = os.environ.get("T1_FAKE_PLAN")
if plan_path and os.path.exists(plan_path):
    with open(plan_path, encoding="utf-8") as fh:
        plan = json.load(fh)
    command = sys.argv[2] if len(sys.argv) > 2 else ""
    for entry in plan:
        if entry.get("command") == command:
            if entry.get("out"):
                sys.stdout.write(entry["out"])
            if entry.get("err"):
                sys.stderr.write(entry["err"])
            raise SystemExit(entry.get("status", 0))
raise SystemExit(0)
'''

FAKE_CLI = r'''from pathlib import Path
Path(__file__).with_name("cli-ran.marker").write_text("ran\n", encoding="utf-8")
raise SystemExit(99)
'''


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_fake_project(root: Path, with_python: bool = True) -> SimpleNamespace:
    root.mkdir(parents=True, exist_ok=True)
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    script = scripts / "t1-run-tools.sh"
    script.write_bytes(REAL_SCRIPT.read_bytes())
    _make_executable(script)
    py = root / ".venv" / "bin" / "python"
    if with_python:
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(FAKE_PYTHON, encoding="utf-8")
        _make_executable(py)
    cli = root / "src" / "cli.py"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text(FAKE_CLI, encoding="utf-8")
    return SimpleNamespace(
        root=root,
        script=script,
        py=py,
        cli=cli,
        record=root.parent / "invocations.jsonl",
        plan=root.parent / "plan.json",
    )


@pytest.fixture
def fake_project(tmp_path) -> SimpleNamespace:
    return _make_fake_project(tmp_path / "project")


def _write_plan(plan: Path, entries) -> None:
    plan.write_text(json.dumps(entries), encoding="utf-8")


def _run_script(script, record, plan, *args, cwd=None):
    env = dict(os.environ)
    env["T1_FAKE_RECORD"] = str(record)
    env["T1_FAKE_PLAN"] = str(plan)
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )


def _invocations(record: Path):
    if not record.exists():
        return []
    return [
        json.loads(line)
        for line in record.read_text(encoding="utf-8").splitlines()
    ]


def _success_plan(run_dir: Path) -> list[dict]:
    return [
        {
            "command": "prepare-t1-run",
            "out": f"T1 run prepared\nRun directory: {run_dir}\nC: 1.0\n",
            "status": 0,
        },
        {"command": "train-t1", "out": "T1 training: OK\n", "status": 0},
        {"command": "predict-t1", "out": "T1 prediction: OK\n", "status": 0},
        {"command": "evaluate-t1-run", "out": "Evaluation: OK\n", "status": 0},
    ]


class TestSuccessfulRuns:
    def test_default_c_run_four_commands_in_order(self, fake_project, tmp_path):
        run_dir = tmp_path / "runs" / "run-a"
        _write_plan(fake_project.plan, _success_plan(run_dir))
        result = _run_script(
            fake_project.script, fake_project.record, fake_project.plan
        )
        assert result.returncode == 0, result.stderr
        invocations = _invocations(fake_project.record)
        assert [inv["argv"][2] for inv in invocations] == [
            "prepare-t1-run",
            "train-t1",
            "predict-t1",
            "evaluate-t1-run",
        ]
        assert invocations[0]["argv"][3] == "--C=1.0"
        run_dirs = {inv["argv"][4] for inv in invocations[1:]}
        assert run_dirs == {str(run_dir)}
        assert all(
            inv["argv"][3] == "--run-dir" for inv in invocations[1:]
        )
        assert not any(
            "export-t1-milestone" in inv["argv"] for inv in invocations
        )
        assert "T1.1 run complete" in result.stdout
        assert f"Run directory: {run_dir}" in result.stdout
        files = sorted(
            str(path.relative_to(fake_project.root))
            for path in fake_project.root.rglob("*")
            if path.is_file()
        )
        assert files == [
            ".venv/bin/python",
            "scripts/t1-run-tools.sh",
            "src/cli.py",
        ]

    def test_supplied_c_spaces_in_paths_and_foreign_cwd(self, tmp_path):
        base = tmp_path / "fake project area"
        project = _make_fake_project(base / "my project")
        run_dir = base / "run area" / "run one"
        elsewhere = base / "elsewhere"
        elsewhere.mkdir(parents=True)
        plan = _success_plan(run_dir)
        plan[0]["out"] = (
            f"T1 run prepared\nRun directory: {run_dir}\nC: 0.1\n"
        )
        _write_plan(project.plan, plan)
        result = _run_script(
            project.script, project.record, project.plan, "0.1", cwd=elsewhere
        )
        assert result.returncode == 0, result.stderr
        invocations = _invocations(project.record)
        assert len(invocations) == 4
        assert invocations[0]["argv"][3] == "--C=0.1"
        for invocation in invocations:
            assert Path(invocation["argv"][0]) == project.py
            assert Path(invocation["cwd"]) == project.root
        run_dirs = {inv["argv"][4] for inv in invocations[1:]}
        assert run_dirs == {str(run_dir)}
        assert "T1 training: OK" in result.stdout
        assert "T1.1 run complete" in result.stdout


class TestStageFailures:
    @pytest.mark.parametrize(
        ("stage", "command", "status", "expected_len"),
        [
            ("prepare", "prepare-t1-run", 7, 1),
            ("train", "train-t1", 17, 2),
            ("predict", "predict-t1", 23, 3),
            ("evaluate", "evaluate-t1-run", 29, 4),
        ],
    )
    def test_failure_stops_pipeline_and_reports_status(
        self, fake_project, tmp_path, stage, command, status, expected_len
    ):
        run_dir = tmp_path / "runs" / "run-a"
        entries = _success_plan(run_dir)
        for entry in entries:
            if entry["command"] == command:
                entry["out"] = f"{stage} stdout text\n"
                entry["err"] = f"{stage} stderr text\n"
                entry["status"] = status
        _write_plan(fake_project.plan, entries)
        result = _run_script(
            fake_project.script, fake_project.record, fake_project.plan
        )
        assert result.returncode == status
        invocations = _invocations(fake_project.record)
        assert [inv["argv"][2] for inv in invocations] == [
            entry["command"] for entry in entries[:expected_len]
        ]
        if stage == "prepare":
            assert "prepare stdout text" in result.stdout
            assert "prepare stderr text" in result.stderr
        else:
            assert str(run_dir) in result.stderr
            assert f"stage {command}" in result.stderr


class TestReportContract:
    @pytest.mark.parametrize("kind", ["missing", "duplicate"])
    def test_refuses_without_continuation(self, fake_project, tmp_path, kind):
        run_dir = tmp_path / "runs" / "run-a"
        if kind == "missing":
            out = "T1 run prepared\nNo directory line here.\n"
        else:
            out = (
                f"T1 run prepared\nRun directory: {run_dir}\n"
                f"Run directory: {tmp_path / 'runs' / 'other'}\n"
            )
        _write_plan(
            fake_project.plan,
            [{"command": "prepare-t1-run", "out": out, "status": 0}],
        )
        result = _run_script(
            fake_project.script, fake_project.record, fake_project.plan
        )
        assert result.returncode != 0
        assert "exactly one absolute Run directory" in result.stderr
        assert len(_invocations(fake_project.record)) == 1


class TestInterface:
    def test_too_many_arguments_start_no_experiment(self, fake_project):
        result = _run_script(
            fake_project.script,
            fake_project.record,
            fake_project.plan,
            "1",
            "2",
        )
        assert result.returncode == 2
        assert "Usage" in result.stderr
        assert not fake_project.record.exists()

    def test_sourced_invocation_rejected_without_killing_caller(
        self, fake_project
    ):
        env = dict(os.environ)
        env["T1_FAKE_RECORD"] = str(fake_project.record)
        env["T1_FAKE_PLAN"] = str(fake_project.plan)
        result = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; echo after-source',
                "probe",
                str(fake_project.script),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert "after-source" in result.stdout
        assert "do not source" in result.stderr
        assert not fake_project.record.exists()

    def test_missing_project_interpreter_without_system_fallback(self, tmp_path):
        project = _make_fake_project(tmp_path / "project", with_python=False)
        result = _run_script(
            project.script, project.record, project.plan
        )
        assert result.returncode != 0
        assert "interpreter not executable" in result.stderr
        assert not project.record.exists()
        assert not (project.root / "src" / "cli-ran.marker").exists()