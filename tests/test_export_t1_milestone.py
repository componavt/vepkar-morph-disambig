"""Offline CLI tests for the export-t1-milestone command."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CLI = ROOT / "src" / "cli.py"
sys.path.insert(0, str(SRC))

TAG = "v2026.09"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _metrics_bytes():
    payload = {
        "format": "vepkar-metrics",
        "format_version": 2,
        "method": "t1.1",
        "split": "dev",
        "parameters": {"C": 1.0},
        "word_count": 3,
        "candidate_count": 5,
        "metrics": {
            "top1_accuracy": 0.5,
            "mrr": 0.6,
            "top3_accuracy": 1.0,
        },
        "data": {"tag": TAG, "commit": "c", "split_sha256": "c" * 64},
        "predictions_sha256": "0" * 64,
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def _make_run(tmp_path, name="t1-C1-run"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "dev.csv").write_bytes(b"w,c,g,r,s\n1,10,G,1,0\n")
    (run_dir / "metrics.json").write_bytes(_metrics_bytes())
    return run_dir


class TestParser:
    def test_help_lists_run_dir_option(self):
        result = run_cli("export-t1-milestone", "--help")
        assert result.returncode == 0
        assert "export-t1-milestone" in result.stdout
        assert "--run-dir" in result.stdout

    def test_requires_run_dir(self):
        result = run_cli("export-t1-milestone")
        assert result.returncode != 0
        assert "--run-dir" in result.stderr

    def test_rejects_force_option(self, tmp_path):
        result = run_cli(
            "export-t1-milestone", "--run-dir", str(tmp_path), "--force"
        )
        assert result.returncode != 0
        assert "--force" in result.stderr

    def test_rejects_abbreviated_run_option(self, tmp_path):
        result = run_cli(
            "export-t1-milestone", "--run", str(tmp_path)
        )
        assert result.returncode != 0
        assert "--run" in result.stderr


class TestHandler:
    def test_cli_export_success_redirects_milestones(
        self, tmp_path, monkeypatch, capsys
    ):
        import cli
        import commands.export_t1_milestone as export_pkg

        run_dir = _make_run(tmp_path)
        milestones = tmp_path / "milestones"
        monkeypatch.setattr(export_pkg, "_default_milestones_dir", lambda: milestones)
        status = cli.main(["export-t1-milestone", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 0
        assert captured.err == ""
        out_lines = captured.out.splitlines()
        assert out_lines[0] == "T1 milestone export: OK"
        assert out_lines[1] == f"Run directory: {run_dir.resolve()}"
        expected = milestones / TAG / run_dir.name
        assert out_lines[2] == f"Milestone directory: {expected}"
        assert expected.is_dir()
        assert sorted(p.name for p in expected.iterdir()) == [
            "README.md",
            "dev.csv.zst",
            "metrics.json",
        ]

    def test_cli_export_error_diagnostic_status_one(
        self, tmp_path, monkeypatch, capsys
    ):
        import cli
        import commands.export_t1_milestone as export_pkg

        run_dir = tmp_path / "incomplete-run"
        run_dir.mkdir()
        (run_dir / "dev.csv").write_bytes(b"x\n")
        milestones = tmp_path / "milestones"
        monkeypatch.setattr(export_pkg, "_default_milestones_dir", lambda: milestones)
        status = cli.main(["export-t1-milestone", "--run-dir", str(run_dir)])
        captured = capsys.readouterr()
        assert status == 1
        assert "error:" in captured.err
        assert "metrics.json" in captured.err
        assert "T1 milestone export: OK" not in captured.out
        assert "Milestone directory:" not in captured.out
        assert "Traceback" not in captured.err
        assert not milestones.exists()