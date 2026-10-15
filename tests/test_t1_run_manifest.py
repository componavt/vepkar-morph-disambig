"""Focused tests for the T1 run passport collection and publication."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import cli  # noqa: E402
import commands.prepare_t1_run as prepare_pkg  # noqa: E402
import t1_features.run_manifest as run_manifest  # noqa: E402
from t1_features.run_manifest import (  # noqa: E402
    RunManifestError,
    collect_run_manifest,
    write_run_manifest,
)

GIT_AVAILABLE = (
    subprocess.run(["git", "--version"], capture_output=True).returncode == 0
)

_SPLIT_CONTENT = "language,text_id,split\nvep,1,dev\nvep,2,train\nkrl,1,dev\n"

_WARNING = (
    "Recorded local changes; commit IDs alone do not identify\n"
    "the exact executed code/data. Review changes before the experiment."
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _commit_all(root: Path, message: str = "fixture") -> None:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        message,
    )


def _repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _commit_all(root)
    return root


def _valid_code_repo(tmp_path: Path, name: str = "code") -> Path:
    return _repo(
        tmp_path,
        name,
        {"README.md": "fixture code repo\n", "src/one.py": "print(1)\n"},
    )


def _tagged_data_repo(tmp_path: Path, name: str = "data") -> Path:
    root = _repo(
        tmp_path,
        name,
        {"README.md": "fixture data repo\n", "corpus/placeholder.txt": "x\n"},
    )
    _git(root, "tag", "v2026.09")
    return root


def _split_file(
    tmp_path: Path, name: str = "splits.csv", content: str = _SPLIT_CONTENT
) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _expected_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _stable_manifest() -> dict:
    return {
        "format": "vepkar-t1-run",
        "format_version": 1,
        "stage": "prepared",
        "code": {
            "root": "/code",
            "commit": "c" * 40,
            "status_porcelain": "",
            "dirty": False,
        },
        "data": {
            "root": "/data",
            "tag": "v2026.09",
            "commit": "d" * 40,
            "status_porcelain": "",
            "dirty": False,
        },
        "split": {"path": "/split.csv", "sha256": "0" * 64},
        "environment": _expected_environment(),
        "parameters": {"C": 1.0, "target_split": "dev"},
    }


def _collect(code, data, split, C=1.0):
    return collect_run_manifest(
        code_dir=code,
        data_dir=data,
        data_tag="v2026.09",
        split_file=split,
        C=C,
    )


def test_run_manifest_error_is_value_error():
    assert issubclass(RunManifestError, ValueError)
    assert RunManifestError is not ValueError


def test_parser_wires_prepare_t1_run_arguments(tmp_path):
    from cli import build_parser

    parser = build_parser()
    output = tmp_path / "run.json"
    data = tmp_path / "data"
    split = _split_file(tmp_path)
    args = parser.parse_args(
        [
            "prepare-t1-run",
            "--output",
            str(output),
            "--C",
            "2.5",
            "--data-dir",
            str(data),
            "--split-file",
            str(split),
        ]
    )
    assert args.command == "prepare-t1-run"
    assert args.output == output
    assert args.C == 2.5
    assert args.data_dir == data
    assert args.split_file == split


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestCollectSchema:
    def test_exact_schema_commits_tag_paths_and_sha256(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)

        manifest = collect_run_manifest(
            code_dir=code,
            data_dir=data,
            data_tag="v2026.09",
            split_file=split,
            C=2.5,
        )

        code_commit = _git(code, "rev-parse", "HEAD").stdout.strip()
        data_commit = _git(data, "rev-parse", "HEAD").stdout.strip()
        expected_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()

        assert list(manifest) == [
            "format",
            "format_version",
            "stage",
            "code",
            "data",
            "split",
            "environment",
            "parameters",
        ]
        assert manifest["format"] == "vepkar-t1-run"
        assert manifest["format_version"] == 1
        assert manifest["stage"] == "prepared"
        assert list(manifest["code"]) == [
            "root",
            "commit",
            "status_porcelain",
            "dirty",
        ]
        assert manifest["code"] == {
            "root": str(code.resolve()),
            "commit": code_commit,
            "status_porcelain": "",
            "dirty": False,
        }
        assert list(manifest["data"]) == [
            "root",
            "tag",
            "commit",
            "status_porcelain",
            "dirty",
        ]
        assert manifest["data"] == {
            "root": str(data.resolve()),
            "tag": "v2026.09",
            "commit": data_commit,
            "status_porcelain": "",
            "dirty": False,
        }
        assert manifest["split"] == {
            "path": str(split.resolve()),
            "sha256": expected_sha256,
        }
        assert manifest["environment"] == _expected_environment()
        assert manifest["parameters"] == {"C": 2.5, "target_split": "dev"}


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestWorkingTreeState:
    def test_clean_trees_record_empty_porcelain(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        manifest = _collect(code, data, split)
        assert manifest["code"]["status_porcelain"] == ""
        assert manifest["code"]["dirty"] is False
        assert manifest["data"]["status_porcelain"] == ""
        assert manifest["data"]["dirty"] is False

    def test_tracked_and_untracked_changes_preserve_leading_spaces(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)

        (data / "README.md").write_text("changed\n", encoding="utf-8")
        (data / "new.txt").write_text("untracked\n", encoding="utf-8")

        manifest = _collect(code, data, split)
        porcelain = manifest["data"]["status_porcelain"]
        assert " M README.md" in porcelain
        assert "?? new.txt" in porcelain
        assert manifest["data"]["dirty"] is True
        assert manifest["code"]["dirty"] is False


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestSplitFingerprint:
    def test_bytes_change_changes_fingerprint_but_moves_preserve_it(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        first = _split_file(tmp_path, "first.csv", "language,text_id,split\nvep,1,dev\n")
        moved = _split_file(tmp_path, "second.csv", "language,text_id,split\nvep,1,dev\n")
        digest = hashlib.sha256(first.read_bytes()).hexdigest()

        first_manifest = _collect(code, data, first)
        moved_manifest = _collect(code, data, moved)
        assert first_manifest["split"]["sha256"] == digest
        assert moved_manifest["split"]["sha256"] == digest
        assert first_manifest["split"]["path"] == str(first.resolve())
        assert moved_manifest["split"]["path"] == str(moved.resolve())

        changed = _split_file(tmp_path, "third.csv", "language,text_id,split\nvep,1,test\n")
        changed_manifest = _collect(code, data, changed)
        assert changed_manifest["split"]["sha256"] != digest


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestRepoIsolation:
    def test_code_and_data_records_are_independent(self, tmp_path):
        code = _repo(tmp_path, "code", {"README.md": "code repo\n"})
        data = _repo(tmp_path, "data", {"README.md": "data repo fetched\n"})
        _git(data, "tag", "v2026.09")
        split = _split_file(tmp_path)

        manifest = _collect(code, data, split)
        assert manifest["code"]["root"] == str(code.resolve())
        assert manifest["code"]["commit"] == _git(code, "rev-parse", "HEAD").stdout.strip()
        assert manifest["data"]["root"] == str(data.resolve())
        assert manifest["data"]["commit"] == _git(data, "rev-parse", "HEAD").stdout.strip()

    def test_non_root_directory_rejected(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        inside = code / "src"
        with pytest.raises(RunManifestError) as exc_info:
            _collect(code, inside, split)
        assert "not the intended repository root" in str(exc_info.value)
        assert str(inside) in str(exc_info.value)

    def test_same_repository_for_code_and_data_rejected(self, tmp_path):
        code = _valid_code_repo(tmp_path)
        split = _split_file(tmp_path)
        with pytest.raises(RunManifestError) as exc_info:
            _collect(code, code, split)
        assert "separate repositories" in str(exc_info.value)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestPrepareT1RunCommand:
    def test_cli_success_with_tiny_tagged_data_and_no_table_loading(
        self, tmp_path, monkeypatch, capsys
    ):
        def forbidden(lang, data_dir):
            raise AssertionError("corpus tables must not be loaded")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)
        monkeypatch.setattr(
            benchmark_context, "build_language_instances", forbidden
        )

        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        output = tmp_path / "run.json"
        status = cli.main(
            [
                "prepare-t1-run",
                "--output",
                str(output),
                "--C",
                "0.5",
                "--data-dir",
                str(data),
                "--split-file",
                str(split),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert "T1 run metadata prepared" in captured.out
        assert f"Run passport: {output}" in captured.out
        code_commit = _git(ROOT, "rev-parse", "HEAD").stdout.strip()
        data_commit = _git(data, "rev-parse", "HEAD").stdout.strip()
        assert f"Code commit: {code_commit}" in captured.out
        assert "Data tag: v2026.09" in captured.out
        assert f"Data commit: {data_commit}" in captured.out
        expected_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()
        assert f"Split SHA-256: {expected_sha256}" in captured.out
        assert "C: 0.5" in captured.out
        assert "Target split: dev" in captured.out
        assert "error:" not in captured.err
        assert "Traceback" not in captured.err
        assert "Traceback" not in captured.out

        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["format"] == "vepkar-t1-run"
        assert manifest["format_version"] == 1
        assert manifest["stage"] == "prepared"
        assert manifest["code"]["root"] == str(ROOT)
        assert manifest["code"]["commit"] == code_commit
        assert manifest["data"] == {
            "root": str(data.resolve()),
            "tag": "v2026.09",
            "commit": data_commit,
            "status_porcelain": "",
            "dirty": False,
        }
        assert manifest["split"] == {
            "path": str(split.resolve()),
            "sha256": expected_sha256,
        }
        assert manifest["environment"] == _expected_environment()
        assert manifest["parameters"] == {"C": 0.5, "target_split": "dev"}

    def test_missing_split_file_fails_without_output(self, tmp_path, capsys):
        data = _tagged_data_repo(tmp_path)
        missing = tmp_path / "missing-splits.csv"
        output = tmp_path / "run.json"
        status = cli.main(
            [
                "prepare-t1-run",
                "--output",
                str(output),
                "--C",
                "1.0",
                "--data-dir",
                str(data),
                "--split-file",
                str(missing),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot read split file" in captured.err
        assert str(missing) in captured.err
        assert not output.exists()
        assert "T1 run metadata prepared" not in captured.out
        assert "Traceback" not in captured.err

    def test_data_dir_without_git_repo_fails(self, tmp_path, capsys):
        data = tmp_path / "not-a-repo"
        (data / "corpus").mkdir(parents=True)
        (data / "corpus" / "x.txt").write_text("x", encoding="utf-8")
        split = _split_file(tmp_path)
        output = tmp_path / "run.json"
        status = cli.main(
            [
                "prepare-t1-run",
                "--output",
                str(output),
                "--C",
                "1.0",
                "--data-dir",
                str(data),
                "--split-file",
                str(split),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "Could not read the dictorpus-data checkout tag" in captured.err
        assert str(data) in captured.err
        assert not output.exists()
        assert "Traceback" not in captured.err

    @pytest.mark.parametrize("C", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_C_rejected_before_any_work(self, tmp_path, capsys, C):
        output = tmp_path / "run.json"
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(
                C=C, output=output, data_dir=None, split_file=None
            )
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "C must be finite and positive" in captured.err
        assert not output.exists()

    def test_existing_destination_protected(self, tmp_path, capsys):
        output = tmp_path / "run.json"
        output.write_text("original\n", encoding="utf-8")
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(
                C=1.0, output=output, data_dir=None, split_file=None
            )
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "already exists" in captured.err
        assert output.read_text(encoding="utf-8") == "original\n"

    def test_dangling_symlink_destination_protected(self, tmp_path, capsys):
        output = tmp_path / "run.json"
        missing = tmp_path / "missing.json"
        try:
            output.symlink_to(missing)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation is unavailable in this environment")
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(
                C=1.0, output=output, data_dir=None, split_file=None
            )
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "already exists" in captured.err
        assert output.is_symlink()
        assert output.readlink() == missing
        assert not missing.exists()

    def test_missing_parent_not_created(self, tmp_path, capsys):
        output = tmp_path / "no-such-dir" / "run.json"
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(
                C=1.0, output=output, data_dir=None, split_file=None
            )
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "output directory does not exist" in captured.err
        assert str(output.parent) in captured.err
        assert not (tmp_path / "no-such-dir").exists()


class TestDirtyWarning:
    @staticmethod
    def _run(tmp_path, monkeypatch, capsys, dirty: bool):
        manifest = _stable_manifest()
        if dirty:
            manifest["code"]["dirty"] = True
            manifest["code"]["status_porcelain"] = " M src/one.py"
            manifest["data"]["dirty"] = True
            manifest["data"]["status_porcelain"] = "?? new.txt"
        output = tmp_path / "run.json"
        monkeypatch.setattr(
            prepare_pkg,
            "collect_run_manifest",
            lambda manifest=manifest, **kwargs: manifest,
        )
        monkeypatch.setattr(
            prepare_pkg,
            "_preflight_benchmark",
            lambda data_dir: (data_dir, "v2026.09"),
        )
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(
                C=1.0,
                output=output,
                data_dir=tmp_path,
                split_file=tmp_path / "splits.csv",
            )
        )
        return status, output, capsys.readouterr()

    def test_clean_trees_record_without_warning(self, tmp_path, monkeypatch, capsys):
        status, output, captured = self._run(
            tmp_path, monkeypatch, capsys, dirty=False
        )
        assert status == 0, captured.err
        assert output.is_file()
        assert _WARNING not in captured.err

    def test_dirty_trees_warn_on_stderr(self, tmp_path, monkeypatch, capsys):
        status, output, captured = self._run(
            tmp_path, monkeypatch, capsys, dirty=True
        )
        assert status == 0, captured.err
        assert output.is_file()
        assert _WARNING in captured.err


class TestPublication:
    def test_writes_exact_json_text(self, tmp_path):
        manifest = _stable_manifest()
        output = tmp_path / "run.json"
        write_run_manifest(manifest, output)
        expected = (
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        )
        assert output.read_text(encoding="utf-8") == expected
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]

    @pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
    def test_existing_destination_preserved(self, tmp_path, kind):
        output = tmp_path / "run.json"
        if kind == "file":
            output.write_text("precious\n", encoding="utf-8")
        elif kind == "directory":
            output.mkdir()
        else:
            output.symlink_to(tmp_path / "missing-target")
        with pytest.raises(RunManifestError):
            write_run_manifest(_stable_manifest(), output)
        if kind == "file":
            assert output.read_text(encoding="utf-8") == "precious\n"
        elif kind == "directory":
            assert output.is_dir()
        else:
            assert output.is_symlink()
            assert not (tmp_path / "missing-target").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]

    def test_missing_parent_rejected_without_creation(self, tmp_path):
        output = tmp_path / "missing" / "run.json"
        with pytest.raises(RunManifestError):
            write_run_manifest(_stable_manifest(), output)
        assert not (tmp_path / "missing").exists()

    def test_competing_destination_keeps_foreign_bytes(self, tmp_path, monkeypatch):
        output = tmp_path / "run.json"
        original_link = os.link
        foreign = b"foreign competing destination bytes\n"

        def competing_link(source, destination):
            with open(destination, "wb") as fh:
                fh.write(foreign)
            return original_link(source, destination)

        monkeypatch.setattr(run_manifest.os, "link", competing_link)
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(_stable_manifest(), output)
        assert isinstance(exc_info.value.__cause__, OSError)
        assert output.read_bytes() == foreign
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]

    def test_post_publication_cleanup_failure_reports_both_paths(
        self, tmp_path, monkeypatch
    ):
        output = tmp_path / "run.json"
        original_link = os.link
        original_unlink = Path.unlink
        published = []

        def recording_link(source, destination):
            published.append(Path(source))
            return original_link(source, destination)

        def failing_unlink(self, *args, **kwargs):
            if self in published:
                raise OSError("simulated cleanup denial")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(run_manifest.os, "link", recording_link)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        manifest = _stable_manifest()
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(manifest, output)
        error = exc_info.value
        assert "published" in str(error).lower()
        assert str(output) in str(error)
        assert len(published) == 1
        assert str(published[0]) in str(error)
        assert isinstance(error.__cause__, OSError)
        expected = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        assert output.read_text(encoding="utf-8") == expected
        temps = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert len(temps) == 1
        original_unlink(temps[0])

    def test_pre_publication_failure_leaves_no_output(self, tmp_path, monkeypatch):
        output = tmp_path / "run.json"
        original_unlink = Path.unlink
        temp_links = []

        def failing_link(source, destination):
            temp_links.append(Path(source))
            raise OSError("publication denied")

        monkeypatch.setattr(run_manifest.os, "link", failing_link)
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(_stable_manifest(), output)
        assert "publication denied" in str(exc_info.value.__cause__)
        assert not output.exists()
        assert len(temp_links) == 1
        assert not temp_links[0].exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestEndToEndSideEffects:
    def test_publication_leaves_only_json_and_inputs_unchanged(self, tmp_path, capsys):
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        output = tmp_path / "run.json"
        split_before = split.read_bytes()
        data_files = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        before = {path.name for path in tmp_path.iterdir()}
        status = cli.main(
            [
                "prepare-t1-run",
                "--output",
                str(output),
                "--C",
                "1.0",
                "--data-dir",
                str(data),
                "--split-file",
                str(split),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        after = {path.name for path in tmp_path.iterdir()}
        assert after == before | {"run.json"}
        assert split.read_bytes() == split_before
        after_files = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        assert after_files == data_files
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))