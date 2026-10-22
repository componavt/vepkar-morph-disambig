"""Compact offline tests for loading and verifying an existing T1 run."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import t1_features.run_context as run_context  # noqa: E402
from t1_features.run_context import (  # noqa: E402
    T1RunContext,
    T1RunContextError,
    load_t1_run_context,
)
from t1_features.run_manifest import (  # noqa: E402
    RunManifestError,
    collect_run_manifest,
)

_SPLIT_CONTENT = "language,text_id,split\nvep,1,dev\nvep,2,train\nkrl,1,dev\n"
_TAG = "v2026.09"


def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True
        )
    except OSError:
        return False
    return result.returncode == 0


GIT_AVAILABLE = _git_available()


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


def _data_repo(tmp_path: Path) -> Path:
    root = _repo(
        tmp_path,
        "data",
        {"README.md": "fixture data repo\n", "corpus/placeholder.txt": "x\n"},
    )
    _git(root, "tag", _TAG)
    return root


def _stable_manifest() -> dict:
    return {
        "format": "vepkar-t1-run",
        "format_version": 4,
        "stage": "prepared",
        "data": {
            "root": "/data",
            "tag": _TAG,
            "commit": "d" * 40,
        },
        "split": {"path": "/split.csv", "sha256": "0" * 64},
        "parameters": {"C": 1.0, "target_split": "dev"},
    }


def _write_passport(run_dir: Path, manifest: dict) -> None:
    (run_dir / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepared_run(tmp_path: Path):
    data = _data_repo(tmp_path)
    split = tmp_path / "splits.csv"
    split.write_text(_SPLIT_CONTENT, encoding="utf-8")
    manifest = collect_run_manifest(
        data_dir=data,
        data_tag=_TAG,
        split_file=split,
        C=1.0,
    )
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    _write_passport(run_dir, manifest)
    return data, split, manifest, run_dir


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestLoadT1RunContext:
    def test_matching_identities_return_exact_manifest_and_run_path(
        self, tmp_path
    ):
        data, split, manifest, run_dir = _prepared_run(tmp_path)

        context = load_t1_run_context(run_dir)

        assert isinstance(context, T1RunContext)
        assert context.run_dir == run_dir.resolve()
        assert context.run_dir.is_absolute()
        assert context.manifest == manifest
        assert "code" not in context.manifest
        assert context.manifest["data"]["tag"] == _TAG
        assert context.manifest["data"]["commit"] == _git(
            data, "rev-parse", "HEAD"
        ).stdout.strip()

    def test_relative_and_tilde_run_paths_resolve(self, tmp_path, monkeypatch):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        monkeypatch.chdir(elsewhere)
        relative = load_t1_run_context(Path("../runs/run-a"))
        assert relative.run_dir == run_dir.resolve()
        assert relative.manifest == manifest

        monkeypatch.setenv("HOME", str(tmp_path))
        expanded = load_t1_run_context(Path("~/runs/run-a"))
        assert expanded.run_dir == run_dir.resolve()
        assert expanded.manifest == manifest

    def test_just_prepared_run_with_only_run_json_accepted(self, tmp_path):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        assert sorted(p.name for p in run_dir.iterdir()) == ["run.json"]

        context = load_t1_run_context(run_dir)

        assert context.manifest == manifest

    def test_no_source_code_git_identity_queried_or_corpus_table_loading(
        self, tmp_path, monkeypatch
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("corpus tables must not be loaded")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)
        monkeypatch.setattr(
            benchmark_context, "build_language_instances", forbidden
        )
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        original_run = subprocess.run
        invocations = []

        def spy_run(command, *args, **kwargs):
            invocations.append(command)
            return original_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", spy_run)

        context = load_t1_run_context(run_dir)

        assert context.manifest == manifest
        git_calls = [cmd for cmd in invocations if cmd and cmd[0] == "git"]
        assert git_calls
        operations = []
        for command in git_calls:
            if len(command) == 2 and command[1] == "--version":
                continue
            try:
                target_index = command.index("-C")
            except ValueError:
                pytest.fail(
                    f"unidentified Git call without a repository target: {command}"
                )
            target = Path(command[target_index + 1]).resolve()
            assert target == data.resolve()
            operations.append(command[target_index + 2])
        assert "rev-parse" in operations
        assert not {"status", "diff", "diff-index"} & set(operations)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestIdentityMismatches:
    def test_recorded_data_tag_mismatch_rejected(self, tmp_path, monkeypatch):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        manifest["data"]["tag"] = "v2026.08"
        _write_passport(run_dir, manifest)

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        message = str(exc_info.value)
        assert "data tag" in message
        assert "v2026.08" in message
        assert "v2026.09" in message

    def test_recorded_data_commit_mismatch_rejected(self, tmp_path, monkeypatch):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        manifest["data"]["commit"] = "d" * 40
        _write_passport(run_dir, manifest)

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        message = str(exc_info.value)
        assert "data commit" in message
        assert "d" * 40 in message
        assert _git(data, "rev-parse", "HEAD").stdout.strip() in message

    def test_changed_split_bytes_cause_fingerprint_mismatch(
        self, tmp_path, monkeypatch
    ):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        split.write_text("language,text_id,split\nvep,1,test\n", encoding="utf-8")

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        message = str(exc_info.value)
        assert "split" in message
        assert "SHA-256" in message
        assert manifest["split"]["sha256"] in message


class TestRunPathAndPassportErrors:
    def test_missing_run_directory_reports_path(self, tmp_path):
        missing = tmp_path / "no-such-run"

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(missing)

        assert str(missing.resolve()) in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    def test_non_directory_run_path_reports_path(self, tmp_path):
        not_a_dir = tmp_path / "file-run"
        not_a_dir.write_text("x\n", encoding="utf-8")

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(not_a_dir)

        assert str(not_a_dir.resolve()) in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    def test_missing_passport_reports_paths_and_cause(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        assert str(run_dir) in str(exc_info.value)
        assert str(run_dir / "run.json") in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RunManifestError)

    def test_malformed_passport_reports_paths_and_cause(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        (run_dir / "run.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        assert str(run_dir / "run.json") in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RunManifestError)

    def test_unsupported_passport_version_reports_paths_and_cause(self, tmp_path):
        run_dir = tmp_path / "run-a"
        run_dir.mkdir()
        manifest = _stable_manifest()
        manifest["format_version"] = 2
        _write_passport(run_dir, manifest)

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        assert str(run_dir / "run.json") in str(exc_info.value)
        assert "format_version" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RunManifestError)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestNoSideEffects:
    def test_loaded_manifest_passport_bytes_inputs_and_entries_unchanged(
        self, tmp_path
    ):
        data, split, manifest, run_dir = _prepared_run(tmp_path)
        passport_bytes = (run_dir / "run.json").read_bytes()
        split_bytes = split.read_bytes()
        data_files = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        entries_before = sorted(p.name for p in run_dir.iterdir())
        manifest_before = copy.deepcopy(manifest)

        context = load_t1_run_context(run_dir)

        assert context.manifest == manifest_before
        assert (run_dir / "run.json").read_bytes() == passport_bytes
        assert split.read_bytes() == split_bytes
        after_data = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        assert after_data == data_files
        assert sorted(p.name for p in run_dir.iterdir()) == entries_before