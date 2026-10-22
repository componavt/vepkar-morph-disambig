"""Compact offline tests for loading and verifying an existing T1 run."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import warnings
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


def _code_repo(tmp_path: Path) -> Path:
    return _repo(
        tmp_path,
        "code",
        {"README.md": "fixture code repo\n", "src/one.py": "print(1)\n"},
    )


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
        "format_version": 3,
        "stage": "prepared",
        "code": {"root": "/code", "commit": "c" * 40},
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


def _prepared_run(tmp_path: Path, monkeypatch):
    code = _code_repo(tmp_path)
    data = _data_repo(tmp_path)
    split = tmp_path / "splits.csv"
    split.write_text(_SPLIT_CONTENT, encoding="utf-8")
    manifest = collect_run_manifest(
        code_dir=code,
        data_dir=data,
        data_tag=_TAG,
        split_file=split,
        C=1.0,
    )
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    _write_passport(run_dir, manifest)
    monkeypatch.setattr(run_context, "_PROJECT_ROOT", code)
    return code, data, split, manifest, run_dir


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestLoadT1RunContext:
    def test_matching_identities_return_exact_manifest_run_path_and_code_commit(
        self, tmp_path, monkeypatch
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        expected_commit = _git(code, "rev-parse", "HEAD").stdout.strip()

        context = load_t1_run_context(run_dir)

        assert isinstance(context, T1RunContext)
        assert context.run_dir == run_dir.resolve()
        assert context.run_dir.is_absolute()
        assert context.manifest == manifest
        assert context.manifest["code"]["commit"] == expected_commit
        assert context.manifest["data"]["tag"] == _TAG
        assert context.manifest["data"]["commit"] == _git(
            data, "rev-parse", "HEAD"
        ).stdout.strip()
        assert context.current_code_commit == expected_commit

    def test_relative_and_tilde_run_paths_resolve(self, tmp_path, monkeypatch):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
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

    def test_just_prepared_run_with_only_run_json_accepted(
        self, tmp_path, monkeypatch
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        assert sorted(p.name for p in run_dir.iterdir()) == ["run.json"]

        context = load_t1_run_context(run_dir)

        assert context.manifest == manifest
        assert context.current_code_commit == manifest["code"]["commit"]

    def test_new_code_commit_warns_and_returns_by_default(
        self, tmp_path, monkeypatch
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        (code / "src" / "two.py").write_text("print(2)\n", encoding="utf-8")
        _commit_all(code, "second commit")
        new_commit = _git(code, "rev-parse", "HEAD").stdout.strip()

        with pytest.warns(UserWarning) as records:
            context = load_t1_run_context(run_dir)

        assert len(records) == 1
        message = str(records[0].message)
        assert str(run_dir.resolve()) in message
        assert manifest["code"]["commit"] in message
        assert new_commit in message
        assert "may continue" in message
        assert context.current_code_commit == new_commit
        assert context.manifest == manifest
        assert context.run_dir == run_dir.resolve()

    def test_new_code_commit_with_strict_code_warns_and_raises(
        self, tmp_path, monkeypatch
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        (code / "src" / "two.py").write_text("print(2)\n", encoding="utf-8")
        _commit_all(code, "second commit")
        new_commit = _git(code, "rev-parse", "HEAD").stdout.strip()

        with pytest.warns(UserWarning) as records:
            with pytest.raises(T1RunContextError) as exc_info:
                load_t1_run_context(run_dir, strict_code=True)

        assert len(records) == 1
        message = str(exc_info.value)
        assert "strict" in message.lower()
        assert "refusal" in message.lower()
        assert manifest["code"]["commit"] in message
        assert new_commit in message
        assert isinstance(exc_info.value, ValueError)

    def test_matching_code_emits_no_application_warning(
        self, tmp_path, monkeypatch, recwarn
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)

        context = load_t1_run_context(run_dir)

        assert context.current_code_commit == manifest["code"]["commit"]
        assert len(recwarn) == 0


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestIdentityMismatches:
    def test_recorded_data_tag_mismatch_rejected(self, tmp_path, monkeypatch):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        manifest["data"]["tag"] = "v2026.08"
        _write_passport(run_dir, manifest)

        with pytest.raises(T1RunContextError) as exc_info:
            load_t1_run_context(run_dir)

        message = str(exc_info.value)
        assert "data tag" in message
        assert "v2026.08" in message
        assert "v2026.09" in message

    def test_recorded_data_commit_mismatch_rejected(self, tmp_path, monkeypatch):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
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
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
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
        self, tmp_path, monkeypatch
    ):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        passport_bytes = (run_dir / "run.json").read_bytes()
        split_bytes = split.read_bytes()
        data_files = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        code_files = {
            path.relative_to(code): path.read_bytes()
            for path in code.rglob("*")
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
        after_code = {
            path.relative_to(code): path.read_bytes()
            for path in code.rglob("*")
            if path.is_file()
        }
        assert after_data == data_files
        assert after_code == code_files
        assert sorted(p.name for p in run_dir.iterdir()) == entries_before

    def test_malformed_local_toml_has_no_effect(self, tmp_path, monkeypatch):
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
        (code / "t1.local.toml").write_text(
            "[paths\nmalformed =", encoding="utf-8"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = load_t1_run_context(run_dir)

        assert context.manifest == manifest
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]

    def test_no_corpus_table_loading_and_no_status_or_diff_git_calls(
        self, tmp_path, monkeypatch
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("corpus tables must not be loaded")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)
        monkeypatch.setattr(
            benchmark_context, "build_language_instances", forbidden
        )
        code, data, split, manifest, run_dir = _prepared_run(tmp_path, monkeypatch)
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
            try:
                c_index = command.index("-C")
            except ValueError:
                continue
            operations.append(command[c_index + 2])
        assert operations
        assert "rev-parse" in operations
        assert not {"status", "diff", "diff-index"} & set(operations)


class TestGitAvailabilityHelper:
    def test_reports_true_for_successful_git(self):
        if not GIT_AVAILABLE:
            pytest.skip("Git unavailable")
        assert _git_available() is True