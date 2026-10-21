"""Focused tests for the T1 run passport collection and publication."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    load_run_manifest,
    write_run_manifest,
)

def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True
        )
    except OSError:
        return False
    return result.returncode == 0


GIT_AVAILABLE = _git_available()

_SPLIT_CONTENT = "language,text_id,split\nvep,1,dev\nvep,2,train\nkrl,1,dev\n"
_TAG = "v2026.09"


class _WriteAndCloseFailingStream:
    def __init__(self, stream, write_error, close_error):
        self._stream = stream
        self._write_error = write_error
        self._close_error = close_error

    def write(self, text):
        raise self._write_error

    def close(self):
        self._stream.close()
        raise self._close_error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


class _CloseFailingStream:
    def __init__(self, stream, close_error):
        self._stream = stream
        self._close_error = close_error

    def write(self, text):
        return self._stream.write(text)

    def close(self):
        self._stream.close()
        raise self._close_error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


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


def _default_split(code_root: Path, content: str = _SPLIT_CONTENT) -> Path:
    path = code_root / "data" / "derived" / f"splits_{_TAG}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _commit_all(code_root)
    return path


def _created_run_dir(captured) -> Path:
    run_line = next(
        line
        for line in captured.out.splitlines()
        if line.startswith("Run directory: ")
    )
    return Path(run_line.split(": ", 1)[1])


def _stable_manifest() -> dict:
    return {
        "format": "vepkar-t1-run",
        "format_version": 3,
        "stage": "prepared",
        "code": {"root": "/code", "commit": "c" * 40},
        "data": {
            "root": "/data",
            "tag": "v2026.09",
            "commit": "d" * 40,
        },
        "split": {"path": "/split.csv", "sha256": "0" * 64},
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


def test_parser_defaults_and_explicit_prepare_t1_run_arguments(tmp_path):
    from cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["prepare-t1-run"])
    assert args.command == "prepare-t1-run"
    assert args.run_dir is None
    assert args.C == 1.0
    assert args.data_dir is None
    assert args.split_file is None
    assert not hasattr(args, "output")

    run_dir = tmp_path / "runs" / "run-a"
    data = tmp_path / "data"
    split = _split_file(tmp_path)
    args = parser.parse_args(
        [
            "prepare-t1-run",
            "--run-dir",
            str(run_dir),
            "--C",
            "2.5",
            "--data-dir",
            str(data),
            "--split-file",
            str(split),
        ]
    )
    assert args.run_dir == run_dir
    assert args.C == 2.5
    assert args.data_dir == data
    assert args.split_file == split


@pytest.mark.parametrize("option", ["--output", "--runs-dir"])
def test_prepare_t1_run_rejects_removed_output_and_runs_dir(option, tmp_path):
    from cli import build_parser

    parser = build_parser()
    target = tmp_path / "run.json"
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["prepare-t1-run", option, str(target)])
    assert excinfo.value.code == 2


def test_prepare_t1_run_help_describes_optional_run_dir(capsys):
    from cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["prepare-t1-run", "--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--run-dir" in captured.out
    assert "--output" not in captured.out
    assert "--runs-dir" not in captured.out


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
            "parameters",
        ]
        assert manifest["format"] == "vepkar-t1-run"
        assert manifest["format_version"] == 3
        assert manifest["stage"] == "prepared"
        assert list(manifest["code"]) == ["root", "commit"]
        assert manifest["code"] == {
            "root": str(code.resolve()),
            "commit": code_commit,
        }
        assert list(manifest["data"]) == ["root", "tag", "commit"]
        assert manifest["data"] == {
            "root": str(data.resolve()),
            "tag": "v2026.09",
            "commit": data_commit,
        }
        assert manifest["split"] == {
            "path": str(split.resolve()),
            "sha256": expected_sha256,
        }
        assert manifest["parameters"] == {"C": 2.5, "target_split": "dev"}


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
    @staticmethod
    def _guard_tables(monkeypatch):
        def forbidden(lang, data_dir):
            raise AssertionError("corpus tables must not be loaded")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)
        monkeypatch.setattr(
            benchmark_context, "build_language_instances", forbidden
        )

    def test_cli_automatic_creation_with_published_split_and_missing_toml_notice(
        self, tmp_path, monkeypatch, capsys
    ):
        self._guard_tables(monkeypatch)
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)

        status = cli.main(
            ["prepare-t1-run", "--C", "0.5", "--data-dir", str(data)]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        notice = (
            "Local configuration not found: "
            f"{code / 't1.local.toml'}; using defaults and explicit overrides."
        )
        assert captured.err.count(notice) == 1
        assert "T1 run prepared" in captured.out
        created = _created_run_dir(captured)
        assert f"Run directory: {created}" in captured.out
        assert created.parent == code / "work" / "runs"
        assert created.is_absolute()
        assert created.is_dir()
        assert sorted(p.name for p in created.iterdir()) == ["run.json"]
        manifest = load_run_manifest(created / "run.json")
        assert manifest["data"]["root"] == str(data)
        assert manifest["data"]["tag"] == _TAG
        assert manifest["split"]["path"] == str(split.resolve())
        assert manifest["split"]["sha256"] == hashlib.sha256(
            split.read_bytes()
        ).hexdigest()
        assert manifest["parameters"] == {"C": 0.5, "target_split": "dev"}
        code_commit = _git(code, "rev-parse", "HEAD").stdout.strip()
        assert manifest["code"]["commit"] == code_commit
        assert manifest["code"]["root"] == str(code)
        assert "Data tag: v2026.09" in captured.out
        assert "C: 0.5" in captured.out
        assert "Target split: dev" in captured.out
        assert "error:" not in captured.err
        assert "Traceback" not in captured.err
        assert "Traceback" not in captured.out

    def test_cli_toml_paths_with_explicit_overrides_precedence(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data_toml = _tagged_data_repo(tmp_path, name="data-toml")
        data_cli = _tagged_data_repo(tmp_path, name="data-cli")
        split_toml = _split_file(tmp_path, "split-toml.csv")
        split_cli = _split_file(tmp_path, "split-cli.csv")
        (code / "t1.local.toml").write_text(
            "[paths]\n"
            f'data_dir = "{data_toml}"\n'
            'runs_dir = "toml-runs"\n'
            f'split_file = "{split_toml}"\n',
            encoding="utf-8",
        )
        _commit_all(code, "configure")
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)

        status = cli.main(
            [
                "prepare-t1-run",
                "--data-dir",
                str(data_cli),
                "--split-file",
                str(split_cli),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert captured.err == ""
        created = _created_run_dir(captured)
        assert created.parent == code / "toml-runs"
        manifest = load_run_manifest(created / "run.json")
        assert manifest["data"]["root"] == str(data_cli)
        assert manifest["data"]["tag"] == _TAG
        assert manifest["split"]["path"] == str(split_cli.resolve())
        assert manifest["split"]["path"] != str(split_toml.resolve())

    def test_cli_explicit_new_nested_run_directory(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        run_dir = tmp_path / "nested" / "runs" / "run-a"
        status = cli.main(
            [
                "prepare-t1-run",
                "--run-dir",
                str(run_dir),
                "--data-dir",
                str(data),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        assert f"Run directory: {run_dir}" in captured.out
        assert run_dir.is_dir()
        assert sorted(p.name for p in run_dir.iterdir()) == ["run.json"]
        assert load_run_manifest(run_dir / "run.json")["parameters"]["C"] == 1.0

    def test_cli_existing_final_run_path_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        existing = tmp_path / "existing" / "run"
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("keep\n", encoding="utf-8")
        status = cli.main(
            [
                "prepare-t1-run",
                "--run-dir",
                str(existing),
                "--data-dir",
                str(data),
            ]
        )
        captured = capsys.readouterr()
        assert status == 1
        assert str(existing) in captured.err
        assert "already exists" in captured.err
        assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep\n"
        assert not (existing / "run.json").exists()
        assert "T1 run prepared" not in captured.out

    def test_cli_data_dir_without_git_repo_fails(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        data = tmp_path / "not-a-repo"
        (data / "corpus").mkdir(parents=True)
        (data / "corpus" / "x.txt").write_text("x", encoding="utf-8")
        status = cli.main(["prepare-t1-run", "--data-dir", str(data)])
        captured = capsys.readouterr()
        assert status == 1
        assert "Could not read the dictorpus-data checkout tag" in captured.err
        assert str(data) in captured.err
        assert not (code / "work" / "runs").exists()
        assert "T1 run prepared" not in captured.out
        assert "Traceback" not in captured.err

    def test_cli_missing_published_split_fails(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        status = cli.main(["prepare-t1-run", "--data-dir", str(data)])
        captured = capsys.readouterr()
        assert status == 1
        assert "cannot read split file" in captured.err
        published = code / "data" / "derived" / f"splits_{_TAG}.csv"
        assert str(published) in captured.err
        assert not (code / "work" / "runs").exists()
        assert "T1 run prepared" not in captured.out
        assert "Traceback" not in captured.err

    def test_cli_bad_configuration_rejected(self, tmp_path, monkeypatch, capsys):
        code = _valid_code_repo(tmp_path)
        (code / "t1.local.toml").write_text(
            '[paths]\nunknown_key = "x"\n', encoding="utf-8"
        )
        _commit_all(code, "bad config")
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        status = cli.main(["prepare-t1-run"])
        captured = capsys.readouterr()
        assert status == 1
        assert "Invalid T1 local configuration" in captured.err
        assert str(code / "t1.local.toml") in captured.err
        assert not (code / "work" / "runs").exists()
        assert "T1 run prepared" not in captured.out
        assert "Traceback" not in captured.err

    @pytest.mark.parametrize(
        "C",
        [
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            pytest.param(True, id="boolean"),
            pytest.param(10**1000, id="oversized-integer"),
        ],
    )
    def test_handler_invalid_C_rejected_before_configuration_or_creation(
        self, monkeypatch, capsys, C
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("no configuration or creation work is allowed")

        monkeypatch.setattr(prepare_pkg, "load_t1_config", forbidden)
        monkeypatch.setattr(prepare_pkg, "create_t1_run", forbidden)
        status = prepare_pkg._run_prepare_t1_run(
            argparse.Namespace(C=C, run_dir=None, data_dir=None, split_file=None)
        )
        captured = capsys.readouterr()
        assert status == 1
        assert "C must be finite and positive" in captured.err
        assert "T1 run prepared" not in captured.out
        assert "Run directory:" not in captured.out

    def test_cli_passport_reread_failure_reports_created_run(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)

        def failing_load(path):
            raise RunManifestError("distinctive reread explanation")

        monkeypatch.setattr(prepare_pkg, "load_run_manifest", failing_load)
        status = cli.main(["prepare-t1-run", "--data-dir", str(data)])
        captured = capsys.readouterr()
        assert status == 1
        assert "distinctive reread explanation" in captured.err
        runs = code / "work" / "runs"
        assert runs.is_dir()
        created = next(path for path in runs.iterdir())
        assert str(created) in captured.err
        assert "T1 run prepared" not in captured.out
        assert "Run directory:" not in captured.out
        assert (created / "run.json").is_file()
        load_run_manifest(created / "run.json")
        assert sorted(p.name for p in created.iterdir()) == ["run.json"]


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


class TestWriteFailureLifecycle:
    def test_fdopen_failure_closes_descriptor_once_and_removes_temp(
        self, tmp_path, monkeypatch
    ):
        output = tmp_path / "run.json"
        original_close = os.close
        captured = {}
        close_calls = []

        def failing_fdopen(fd, *args, **kwargs):
            captured["descriptor"] = fd
            raise OSError("simulated stream open denial")

        def recording_close(fd):
            close_calls.append(fd)
            return original_close(fd)

        monkeypatch.setattr(run_manifest.os, "fdopen", failing_fdopen)
        monkeypatch.setattr(run_manifest.os, "close", recording_close)
        try:
            with pytest.raises(RunManifestError) as exc_info:
                write_run_manifest(_stable_manifest(), output)
            assert "simulated stream open denial" in str(exc_info.value.__cause__)
            assert isinstance(exc_info.value.__cause__, OSError)
            assert not output.exists()
            descriptor = captured["descriptor"]
            assert close_calls.count(descriptor) == 1
            with pytest.raises(OSError):
                os.fstat(descriptor)
            assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))
        finally:
            descriptor = captured.get("descriptor")
            if descriptor is not None:
                try:
                    os.fstat(descriptor)
                except OSError:
                    pass
                else:
                    original_close(descriptor)
            for leftover in tmp_path.glob(".vepkar-t1-run-*.tmp"):
                try:
                    leftover.unlink()
                except OSError:
                    pass

    def test_write_and_close_failure_keeps_write_cause_and_notes_close(
        self, tmp_path, monkeypatch
    ):
        output = tmp_path / "run.json"
        original_fdopen = os.fdopen
        original_link = os.link
        write_error = OSError("simulated write denial")
        close_error = OSError("simulated close denial")
        link_calls = []

        def wrapping_fdopen(fd, *args, **kwargs):
            stream = original_fdopen(fd, *args, **kwargs)
            return _WriteAndCloseFailingStream(stream, write_error, close_error)

        def recording_link(source, destination):
            link_calls.append((source, destination))
            return original_link(source, destination)

        monkeypatch.setattr(run_manifest.os, "fdopen", wrapping_fdopen)
        monkeypatch.setattr(run_manifest.os, "link", recording_link)
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(_stable_manifest(), output)
        assert exc_info.value.__cause__ is write_error
        assert "simulated write denial" in str(exc_info.value)
        notes = getattr(exc_info.value, "__notes__", ())
        assert any("simulated close denial" in note for note in notes)
        assert link_calls == []
        assert not output.exists()
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))

    def test_close_failure_after_successful_write_never_publishes(
        self, tmp_path, monkeypatch
    ):
        output = tmp_path / "run.json"
        original_fdopen = os.fdopen
        original_link = os.link
        close_error = OSError("simulated close denial")
        link_calls = []

        def wrapping_fdopen(fd, *args, **kwargs):
            stream = original_fdopen(fd, *args, **kwargs)
            return _CloseFailingStream(stream, close_error)

        def recording_link(source, destination):
            link_calls.append((source, destination))
            return original_link(source, destination)

        monkeypatch.setattr(run_manifest.os, "fdopen", wrapping_fdopen)
        monkeypatch.setattr(run_manifest.os, "link", recording_link)
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(_stable_manifest(), output)
        assert exc_info.value.__cause__ is close_error
        assert "simulated close denial" in str(exc_info.value)
        assert link_calls == []
        assert not output.exists()
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))

    def test_non_oserror_write_failure_survives_close_failure(
        self, tmp_path, monkeypatch
    ):
        output = tmp_path / "run.json"
        original_fdopen = os.fdopen
        original_link = os.link
        write_error = RuntimeError("simulated encoding denial")
        close_error = OSError("simulated close denial")
        link_calls = []

        def wrapping_fdopen(fd, *args, **kwargs):
            stream = original_fdopen(fd, *args, **kwargs)
            return _WriteAndCloseFailingStream(stream, write_error, close_error)

        def recording_link(source, destination):
            link_calls.append((source, destination))
            return original_link(source, destination)

        monkeypatch.setattr(run_manifest.os, "fdopen", wrapping_fdopen)
        monkeypatch.setattr(run_manifest.os, "link", recording_link)
        with pytest.raises(RuntimeError) as exc_info:
            write_run_manifest(_stable_manifest(), output)
        assert exc_info.value is write_error
        assert not isinstance(exc_info.value, RunManifestError)
        notes = getattr(exc_info.value, "__notes__", ())
        assert any("simulated close denial" in note for note in notes)
        assert link_calls == []
        assert not output.exists()
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestEndToEndSideEffects:
    def test_publication_leaves_only_json_and_inputs_unchanged(
        self, tmp_path, monkeypatch, capsys
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _default_split(code)
        monkeypatch.setattr(prepare_pkg, "_PROJECT_ROOT", code)
        split_before = split.read_bytes()
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
        before = {path.name for path in tmp_path.iterdir()}
        status = cli.main(
            [
                "prepare-t1-run",
                "--C",
                "1.0",
                "--data-dir",
                str(data),
            ]
        )
        captured = capsys.readouterr()
        assert status == 0, captured.err
        created = _created_run_dir(captured)
        after = {path.name for path in tmp_path.iterdir()}
        assert after == before
        assert split.read_bytes() == split_before
        after_files = {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        }
        assert after_files == data_files
        after_code_files = {
            path.relative_to(code): path.read_bytes()
            for path in code.rglob("*")
            if path.is_file()
        }
        for relative, content in code_files.items():
            assert after_code_files[relative] == content
        assert set(after_code_files) - set(code_files) == {
            created.relative_to(code) / "run.json"
        }
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))
        assert (created / "run.json").is_file()


class TestGitAvailabilityHelper:
    def test_reports_true_for_successful_git(self):
        if not GIT_AVAILABLE:
            pytest.skip("Git unavailable")
        assert _git_available() is True

    def test_reports_false_for_nonzero_status(self, monkeypatch):
        def nonzero_run(command, *args, **kwargs):
            return subprocess.CompletedProcess(
                args=command, returncode=2, stdout="", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", nonzero_run)
        assert _git_available() is False

    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
    def test_reports_false_for_launch_failure(self, monkeypatch, error):
        def raising_run(command, *args, **kwargs):
            raise error("simulated missing git")

        monkeypatch.setattr(subprocess, "run", raising_run)
        assert _git_available() is False


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
class TestGitLaunchFailure:
    def test_launch_failure_wrapped_with_root_operation_and_cause(
        self, tmp_path, monkeypatch, error
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)

        def raising_run(command, *args, **kwargs):
            raise error("simulated git launch denial")

        monkeypatch.setattr(subprocess, "run", raising_run)
        with pytest.raises(RunManifestError) as exc_info:
            _collect(code, data, split)
        message = str(exc_info.value)
        assert "cannot run Git" in message
        assert str(code) in message
        assert "rev-parse" in message
        assert "simulated git launch denial" in message
        assert isinstance(exc_info.value.__cause__, error)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestNonMutatingGitObservation:
    def test_collection_records_identities_with_rev_parse_only(
        self, tmp_path, monkeypatch
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        code_commit = _git(code, "rev-parse", "HEAD").stdout.strip()
        data_commit = _git(data, "rev-parse", "HEAD").stdout.strip()
        original_run = subprocess.run
        invocations = []

        def spy_run(command, *args, **kwargs):
            invocations.append(command)
            return original_run(command, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", spy_run)
        manifest = _collect(code, data, split)
        assert manifest["code"]["root"] == str(code.resolve())
        assert manifest["code"]["commit"] == code_commit
        assert manifest["data"]["root"] == str(data.resolve())
        assert manifest["data"]["commit"] == data_commit
        git_calls = [cmd for cmd in invocations if cmd and cmd[0] == "git"]
        assert len(git_calls) == 4
        assert all(
            command[1] == "--no-optional-locks"
            and command[2] == "-C"
            and (str(code) in command or str(data) in command)
            for command in git_calls
        )
        assert [command[4] for command in git_calls] == ["rev-parse"] * 4
        assert not any(
            "status" in argument
            for command in git_calls
            for argument in command
        )


def _set_key(manifest: dict, *path, value) -> dict:
    target = manifest
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    return manifest


def _drop_key(manifest: dict, *path) -> dict:
    target = manifest
    for part in path[:-1]:
        target = target[part]
    del target[path[-1]]
    return manifest


def _write_raw(tmp_path: Path, manifest, name: str = "run.json") -> Path:
    output = tmp_path / name
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


class TestLoadRoundTrip:
    def test_write_load_round_trip_preserves_manifest(self, tmp_path):
        manifest = _stable_manifest()
        output = tmp_path / "run.json"
        write_run_manifest(manifest, output)
        assert load_run_manifest(output) == manifest


class TestLoadNonMutating:
    def test_loading_leaves_file_bytes_and_directory_entries_unchanged(
        self, tmp_path
    ):
        output = tmp_path / "run.json"
        output.write_text(
            json.dumps(_stable_manifest(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        before = output.read_bytes()
        entries_before = sorted(p.name for p in tmp_path.iterdir())
        load_run_manifest(output)
        assert output.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == entries_before


class TestLoadVersionRejection:
    @pytest.mark.parametrize(
        "version",
        [1, 2, 0, -1, "2", "3", 2.0, 3.0, None, True],
        ids=[
            "v1",
            "v2",
            "zero",
            "negative",
            "string-two",
            "string-three",
            "float-two",
            "float-three",
            "null",
            "boolean",
        ],
    )
    def test_unsupported_format_versions_rejected(self, tmp_path, version):
        manifest = _stable_manifest()
        manifest["format_version"] = version
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "format_version" in str(exc_info.value)

    def test_format_version_3_accepted(self, tmp_path):
        output = _write_raw(tmp_path, _stable_manifest())
        assert load_run_manifest(output)["format_version"] == 3


class TestSchemaRejection:
    @pytest.mark.parametrize(
        "drop_path",
        [
            ("parameters",),
            ("split", "sha256"),
            ("code", "commit"),
            ("data", "tag"),
            ("stage",),
        ],
        ids=["top-level", "split", "code", "data", "stage"],
    )
    def test_missing_keys_rejected(self, tmp_path, drop_path):
        manifest = _drop_key(_stable_manifest(), *drop_path)
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "missing keys" in str(exc_info.value)

    @pytest.mark.parametrize(
        "key_path",
        [
            ("surprise",),
            ("code", "surprise"),
            ("data", "surprise"),
            ("split", "surprise"),
            ("parameters", "surprise"),
        ],
        ids=["top-level", "code", "data", "split", "parameters"],
    )
    def test_extra_keys_rejected(self, tmp_path, key_path):
        manifest = _set_key(_stable_manifest(), *key_path, value=1)
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "unexpected keys" in str(exc_info.value)

    @pytest.mark.parametrize(
        "key_path",
        [
            ("code",),
            ("data",),
            ("split",),
            ("parameters",),
        ],
        ids=["code", "data", "split", "parameters"],
    )
    def test_non_mapping_sections_rejected(self, tmp_path, key_path):
        manifest = _set_key(_stable_manifest(), *key_path, value=["not", "a", "mapping"])
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "must be a dictionary" in str(exc_info.value)
        assert ".".join(key_path) in str(exc_info.value)

    def test_non_dictionary_payload_rejected(self, tmp_path):
        output = tmp_path / "run.json"
        output.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert str(output) in str(exc_info.value)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda manifest: _set_key(manifest, "format", value="other"),
            lambda manifest: _set_key(manifest, "stage", value="done"),
            lambda manifest: _set_key(manifest, "code", "root", value=7),
            lambda manifest: _set_key(manifest, "code", "root", value=""),
            lambda manifest: _set_key(manifest, "code", "commit", value=""),
            lambda manifest: _set_key(manifest, "data", "tag", value=""),
            lambda manifest: _set_key(manifest, "split", "path", value=""),
            lambda manifest: _set_key(manifest, "parameters", "target_split", value="train"),
        ],
        ids=[
            "wrong-format",
            "wrong-stage",
            "numeric-root",
            "empty-root",
            "empty-commit",
            "empty-tag",
            "empty-split-path",
            "wrong-target-split",
        ],
    )
    def test_representative_incorrect_values_rejected(self, tmp_path, mutate):
        manifest = _stable_manifest()
        mutate(manifest)
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError):
            load_run_manifest(output)


class TestInvalidCRejection:
    @pytest.mark.parametrize(
        "C",
        [True, float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 0, "1.0", None],
        ids=["boolean", "nan", "infinity", "negative-infinity", "zero", "negative", "int-zero", "string", "null"],
    )
    def test_invalid_C_rejected_on_load(self, tmp_path, C):
        manifest = _stable_manifest()
        manifest["parameters"]["C"] = C
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "C" in str(exc_info.value)

    @pytest.mark.parametrize("C", [1.5, 2, 1e-3, 10**30])
    def test_valid_positive_C_accepted_on_load(self, tmp_path, C):
        manifest = _stable_manifest()
        manifest["parameters"]["C"] = C
        output = _write_raw(tmp_path, manifest)
        assert load_run_manifest(output)["parameters"]["C"] == C


class TestOversizedIntegerC:
    @staticmethod
    def _overflow_cause(error):
        cause = error
        while cause is not None:
            if isinstance(cause, OverflowError):
                return cause
            cause = cause.__cause__
        raise AssertionError("OverflowError not in exception chain")

    def test_load_rejects_oversized_integer_C_with_path_and_cause(self, tmp_path):
        manifest = _stable_manifest()
        manifest["parameters"]["C"] = 10**1000
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        message = str(exc_info.value)
        assert str(output) in message
        assert "parameters.C" in message
        self._overflow_cause(exc_info.value)

    def test_write_rejects_oversized_integer_C_before_file_creation(self, tmp_path):
        manifest = _stable_manifest()
        manifest["parameters"]["C"] = 10**1000
        output = tmp_path / "run.json"
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(manifest, output)
        assert "parameters.C" in str(exc_info.value)
        self._overflow_cause(exc_info.value)
        assert not output.exists()
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))
        assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestCollectRejectsInvalidCBeforeRepoAccess:
    @pytest.mark.parametrize(
        "C", [10**1000, True], ids=["oversized-int", "boolean"]
    )
    def test_invalid_C_rejected_before_repository_access(
        self, tmp_path, monkeypatch, C
    ):
        code = _valid_code_repo(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _split_file(tmp_path)
        calls = []

        def raising_root(*args, **kwargs):
            calls.append(args)
            raise AssertionError("repository access must not happen")

        monkeypatch.setattr(run_manifest, "_repository_root", raising_root)
        with pytest.raises(RunManifestError) as exc_info:
            _collect(code, data, split, C=C)
        assert "C" in str(exc_info.value)
        assert calls == []


class TestSplitFingerprintValidation:
    @pytest.mark.parametrize(
        "fingerprint",
        [
            "0" * 63,
            "0" * 65,
            "g" + "0" * 63,
            "0" * 32 + "A" + "0" * 31,
            "",
        ],
        ids=["too-short", "too-long", "non-hex", "uppercase", "empty"],
    )
    def test_invalid_split_sha256_rejected(self, tmp_path, fingerprint):
        manifest = _stable_manifest()
        manifest["split"]["sha256"] = fingerprint
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert "split.sha256" in str(exc_info.value)


class TestLoadErrorsPathAware:
    def test_missing_file_reports_path_and_cause(self, tmp_path):
        missing = tmp_path / "missing.json"
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(missing)
        assert str(missing) in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    @pytest.mark.parametrize(
        "content, cause_type",
        [
            (b'{"format":', json.JSONDecodeError),
            (b"\xff\xfe", UnicodeDecodeError),
        ],
        ids=["malformed-json", "invalid-utf8"],
    )
    def test_parse_failure_reports_path_and_cause(
        self, tmp_path, content, cause_type
    ):
        output = tmp_path / "run.json"
        output.write_bytes(content)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert str(output) in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, cause_type)

    def test_schema_error_reports_path(self, tmp_path):
        manifest = _stable_manifest()
        manifest["format_version"] = 1
        output = _write_raw(tmp_path, manifest)
        with pytest.raises(RunManifestError) as exc_info:
            load_run_manifest(output)
        assert str(output) in str(exc_info.value)


class TestWriterRejectsInvalidManifest:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda manifest: _set_key(manifest, "parameters", "C", value=-1.0),
            lambda manifest: _set_key(manifest, "split", "sha256", value="not-hex"),
            lambda manifest: _set_key(manifest, "format_version", value=1),
            lambda manifest: _set_key(manifest, "environment", value={}),
        ],
        ids=["invalid-C", "invalid-fingerprint", "v1", "environment"],
    )
    def test_invalid_manifest_rejected_before_any_file_creation(
        self, tmp_path, mutate
    ):
        manifest = _stable_manifest()
        mutate(manifest)
        output = tmp_path / "run.json"
        with pytest.raises(RunManifestError) as exc_info:
            write_run_manifest(manifest, output)
        assert "invalid run manifest" in str(exc_info.value)
        assert not output.exists()
        assert not list(tmp_path.glob(".vepkar-t1-run-*.tmp"))
        assert sorted(p.name for p in tmp_path.iterdir()) == []