"""Offline tests for T1 run-directory creation and passport publication."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
import t1_features.workspace as workspace  # noqa: E402
from t1_features.config import load_t1_config  # noqa: E402
from t1_features.run_manifest import (  # noqa: E402
    RunManifestError,
    load_run_manifest,
)
from t1_features.workspace import (  # noqa: E402
    T1WorkspaceError,
    create_t1_run,
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


def _project_root(tmp_path: Path, name: str = "code") -> Path:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tagged_data_repo(tmp_path: Path, name: str = "data") -> Path:
    root = _repo(
        tmp_path,
        name,
        {"README.md": "fixture data repo\n", "corpus/placeholder.txt": "x\n"},
    )
    _git(root, "tag", _TAG)
    return root


def _config(tmp_path: Path, suffix: str = "") -> object:
    code = _project_root(tmp_path, name=f"code{suffix}")
    data = _tagged_data_repo(tmp_path, name=f"data{suffix}")
    return load_t1_config(
        code,
        data_dir=data,
        runs_dir=tmp_path / f"runs{suffix}",
    )


def _default_split(code_root: Path, content: str = _SPLIT_CONTENT) -> Path:
    path = code_root / "data" / "derived" / f"splits_{_TAG}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestAutomaticCreation:
    @pytest.mark.parametrize(
        ("C", "label"),
        [(1.0, "t1-C1-"), (0.1, "t1-C0.1-"), (2.5, "t1-C2.5-")],
    )
    def test_creates_run_with_readable_label_and_valid_passport(
        self, tmp_path, C, label
    ):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        created = create_t1_run(config, C=C)
        assert created.is_absolute()
        assert created.parent == config.runs_dir
        assert created.name.startswith(label)
        assert re.fullmatch(r"[A-Za-z0-9_]+", created.name[len(label) :])
        assert sorted(p.name for p in created.iterdir()) == ["run.json"]
        manifest = load_run_manifest(created / "run.json")
        assert manifest["format_version"] == 4
        assert manifest["stage"] == "prepared"
        assert manifest["parameters"]["C"] == C
        assert manifest["parameters"]["target_split"] == "dev"

    def test_two_automatic_calls_create_distinct_runs(self, tmp_path):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        first = create_t1_run(config)
        first_bytes = (first / "run.json").read_bytes()
        second = create_t1_run(config)
        assert second != first
        assert (first / "run.json").read_bytes() == first_bytes
        assert sorted(config.runs_dir.iterdir()) == sorted([first, second])


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestExplicitPaths:
    def test_absolute_path_with_missing_parents(self, tmp_path):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        target = tmp_path / "alpha" / "beta" / "run-abs"
        created = create_t1_run(config, run_dir=target)
        assert created == target
        assert created.is_absolute()
        assert load_run_manifest(created / "run.json")["parameters"]["C"] == 1.0

    def test_relative_path_resolved_against_cwd(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        work = tmp_path / "workdir"
        work.mkdir()
        monkeypatch.chdir(work)
        created = create_t1_run(config, run_dir=Path("custom/runs/run-rel"))
        assert created.is_absolute()
        assert created.resolve() == (work / "custom" / "runs" / "run-rel").resolve()
        assert load_run_manifest(created / "run.json")["parameters"]["C"] == 1.0

    def test_tilde_relative_to_home_not_repo(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        created = create_t1_run(config, run_dir=Path("~/runs/from-home"))
        assert created.resolve() == (home / "runs" / "from-home").resolve()
        assert load_run_manifest(created / "run.json")["parameters"]["C"] == 1.0


def _make_existing_final(tmp_path: Path, kind: str) -> Path:
    target = tmp_path / "existing" / "run"
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "file":
        target.write_text("precious\n", encoding="utf-8")
    elif kind == "empty-directory":
        target.mkdir()
    elif kind == "nonempty-directory":
        target.mkdir()
        (target / "keep.txt").write_text("keep\n", encoding="utf-8")
    elif kind == "valid-symlink":
        link_target = tmp_path / "existing" / "linked"
        link_target.mkdir()
        target.symlink_to(link_target)
    else:
        target.symlink_to(tmp_path / "existing" / "missing")
    return target


@pytest.mark.parametrize(
    "kind",
    ["file", "empty-directory", "nonempty-directory", "valid-symlink", "dangling-symlink"],
)
def test_existing_final_path_rejected_early_and_preserved(tmp_path, kind):
    config = load_t1_config(tmp_path / "norepo", runs_dir=tmp_path / "unused")
    target = _make_existing_final(tmp_path, kind)
    with pytest.raises(T1WorkspaceError) as exc_info:
        create_t1_run(config, run_dir=target)
    assert str(target) in str(exc_info.value)
    assert "already exists" in str(exc_info.value)
    if kind == "file":
        assert target.read_text(encoding="utf-8") == "precious\n"
    elif kind == "empty-directory":
        assert target.is_dir()
        assert sorted(target.iterdir()) == []
    elif kind == "nonempty-directory":
        assert target.is_dir()
        assert (target / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    elif kind == "valid-symlink":
        assert target.is_symlink()
        assert target.readlink() == (tmp_path / "existing" / "linked")
        assert (tmp_path / "existing" / "linked").is_dir()
    else:
        assert target.is_symlink()
        assert not (tmp_path / "existing" / "missing").exists()
    assert not (target / "run.json").exists()


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestSplitResolution:
    def test_default_tag_split_and_configured_split_paths_and_hashes(
        self, tmp_path
    ):
        config = _config(tmp_path)
        default_split = _default_split(config.repo_root)
        created = create_t1_run(config)
        manifest = load_run_manifest(created / "run.json")
        assert manifest["split"]["path"] == str(default_split.resolve())
        assert manifest["split"]["sha256"] == _digest(default_split)

        explicit = tmp_path / "custom-split.csv"
        explicit.write_text(_SPLIT_CONTENT, encoding="utf-8")
        configured = _config(tmp_path, suffix="-2")
        configured = load_t1_config(
            configured.repo_root,
            data_dir=configured.data_dir,
            runs_dir=tmp_path / "runs-2",
            split_file=explicit,
        )
        created2 = create_t1_run(configured)
        manifest2 = load_run_manifest(created2 / "run.json")
        assert manifest2["split"]["path"] == str(explicit.resolve())
        assert manifest2["split"]["sha256"] == _digest(explicit)


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestPreparationFailures:
    @pytest.mark.parametrize(
        "C",
        [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "1.0", None],
    )
    def test_invalid_C_rejected_before_repository_or_creation(self, tmp_path, C):
        config = load_t1_config(tmp_path / "norepo", runs_dir=tmp_path / "runs")
        with pytest.raises(T1WorkspaceError):
            create_t1_run(config, C=C)
        assert not (tmp_path / "runs").exists()

    def test_oversized_integer_C_rejected_without_overflowerror_leak(
        self, tmp_path
    ):
        config = load_t1_config(tmp_path / "norepo", runs_dir=tmp_path / "runs")
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config, C=10**1000)
        assert "C" in str(exc_info.value)
        cause = exc_info.value
        while cause is not None:
            if isinstance(cause, OverflowError):
                break
            cause = cause.__cause__
        assert isinstance(cause, OverflowError)
        assert not (tmp_path / "runs").exists()

    def test_preflight_failure_creates_no_hierarchy(self, tmp_path):
        code = _project_root(tmp_path)
        config = load_t1_config(
            code,
            data_dir=tmp_path / "no-such-data",
            runs_dir=tmp_path / "runs",
        )
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config)
        assert "preflight" in str(exc_info.value) or "missing" in str(exc_info.value)
        assert not (tmp_path / "runs").exists()

    def test_missing_split_creates_no_hierarchy(self, tmp_path):
        config = _config(tmp_path)
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config)
        assert "split" in str(exc_info.value)
        assert not config.runs_dir.exists()


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestRollback:
    def test_pre_publication_failure_removes_empty_run_directory(self, tmp_path, monkeypatch):
        config = _config(tmp_path)
        _default_split(config.repo_root)

        def failing_write(manifest, output_path):
            raise RunManifestError("simulated publication denial")

        monkeypatch.setattr(workspace, "write_run_manifest", failing_write)
        with pytest.raises(T1WorkspaceError):
            create_t1_run(config)
        assert config.runs_dir.is_dir()
        assert sorted(config.runs_dir.iterdir()) == []

    def test_writer_failure_with_retained_file_reports_run_directory(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        target = tmp_path / "runs" / "kept-run"
        original_write = workspace.write_run_manifest

        def publish_then_fail(manifest, output_path):
            original_write(manifest, output_path)
            raise RunManifestError("simulated post-publication failure")

        monkeypatch.setattr(workspace, "write_run_manifest", publish_then_fail)
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config, run_dir=target)
        combined = str(exc_info.value) + "\n".join(
            getattr(exc_info.value, "__notes__", ())
        )
        assert "retained" in combined
        assert str(target) in combined
        assert target.is_dir()
        assert (target / "run.json").is_file()
        load_run_manifest(target / "run.json")
        assert sorted(p.name for p in target.iterdir()) == ["run.json"]

    def test_directory_creation_failure_preserves_reason_and_foreign_paths(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        target = tmp_path / "runs" / "run-a"
        foreign = tmp_path / "runs" / "foreign.txt"
        foreign.parent.mkdir()
        foreign.write_text("keep\n", encoding="utf-8")
        original_mkdir = Path.mkdir

        def denying_mkdir(self, *args, **kwargs):
            if "parents" not in kwargs and self == target:
                raise PermissionError("simulated final directory denial")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", denying_mkdir)
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config, run_dir=target)
        assert isinstance(exc_info.value.__cause__, PermissionError)
        assert "simulated final directory denial" in str(exc_info.value)
        assert not target.exists()
        assert foreign.read_text(encoding="utf-8") == "keep\n"
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
            "foreign.txt"
        ]

    def test_actual_final_directory_creation_collision(
        self, tmp_path, monkeypatch
    ):
        config = _config(tmp_path)
        _default_split(config.repo_root)
        target = tmp_path / "runs" / "run-collision"
        marker = b"foreign marker bytes before the passport\n"
        original_mkdir = Path.mkdir
        staged = False

        def colliding_mkdir(self, *args, **kwargs):
            nonlocal staged
            if "parents" not in kwargs and self == target and not staged:
                staged = True
                original_mkdir(self, *args, **kwargs)
                (self / "marker.bin").write_bytes(marker)
                original_mkdir(self, *args, **kwargs)
            return original_mkdir(self, *args, **kwargs)

        writer_calls = []
        original_write = workspace.write_run_manifest

        def recording_write(manifest, output_path):
            writer_calls.append(output_path)
            return original_write(manifest, output_path)

        monkeypatch.setattr(Path, "mkdir", colliding_mkdir)
        monkeypatch.setattr(workspace, "write_run_manifest", recording_write)
        with pytest.raises(T1WorkspaceError) as exc_info:
            create_t1_run(config, run_dir=target)
        error = exc_info.value
        assert str(target) in str(error)
        assert "already exists" in str(error)
        assert "concurrent" in str(error)
        assert isinstance(error.__cause__, FileExistsError)
        assert target.is_dir()
        assert (target / "marker.bin").read_bytes() == marker
        assert sorted(p.name for p in target.iterdir()) == ["marker.bin"]
        assert not (target / "run.json").exists()
        assert writer_calls == []


@pytest.mark.skipif(not GIT_AVAILABLE, reason="Git unavailable")
class TestNoSideEffects:
    def test_inputs_unchanged_and_no_table_loading_or_output(
        self, tmp_path, monkeypatch, capsys
    ):
        def forbidden(lang, data_dir):
            raise AssertionError("corpus tables must not be loaded")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", forbidden)
        monkeypatch.setattr(benchmark_context, "build_language_instances", forbidden)

        code = _project_root(tmp_path)
        data = _tagged_data_repo(tmp_path)
        split = _default_split(code)
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
        config = load_t1_config(code, data_dir=data, runs_dir=tmp_path / "runs")
        capsys.readouterr()
        created = create_t1_run(config)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert split.read_bytes() == split_before
        assert {
            path.relative_to(data): path.read_bytes()
            for path in data.rglob("*")
            if path.is_file()
        } == data_files
        assert {
            path.relative_to(code): path.read_bytes()
            for path in code.rglob("*")
            if path.is_file()
        } == code_files
        assert load_run_manifest(created / "run.json")
        assert not list(tmp_path.rglob(".vepkar-t1-run-*.tmp"))
        assert sorted(p.name for p in config.runs_dir.iterdir()) == [created.name]