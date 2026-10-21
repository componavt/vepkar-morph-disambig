"""Offline tests for machine-local T1 configuration and split resolution."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from t1_features.config import (  # noqa: E402
    T1ConfigError,
    load_t1_config,
    resolve_t1_split_file,
)

CONFIG_NAME = "t1.local.toml"


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / CONFIG_NAME
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_missing_config_uses_defaults(tmp_path, capsys):
    config = load_t1_config(tmp_path)
    assert config.config_found is False
    assert config.config_path == (tmp_path / CONFIG_NAME).resolve()
    assert config.repo_root == tmp_path.resolve()
    assert config.data_dir == (tmp_path / "data" / "dictorpus-data").resolve()
    assert config.runs_dir == (tmp_path / "work" / "runs").resolve()
    assert config.split_file is None
    assert capsys.readouterr().out == ""


def test_empty_config_file_uses_defaults(tmp_path):
    write_config(tmp_path, "")
    config = load_t1_config(tmp_path)
    assert config.config_found is True
    assert config.data_dir == (tmp_path / "data" / "dictorpus-data").resolve()
    assert config.runs_dir == (tmp_path / "work" / "runs").resolve()
    assert config.split_file is None


def test_partial_paths_table_merges_defaults(tmp_path):
    write_config(tmp_path, '[paths]\ndata_dir = "data/custom-data"\n')
    config = load_t1_config(tmp_path)
    assert config.data_dir == (tmp_path / "data" / "custom-data").resolve()
    assert config.runs_dir == (tmp_path / "work" / "runs").resolve()
    assert config.split_file is None


def test_full_toml_settings(tmp_path):
    write_config(
        tmp_path,
        '[paths]\n'
        'data_dir = "data/custom-data"\n'
        'runs_dir = "work/custom-runs"\n'
        'split_file = "data/derived/splits_custom.csv"\n',
    )
    config = load_t1_config(tmp_path)
    assert config.data_dir == (tmp_path / "data" / "custom-data").resolve()
    assert config.runs_dir == (tmp_path / "work" / "custom-runs").resolve()
    assert config.split_file == (
        tmp_path / "data" / "derived" / "splits_custom.csv"
    ).resolve()


@pytest.mark.parametrize(
    "key, toml_value, override_value",
    [
        ("data_dir", "data/toml-data", "override-data"),
        ("runs_dir", "work/toml-runs", "override-runs"),
        ("split_file", "data/split.csv", "override-split.csv"),
    ],
)
def test_explicit_override_wins_over_toml(
    tmp_path, monkeypatch, key, toml_value, override_value
):
    fixed = {
        "data_dir": "data/toml-data",
        "runs_dir": "work/toml-runs",
        "split_file": "data/split.csv",
    }
    fixed[key] = toml_value
    write_config(
        tmp_path,
        "[paths]\n"
        + "".join(f'{k} = {v!r}\n' for k, v in fixed.items()),
    )
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    config = load_t1_config(tmp_path, **{key: Path(override_value)})
    assert getattr(config, key) == (work_dir / override_value).resolve()
    for other in ("data_dir", "runs_dir", "split_file"):
        if other != key:
            assert getattr(config, other) == (
                tmp_path / fixed[other]
            ).resolve()


def test_config_paths_repo_relative_overrides_cwd_relative(tmp_path, monkeypatch):
    write_config(tmp_path, '[paths]\ndata_dir = "data/configured-data"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_t1_config(tmp_path, runs_dir=Path("local/runs"))
    assert config.data_dir == (tmp_path / "data" / "configured-data").resolve()
    assert config.runs_dir == (elsewhere / "local" / "runs").resolve()


def test_absolute_spaces_and_home_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    write_config(
        tmp_path,
        '[paths]\n'
        'data_dir = "~/data with spaces"\n'
        'runs_dir = "/absolute/runs dir"\n',
    )
    config = load_t1_config(tmp_path)
    assert config.data_dir == (tmp_path / "home" / "data with spaces").resolve()
    assert config.runs_dir == Path("/absolute/runs dir").resolve()


@pytest.mark.parametrize(
    "content, match",
    [
        ('[paths]\ndata_dir = "unterminated', "Cannot read T1 local configuration"),
        ('data_dir = "top-level"', "unknown top-level key"),
        ('[other]\nx = 1', "unknown top-level key"),
        ('paths = "not-a-table"', "must be a table"),
        ('paths = ["a"]', "must be a table"),
        ('[paths]\nunknown_key = "x"', "unknown key"),
        ('[paths]\ndata_dir = 5', "must be a nonempty string"),
        ('[paths]\nruns_dir = ["a"]', "must be a nonempty string"),
        ('[paths]\nsplit_file = true', "must be a nonempty string"),
        ('[paths]\ndata_dir = ""', "must be a nonempty string"),
        ('[paths]\ndata_dir = "   "', "must be a nonempty string"),
    ],
)
def test_invalid_config_rejected(tmp_path, content, match):
    write_config(tmp_path, content)
    with pytest.raises(T1ConfigError, match=match):
        load_t1_config(tmp_path)


def test_invalid_config_rejected_despite_overrides(tmp_path):
    write_config(tmp_path, '[paths]\nunknown_key = "x"\n')
    with pytest.raises(T1ConfigError, match="unknown key"):
        load_t1_config(tmp_path, data_dir=Path("override"), runs_dir=Path("override"))


def test_unreadable_config_raises(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, "[paths]\ndata_dir = 'x'\n")
    real_open = Path.open

    def deny_read(self, *args, **kwargs):
        if self.resolve() == config_path.resolve():
            raise PermissionError("permission denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_read)
    with pytest.raises(T1ConfigError, match="permission denied"):
        load_t1_config(tmp_path)


def test_broken_symlink_config_raises(tmp_path):
    (tmp_path / CONFIG_NAME).symlink_to(tmp_path / "missing-target")
    with pytest.raises(T1ConfigError, match="broken symlink"):
        load_t1_config(tmp_path)


def test_configured_split_takes_precedence(tmp_path):
    write_config(
        tmp_path, '[paths]\nsplit_file = "data/derived/splits_configured.csv"\n'
    )
    config = load_t1_config(tmp_path)
    assert resolve_t1_split_file(config, "tag-1") == (
        tmp_path / "data" / "derived" / "splits_configured.csv"
    ).resolve()


def test_published_split_filename_follows_tag(tmp_path):
    config = load_t1_config(tmp_path)
    first = resolve_t1_split_file(config, "tag-a")
    second = resolve_t1_split_file(config, "tag-b")
    assert first == (tmp_path / "data" / "derived" / "splits_tag-a.csv").resolve()
    assert second == (tmp_path / "data" / "derived" / "splits_tag-b.csv").resolve()
    assert first != second


@pytest.mark.parametrize("tag", ["", "a/b", "a\\b", "a/b/c"])
def test_split_resolution_rejects_invalid_tags(tmp_path, tag):
    config = load_t1_config(tmp_path)
    with pytest.raises(T1ConfigError):
        resolve_t1_split_file(config, tag)


def test_split_resolution_does_not_require_existing_file(tmp_path):
    config = load_t1_config(tmp_path)
    resolved = resolve_t1_split_file(config, "fresh-tag")
    assert not resolved.exists()
    assert resolved == (tmp_path / "data" / "derived" / "splits_fresh-tag.csv").resolve()


def test_loading_defaults_creates_no_files(tmp_path):
    before = sorted(tmp_path.rglob("*"))
    load_t1_config(tmp_path)
    assert sorted(tmp_path.rglob("*")) == before
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "data").exists()


def test_loading_does_not_modify_config_bytes(tmp_path):
    content = b'[paths]\ndata_dir = "data/x"\n'
    config_path = write_config(tmp_path, content.decode())
    load_t1_config(tmp_path)
    assert config_path.read_bytes() == content