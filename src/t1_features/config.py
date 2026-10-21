"""Machine-local T1 configuration and deterministic path resolution.

T1 runs read an optional ``<repo_root>/t1.local.toml`` file holding
machine-local path overrides.  Paths are resolved deterministically:
explicit keyword overrides beat TOML values, which beat built-in defaults.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from core.splits import split_csv_path

CONFIG_FILENAME = "t1.local.toml"


class T1ConfigError(ValueError):
    """Invalid, unreadable, or unusable T1 local configuration or split tag."""


@dataclass(frozen=True)
class T1Config:
    """Resolved T1 paths.

    ``repo_root`` is absolute.  ``data_dir`` and ``runs_dir`` are absolute
    regardless of the filesystem.  ``split_file`` is ``None`` unless
    explicitly configured or overridden.  ``config_path`` names the
    configuration file even when no such file exists, and ``config_found``
    reports whether it was actually read.
    """

    repo_root: Path
    data_dir: Path
    split_file: Path | None
    runs_dir: Path
    config_path: Path
    config_found: bool


def _resolve_path(value: Path | str, base: Path) -> Path:
    """Resolve one configured path against ``base``.

    A leading ``~`` is expanded with :meth:`Path.expanduser`; relative paths
    are joined to ``base``; absolute paths keep their target.  The result is
    normalized without requiring any destination to exist.  ``$VARIABLE``
    text is never expanded.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_paths(data: dict, config_path: Path) -> dict[str, str]:
    """Validate the decoded TOML mapping and return the ``[paths]`` values."""
    unknown = sorted(set(data) - {"paths"})
    if unknown:
        raise T1ConfigError(
            f"Invalid T1 local configuration {config_path}: unknown "
            f"top-level key(s) {', '.join(repr(key) for key in unknown)}"
        )
    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        raise T1ConfigError(
            f"Invalid T1 local configuration {config_path}: `paths` must be "
            "a table"
        )
    unknown = sorted(set(paths) - {"data_dir", "runs_dir", "split_file"})
    if unknown:
        raise T1ConfigError(
            f"Invalid T1 local configuration {config_path}: unknown key(s) "
            f"in [paths] {', '.join(repr(key) for key in unknown)}"
        )
    values: dict[str, str] = {}
    for key in ("data_dir", "runs_dir", "split_file"):
        if key not in paths:
            continue
        value = paths[key]
        if not isinstance(value, str) or not value or value.isspace():
            raise T1ConfigError(
                f"Invalid T1 local configuration {config_path}: "
                f"`paths.{key}` must be a nonempty string"
            )
        values[key] = value
    return values


def _pick_path(
    override: Path | None,
    configured: str | None,
    default: Path | None,
    repo_root: Path,
) -> Path | None:
    """Apply explicit override > TOML value > built-in default."""
    if override is not None:
        return _resolve_path(override, Path.cwd())
    if configured is not None:
        return _resolve_path(configured, repo_root)
    if default is not None:
        return _resolve_path(default, repo_root)
    return None


def load_t1_config(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
    split_file: Path | None = None,
    runs_dir: Path | None = None,
) -> T1Config:
    """Load and resolve T1 configuration paths.

    Reads ``<repo_root>/t1.local.toml`` when present; a genuinely missing
    file is not an error and leaves ``config_found`` False.  TOML values are
    resolved relative to ``repo_root``; explicit overrides are resolved
    relative to the caller's current working directory.
    """
    root = Path(repo_root).expanduser().resolve()
    config_path = root / CONFIG_FILENAME
    config_found = True
    try:
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except FileNotFoundError:
        if config_path.is_symlink():
            raise T1ConfigError(
                f"T1 local configuration is a broken symlink: {config_path}"
            ) from None
        config_found = False
        data = {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise T1ConfigError(
            f"Cannot read T1 local configuration {config_path}: {exc}"
        ) from exc
    values = _parse_paths(data, config_path)
    return T1Config(
        repo_root=root,
        data_dir=_pick_path(
            data_dir, values.get("data_dir"), root / "data" / "dictorpus-data", root
        ),
        runs_dir=_pick_path(
            runs_dir, values.get("runs_dir"), root / "work" / "runs", root
        ),
        split_file=_pick_path(split_file, values.get("split_file"), None, root),
        config_path=config_path,
        config_found=config_found,
    )


def resolve_t1_split_file(config: T1Config, tag: str) -> Path:
    """Return the split CSV path for a data ``tag``.

    An explicitly configured or overridden ``split_file`` wins; otherwise the
    published split for the actual data tag is
    ``<repo_root>/data/derived/splits_<tag>.csv``.  The resulting file is not
    required to exist.
    """
    if not isinstance(tag, str) or not tag:
        raise T1ConfigError("split tag must be a nonempty string")
    if "/" in tag or "\\" in tag:
        raise T1ConfigError(
            f"split tag must not contain path separators: {tag!r}"
        )
    if config.split_file is not None:
        return config.split_file
    return split_csv_path(config.repo_root / "data" / "derived", tag)