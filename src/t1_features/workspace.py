"""Create a fresh T1 run directory and publish its version-4 passport.

The run directory is created only after all recorded observations (data
repository snapshot, split fingerprint, parameters) have been collected
successfully; the passport is then published into the directory as
``run.json``.  Preparation never loads corpus tables and never modifies the
data checkout.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from benchmark_context import _preflight_benchmark
from core.data import DataError
from core.instances import CorpusTagError
from t1_features.config import T1Config, T1ConfigError, resolve_t1_split_file
from t1_features.run_manifest import (
    RunManifestError,
    collect_run_manifest,
    write_run_manifest,
)


class T1WorkspaceError(ValueError):
    """A T1 run directory could not be prepared or its passport published."""


def _validate_C(C) -> None:
    if type(C) is bool or not isinstance(C, (int, float)):
        raise T1WorkspaceError(
            f"invalid C: expected a finite positive number, got {C!r}"
        )
    try:
        finite = math.isfinite(C)
    except OverflowError as exc:
        raise T1WorkspaceError(
            f"invalid C: the value overflows the finite representation, "
            f"got {C!r}"
        ) from exc
    if not finite or C <= 0:
        raise T1WorkspaceError(
            f"invalid C: must be finite and positive, got {C!r}"
        )


def _run_label(C) -> str:
    """Readable directory-name label for the regularization strength."""
    if type(C) is int:
        return str(C)
    return format(C, "g")


def _resolve_explicit_run_dir(run_dir: Path) -> Path:
    target = Path(run_dir).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    return target


def _rollback_run_dir(run_dir: Path) -> str | None:
    """Remove an empty run directory; return a retention note or None."""
    try:
        run_dir.rmdir()
    except FileNotFoundError:
        return None
    except OSError as exc:
        nonempty = False
        try:
            nonempty = run_dir.is_dir() and any(run_dir.iterdir())
        except OSError:
            nonempty = True
        if nonempty:
            return (
                f"run directory retained with contents after passport "
                f"publication failure (rmdir failed: {exc}): {run_dir}"
            )
        return (
            f"failed to remove empty run directory after passport "
            f"publication failure (rmdir failed: {exc}): {run_dir}"
        )
    return None


def _create_automatic_run_dir(runs_dir: Path, C) -> Path:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise T1WorkspaceError(
            f"cannot create runs directory {runs_dir}: {exc}"
        ) from exc
    try:
        name = tempfile.mkdtemp(prefix=f"t1-C{_run_label(C)}-", dir=runs_dir)
    except OSError as exc:
        raise T1WorkspaceError(
            f"cannot create a new run directory under {runs_dir}: {exc}"
        ) from exc
    return Path(name)


def _create_explicit_run_dir(target: Path) -> Path:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise T1WorkspaceError(
            f"cannot create parent directories for run directory {target}: {exc}"
        ) from exc
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise T1WorkspaceError(
            f"run directory already exists (appeared concurrently): {target}"
        ) from exc
    except OSError as exc:
        raise T1WorkspaceError(
            f"cannot create run directory {target}: {exc}"
        ) from exc
    return target


def create_t1_run(
    config: T1Config,
    *,
    C: float = 1.0,
    run_dir: Path | None = None,
) -> Path:
    """Prepare one T1 run directory and publish its passport under it.

    With ``run_dir`` None a unique directory ``t1-C<label>-<suffix>`` is
    created under ``config.runs_dir``.  With an explicit ``run_dir``, a
    leading ``~`` is expanded, a relative path is resolved against the
    caller's current directory, and an already-existing final path is
    rejected before any other work.
    """
    _validate_C(C)
    if run_dir is not None:
        target = _resolve_explicit_run_dir(run_dir)
        if target.is_symlink() or target.exists():
            raise T1WorkspaceError(f"run directory already exists: {target}")
    else:
        target = None
    try:
        data_dir, tag = _preflight_benchmark(config.data_dir)
    except (DataError, CorpusTagError) as exc:
        raise T1WorkspaceError(
            f"cannot preflight benchmark data checkout {config.data_dir}: {exc}"
        ) from exc
    try:
        split_path = resolve_t1_split_file(config, tag)
    except T1ConfigError as exc:
        raise T1WorkspaceError(
            f"cannot resolve split file for data tag {tag!r}: {exc}"
        ) from exc
    try:
        manifest = collect_run_manifest(
            data_dir=data_dir,
            data_tag=tag,
            split_file=split_path,
            C=C,
        )
    except RunManifestError as exc:
        raise T1WorkspaceError(f"cannot collect run manifest: {exc}") from exc
    if target is None:
        created = _create_automatic_run_dir(config.runs_dir, C)
    else:
        created = _create_explicit_run_dir(target)
    passport = created / "run.json"
    try:
        write_run_manifest(manifest, passport)
    except RunManifestError as exc:
        note = _rollback_run_dir(created)
        error = T1WorkspaceError(f"cannot write run passport {passport}: {exc}")
        if note is not None:
            error.add_note(note)
        raise error from exc
    except BaseException as exc:
        note = _rollback_run_dir(created)
        if note is not None:
            exc.add_note(note)
        raise
    return created