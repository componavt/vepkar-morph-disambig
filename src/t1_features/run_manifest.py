"""Record provenance and intended settings of a forthcoming T1 dev experiment.

This module prepares metadata only.  It does not train, predict, evaluate,
or certify that a dirty working tree is reproducible.  All observations
(Git state, split fingerprint, environment, parameters) are collected
*before* the passport file itself is created, so the recorded state never
contains the passport.  Git status describes working-tree/index
differences, not a complete content archive, and ignored files are not
represented at all.  https://git-scm.com/docs/git-status  The split
SHA-256 fingerprints the raw file bytes, not semantic split equivalence.
https://www.gnu.org/s/coreutils/manual/html_node/sha2-utilities.html
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

_FORMAT_NAME = "vepkar-t1-run"
_FORMAT_VERSION = 1
_STAGE = "prepared"
_TARGET_SPLIT = "dev"
_ENVIRONMENT_KEYS = ("python", "numpy", "scipy", "scikit-learn", "joblib")
_TEMP_PREFIX = ".vepkar-t1-run-"
_TEMP_SUFFIX = ".tmp"
_CHUNK_BYTES = 65536


class RunManifestError(ValueError):
    """Expected collection or publication failure for a T1 run passport."""


def _current_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _git(root: Path, *args: str) -> str:
    command = ["git", "--no-optional-locks", "-C", str(root), *args]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as exc:
        raise RunManifestError(
            f"cannot run Git for {root}: git {' '.join(args)}: {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        raise RunManifestError(
            f"Git operation failed for {root}: git {' '.join(args)}: {detail}"
        ) from exc
    return result.stdout


def _repository_root(supplied: Path) -> Path:
    conveyed = Path(_git(supplied, "rev-parse", "--show-toplevel").strip())
    expected = supplied.resolve()
    if conveyed != expected:
        raise RunManifestError(
            f"{supplied} is inside a Git repository rooted at {conveyed}, "
            "not the intended repository root"
        )
    return conveyed


def _git_snapshot(root: Path) -> dict[str, object]:
    commit = _git(root, "rev-parse", "HEAD").strip()
    porcelain = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ).rstrip("\n")
    return {
        "root": str(root),
        "commit": commit,
        "status_porcelain": porcelain,
        "dirty": bool(porcelain),
    }


def _split_sha256(path: Path) -> str:
    resolved = path.resolve()
    digest = hashlib.sha256()
    try:
        with open(resolved, "rb") as fh:
            while True:
                block = fh.read(_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise RunManifestError(f"cannot read split file: {path}: {exc}") from exc
    return digest.hexdigest()


def collect_run_manifest(
    *,
    code_dir: Path,
    data_dir: Path,
    data_tag: str,
    split_file: Path,
    C: float,
) -> dict:
    """Collect all recorded observations into one prepared run passport."""
    if not math.isfinite(C) or C <= 0:
        raise RunManifestError(f"C must be finite and positive, got {C!r}")
    code_root = _repository_root(code_dir)
    data_root = _repository_root(data_dir)
    if code_root == data_root:
        raise RunManifestError(
            f"code and data must be separate repositories; both are {code_root}"
        )
    code = _git_snapshot(code_root)
    data_snapshot = _git_snapshot(data_root)
    data = {
        "root": data_snapshot["root"],
        "tag": data_tag,
        "commit": data_snapshot["commit"],
        "status_porcelain": data_snapshot["status_porcelain"],
        "dirty": data_snapshot["dirty"],
    }
    return {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "stage": _STAGE,
        "code": code,
        "data": data,
        "split": {
            "path": str(split_file.resolve()),
            "sha256": _split_sha256(split_file),
        },
        "environment": _current_environment(),
        "parameters": {"C": C, "target_split": _TARGET_SPLIT},
    }


def write_run_manifest(manifest: dict, output_path: Path) -> None:
    """Publish the passport text to ``output_path`` without overwriting.

    The destination must not exist and its parent directory must already
    exist; nothing else is created.  The text is written to a temporary
    sibling file that is then hard-linked into place.  All observations
    were collected by :func:`collect_run_manifest` before publication, so
    the passport never appears in the recorded state.
    """
    if output_path.is_symlink() or output_path.exists():
        raise RunManifestError(f"destination already exists: {output_path}")
    parent = output_path.parent
    if not parent.is_dir():
        raise RunManifestError(f"parent directory does not exist: {parent}")
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent)
        )
    except OSError as exc:
        raise RunManifestError(
            f"cannot write run passport {output_path}: create temporary "
            f"file: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise RunManifestError(
                f"cannot write run passport {output_path}: open temporary "
                f"file {temp_path}: {exc}"
            ) from exc
        write_failed = False
        try:
            with stream:
                try:
                    stream.write(text)
                except BaseException:
                    write_failed = True
                    raise
        except OSError as exc:
            if not write_failed:
                raise RunManifestError(
                    f"cannot write run passport {output_path}: close "
                    f"temporary file {temp_path}: {exc}"
                ) from exc
            primary = exc.__context__
            if primary is not None and not isinstance(primary, OSError):
                primary.add_note(
                    f"failed to close temporary run passport stream "
                    f"{temp_path}: {exc}"
                )
                raise primary
            if primary is not None:
                error = RunManifestError(
                    f"cannot write run passport {output_path}: write "
                    f"temporary file {temp_path}: {primary}"
                )
                error.add_note(
                    f"failed to close temporary run passport stream "
                    f"{temp_path}: {exc}"
                )
                raise error from primary
            raise RunManifestError(
                f"cannot write run passport {output_path}: write "
                f"temporary file {temp_path}: {exc}"
            ) from exc
        try:
            os.link(temp_path, output_path)
        except OSError as exc:
            raise RunManifestError(
                f"cannot write run passport {output_path}: publish "
                f"{temp_path}: {exc}"
            ) from exc
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            exc.add_note(
                f"failed to clean up temporary run passport {temp_path}: "
                f"{cleanup_exc}"
            )
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RunManifestError(
            f"run passport published at {output_path} but removal of "
            f"temporary {temp_path} failed"
        ) from exc