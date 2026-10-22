"""Record provenance and intended settings of a forthcoming T1 dev experiment.

This module prepares metadata only.  It does not train, predict, or
evaluate.  All observations (data repository identity, split fingerprint,
parameters) are collected *before* the passport file itself is created, so
the recorded state never contains the passport.  The split SHA-256
fingerprints the raw file bytes, not semantic split equivalence.
https://www.gnu.org/s/coreutils/manual/html_node/sha2-utilities.html
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

_FORMAT_NAME = "vepkar-t1-run"
_FORMAT_VERSION = 4
_STAGE = "prepared"
_TARGET_SPLIT = "dev"
_TEMP_PREFIX = ".vepkar-t1-run-"
_TEMP_SUFFIX = ".tmp"
_CHUNK_BYTES = 65536
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = (
    "format",
    "format_version",
    "stage",
    "data",
    "split",
    "parameters",
)
_DATA_KEYS = ("root", "tag", "commit")
_SPLIT_KEYS = ("path", "sha256")
_PARAMETER_KEYS = ("C", "target_split")


class RunManifestError(ValueError):
    """Expected collection or publication failure for a T1 run passport."""


def _require_keys(mapping: dict, expected, location: str) -> None:
    missing = [key for key in expected if key not in mapping]
    extra = [key for key in mapping if key not in expected]
    if not missing and not extra:
        return
    problems = []
    if missing:
        problems.append(f"missing keys {missing}")
    if extra:
        problems.append(f"unexpected keys {extra}")
    raise RunManifestError(
        f"invalid run manifest: {location}: " + "; ".join(problems)
    )


def _require_mapping(value, location: str) -> dict:
    if not isinstance(value, dict):
        raise RunManifestError(
            f"invalid run manifest: {location} must be a dictionary, "
            f"got {value!r}"
        )
    return value


def _require_nonempty_string(value, location: str) -> None:
    if not isinstance(value, str) or not value:
        raise RunManifestError(
            f"invalid run manifest: {location} must be a nonempty string, "
            f"got {value!r}"
        )


def _validate_repo(section: dict, location: str, *, has_tag: bool) -> None:
    _require_nonempty_string(section["root"], f"{location}.root")
    _require_nonempty_string(section["commit"], f"{location}.commit")
    if has_tag:
        _require_nonempty_string(section["tag"], f"{location}.tag")


def _validate_positive_finite(value) -> None:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise RunManifestError(
            f"invalid run manifest: parameters.C must be a finite positive "
            f"number, got {value!r}"
        )
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise RunManifestError(
            f"invalid run manifest: parameters.C overflows the finite "
            f"representation, got {value!r}"
        ) from exc
    if not finite or value <= 0:
        raise RunManifestError(
            f"invalid run manifest: parameters.C must be finite and "
            f"positive, got {value!r}"
        )


def _validate_manifest(manifest: object) -> None:
    """Require the exact version-4 T1 run passport schema."""
    if not isinstance(manifest, dict):
        raise RunManifestError("invalid run manifest: expected a dictionary")
    _require_keys(manifest, _TOP_LEVEL_KEYS, "manifest")
    if manifest["format"] != _FORMAT_NAME:
        raise RunManifestError(
            f"invalid run manifest: format must be {_FORMAT_NAME!r}, "
            f"got {manifest['format']!r}"
        )
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != _FORMAT_VERSION
    ):
        raise RunManifestError(
            f"invalid run manifest: format_version must be the integer "
            f"{_FORMAT_VERSION}, got {manifest['format_version']!r}"
        )
    if manifest["stage"] != _STAGE:
        raise RunManifestError(
            f"invalid run manifest: stage must be {_STAGE!r}, "
            f"got {manifest['stage']!r}"
        )
    data = _require_mapping(manifest["data"], "manifest.data")
    _require_keys(data, _DATA_KEYS, "manifest.data")
    _validate_repo(data, "manifest.data", has_tag=True)
    split = _require_mapping(manifest["split"], "manifest.split")
    _require_keys(split, _SPLIT_KEYS, "manifest.split")
    _require_nonempty_string(split["path"], "manifest.split.path")
    sha256 = split["sha256"]
    if (
        type(sha256) is not str
        or _SHA256_PATTERN.fullmatch(sha256) is None
    ):
        raise RunManifestError(
            f"invalid run manifest: manifest.split.sha256 must be a "
            f"64-character hexadecimal SHA-256, got {sha256!r}"
        )
    parameters = _require_mapping(manifest["parameters"], "manifest.parameters")
    _require_keys(parameters, _PARAMETER_KEYS, "manifest.parameters")
    _validate_positive_finite(parameters["C"])
    if parameters["target_split"] != _TARGET_SPLIT:
        raise RunManifestError(
            f"invalid run manifest: parameters.target_split must be "
            f"{_TARGET_SPLIT!r}, got {parameters['target_split']!r}"
        )


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
    return {
        "root": str(root),
        "commit": commit,
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
    data_dir: Path,
    data_tag: str,
    split_file: Path,
    C: float,
) -> dict:
    """Collect all recorded observations into one prepared run passport."""
    _validate_positive_finite(C)
    data_root = _repository_root(data_dir)
    data_snapshot = _git_snapshot(data_root)
    data = {
        "root": data_snapshot["root"],
        "tag": data_tag,
        "commit": data_snapshot["commit"],
    }
    manifest = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "stage": _STAGE,
        "data": data,
        "split": {
            "path": str(split_file.resolve()),
            "sha256": _split_sha256(split_file),
        },
        "parameters": {"C": C, "target_split": _TARGET_SPLIT},
    }
    _validate_manifest(manifest)
    return manifest


def load_run_manifest(path: Path) -> dict:
    """Load and validate a run passport without modifying it."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"cannot load run passport {path}: {exc}") from exc
    try:
        _validate_manifest(payload)
    except RunManifestError as exc:
        raise RunManifestError(f"invalid run passport {path}: {exc}") from exc
    return payload


def write_run_manifest(manifest: dict, output_path: Path) -> None:
    """Publish the passport text to ``output_path`` without overwriting.

    The destination must not exist and its parent directory must already
    exist; nothing else is created.  The text is written to a temporary
    sibling file that is then hard-linked into place.  All observations
    were collected by :func:`collect_run_manifest` before publication, so
    the passport never appears in the recorded state.
    """
    _validate_manifest(manifest)
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
