"""Load an existing T1 run and verify its recorded data identities.

This small shared preparation API is used by a later consuming command.  It
loads the version-4 run passport of an existing T1 run and compares the
recorded data tag, data commit, and split fingerprint against the currently
checked-out data repository.  It never trains, predicts, evaluates, or
modifies any input, and it never observes the source-code repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_context import _preflight_benchmark
from core.data import DataError
from core.instances import CorpusTagError
from t1_features.run_manifest import (
    RunManifestError,
    collect_run_manifest,
    load_run_manifest,
)


class T1RunContextError(ValueError):
    """Expected failure while loading or verifying an existing T1 run."""


@dataclass(frozen=True)
class T1RunContext:
    run_dir: Path
    manifest: dict


def load_t1_run_context(run_dir: Path) -> T1RunContext:
    """Load and verify the data identities recorded in an existing T1 run.

    ``run_dir`` may start with ``~`` or be relative to the caller's current
    directory.  Only ``<run_dir>/run.json`` is read; no model artifacts,
    dev split, metrics, configuration, or defaults are required.
    """
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise T1RunContextError(
            f"T1 run directory does not exist or is not a directory: "
            f"{resolved_run_dir}"
        )
    run_passport = resolved_run_dir / "run.json"
    try:
        manifest = load_run_manifest(run_passport)
    except RunManifestError as exc:
        raise T1RunContextError(
            f"cannot load T1 run passport {run_passport}: {exc}"
        ) from exc

    data_root = Path(manifest["data"]["root"])
    try:
        data_dir, tag = _preflight_benchmark(data_root)
    except (DataError, CorpusTagError) as exc:
        raise T1RunContextError(
            f"cannot verify the recorded data checkout for run "
            f"{resolved_run_dir}: {exc}"
        ) from exc

    try:
        current = collect_run_manifest(
            data_dir=data_dir,
            data_tag=tag,
            split_file=Path(manifest["split"]["path"]),
            C=manifest["parameters"]["C"],
        )
    except RunManifestError as exc:
        raise T1RunContextError(
            f"cannot collect current experiment identities for run "
            f"{resolved_run_dir}: {exc}"
        ) from exc

    if current["data"]["tag"] != manifest["data"]["tag"]:
        raise T1RunContextError(
            f"T1 run {resolved_run_dir}: recorded data tag "
            f"{manifest['data']['tag']!r} does not match actual data tag "
            f"{current['data']['tag']!r}"
        )
    if current["data"]["commit"] != manifest["data"]["commit"]:
        raise T1RunContextError(
            f"T1 run {resolved_run_dir}: recorded data commit "
            f"{manifest['data']['commit']!r} does not match actual data "
            f"commit {current['data']['commit']!r}"
        )
    if current["split"]["sha256"] != manifest["split"]["sha256"]:
        raise T1RunContextError(
            f"T1 run {resolved_run_dir}: recorded split SHA-256 "
            f"{manifest['split']['sha256']!r} does not match actual split "
            f"SHA-256 {current['split']['sha256']!r}"
        )

    return T1RunContext(
        run_dir=resolved_run_dir,
        manifest=manifest,
    )