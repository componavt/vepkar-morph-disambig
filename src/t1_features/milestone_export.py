"""Export one completed T1.1 dev result into a compact milestone directory.

The exporter reads only ``<run_dir>/metrics.json`` and ``<run_dir>/dev.csv``
from an already evaluated run.  It copies the raw metrics bytes unchanged,
streams the dev predictions into a level-22 zstandard frame, and renders one
short fixed English README.  All three files are staged in an owned temporary
sibling directory before an exclusively created destination directory is
published.  This module performs no evaluation, no integrity verification,
and never modifies the source run.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import zstandard

_FORMAT_NAME = "vepkar-metrics"
_FORMAT_VERSION = 2
_METHOD = "t1.1"
_TARGET_SPLIT = "dev"
_COMPRESSION_LEVEL = 22
_CHUNK_BYTES = 65536
_TEMP_PREFIX = ".vepkar-t1-milestone-"
_TEMP_SUFFIX = ".tmp"
_OUTPUT_NAMES = ("README.md", "metrics.json", "dev.csv.zst")
_README_LINK = "../../../../README.md#t1-feature-based-model"

_README_TEMPLATE = """# T1.1: feature-based candidate ranking

Binary logistic regression with morphological, suffix, and
immediate-context features, without train-frequency features.

- Data tag: {tag}
- C: {C}
- Evaluation split: dev

This is a development-set result, not a final test evaluation.

| Metric | Value |
| --- | ---: |
| Word occurrences | {word_count} |
| Candidate rows | {candidate_count} |
| Top-1 accuracy | {top1_accuracy} |
| MRR | {mrr} |
| Top-3 accuracy | {top3_accuracy} |

- `metrics.json` — evaluation metrics and experiment settings in the
  vepkar-metrics JSON format.
- `dev.csv.zst` — the run's dev predictions, compressed with zstandard.

The trained model is not included in this directory.

See the [T1 feature-based model instructions]({link}) for the full
experimental workflow.
"""


class T1MilestoneExportError(ValueError):
    """A T1 milestone could not be inspected, staged, or published."""


def _extract_fields(metrics, path: Path) -> dict:
    if not isinstance(metrics, dict):
        raise T1MilestoneExportError(
            f"malformed T1 metrics in {path}: expected a JSON object, "
            f"got {type(metrics).__name__}"
        )
    if metrics.get("format") != _FORMAT_NAME:
        raise T1MilestoneExportError(
            f"unsupported T1 metrics in {path}: expected format "
            f"{_FORMAT_NAME!r}, got {metrics.get('format')!r}"
        )
    version = metrics.get("format_version")
    if type(version) is not int or version != _FORMAT_VERSION:
        raise T1MilestoneExportError(
            f"unsupported T1 metrics in {path}: expected format_version "
            f"{_FORMAT_VERSION} (integer), got {version!r}"
        )
    if metrics.get("method") != _METHOD:
        raise T1MilestoneExportError(
            f"unsupported T1 metrics in {path}: expected method "
            f"{_METHOD!r}, got {metrics.get('method')!r}"
        )
    if metrics.get("split") != _TARGET_SPLIT:
        raise T1MilestoneExportError(
            f"unsupported T1 metrics in {path}: expected split "
            f"{_TARGET_SPLIT!r}, got {metrics.get('split')!r}"
        )
    data = metrics.get("data")
    if not isinstance(data, dict):
        raise T1MilestoneExportError(
            f"unusable T1 metrics in {path}: data is not an object"
        )
    tag = data.get("tag")
    if not isinstance(tag, str) or not tag:
        raise T1MilestoneExportError(
            f"unusable data tag in {path}: expected a nonempty string, "
            f"got {tag!r}"
        )
    if tag in (".", "..") or "/" in tag or "\\" in tag or "\x00" in tag:
        raise T1MilestoneExportError(
            f"unsafe data tag in {path}: {tag!r}"
        )
    parameters = metrics.get("parameters")
    if not isinstance(parameters, dict):
        raise T1MilestoneExportError(
            f"unusable T1 metrics in {path}: parameters is not an object"
        )
    sub = metrics.get("metrics")
    if not isinstance(sub, dict):
        raise T1MilestoneExportError(
            f"unusable T1 metrics in {path}: metrics is not an object"
        )
    word_count = metrics.get("word_count")
    if type(word_count) is not int:
        raise T1MilestoneExportError(
            f"unusable word_count in {path}: expected an integer, "
            f"got {word_count!r}"
        )
    candidate_count = metrics.get("candidate_count")
    if type(candidate_count) is not int:
        raise T1MilestoneExportError(
            f"unusable candidate_count in {path}: expected an integer, "
            f"got {candidate_count!r}"
        )
    C = parameters.get("C")
    if type(C) is bool or not isinstance(C, (int, float)):
        raise T1MilestoneExportError(
            f"unusable parameters.C in {path}: expected a number, got {C!r}"
        )
    top1 = sub.get("top1_accuracy")
    if type(top1) is bool or not isinstance(top1, (int, float)):
        raise T1MilestoneExportError(
            f"unusable metrics.top1_accuracy in {path}: expected a number, "
            f"got {top1!r}"
        )
    mrr = sub.get("mrr")
    if type(mrr) is bool or not isinstance(mrr, (int, float)):
        raise T1MilestoneExportError(
            f"unusable metrics.mrr in {path}: expected a number, got {mrr!r}"
        )
    top3 = sub.get("top3_accuracy")
    if type(top3) is bool or not isinstance(top3, (int, float)):
        raise T1MilestoneExportError(
            f"unusable metrics.top3_accuracy in {path}: expected a number, "
            f"got {top3!r}"
        )
    return {
        "tag": tag,
        "C": C,
        "word_count": word_count,
        "candidate_count": candidate_count,
        "top1_accuracy": top1,
        "mrr": mrr,
        "top3_accuracy": top3,
    }


def _render_readme(fields: dict) -> str:
    return _README_TEMPLATE.format(
        tag=fields["tag"],
        C=str(fields["C"]),
        word_count=fields["word_count"],
        candidate_count=fields["candidate_count"],
        top1_accuracy=f"{fields['top1_accuracy']:.4f}",
        mrr=f"{fields['mrr']:.4f}",
        top3_accuracy=f"{fields['top3_accuracy']:.4f}",
        link=_README_LINK,
    )


def _copy_metrics(metrics_bytes: bytes, output: Path, source: Path) -> None:
    try:
        output.write_bytes(metrics_bytes)
    except OSError as exc:
        raise T1MilestoneExportError(
            f"cannot copy T1 metrics {source} to milestone staging "
            f"{output}: {exc}"
        ) from exc


def _compress_dev_csv(source: Path, output: Path) -> None:
    try:
        compressor = zstandard.ZstdCompressor(level=_COMPRESSION_LEVEL)
        cobj = compressor.compressobj()
        with source.open("rb") as src, output.open("wb") as dst:
            while True:
                block = src.read(_CHUNK_BYTES)
                if not block:
                    break
                dst.write(cobj.compress(block))
            dst.write(cobj.flush())
    except (OSError, zstandard.ZstdError) as exc:
        raise T1MilestoneExportError(
            f"cannot compress T1 dev predictions {source} into milestone "
            f"{output}: {exc}"
        ) from exc


def _write_readme(fields: dict, output: Path) -> None:
    try:
        with output.open("w", encoding="utf-8") as stream:
            stream.write(_render_readme(fields))
    except OSError as exc:
        raise T1MilestoneExportError(
            f"cannot write milestone README {output}: {exc}"
        ) from exc


def _cleanup_notes(
    destination: Path,
    staging: Path | None,
    created_destination: bool,
    moved: list[str],
) -> list[str]:
    notes: list[str] = []
    if staging is not None:
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            notes.append(
                f"failed to remove temporary staging directory {staging}: {exc}"
            )
    if created_destination:
        for name in moved:
            path = destination / name
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                notes.append(
                    f"failed to remove milestone file {path}: {exc}"
                )
        try:
            destination.rmdir()
        except OSError as exc:
            notes.append(
                f"failed to remove milestone directory {destination}: {exc}"
            )
    return notes


def export_t1_milestone(
    run_dir: Path,
    *,
    milestones_dir: Path,
) -> Path:
    """Export one finished T1.1 dev result and return its destination.

    ``run_dir`` must contain ``metrics.json`` (a version-2 vepkar-metrics
    payload for method ``t1.1`` on split ``dev``) and ``dev.csv``.  The
    destination ``milestones_dir / <data tag> / <run directory basename>``
    is created exclusively and must not already exist; it receives exactly
    ``README.md``, ``metrics.json``, and ``dev.csv.zst``.
    """
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise T1MilestoneExportError(
            f"T1 run directory not found: {resolved}"
        )
    metrics_path = resolved / "metrics.json"
    predictions_path = resolved / "dev.csv"
    if not metrics_path.exists():
        raise T1MilestoneExportError(
            f"T1 run {resolved} is missing metrics.json: {metrics_path}"
        )
    if not predictions_path.exists():
        raise T1MilestoneExportError(
            f"T1 run {resolved} is missing dev.csv: {predictions_path}"
        )
    try:
        metrics_bytes = metrics_path.read_bytes()
    except OSError as exc:
        raise T1MilestoneExportError(
            f"cannot read T1 metrics {metrics_path}: {exc}"
        ) from exc
    try:
        text = metrics_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise T1MilestoneExportError(
            f"cannot decode T1 metrics {metrics_path} as UTF-8: {exc}"
        ) from exc
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise T1MilestoneExportError(
            f"cannot parse T1 metrics JSON {metrics_path}: {exc}"
        ) from exc
    fields = _extract_fields(parsed, metrics_path)
    destination = Path(milestones_dir) / fields["tag"] / resolved.name
    if destination.is_symlink() or destination.exists():
        raise T1MilestoneExportError(
            f"milestone destination already exists: {destination}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise T1MilestoneExportError(
            f"cannot create milestone parent directory "
            f"{destination.parent}: {exc}"
        ) from exc
    staging = None
    created_destination = False
    moved: list[str] = []
    try:
        try:
            staging = Path(
                tempfile.mkdtemp(
                    prefix=_TEMP_PREFIX,
                    suffix=_TEMP_SUFFIX,
                    dir=str(destination.parent),
                )
            )
        except OSError as exc:
            raise T1MilestoneExportError(
                f"cannot create temporary staging directory under "
                f"{destination.parent}: {exc}"
            ) from exc
        _copy_metrics(metrics_bytes, staging / "metrics.json", metrics_path)
        _compress_dev_csv(predictions_path, staging / "dev.csv.zst")
        _write_readme(fields, staging / "README.md")
        try:
            os.mkdir(destination)
        except FileExistsError as exc:
            raise T1MilestoneExportError(
                f"milestone destination already exists: {destination}"
            ) from exc
        except OSError as exc:
            raise T1MilestoneExportError(
                f"cannot create milestone destination directory "
                f"{destination}: {exc}"
            ) from exc
        created_destination = True
        for name in _OUTPUT_NAMES:
            source = staging / name
            target = destination / name
            try:
                os.link(source, target)
            except OSError as exc:
                raise T1MilestoneExportError(
                    f"cannot publish milestone file {target}: {exc}"
                ) from exc
            moved.append(name)
            try:
                source.unlink()
            except OSError as exc:
                raise T1MilestoneExportError(
                    f"cannot remove staged milestone file {source} after "
                    f"publishing {target}: {exc}"
                ) from exc
    except BaseException as exc:
        for note in _cleanup_notes(
            destination, staging, created_destination, moved
        ):
            exc.add_note(note)
        raise
    try:
        staging.rmdir()
    except OSError as exc:
        error = T1MilestoneExportError(
            f"milestone published at {destination} but temporary staging "
            f"removal failed: {staging}"
        )
        error.add_note(str(exc))
        raise error from exc
    return destination