#!/usr/bin/env python3
"""Build checked A1 diagnostic tables from the frozen T1 dev predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_context import determine_data_tag  # noqa: E402
from commands.evaluate_predictions import load_ranked_predictions  # noqa: E402
from core.data import SUPPORTED_LANGUAGES  # noqa: E402
from core.predictions import validate_predictions  # noqa: E402
from t1_features.data import load_t1_inputs  # noqa: E402
from t1_features.error_analysis import (  # noqa: E402
    FROZEN_PREDICTIONS_SHA256,
    T1ErrorAnalysisError,
    build_t1_error_tables,
    resolve_recorded_prediction_path,
    validate_t1_error_table_invariants,
    validate_frozen_prediction_hash,
)

DEFAULT_EXPERIMENT = ROOT / "work/experiments/v2026.11/context-ablation-k5-v1"
DEFAULT_RUN_DIR = DEFAULT_EXPERIMENT / "runs/both/c-7"
DEFAULT_OUTPUT = ROOT / "work/experiments/v2026.11/t1-dev-error-analysis-a1-v3"

CSV_FIELDS = {
    "instances": (
        "language", "text_id", "sentence_id", "word_id", "surface_form", "normalized_form",
        "normalization", "target_position", "candidate_count", "distinct_gramset_count",
        "gold_wordform_id", "gold_gramset", "gold_rank", "gold_score", "top1_wordform_id",
        "top1_gramset", "top1_rank", "top1_score", "top1_minus_gold_score",
        "same_wordform_id_as_gold", "same_gramset_as_gold", "is_correct", "error_category",
        "sentence", "left_neighbor", "right_neighbor",
    ),
    "candidates": (
        "language", "text_id", "sentence_id", "word_id", "surface_form", "wordform_id",
        "gramset", "rank", "score", "is_gold", "is_top1",
    ),
    "conflict_opportunities": (
        "language", "text_id", "sentence_id", "word_id", "gold_gramset",
        "alternative_gramset", "alternative_candidate_count", "selected_error",
    ),
    "conflict_summary": (
        "gold_gramset", "alternative_gramset", "opportunity_count", "selected_error_count",
        "selected_error_rate",
    ),
    "same_gramset_alternatives": (
        "language", "text_id", "sentence_id", "word_id", "gold_wordform_id", "gold_gramset",
        "alternative_wordform_id", "alternative_gramset", "is_top1", "selected_error",
    ),
    "confusions": ("gold_gramset", "top1_gramset", "confusion_count"),
    "sentence_summary": (
        "language", "text_id", "sentence_id", "benchmark_instance_count", "error_count",
        "error_rate", "distinct_directed_confusion_count", "sentence",
    ),
    "sentence_instance_links": (
        "language", "text_id", "sentence_id", "word_id", "word", "target_position",
        "gold_gramset", "top1_gramset", "gold_rank", "is_correct", "error_category",
    ),
    "sentence_error_links": (
        "language", "text_id", "sentence_id", "word_id", "word", "target_position",
        "gold_gramset", "top1_gramset", "gold_rank", "is_correct", "error_category",
    ),
    "form_summary": (
        "language", "normalized_form", "normalization", "instance_count", "error_count",
        "text_count", "sentence_count", "distinct_gold_gramset_count", "distinct_directed_error_count",
    ),
    "form_instance_links": (
        "language", "normalized_form", "text_id", "sentence_id", "word_id", "gold_gramset",
        "top1_gramset", "gold_rank", "is_correct", "error_category",
    ),
}

OUTPUT_NAMES = {
    "instances": "instances.csv",
    "candidates": "candidates.csv",
    "conflict_opportunities": "conflict-opportunities.csv",
    "conflict_summary": "conflict-summary.csv",
    "same_gramset_alternatives": "same-gramset-alternatives.csv",
    "confusions": "confusions.csv",
    "sentence_summary": "sentence-summary.csv",
    "sentence_instance_links": "sentence-instance-links.csv",
    "sentence_error_links": "sentence-error-links.csv",
    "form_summary": "form-summary.csv",
    "form_instance_links": "form-instance-links.csv",
}


class A1InputError(ValueError):
    """A1's frozen inputs or recorded provenance do not pass preflight."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(args: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        return ""
    return result.stdout.strip()


def _git_file_sha(commit: str, relative_path: str) -> str | None:
    if not commit:
        return None
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _features_hash_in_history(target: str) -> bool:
    commits = _git_text(["log", "--all", "--format=%H", "--", "src/t1_features/features.py"])
    for commit in commits.splitlines():
        digest = _git_file_sha(commit, "src/t1_features/features.py")
        if digest == target:
            return True
    return False


def _load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise A1InputError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A1InputError(f"metadata must contain a JSON object: {path}")
    return value


def _verify_run_metadata(
    run: dict, experiment: dict, run_dir: Path, experiment_dir: Path, prediction_path: Path
) -> None:
    parameters = run.get("parameters", {})
    required = {
        "variant": "both",
        "C": 7.0,
        "target_split": "dev",
        "suffix_max_length": 5,
        "include_affix": False,
        "include_left_context": True,
        "include_right_context": True,
    }
    observed = {"variant": run.get("variant"), **parameters}
    for key, expected in required.items():
        if observed.get(key) != expected:
            raise A1InputError(
                f"run.json configuration mismatch for {key}: {observed.get(key)!r}; expected {expected!r}"
            )
    run_prediction_path = run.get("predictions")
    if not isinstance(run_prediction_path, str) or not run_prediction_path.strip():
        raise A1InputError("run.json is missing its predictions path")
    recorded_prediction = (run_dir / run_prediction_path).resolve()
    if recorded_prediction != prediction_path.resolve():
        raise A1InputError(
            f"run.json points to {recorded_prediction}, not selected predictions {prediction_path}"
        )

    protocol = experiment.get("protocol", {})
    if protocol.get("data_tag") != "v2026.11" or protocol.get("split") != "dev":
        raise A1InputError("experiment summary does not identify v2026.11 dev")
    if protocol.get("suffix_max_length") != 5 or protocol.get("include_affix") is not False:
        raise A1InputError("experiment summary has a different suffix/affix configuration")
    if protocol.get("context_positions") != [-1, 1]:
        raise A1InputError("experiment summary does not identify the ±1 context")
    if protocol.get("test_used") is not False:
        raise A1InputError("experiment summary does not confirm test_used=false")
    variant = protocol.get("variants", {}).get("both", {})
    if variant.get("include_left_context") is not True or variant.get("include_right_context") is not True:
        raise A1InputError("experiment summary does not identify the both-context variant")
    selected_c = experiment.get("selected_C", {}).get("both", {}).get("C")
    if selected_c != 7.0:
        raise A1InputError(f"experiment summary selected C={selected_c!r}, expected C=7")
    try:
        recorded_path = resolve_recorded_prediction_path(
            experiment, experiment_dir, variant="both", c_value="7.0"
        )
    except T1ErrorAnalysisError as exc:
        raise A1InputError(str(exc)) from exc
    if recorded_path != prediction_path.resolve():
        raise A1InputError(
            "experiment summary prediction path does not match selected dev.csv: "
            f"{recorded_path} != {prediction_path.resolve()}"
        )


def _benchmark_input_hashes(data_dir: Path) -> dict[str, str]:
    hashes = {}
    for language in SUPPORTED_LANGUAGES:
        for stem in ("texts", "sentences", "words", "candidate_analyses"):
            path = data_dir / "corpus" / f"{stem}_{language}.csv.zst"
            if not path.is_file():
                raise A1InputError(f"benchmark input file is missing: {path}")
            hashes[str(path)] = _sha256(path)
    return hashes


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _build_provenance(
    *, data_dir: Path, run_dir: Path, run_path: Path, summary_path: Path,
    prediction_path: Path, run: dict, experiment: dict,
) -> dict:
    recorded_sources = experiment.get("source_sha256", {})
    source_status = {}
    for relative, recorded_hash in sorted(recorded_sources.items()):
        path = ROOT / relative
        available_hash = _sha256(path) if path.is_file() else None
        source_status[relative] = {
            "sha256_recorded": recorded_hash,
            "sha256_current": available_hash,
            "matches_recorded": available_hash == recorded_hash,
        }

    features_path = ROOT / "src/t1_features/features.py"
    recorded_features_hash = recorded_sources.get("src/t1_features/features.py")
    current_head = _git_text(["rev-parse", "HEAD"])
    recorded_head = experiment.get("repository_head")
    bundle_path = run_dir / "model.bundle"
    try:
        data_commit = _git_text(["-C", str(data_dir), "rev-parse", "HEAD"])
    except Exception:
        data_commit = ""
    return {
        "predictions_frozen": True,
        "training_reproducibility_confirmed": False,
        "predictions_sha256": _sha256(prediction_path),
        "run_json_path": str(run_path),
        "run_json_sha256": _sha256(run_path),
        "experiment_summary_path": str(summary_path),
        "experiment_summary_sha256": _sha256(summary_path),
        "model_bundle_path_expected": str(bundle_path),
        "model_bundle_exists": bundle_path.is_file(),
        "features_source": {
            "sha256_recorded": recorded_features_hash,
            "sha256_available_worktree": _sha256(features_path) if features_path.is_file() else None,
            "sha256_at_recorded_repository_head": _git_file_sha(
                str(recorded_head), "src/t1_features/features.py"
            ),
            "recorded_hash_found_in_available_git_history": (
                _features_hash_in_history(str(recorded_features_hash))
                if recorded_features_hash else False
            ),
        },
        "recorded_source_hash_checks": source_status,
        "repository_head_at_a1": current_head,
        "repository_head_recorded_by_experiment": recorded_head,
        "source_data_tag": determine_data_tag(data_dir),
        "source_data_commit_current": data_commit,
        "source_data_commit_recorded": experiment.get("data", {}).get("commit"),
        "configuration_as_recorded": {
            "run_json": run.get("parameters"),
            "variant": run.get("variant"),
            "experiment_selected_C": experiment.get("selected_C", {}).get("both", {}).get("C"),
            "test_used_in_original_experiment": experiment.get("protocol", {}).get("test_used"),
        },
        "test_predictions_read": False,
        "test_metrics_computed": False,
        "test_analyzed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dictorpus-data")
    parser.add_argument("--tag", default="v2026.11")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    run_dir = args.run_dir.resolve()
    experiment_dir = args.experiment_dir.resolve()
    run_path = run_dir / "run.json"
    summary_path = experiment_dir / "summary.json"
    prediction_path = run_dir / "dev.csv"
    if not prediction_path.is_file():
        parser.error(f"frozen dev predictions do not exist: {prediction_path}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"refusing to overwrite existing A1 output: {args.output_dir}")
    actual_prediction_hash = _sha256(prediction_path)
    try:
        validate_frozen_prediction_hash(actual_prediction_hash)
    except T1ErrorAnalysisError as exc:
        parser.error(str(exc))
    run = _load_json(run_path)
    experiment = _load_json(summary_path)
    try:
        _verify_run_metadata(run, experiment, run_dir, experiment_dir, prediction_path)
        if args.tag != "v2026.11" or determine_data_tag(data_dir) != args.tag:
            raise A1InputError("selected dictorpus-data checkout is not v2026.11")
        expected_data_commit = experiment.get("data", {}).get("commit")
        actual_data_commit = _git_text(["-C", str(data_dir), "rev-parse", "HEAD"])
        if actual_data_commit != expected_data_commit:
            raise A1InputError(
                f"dictorpus-data commit mismatch: {actual_data_commit!r} != {expected_data_commit!r}"
            )

        inputs = load_t1_inputs(data_dir, args.tag, split="dev")
        loaded_predictions = tuple(load_ranked_predictions(prediction_path))
        validation = validate_predictions(
            prediction_path,
            "dev",
            inputs.benchmark.instances,
            inputs.benchmark.split_rows,
        )
        if not validation.is_valid:
            raise A1InputError(
                f"frozen predictions failed project validation: {validation.errors!r}"
            )
        if validation.expected_word_count != 2613 or validation.expected_candidate_count != 6519:
            raise A1InputError(
                "dev benchmark control totals differ from the prior independent check: "
                f"{validation.expected_word_count} usages, {validation.expected_candidate_count} candidates"
            )
        if validation.predicted_word_count != run.get("predicted_word_count"):
            raise A1InputError("run.json word count disagrees with validated predictions")
        if validation.predicted_candidate_count != run.get("predicted_candidate_count"):
            raise A1InputError("run.json candidate count disagrees with validated predictions")

        tables = build_t1_error_tables(inputs.instances, loaded_predictions, inputs.context_words)
        counts = validate_t1_error_table_invariants(tables)
        if counts["instances"] != 2613 or counts["candidates"] != 6519:
            raise A1InputError(f"A1 table counts disagree with controls: {counts!r}")
        if counts["correct_top1"] != 2255 or counts["errors"] != 358:
            raise A1InputError(f"Top-1 controls disagree with the prior check: {counts!r}")
        if counts["different_gramset_errors"] != 347 or counts["same_gramset_errors"] != 11:
            raise A1InputError(f"gramset error controls disagree with the prior check: {counts!r}")
    except (A1InputError, T1ErrorAnalysisError, ValueError) as exc:
        parser.error(str(exc))

    input_hashes = _benchmark_input_hashes(data_dir)
    provenance = _build_provenance(
        data_dir=data_dir,
        run_dir=run_dir,
        run_path=run_path,
        summary_path=summary_path,
        prediction_path=prediction_path,
        run=run,
        experiment=experiment,
    )
    split_path = Path(inputs.benchmark.split_path)
    if _sha256(split_path) != experiment.get("split", {}).get("sha256"):
        parser.error("split checksum differs from the original experiment metadata")
    input_hashes[str(split_path)] = _sha256(split_path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_checksums = {}
    for table_name, filename in OUTPUT_NAMES.items():
        path = args.output_dir / filename
        _write_csv(path, CSV_FIELDS[table_name], tables[table_name])
        output_checksums[filename] = _sha256(path)

    error_categories = {}
    for category in (
        "same_wordform_id_different_gramset",
        "different_wordform_id_different_gramset",
        "different_wordform_id_same_gramset",
    ):
        error_categories[category] = sum(
            row["error_category"] == category for row in tables["instances"]
        )
    if sum(error_categories.values()) != counts["errors"]:
        parser.error(f"error identity categories do not sum to all errors: {error_categories!r}")

    metadata = {
        "study": "T1 error analysis A1",
        "source_data_tag": args.tag,
        "target_split": "dev",
        "configuration_as_recorded": provenance["configuration_as_recorded"],
        "prediction_source": {
            "path": str(prediction_path),
            "sha256": provenance["predictions_sha256"],
            "expected_sha256": FROZEN_PREDICTIONS_SHA256,
            "frozen": True,
        },
        "run_json": {
            "path": str(run_path),
            "sha256": provenance["run_json_sha256"],
        },
        "experiment_summary": {
            "path": str(summary_path),
            "sha256": provenance["experiment_summary_sha256"],
        },
        "split_file": {"path": str(split_path), "sha256": _sha256(split_path)},
        "benchmark_source_files_sha256": input_hashes,
        "provenance_limitations": provenance,
        "prediction_validation": {
            "is_valid": validation.is_valid,
            "expected_word_count": validation.expected_word_count,
            "expected_candidate_count": validation.expected_candidate_count,
            "predicted_word_count": validation.predicted_word_count,
            "predicted_candidate_count": validation.predicted_candidate_count,
            "score_values_finite": True,
            "candidate_sets_exact": True,
            "saved_ranks_match_score_and_tie_breaks": True,
            "input_rank_order_preserved": True,
        },
        "counts": counts,
        "error_identity_breakdown": error_categories,
        "conflict_direction": "gold_gramset G -> alternative_gramset H",
        "same_gramset_alternatives_are_separate": True,
        "form_grouping": "language + normalized corpus word field; no additional normalization",
        "tables": {name: len(rows) for name, rows in tables.items()},
        "output_sha256": output_checksums,
        "test_analyzed": False,
        "model_retrained": False,
        "new_predictions_generated": False,
        "gold_or_candidates_modified": False,
        "model_bundle_loaded": False,
        "model_features_or_predictions_recomputed": False,
        "analysis_code_sha256": {
            "script": _sha256(Path(__file__)),
            "module": _sha256(ROOT / "src/t1_features/error_analysis.py"),
        },
        "repository_head_at_analysis": provenance["repository_head_at_a1"],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "output_dir": str(args.output_dir),
        "counts": counts,
        "error_identity_breakdown": error_categories,
        "prediction_validation": metadata["prediction_validation"],
        "provenance_limitations": {
            "training_reproducibility_confirmed": provenance["training_reproducibility_confirmed"],
            "model_bundle_exists": provenance["model_bundle_exists"],
            "features_source": provenance["features_source"],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
