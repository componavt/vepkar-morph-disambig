"""Predict dev candidate rankings with a saved T1 model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.exceptions import NotFittedError

from commands.prediction_reports import _print_validation_failure
from core.data import DataError
from core.predictions import BenchmarkIntegrityError
from core.splits import SplitError
from t1_features.data import load_t1_inputs
from t1_features.features import FeatureConstructionError, build_candidate_examples
from t1_features.model import score_candidate_examples
from t1_features.model_io import ModelBundleError, load_model_bundle
from t1_features.prediction_io import (
    T1PredictionPublishedCleanupError,
    T1PredictionWriteError,
    write_dev_predictions,
)
from t1_features.ranking import rank_candidate_scores
from t1_features.run_context import T1RunContextError, load_t1_run_context


def _run_predict_t1(args: argparse.Namespace) -> int:
    try:
        context = load_t1_run_context(args.run_dir)
    except T1RunContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output_path = context.run_dir / "dev.csv"
    if output_path.is_dir():
        print("error: output path is a directory:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if output_path.exists() or output_path.is_symlink():
        print("error: output file already exists:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    model_path = context.run_dir / "model.bundle"
    try:
        model = load_model_bundle(model_path)
    except ModelBundleError as exc:
        print("error: cannot load T1 model bundle:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    data_dir = Path(context.manifest["data"]["root"])
    tag = context.manifest["data"]["tag"]
    split_file = Path(context.manifest["split"]["path"])
    try:
        inputs = load_t1_inputs(
            data_dir,
            tag,
            split="dev",
            split_file_override=split_file,
        )
    except DataError:
        return 1
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BenchmarkIntegrityError as exc:
        print("error: strict benchmark is internally inconsistent:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    if not inputs.instances:
        print("error: no dev word occurrences selected for prediction", file=sys.stderr)
        return 1
    try:
        examples = build_candidate_examples(
            inputs.instances,
            inputs.context_words,
            mode="predict",
        )
    except FeatureConstructionError as exc:
        print("error: T1 feature construction failed:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        scores = score_candidate_examples(model, examples)
    except NotFittedError as exc:
        print("error: T1 model is not fitted:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print("error: T1 scoring failed:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        rows = rank_candidate_scores(examples, scores)
    except BenchmarkIntegrityError as exc:
        print("error: strict benchmark is internally inconsistent:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print("error: T1 ranking failed:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        validation = write_dev_predictions(
            rows,
            output_path,
            benchmark=inputs.benchmark,
        )
    except T1PredictionPublishedCleanupError as exc:
        print(
            "error: T1 dev predictions were published, but temporary cleanup failed:",
            file=sys.stderr,
        )
        print(f"  output: {exc.output_path}", file=sys.stderr)
        print(f"  temporary: {exc.temp_path}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except T1PredictionWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        if exc.validation is not None:
            _print_validation_failure(exc.validation)
        return 1
    print("T1 prediction: OK")
    print(f"Run directory: {context.run_dir}")
    print(f"Model: {model_path}")
    print(f"Data tag: {tag}")
    print(f"Split file: {inputs.benchmark.split_path}")
    print("Split: dev")
    print(f"Predicted word occurrences: {validation.predicted_word_count}")
    print(f"Predicted candidate rows: {validation.predicted_candidate_count}")
    print(f"Output: {output_path}")
    print("Validation: OK")
    return 0
