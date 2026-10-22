"""Train the T1 shared model on the real train split and save one model bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from core.data import DataError
from core.predictions import BenchmarkIntegrityError
from core.splits import SplitError
from t1_features.data import load_t1_inputs
from t1_features.features import FeatureConstructionError, build_candidate_examples
from t1_features.model import train_candidate_model
from t1_features.model_io import (
    ModelBundleError,
    ModelBundlePublishedCleanupError,
    save_model_bundle,
)
from t1_features.run_context import T1RunContextError, load_t1_run_context


def _run_train_t1(args: argparse.Namespace) -> int:
    try:
        context = load_t1_run_context(args.run_dir)
    except T1RunContextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    manifest = context.manifest
    output_path = context.run_dir / "model.bundle"
    if output_path.is_dir():
        print("error: model destination is a directory:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    if output_path.exists() or output_path.is_symlink():
        print("error: model already exists:", file=sys.stderr)
        print(f"  {output_path}", file=sys.stderr)
        return 1
    data_dir = Path(manifest["data"]["root"])
    tag = manifest["data"]["tag"]
    split_file = Path(manifest["split"]["path"])
    C = manifest["parameters"]["C"]
    try:
        inputs = load_t1_inputs(
            data_dir,
            tag,
            split="train",
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
    try:
        examples = build_candidate_examples(
            inputs.instances,
            inputs.context_words,
            mode="train",
        )
    except FeatureConstructionError as exc:
        print("error: T1 feature construction failed:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        model = train_candidate_model(examples, C=C)
    except ValueError as exc:
        print("error: T1 training failed:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    except ConvergenceWarning as exc:
        print("error: T1 training did not converge:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1
    try:
        save_model_bundle(model, output_path)
    except ModelBundlePublishedCleanupError as exc:
        print(
            "error: T1 model bundle was published, but temporary cleanup failed:",
            file=sys.stderr,
        )
        print(f"  model: {exc.path}", file=sys.stderr)
        print(f"  temporary: {exc.temp_path}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"  {exc.__cause__}", file=sys.stderr)
        return 1
    except ModelBundleError as exc:
        print("error: cannot save T1 model bundle:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(f"  {note}", file=sys.stderr)
        return 1
    print("T1 training: OK")
    print(f"Run directory: {context.run_dir}")
    print(f"Data tag: {tag}")
    print(f"Split file: {inputs.benchmark.split_path}")
    print("Split: train")
    print(f"Training word occurrences: {len(inputs.instances)}")
    print(f"Training candidate examples: {len(examples)}")
    print(f"Feature columns: {len(model.vectorizer.vocabulary_)}")
    print(f"C: {C}")
    print(f"Model: {output_path}")
    return 0
