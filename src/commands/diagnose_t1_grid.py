"""Build the approved word-level T1 C-grid diagnostic table on dev."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from benchmark_context import _preflight_benchmark, load_benchmark_context
from commands.evaluate_predictions import load_ranked_predictions
from core.data import DataError, read_corpus_tables
from core.instances import CorpusTagError, Instance
from core.predictions import (
    BenchmarkIntegrityError,
    PredictionCsvParseError,
    PredictionFileReadError,
    PredictionRow,
    validate_predictions,
)
from core.splits import SplitError
from t1_features.data import load_t1_inputs
from t1_features.grid_diagnostics import (
    T1_C_GRID,
    T1GridDiagnosticError,
    summarize_trajectory,
)
from t1_features.run_context import T1RunContextError, load_t1_run_context

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_COLUMNS = (
    "language",
    "word_id",
    "category",
    "baseline_correct",
    "sentence",
    "sentence_ru",
    "gold_gramset",
    "top1_gramset",
    "gold_rank",
    "score_margin",
    "gold_pair_in_train",
    "candidates",
    "gold_pos",
    "top1_pos",
    "gold_lemma",
    "top1_lemma",
    "trajectory",
    "n_wrong",
    "n_runs",
    "n_flips",
    "c_threshold",
    "flip_direction",
)


class T1GridDiagnosticCommandError(ValueError):
    """The T1 C-grid diagnostic table could not be built or published."""


def _rows_by_word(path: Path) -> dict[int, tuple[PredictionRow, ...]]:
    grouped: dict[int, list[PredictionRow]] = defaultdict(list)
    for row in load_ranked_predictions(path):
        grouped[row.word_id].append(row)
    return {
        word_id: tuple(sorted(rows, key=lambda row: row.rank))
        for word_id, rows in grouped.items()
    }


def _rank_of_gold(instance: Instance, rows: tuple[PredictionRow, ...]) -> int:
    matches = [
        row.rank
        for row in rows
        if (row.wordform_id, row.gramset)
        == (instance.gold_analysis.wordform_id, instance.gold_analysis.gramset)
    ]
    if len(matches) != 1:
        raise T1GridDiagnosticCommandError(
            f"{instance.language} word_id={instance.word_id}: expected one gold "
            f"candidate in predictions, found {len(matches)}"
        )
    return matches[0]


def _nullable_id(value) -> str:
    if pd.isna(value):
        return ""
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def _load_dictionary_labels(data_dir: Path, language: str):
    morphology = data_dir / "dict" / "morphology"
    links_path = morphology / f"lemma_wordforms_{language}.csv.zst"
    lemmas_path = morphology / f"lemmas_{language}.csv.zst"
    try:
        links = pd.read_csv(
            links_path,
            compression="zstd",
            dtype=str,
            keep_default_na=False,
            na_values=[r"\N"],
        )
        lemmas = pd.read_csv(
            lemmas_path,
            compression="zstd",
            dtype=str,
            keep_default_na=False,
            na_values=[r"\N"],
        )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise T1GridDiagnosticCommandError(
            f"cannot read dictionary morphology for {language}: {exc}"
        ) from exc

    lemma_by_id = {}
    for row in lemmas.itertuples(index=False):
        lemma_by_id[str(row.lemma_id)] = (
            "" if pd.isna(row.lemma) else str(row.lemma),
            "" if pd.isna(row.pos) else str(row.pos),
        )
    labels: dict[tuple[int, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    collected: dict[tuple[int, str], tuple[set[str], set[str]]] = {}
    for row in links.itertuples(index=False):
        wordform_id = _nullable_id(row.wordform_id)
        gramset_id = _nullable_id(row.gramset_id)
        lemma_id = _nullable_id(row.lemma_id)
        if not wordform_id or not gramset_id:
            continue
        lemma, pos = lemma_by_id.get(lemma_id, ("", ""))
        key = (int(wordform_id), gramset_id)
        lemma_values, pos_values = collected.setdefault(key, (set(), set()))
        if lemma:
            lemma_values.add(lemma)
        if pos:
            pos_values.add(pos)
    for key, (lemma_values, pos_values) in collected.items():
        labels[key] = (tuple(sorted(pos_values)), tuple(sorted(lemma_values)))
    return labels


def _labels_for_candidate(
    language: str,
    word_id: int,
    wordform_id: int,
    gramset: str,
    candidate_gramset_ids,
    dictionary_labels,
) -> tuple[str, str]:
    gramset_id = candidate_gramset_ids.get(
        (language, word_id, wordform_id, gramset), ""
    )
    if not gramset_id:
        return "", ""
    pos, lemma = dictionary_labels[language].get(
        (wordform_id, gramset_id), ((), ())
    )
    return "|".join(pos), "|".join(lemma)


def _candidate_sources(data_dir: Path, instances: tuple[Instance, ...]):
    needed_languages = sorted({row.language for row in instances})
    gramset_ids = {}
    sentence_ru = {}
    sentence_words: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    dictionary_labels = {}
    for language in needed_languages:
        tables = read_corpus_tables(language, data_dir)
        for row in tables.candidates.itertuples(index=False):
            gramset_id = _nullable_id(row.gramset_id)
            if gramset_id:
                gramset_ids[(language, int(row.word_id), int(row.wordform_id), str(row.gramset))] = gramset_id
        for row in tables.sentences.itertuples(index=False):
            sentence_ru[(language, int(row.sentence_id))] = (
                "" if pd.isna(row.sentence_ru) else str(row.sentence_ru)
            )
        for row in tables.words.itertuples(index=False):
            sentence_words[(language, int(row.sentence_id))].append(
                (int(row.word_number), str(row.word))
            )
        dictionary_labels[language] = _load_dictionary_labels(data_dir, language)
    for words in sentence_words.values():
        words.sort(key=lambda item: item[0])
    return gramset_ids, sentence_ru, sentence_words, dictionary_labels


def _format_candidates(rows: tuple[PredictionRow, ...]) -> str:
    return " | ".join(
        f"{row.rank}:{row.wordform_id}:{row.gramset}:{row.score:.6g}"
        for row in rows
    )


def _build_output_rows(
    instances: tuple[Instance, ...],
    train_instances: tuple[Instance, ...],
    c_values: tuple[float, ...],
    predictions_by_c: dict[float, dict[int, tuple[PredictionRow, ...]]],
    baseline_rows: dict[int, tuple[PredictionRow, ...]],
    candidate_gramset_ids,
    sentence_ru,
    sentence_words,
    dictionary_labels,
):
    train_gold_pairs = {
        (row.language, row.gold_analysis.wordform_id, row.gold_analysis.gramset)
        for row in train_instances
    }
    output = []
    for instance in sorted(instances, key=lambda row: (row.language, row.word_id)):
        trajectory_rows = [predictions_by_c[c][instance.word_id] for c in c_values]
        correct = tuple(
            _rank_of_gold(instance, rows) == 1 for rows in trajectory_rows
        )
        trajectory = summarize_trajectory(correct, c_values)
        selected_rows = predictions_by_c[10.0][instance.word_id]
        top1 = selected_rows[0]
        gold_rank = _rank_of_gold(instance, selected_rows)
        score_margin = selected_rows[0].score - selected_rows[1].score
        baseline_correct = int(
            _rank_of_gold(instance, baseline_rows[instance.word_id]) == 1
        )
        gold_pos, gold_lemma = _labels_for_candidate(
            instance.language,
            instance.word_id,
            instance.gold_analysis.wordform_id,
            instance.gold_analysis.gramset,
            candidate_gramset_ids,
            dictionary_labels,
        )
        top1_pos, top1_lemma = _labels_for_candidate(
            instance.language,
            instance.word_id,
            top1.wordform_id,
            top1.gramset,
            candidate_gramset_ids,
            dictionary_labels,
        )
        sentence_key = (instance.language, instance.sentence_id)
        sentence = " ".join(
            word for _, word in sentence_words.get(sentence_key, ())
        )
        output.append(
            {
                "language": instance.language,
                "word_id": instance.word_id,
                "category": trajectory.category,
                "baseline_correct": baseline_correct,
                "sentence": sentence,
                "sentence_ru": sentence_ru.get(sentence_key, ""),
                "gold_gramset": instance.gold_analysis.gramset,
                "top1_gramset": top1.gramset,
                "gold_rank": gold_rank,
                "score_margin": score_margin,
                "gold_pair_in_train": int(
                    (
                        instance.language,
                        instance.gold_analysis.wordform_id,
                        instance.gold_analysis.gramset,
                    )
                    in train_gold_pairs
                ),
                "candidates": _format_candidates(selected_rows),
                "gold_pos": gold_pos,
                "top1_pos": top1_pos,
                "gold_lemma": gold_lemma,
                "top1_lemma": top1_lemma,
                "trajectory": trajectory.trajectory,
                "n_wrong": trajectory.n_wrong,
                "n_runs": trajectory.n_runs,
                "n_flips": trajectory.n_flips,
                "c_threshold": trajectory.c_threshold,
                "flip_direction": trajectory.flip_direction,
            }
        )
    return output


def _write_csv(output_path: Path, rows) -> None:
    if output_path.is_symlink() or output_path.exists():
        raise T1GridDiagnosticCommandError(
            f"output already exists: {output_path}"
        )
    if not output_path.parent.is_dir():
        raise T1GridDiagnosticCommandError(
            f"output parent directory does not exist: {output_path.parent}"
        )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=".vepkar-t1-grid-", suffix=".tmp", dir=str(output_path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.link(temp_path, output_path)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass


def _run_diagnose_t1_grid(args: argparse.Namespace) -> int:
    try:
        data_dir, tag = _preflight_benchmark(args.data_dir)
        benchmark = load_benchmark_context(data_dir, tag, args.split_file)
        if tag != "v2026.11":
            raise T1GridDiagnosticCommandError(
                f"the approved diagnostic table is pinned to v2026.11, got {tag!r}"
            )
        split_lookup = {
            (language, int(text_id)): split
            for language, text_id, split in benchmark.split_rows
        }
        instances = tuple(
            row
            for row in benchmark.instances
            if split_lookup.get((row.language, int(row.text_id))) == "dev"
        )
        train_instances = tuple(
            row
            for row in benchmark.instances
            if split_lookup.get((row.language, int(row.text_id))) == "train"
        )
        if len(instances) != 2613:
            raise T1GridDiagnosticCommandError(
                f"expected 2613 strict dev instances for v2026.11, found {len(instances)}"
            )
        if len(args.run_dirs) != len(T1_C_GRID):
            raise T1GridDiagnosticCommandError(
                f"expected {len(T1_C_GRID)} T1 run directories, got {len(args.run_dirs)}"
            )

        manifests = []
        predictions_by_c = {}
        for run_dir in args.run_dirs:
            context = load_t1_run_context(run_dir)
            manifest = context.manifest
            C = float(manifest["parameters"]["C"])
            if C in predictions_by_c:
                raise T1GridDiagnosticCommandError(f"duplicate T1 run for C={C:g}")
            if C not in T1_C_GRID:
                raise T1GridDiagnosticCommandError(f"unexpected T1 C={C:g}")
            manifest_split = manifest["split"]
            if manifest["data"]["tag"] != tag:
                raise T1GridDiagnosticCommandError(
                    f"T1 run {context.run_dir} uses data tag {manifest['data']['tag']!r}, expected {tag!r}"
                )
            if manifest["parameters"]["target_split"] != "dev":
                raise T1GridDiagnosticCommandError(
                    f"T1 run {context.run_dir} was prepared for "
                    f"{manifest['parameters']['target_split']!r}, expected 'dev'"
                )
            if Path(manifest_split["path"]).resolve() != benchmark.split_path.resolve():
                raise T1GridDiagnosticCommandError(
                    f"T1 run {context.run_dir} uses a different split file"
                )
            predictions_path = context.run_dir / "dev.csv"
            validation = validate_predictions(
                predictions_path,
                "dev",
                benchmark.instances,
                benchmark.split_rows,
            )
            if not validation.is_valid:
                raise T1GridDiagnosticCommandError(
                    f"invalid dev predictions in {predictions_path}: {validation.errors}"
                )
            predictions_by_c[C] = _rows_by_word(predictions_path)
            manifests.append(manifest)
        if tuple(sorted(predictions_by_c)) != T1_C_GRID:
            raise T1GridDiagnosticCommandError(
                f"T1 runs do not cover the approved C grid {T1_C_GRID!r}"
            )
        if any(
            manifest["data"] != manifests[0]["data"]
            or manifest["split"] != manifests[0]["split"]
            for manifest in manifests[1:]
        ):
            raise T1GridDiagnosticCommandError(
                "T1 runs do not share identical data and split provenance"
            )
        first_parameters = manifests[0]["parameters"]
        feature_config = (
            first_parameters.get("suffix_max_length", 4),
            first_parameters.get("include_affix", False),
        )
        if any(
            (
                manifest["parameters"].get("suffix_max_length", 4),
                manifest["parameters"].get("include_affix", False),
            )
            != feature_config
            for manifest in manifests[1:]
        ):
            raise T1GridDiagnosticCommandError(
                "T1 runs do not share identical suffix and affix features"
            )

        baseline_validation = validate_predictions(
            args.baseline_predictions,
            "dev",
            benchmark.instances,
            benchmark.split_rows,
        )
        if not baseline_validation.is_valid:
            raise T1GridDiagnosticCommandError(
                f"invalid baseline dev predictions: {baseline_validation.errors}"
            )
        baseline_rows = _rows_by_word(args.baseline_predictions)
        inputs = load_t1_inputs(
            data_dir,
            tag,
            split="dev",
            split_file_override=benchmark.split_path,
        )
        candidate_gramset_ids, sentence_ru, sentence_words, dictionary_labels = (
            _candidate_sources(data_dir, inputs.instances)
        )
        rows = _build_output_rows(
            inputs.instances,
            train_instances,
            T1_C_GRID,
            predictions_by_c,
            baseline_rows,
            candidate_gramset_ids,
            sentence_ru,
            sentence_words,
            dictionary_labels,
        )
        _write_csv(args.output, rows)
    except (
        CorpusTagError,
        DataError,
        SplitError,
        BenchmarkIntegrityError,
        PredictionCsvParseError,
        PredictionFileReadError,
        T1RunContextError,
        T1GridDiagnosticCommandError,
        T1GridDiagnosticError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    counts = Counter(row["category"] for row in rows)
    print("T1 grid diagnostics: OK")
    print(f"Data tag: {tag}")
    print(f"C values: {', '.join(f'{C:g}' for C in T1_C_GRID)}")
    print(f"Rows: {len(rows)}")
    print(f"Suffix maximum length: {feature_config[0]}")
    print(f"Dictionary affix feature: {feature_config[1]}")
    print(f"Output: {args.output}")
    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")
    return 0
