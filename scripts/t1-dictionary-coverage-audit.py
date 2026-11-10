"""Describe dictionary coverage of strict T1 train/dev candidate analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.data import CorpusTables, read_corpus_tables
from core.instances import Instance, build_language_instances

LANGUAGES = ("vep", "krl", "olo", "lud")
SPLITS = ("train", "dev")
COUNT_FIELDS = (
    "instances",
    "instances_with_any_link",
    "instances_with_all_links",
    "gold_candidates",
    "gold_candidates_linked",
    "candidate_rows",
    "candidate_rows_linked",
    "unique_candidate_types",
    "unique_candidate_types_linked",
    "candidate_form_ids",
    "dictionary_supported_form_ids",
    "form_ids_multi_lemma",
    "form_ids_multi_gramset",
    "form_ids_multi_dialect",
    "candidate_pairs_multi_lemma",
    "surface_forms_multi_wordform_id",
    "surface_forms_multi_lemma",
    "surface_forms_multi_gramset",
)


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _read_train_dev_splits(path: Path) -> dict[tuple[str, int], str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"language", "text_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Split file must contain {sorted(required)}")

    # Discard all other split rows immediately; no test instances are built or
    # summarized by this audit.
    frame = frame.loc[
        frame["split"].isin(SPLITS), ["language", "text_id", "split"]
    ].copy()
    frame["text_id"] = pd.to_numeric(frame["text_id"], errors="raise").astype(int)
    if frame.duplicated(["language", "text_id"]).any():
        raise ValueError("Train/dev split file has duplicate language/text_id rows")
    return {
        (str(row.language), int(row.text_id)): str(row.split)
        for row in frame.itertuples(index=False)
    }


def _train_dev_tables(
    language: str,
    data_dir: Path,
    split_lookup: dict[tuple[str, int], str],
) -> tuple[CorpusTables, dict[int, str]]:
    all_tables = read_corpus_tables(language, data_dir)
    selected_texts = {
        text_id
        for (lang, text_id), _split in split_lookup.items()
        if lang == language
    }
    if not selected_texts:
        raise ValueError(f"No train/dev text assignments for {language}")

    texts = all_tables.texts.loc[
        all_tables.texts["text_id"].isin(selected_texts)
    ].copy()
    sentences = all_tables.sentences.loc[
        all_tables.sentences["text_id"].isin(selected_texts)
    ].copy()
    sentence_ids = set(int(value) for value in sentences["sentence_id"])
    words = all_tables.words.loc[
        all_tables.words["sentence_id"].isin(sentence_ids)
    ].copy()
    word_ids = set(int(value) for value in words["word_id"])
    candidates = all_tables.candidates.loc[
        all_tables.candidates["word_id"].isin(word_ids)
    ].copy()

    return (
        CorpusTables(
            texts=texts,
            sentences=sentences,
            words=words,
            candidates=candidates,
        ),
        {text_id: split for (lang, text_id), split in split_lookup.items() if lang == language},
    )


def _read_dictionary_links(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="zstd",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "wordform_id",
            "wordform",
            "wordform_for_search",
            "lemma_id",
            "gramset_id",
            "dialect_id",
            "dialect_code",
        ],
    )
    for column in frame.columns:
        frame[column] = frame[column].replace(r"\N", "")
    frame["wordform_id"] = pd.to_numeric(frame["wordform_id"], errors="raise").astype(int)
    return frame


def _distinct_nonempty(values: pd.Series) -> int:
    return int(values.loc[values.ne("")].nunique())


def _candidate_row_index(
    tables: CorpusTables,
    instances: tuple[Instance, ...],
) -> dict[tuple[int, int, str], tuple[int, int]]:
    selected_word_ids = {instance.word_id for instance in instances}
    rows = tables.candidates.loc[
        tables.candidates["word_id"].isin(selected_word_ids)
    ]
    index: dict[tuple[int, int, str], tuple[int, int]] = {}
    for row in rows.itertuples(index=False):
        key = (int(row.word_id), int(row.wordform_id), str(row.gramset))
        if key in index:
            raise ValueError(f"Duplicate candidate identity in strict benchmark: {key}")
        if pd.isna(row.gramset_id):
            raise ValueError(f"Strict benchmark candidate lacks gramset_id: {key}")
        index[key] = (int(row.gramset_id), int(row.relevance))

    expected = {
        (instance.word_id, candidate.wordform_id, candidate.gramset)
        for instance in instances
        for candidate in instance.candidates
    }
    if expected != set(index):
        missing = expected - set(index)
        extra = set(index) - expected
        raise ValueError(
            f"Candidate rows differ from strict benchmark; missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    return index


def _ambiguity_counts(
    language: str,
    split: str,
    candidate_types: set[tuple[str, int, int]],
    dictionary_links: pd.DataFrame,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    form_ids = {wordform_id for _lang, wordform_id, _gramset_id in candidate_types}
    links = dictionary_links.loc[
        dictionary_links["wordform_id"].isin(form_ids)
    ].copy()
    links = links.drop_duplicates(
        [
            "wordform_id",
            "wordform",
            "wordform_for_search",
            "lemma_id",
            "gramset_id",
            "dialect_id",
            "dialect_code",
        ]
    )

    by_id = {
        int(wordform_id): group
        for wordform_id, group in links.groupby("wordform_id", sort=True)
    }
    form_ids_multi_lemma = 0
    form_ids_multi_gramset = 0
    form_ids_multi_dialect = 0
    supported_form_ids = 0
    details: list[dict[str, object]] = []

    for wordform_id in sorted(form_ids):
        group = by_id.get(wordform_id)
        if group is None:
            continue
        supported_form_ids += 1
        lemma_ids = sorted(value for value in group["lemma_id"].unique() if value)
        gramset_ids = sorted(value for value in group["gramset_id"].unique() if value)
        dialect_codes = sorted(value for value in group["dialect_code"].unique() if value)
        multi_lemma = len(lemma_ids) > 1
        multi_gramset = len(gramset_ids) > 1
        multi_dialect = len(dialect_codes) > 1
        form_ids_multi_lemma += int(multi_lemma)
        form_ids_multi_gramset += int(multi_gramset)
        form_ids_multi_dialect += int(multi_dialect)

        if multi_lemma or multi_gramset or multi_dialect:
            words = sorted(value for value in group["wordform"].unique() if value)
            search_forms = sorted(
                value for value in group["wordform_for_search"].unique() if value
            )
            details.append(
                {
                    "language": language,
                    "split": split,
                    "wordform_id": wordform_id,
                    "wordform": "|".join(words),
                    "wordform_for_search": "|".join(search_forms),
                    "lemma_ids": "|".join(lemma_ids),
                    "gramset_ids": "|".join(gramset_ids),
                    "dialect_codes": "|".join(dialect_codes),
                    "multiple_lemmas": multi_lemma,
                    "multiple_gramsets": multi_gramset,
                    "multiple_dialects": multi_dialect,
                }
            )

    pair_lemmas: dict[tuple[int, int], set[str]] = defaultdict(set)
    valid_links = links.loc[links["gramset_id"].str.fullmatch(r"[0-9]+")]
    for row in valid_links.itertuples(index=False):
        if row.lemma_id:
            pair_lemmas[(int(row.wordform_id), int(row.gramset_id))].add(
                str(row.lemma_id)
            )
    candidate_pairs_multi_lemma = sum(
        len(pair_lemmas.get((wordform_id, gramset_id), set())) > 1
        for _lang, wordform_id, gramset_id in candidate_types
    )

    surface_links = links.loc[links["wordform"].ne("")]
    surface_multi_id = 0
    surface_multi_lemma = 0
    surface_multi_gramset = 0
    for _surface, group in surface_links.groupby("wordform", sort=True):
        n_ids = int(group["wordform_id"].nunique())
        n_lemmas = _distinct_nonempty(group["lemma_id"])
        n_gramsets = _distinct_nonempty(
            group.loc[group["gramset_id"].str.fullmatch(r"[0-9]+"), "gramset_id"]
        )
        surface_multi_id += int(n_ids > 1)
        surface_multi_lemma += int(n_lemmas > 1)
        surface_multi_gramset += int(n_gramsets > 1)

    counts = {
        "candidate_form_ids": len(form_ids),
        "dictionary_supported_form_ids": supported_form_ids,
        "form_ids_multi_lemma": form_ids_multi_lemma,
        "form_ids_multi_gramset": form_ids_multi_gramset,
        "form_ids_multi_dialect": form_ids_multi_dialect,
        "candidate_pairs_multi_lemma": int(candidate_pairs_multi_lemma),
        "surface_forms_multi_wordform_id": surface_multi_id,
        "surface_forms_multi_lemma": surface_multi_lemma,
        "surface_forms_multi_gramset": surface_multi_gramset,
    }
    return counts, details


def _rates(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    denominator_columns = (
        ("candidate_row_coverage_pct", "candidate_rows_linked", "candidate_rows"),
        ("instance_any_coverage_pct", "instances_with_any_link", "instances"),
        ("instance_all_coverage_pct", "instances_with_all_links", "instances"),
        ("gold_coverage_pct", "gold_candidates_linked", "gold_candidates"),
        (
            "unique_type_coverage_pct",
            "unique_candidate_types_linked",
            "unique_candidate_types",
        ),
        (
            "candidate_pair_multi_lemma_pct",
            "candidate_pairs_multi_lemma",
            "unique_candidate_types_linked",
        ),
        (
            "form_id_multi_lemma_pct",
            "form_ids_multi_lemma",
            "dictionary_supported_form_ids",
        ),
        (
            "form_id_multi_gramset_pct",
            "form_ids_multi_gramset",
            "dictionary_supported_form_ids",
        ),
        (
            "form_id_multi_dialect_pct",
            "form_ids_multi_dialect",
            "dictionary_supported_form_ids",
        ),
    )
    for output, numerator, denominator in denominator_columns:
        count = int(result.get(denominator, 0))
        result[output] = (
            round(100.0 * int(result.get(numerator, 0)) / count, 4)
            if count
            else None
        )
    return result


def run_audit(data_dir: Path, split_file: Path, output_dir: Path) -> None:
    split_lookup = _read_train_dev_splits(split_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")

    summary_rows: list[dict[str, object]] = []
    ambiguity_rows: list[dict[str, object]] = []
    ambiguity_details: list[dict[str, object]] = []

    for language in LANGUAGES:
        tables, language_splits = _train_dev_tables(language, data_dir, split_lookup)
        benchmark = build_language_instances(language, tables)
        dictionary_links = _read_dictionary_links(
            data_dir / "dict" / "morphology" / f"lemma_wordforms_{language}.csv.zst"
        )
        dictionary_pairs = {
            (int(row.wordform_id), int(row.gramset_id))
            for row in dictionary_links.itertuples(index=False)
            if str(row.gramset_id).isdigit()
        }
        row_index = _candidate_row_index(tables, benchmark.instances)

        for split in SPLITS:
            instances = tuple(
                instance
                for instance in benchmark.instances
                if language_splits.get(instance.text_id) == split
            )
            candidate_types: set[tuple[str, int, int]] = set()
            instance_any = 0
            instance_all = 0
            candidate_rows = 0
            candidate_rows_linked = 0
            gold_candidates = 0
            gold_candidates_linked = 0
            for instance in instances:
                candidate_matches: list[bool] = []
                for candidate in instance.candidates:
                    key = (instance.word_id, candidate.wordform_id, candidate.gramset)
                    gramset_id, relevance = row_index[key]
                    pair = (candidate.wordform_id, gramset_id)
                    matched = pair in dictionary_pairs
                    candidate_matches.append(matched)
                    candidate_types.add((language, candidate.wordform_id, gramset_id))
                    candidate_rows += 1
                    candidate_rows_linked += int(matched)
                    if relevance == 2:
                        gold_candidates += 1
                        gold_candidates_linked += int(matched)

                instance_any += int(any(candidate_matches))
                instance_all += int(all(candidate_matches))

            ambiguity, details = _ambiguity_counts(
                language, split, candidate_types, dictionary_links
            )
            counts: dict[str, object] = {
                "language": language,
                "split": split,
                "instances": len(instances),
                "instances_with_any_link": instance_any,
                "instances_with_all_links": instance_all,
                "gold_candidates": gold_candidates,
                "gold_candidates_linked": gold_candidates_linked,
                "candidate_rows": candidate_rows,
                "candidate_rows_linked": candidate_rows_linked,
                "unique_candidate_types": len(candidate_types),
                "unique_candidate_types_linked": sum(
                    (wordform_id, gramset_id) in dictionary_pairs
                    for _lang, wordform_id, gramset_id in candidate_types
                ),
                **ambiguity,
            }
            summary_rows.append(_rates(counts))
            ambiguity_rows.append(
                _rates({"language": language, "split": split, **ambiguity})
            )
            ambiguity_details.extend(details)

    for split in SPLITS:
        language_rows = [
            row for row in summary_rows if row["split"] == split
        ]
        pooled: dict[str, object] = {"language": "pooled", "split": split}
        for field in COUNT_FIELDS:
            pooled[field] = sum(int(row[field]) for row in language_rows)
        summary_rows.append(_rates(pooled))

        language_ambiguity_rows = [
            row for row in ambiguity_rows if row["split"] == split
        ]
        pooled_ambiguity: dict[str, object] = {"language": "pooled", "split": split}
        for field in COUNT_FIELDS[9:]:
            pooled_ambiguity[field] = sum(
                int(row[field]) for row in language_ambiguity_rows
            )
        ambiguity_rows.append(_rates(pooled_ambiguity))

    summary = pd.DataFrame(summary_rows).sort_values(
        ["split", "language"], key=lambda col: col.map(
            {"train": 0, "dev": 1, "vep": 0, "krl": 1, "olo": 2, "lud": 3, "pooled": 4}
        )
    )
    ambiguity = pd.DataFrame(ambiguity_rows).sort_values(
        ["split", "language"], key=lambda col: col.map(
            {"train": 0, "dev": 1, "vep": 0, "krl": 1, "olo": 2, "lud": 3, "pooled": 4}
        )
    )
    details = pd.DataFrame(ambiguity_details)
    if not details.empty:
        details = details.sort_values(["split", "language", "wordform_id"])

    summary.to_csv(output_dir / "coverage-summary.csv", index=False)
    ambiguity.to_csv(output_dir / "ambiguity-summary.csv", index=False)
    details.to_csv(output_dir / "ambiguous-candidate-forms.csv", index=False)
    split_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    metadata = {
        "data_dir": str(data_dir),
        "data_commit": _git_head(data_dir),
        "split_file": str(split_file),
        "split_sha256": split_hash,
        "splits_analyzed": list(SPLITS),
        "test_analyzed": False,
        "benchmark": "strict T1 instances built from train/dev texts only",
        "join_key": ["language", "wordform_id", "gramset_id"],
        "dictionary_matching": "left-preserving coverage check after unique pair deduplication",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nWrote audit outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "dictorpus-data"
    )
    parser.add_argument(
        "--split-file", type=Path, default=ROOT / "data" / "derived" / "splits_v2026.11.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "work"
        / "experiments"
        / "v2026.11"
        / "dictionary-coverage-audit-v1",
    )
    args = parser.parse_args()
    run_audit(args.data_dir, args.split_file, args.output_dir)


if __name__ == "__main__":
    main()
