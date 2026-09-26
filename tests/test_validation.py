"""Offline tests for cross-table integrity validation and diagnostics."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from test_data import (  # noqa: E402
    CAND_COLS,
    GOOD_CANDIDATES,
    GOOD_SENTENCES,
    GOOD_TEXTS,
    GOOD_WORDS,
    SENT_COLS,
    TEXT_COLS,
    WORD_COLS,
    build_fixture,
    write_zst_csv,
)
from core.data import read_corpus_tables  # noqa: E402
from core.validation import CorpusError, inspect_corpus  # noqa: E402


def load(root: Path):
    return read_corpus_tables("krl", root)


def test_good_fixture_counts_and_diagnostics(tmp_path):
    build_fixture(tmp_path)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.lang == "krl"
    assert (inspection.row_texts, inspection.row_sentences) == (4, 3)
    assert (inspection.row_words, inspection.row_candidates) == (5, 7)
    assert inspection.duplicate_text_id == 0
    assert inspection.duplicate_sentence_id == 0
    assert inspection.duplicate_word_id == 0
    assert inspection.duplicate_candidate_tuple == 0
    assert inspection.orphan_sentences_without_text == 0
    assert inspection.orphan_words_without_sentence == 0
    assert inspection.orphan_candidates_without_word == 0
    assert inspection.words_with_zero_candidates == 1
    assert inspection.words_with_one_candidate == 2
    assert inspection.words_with_multiple_candidates == 2
    assert inspection.words_with_zero_selected == 2
    assert inspection.words_with_one_selected == 2
    assert inspection.words_with_multiple_selected == 1


def test_same_word_wordform_different_gramset_is_not_a_duplicate(tmp_path):
    build_fixture(tmp_path)
    candidates = [
        (100, 9001, "SG+NOM", 2),
        (100, 9001, "SG+GEN", 1),
        (100, 9001, "SG+ACC", 0),
        (101, 9003, "INTERJ", 2),
    ]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.duplicate_candidate_tuple == 0


def test_repeated_candidate_tuple_is_a_validation_error(tmp_path):
    build_fixture(tmp_path)
    candidates = [(100, 9001, "SG+NOM", 2), (100, 9001, "SG+NOM", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    with pytest.raises(CorpusError, match="duplicate candidate tuple"):
        inspect_corpus(load(tmp_path), "krl")


@pytest.mark.parametrize(
    "table,rows,columns,message",
    [
        ("texts_krl.csv.zst",
         [(1, "3", "", "", ""), (1, "3", "", "", "")],
         TEXT_COLS, "duplicate text_id"),
        ("sentences_krl.csv.zst",
         [(10, 1, "<s>a</s>", ""), (10, 1, "<s>b</s>", "")],
         SENT_COLS, "duplicate sentence_id"),
        ("words_krl.csv.zst",
         [(100, 10, 1, "tere"), (100, 10, 1, "tere")],
         WORD_COLS, "duplicate word_id"),
    ],
)
def test_duplicate_primary_keys_are_validation_errors(tmp_path, table, rows, columns, message):
    build_fixture(tmp_path)
    write_zst_csv(tmp_path / "corpus" / table, columns, rows)
    with pytest.raises(CorpusError, match=message):
        inspect_corpus(load(tmp_path), "krl")


def test_orphan_sentence_without_text_is_error(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_SENTENCES, (99, 424242, "<s>ghost</s>", "")]
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, build)
    with pytest.raises(CorpusError, match="without text"):
        inspect_corpus(load(tmp_path), "krl")


def test_orphan_word_without_sentence_is_error(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_WORDS, (9999, 424242, 1, "ghost")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, build)
    with pytest.raises(CorpusError, match="without sentence"):
        inspect_corpus(load(tmp_path), "krl")


def test_orphan_candidate_without_word_is_error(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_CANDIDATES, (9999, 9001, "SG+NOM", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst", CAND_COLS, build)
    with pytest.raises(CorpusError, match="without word"):
        inspect_corpus(load(tmp_path), "krl")


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SRC / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_cli_inspect_data_succeeds_on_fixture(tmp_path):
    build_fixture(tmp_path)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Language: krl" in result.stdout
    assert "TABLES" in result.stdout
    assert "rows: 7" in result.stdout
    assert "words with 0 candidates: 1" in result.stdout
    assert "words with 1 candidate:" in result.stdout
    assert "words with 1 relevance=2" in result.stdout


def test_cli_inspect_data_fails_when_dataset_absent(tmp_path):
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path / "missing"))
    assert result.returncode != 0
    assert "not found" in result.stderr.lower()


def test_cli_inspect_data_rejects_unknown_language(tmp_path):
    result = _run_cli("inspect-data", "eng", "--data-dir", str(tmp_path))
    assert result.returncode != 0
    assert "Unsupported language" in result.stderr


def test_cli_inspect_data_reports_orphan_error_on_fixture(tmp_path):
    build_fixture(tmp_path)
    bad = [*GOOD_SENTENCES, (99, 424242, "<s>ghost</s>", "")]
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, bad)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode != 0
    assert "without text" in result.stderr