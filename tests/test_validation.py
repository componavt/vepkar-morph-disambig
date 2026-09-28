"""Offline tests for cross-table integrity inspection and diagnostics."""

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
from core.validation import inspect_corpus  # noqa: E402

SENTENCE_1_XML = '<s id="1"><w id="1">"Tere", sanoi ""maa""</w></s>'


def load(root: Path):
    return read_corpus_tables("krl", root)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SRC / "cli.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _warn_line(label: str, value: str) -> str:
    return f"{label:<18}  {value}"


def test_good_fixture_counts_and_diagnostics(tmp_path):
    build_fixture(tmp_path)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.lang == "krl"
    assert (inspection.row_texts, inspection.row_sentences) == (4, 3)
    assert (inspection.row_words, inspection.row_candidates) == (5, 7)
    assert inspection.zero_word_number == 0
    assert inspection.empty_gramset == 0
    assert inspection.empty_gramset_relevance_0 == 0
    assert inspection.empty_gramset_relevance_1 == 0
    assert inspection.empty_gramset_relevance_2 == 0
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
    assert inspection.examples == {}
    assert not inspection.has_broken_integrity


def test_zero_word_number_and_empty_gramset_are_counted(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    candidates = [*GOOD_CANDIDATES, (200, 9100, "", 0), (200, 9101, "", 2)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.zero_word_number == 1
    assert inspection.zero_word_number_sentences == 1
    assert inspection.empty_gramset == 2
    assert inspection.empty_gramset_relevance_0 == 1
    assert inspection.empty_gramset_relevance_1 == 0
    assert inspection.empty_gramset_relevance_2 == 1
    assert not inspection.has_broken_integrity
    example = inspection.examples["empty gramset"][0]
    assert len(inspection.examples["empty gramset"]) == 1
    assert example.detail == (
        "word_id=200; wordform_id=9101; word=ko; relevance=2"
    )


def test_empty_gramset_example_first_row_when_no_gold(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    candidates = [*GOOD_CANDIDATES, (200, 9100, "", 0), (200, 9101, "", 1)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.empty_gramset == 2
    assert inspection.empty_gramset_relevance_0 == 1
    assert inspection.empty_gramset_relevance_1 == 1
    assert inspection.empty_gramset_relevance_2 == 0
    example = inspection.examples["empty gramset"][0]
    assert len(inspection.examples["empty gramset"]) == 1
    assert example.detail == (
        "word_id=200; wordform_id=9100; word=ko; relevance=0"
    )


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


def test_repeated_candidate_tuple_is_counted_with_full_key(tmp_path):
    build_fixture(tmp_path)
    candidates = [*GOOD_CANDIDATES, (102, 9006, "N+SG+PAR", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.duplicate_candidate_tuple == 1
    assert inspection.has_broken_integrity
    example = inspection.examples["duplicate candidate tuple"][0]
    assert "word_id=102; wordform_id=9006" in example.detail
    assert "gramset='N+SG+PAR'" in example.detail


@pytest.mark.parametrize(
    "table,columns,row,attr",
    [
        ("texts_krl.csv.zst", TEXT_COLS, (1, "9", "", "", ""), "duplicate_text_id"),
        ("sentences_krl.csv.zst", SENT_COLS, (10, 5, "<s>dup</s>", ""),
         "duplicate_sentence_id"),
        ("words_krl.csv.zst", WORD_COLS, (100, 10, 1, "tere"), "duplicate_word_id"),
    ],
)
def test_duplicate_primary_keys_are_counted(tmp_path, table, columns, row, attr):
    build_fixture(tmp_path)
    base = {
        "texts_krl.csv.zst": GOOD_TEXTS,
        "sentences_krl.csv.zst": GOOD_SENTENCES,
        "words_krl.csv.zst": GOOD_WORDS,
    }[table]
    write_zst_csv(tmp_path / "corpus" / table, columns, [*base, row])
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert getattr(inspection, attr) == 1
    assert inspection.has_broken_integrity


def test_orphan_sentence_without_text_is_counted(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_SENTENCES, (99, 424242, "<s>ghost</s>", "")]
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, build)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.orphan_sentences_without_text == 1
    assert inspection.has_broken_integrity
    example = inspection.examples["sentences without text"][0]
    assert example.detail == "sentence_id=99; text_id=424242"
    assert example.sentence_xml == "<s>ghost</s>"


def test_orphan_word_without_sentence_is_counted(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_WORDS, (201, 424242, 1, "ghost")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, build)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.orphan_words_without_sentence == 1
    assert inspection.has_broken_integrity
    example = inspection.examples["words without sentence"][0]
    assert example.detail == "word_id=201; sentence_id=424242; word=ghost"
    assert example.sentence_xml == (
        "unavailable — sentence_id=424242 absent from sentences"
    )


def test_orphan_candidate_without_word_is_counted(tmp_path):
    build_fixture(tmp_path)
    build = [*GOOD_CANDIDATES, (9999, 9001, "SG+NOM", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, build)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.orphan_candidates_without_word == 1
    assert inspection.has_broken_integrity
    example = inspection.examples["candidates without word"][0]
    assert example.detail == (
        "word_id=9999; wordform_id=9001; word=<unavailable>; relevance=2"
    )
    assert example.sentence_xml == "unavailable — word_id=9999 absent from words"


def test_word_finding_shows_original_sentence_xml(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    inspection = inspect_corpus(load(tmp_path), "krl")
    example = inspection.examples["word_number=0"][0]
    assert example.detail == "word_id=200; sentence_id=10; word=ko"
    assert example.sentence_xml == SENTENCE_1_XML


def test_candidate_finding_xml_through_word_to_sentence(tmp_path):
    build_fixture(tmp_path)
    candidates = [*GOOD_CANDIDATES, (102, 9110, "", 1)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    example = inspection.examples["empty gramset"][0]
    assert example.detail == (
        "word_id=102; wordform_id=9110; word=höyhen; relevance=1"
    )
    assert example.sentence_xml == '<s id="2">Höyhen</s>'


def test_ambiguous_word_never_selects_sentence(tmp_path):
    build_fixture(tmp_path)
    words = [(300, 10, 1, "myrsky"), (300, 11, 1, "myrsky")]
    candidates = [(300, 9300, "", 2)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    inspection = inspect_corpus(load(tmp_path), "krl")
    example = inspection.examples["empty gramset"][0]
    assert example.sentence_xml == "unavailable — word_id=300 is ambiguous"
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 1
    assert "<w" not in result.stdout


def test_exactly_one_example_per_finding(tmp_path):
    build_fixture(tmp_path)
    words = [
        (100, 10, 0, "a"),
        (101, 10, 0, "b"),
        (102, 11, 0, "c"),
        (103, 12, 0, "d"),
        (104, 12, 0, "e"),
    ]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    inspection = inspect_corpus(load(tmp_path), "krl")
    assert inspection.zero_word_number == 5
    assert inspection.zero_word_number_sentences == 3
    assert len(inspection.examples["word_number=0"]) == 1
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 0
    assert result.stdout.count("  word_id=") == 1
    assert "word_number=0: 5 words in 3 sentences" in result.stdout


def test_sentence_xml_printed_on_single_line(tmp_path):
    build_fixture(tmp_path)
    sentences = [*GOOD_SENTENCES, (99, 1, '<s id="9">\n  <w id="7">kiuru</w>\n</s>', "")]
    words = [*GOOD_WORDS, (300, 99, 0, "kiuru")]
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, sentences)
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 0
    lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("  sentence_xml: <s id=\"9\">")
    ]
    assert len(lines) == 1
    assert lines[0] == '  sentence_xml: <s id="9">   <w id="7">kiuru</w> </s>'


def test_blank_line_between_nonzero_finding_blocks(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    candidates = [*GOOD_CANDIDATES, (200, 9100, "", 1)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    i = lines.index("word_number=0: 1 words in 1 sentences")
    assert lines[i + 1].startswith("  word_id=")
    assert lines[i + 2].startswith("  sentence_xml:")
    assert lines[i + 3] == ""
    assert lines[i + 4].startswith("empty gramset")
    j = lines.index(_warn_line("empty gramset", "candidate rows: 1"))
    assert lines[j + 2].startswith("  word_id=200")
    assert lines[j + 3].startswith("  sentence_xml:")
    assert lines[j + 4] == ""
    assert lines[j + 5] == "DUPLICATES"


def test_no_blank_line_after_zero_count_class(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    texts = [*GOOD_TEXTS, (1, "3", "krl-new", "", "")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "texts_krl.csv.zst", TEXT_COLS, texts)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    gram = lines.index(_warn_line("empty gramset", "candidate rows: 0"))
    assert lines[gram + 1] == "DUPLICATES"
    dup = lines.index(f"{'text_id'.ljust(18)}: 1")
    assert lines[dup + 1].startswith("  text_id=")
    assert lines[dup + 2].startswith("  sentence_xml:")
    assert lines[dup + 3] == ""
    assert lines[dup + 4].startswith("sentence_id")


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
    assert "Result: no integrity issues found" in result.stdout


def test_cli_inspect_data_fails_when_dataset_absent(tmp_path):
    missing = tmp_path / "missing"
    result = _run_cli("inspect-data", "krl", "--data-dir", str(missing))
    assert result.returncode == 1
    assert "dictorpus-data" in result.stderr
    assert str(missing) in result.stderr
    assert "fetch-data" not in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_inspect_data_rejects_unknown_language(tmp_path):
    build_fixture(tmp_path)
    result = _run_cli("inspect-data", "eng", "--data-dir", str(tmp_path))
    assert result.returncode != 0
    assert "Unsupported language" in result.stderr


def test_cli_inspect_data_reports_orphan_before_exit(tmp_path):
    build_fixture(tmp_path)
    bad = [*GOOD_SENTENCES, (99, 424242, "<s>ghost</s>", "")]
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, bad)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 1
    assert "sentences without text" in result.stdout
    assert "Result: integrity issues found" in result.stdout
    assert "error:" not in result.stderr


def test_cli_inspect_data_reports_duplicate_before_exit(tmp_path):
    build_fixture(tmp_path)
    candidates = [*GOOD_CANDIDATES, (100, 9001, "SG+NOM", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 1
    assert "word_id=100; wordform_id=9001; gramset='SG+NOM'" in result.stdout
    assert "Result: integrity issues found" in result.stdout


def test_cli_inspect_data_reports_all_findings_before_exit(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko"), (201, 424242, 1, "ghost")]
    candidates = [*GOOD_CANDIDATES, (200, 9100, "", 1), (200, 9102, "", 2)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 1
    out = result.stdout
    assert "word_number=0: 1 words in 1 sentences" in out
    assert _warn_line("empty gramset", "candidate rows: 2") in out
    assert "relevance=0: 0; relevance=1: 1; relevance=2: 1" in out
    assert "words without sentence" in out
    assert "word_id=200; sentence_id=10; word=ko" in out
    assert "word_id=200; wordform_id=9102; word=ko; relevance=2" in out
    assert f"sentence_xml: {SENTENCE_1_XML}" in out
    assert "sentence_xml: unavailable — sentence_id=424242 absent from sentences" in out
    assert "Result: integrity issues found; source data unchanged" in out
    assert "error:" not in result.stderr


def test_zero_positions_and_empty_gramsets_alone_exit_zero(tmp_path):
    build_fixture(tmp_path)
    words = [*GOOD_WORDS, (200, 10, 0, "ko")]
    candidates = [*GOOD_CANDIDATES, (200, 9100, "", 0), (200, 9101, "", 2)]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst",
                  CAND_COLS, candidates)
    result = _run_cli("inspect-data", "krl", "--data-dir", str(tmp_path))
    assert result.returncode == 0
    assert "word_number=0: 1 words in 1 sentences" in result.stdout
    assert _warn_line("empty gramset", "candidate rows: 2") in result.stdout
    assert "Result: no integrity issues found; source data unchanged" in result.stdout