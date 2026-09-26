"""Offline tests for typed loading of the four corpus tables."""

import io
import sys
from pathlib import Path

import pandas as pd
import pytest
import zstandard as zstd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.data import (  # noqa: E402
    CorpusTables,
    DataError,
    read_candidates,
    read_corpus_tables,
    read_sentences,
    read_texts,
    read_words,
)

TEXT_COLS = ["text_id", "corpus_id", "dialect_code", "genre_id", "year_recorded"]
SENT_COLS = ["sentence_id", "text_id", "sentence_xml", "sentence_ru"]
WORD_COLS = ["word_id", "sentence_id", "word_number", "word"]
CAND_COLS = ["word_id", "wordform_id", "gramset", "relevance"]

GOOD_TEXTS = [
    (1, "3", "krl-new", "", ""),
    (2, "8|11", "", "8|11", "1961"),
    (3, "1|4|15", "krl-old", "", "1997"),
    (4, "", "", "", ""),
]
GOOD_SENTENCES = [
    (10, 1, '<s id="1"><w id="1">"Tere", sanoi ""maa""</w></s>', '"Привет", сказала "земля"'),
    (11, 2, "<s id=\"2\">Höyhen</s>", ""),
    (12, 3, "<s id=\"3\">Vesi ja \"maa\"</s>", "Вода и земля"),
]
GOOD_WORDS = [
    (100, 10, 1, "tere"),
    (101, 10, 2, '"maa!"'),
    (102, 11, 1, "höyhen"),
    (103, 12, 1, "vesi, vesi"),
    (104, 12, 2, "jogi"),
]
GOOD_CANDIDATES = [
    (100, 9001, "SG+NOM", 2),
    (100, 9002, "SG+GEN", 1),
    (101, 9003, "INTERJ", 2),
    (102, 9004, "N+SG+NOM", 1),
    (102, 9005, "N+SG+GEN", 2),
    (102, 9006, "N+SG+PAR", 2),
    (103, 9007, "N+SG+NOM", 0),
]


def write_zst_csv(path: Path, columns: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows, columns=columns)
    csv = frame.to_csv(index=False, lineterminator="\n")
    cctx = zstd.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        cctx.copy_stream(io.BytesIO(csv.encode("utf-8")), fh)


def build_fixture(
    root: Path,
    texts=GOOD_TEXTS,
    sentences=GOOD_SENTENCES,
    words=GOOD_WORDS,
    candidates=GOOD_CANDIDATES,
) -> Path:
    corpus = root / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    write_zst_csv(corpus / "texts_krl.csv.zst", TEXT_COLS, texts)
    write_zst_csv(corpus / "sentences_krl.csv.zst", SENT_COLS, sentences)
    write_zst_csv(corpus / "words_krl.csv.zst", WORD_COLS, words)
    write_zst_csv(corpus / "candidate_analyses_krl.csv.zst", CAND_COLS, candidates)
    return root


def test_read_all_four_tables_with_typed_schema(tmp_path):
    build_fixture(tmp_path)
    tables = read_corpus_tables("krl", tmp_path)
    assert isinstance(tables, CorpusTables)

    texts = tables.texts
    assert list(texts.columns) == [
        "text_id", "corpus_ids", "dialect_code", "genre_ids", "year_recorded",
    ]
    assert "corpus_id" not in texts.columns
    assert "genre_id" not in texts.columns
    assert str(texts["text_id"].dtype) == "Int64"
    assert texts["text_id"].tolist() == [1, 2, 3, 4]
    assert str(texts["dialect_code"].dtype) == "string"
    assert texts["dialect_code"].tolist() == ["krl-new", "", "krl-old", ""]
    assert str(texts["year_recorded"].dtype) == "Int64"
    assert pd.isna(texts["year_recorded"].iloc[0])
    assert pd.isna(texts["year_recorded"].iloc[3])
    assert texts["year_recorded"].iloc[1] == 1961
    assert texts["year_recorded"].iloc[2] == 1997

    sentences = tables.sentences
    assert list(sentences.columns) == SENT_COLS
    assert str(sentences["sentence_id"].dtype) == "Int64"
    assert str(sentences["text_id"].dtype) == "Int64"
    assert str(sentences["sentence_xml"].dtype) == "string"
    assert str(sentences["sentence_ru"].dtype) == "string"
    assert sentences["sentence_id"].tolist() == [10, 11, 12]
    assert sentences["sentence_ru"].iloc[1] == ""

    words = tables.words
    assert list(words.columns) == WORD_COLS
    assert str(words["word_id"].dtype) == "Int64"
    assert str(words["sentence_id"].dtype) == "Int64"
    assert str(words["word_number"].dtype) == "Int64"
    assert str(words["word"].dtype) == "string"

    candidates = tables.candidates
    assert list(candidates.columns) == CAND_COLS
    assert str(candidates["word_id"].dtype) == "Int64"
    assert str(candidates["wordform_id"].dtype) == "Int64"
    assert str(candidates["gramset"].dtype) == "string"
    assert str(candidates["relevance"].dtype) == "Int64"
    assert candidates["relevance"].tolist() == [2, 1, 2, 1, 2, 2, 0]


def test_pipe_separated_metadata_is_parsed_to_tuples(tmp_path):
    build_fixture(tmp_path)
    texts = read_texts("krl", tmp_path)
    assert (texts["corpus_ids"].tolist() ==
            [(3,), (8, 11), (1, 4, 15), ()])
    assert texts["genre_ids"].tolist() == [(), (8, 11), (), ()]


def test_utf8_commas_doubled_quotes_and_xml_round_trip(tmp_path):
    build_fixture(tmp_path)
    sentences = read_sentences("krl", tmp_path)
    assert sentences["sentence_xml"].iloc[0] == (
        '<s id="1"><w id="1">"Tere", sanoi ""maa""</w></s>'
    )
    assert sentences["sentence_ru"].iloc[0] == '"Привет", сказала "земля"'
    words = read_words("krl", tmp_path)
    assert words["word"].iloc[3] == "vesi, vesi"
    assert words["word"].iloc[1] == '"maa!"'


def test_column_order_mismatch_rejected(tmp_path):
    build_fixture(tmp_path)
    shuffled = [
        (1, "krl-new", "", "3", ""),
        (2, "", "8|11", "8|11", "1961"),
    ]
    write_zst_csv(tmp_path / "corpus" / "texts_krl.csv.zst",
                  ["text_id", "dialect_code", "genre_id", "corpus_id", "year_recorded"], shuffled)
    with pytest.raises(DataError, match="Schema mismatch.*texts_krl"):
        read_texts("krl", tmp_path)


def test_extra_column_rejected(tmp_path):
    build_fixture(tmp_path)
    rows = [(1, "3", "krl-new", "", "", "junk")]
    columns = TEXT_COLS + ["extra"]
    write_zst_csv(tmp_path / "corpus" / "texts_krl.csv.zst", columns, rows)
    with pytest.raises(DataError, match="Schema mismatch"):
        read_texts("krl", tmp_path)


@pytest.mark.parametrize("value", ["8||11", "|8", "11|", "8|abc", "-1", "0"])
def test_invalid_composite_metadata_rejected(tmp_path, value):
    candidate_meta = [
        (1, value, "", "", ""),
        (2, "3", "", "", ""),
    ]
    write_zst_csv(tmp_path / "corpus" / "texts_krl.csv.zst", TEXT_COLS, candidate_meta)
    with pytest.raises(DataError, match="corpus_id"):
        read_texts("krl", tmp_path)


@pytest.mark.parametrize("column", ["sentence_id", "text_id"])
def test_non_integer_required_sentence_id_rejected(tmp_path, column):
    build_fixture(tmp_path)
    bad_rows = []
    for row in GOOD_SENTENCES:
        values = list(row)
        if row[1] == 1:
            values[0 if column == "sentence_id" else 1] = "abc"
        bad_rows.append(tuple(values))
    write_zst_csv(tmp_path / "corpus" / "sentences_krl.csv.zst", SENT_COLS, bad_rows)
    with pytest.raises(DataError, match=column):
        read_sentences("krl", tmp_path)


@pytest.mark.parametrize("value", ["0", "-5", "", "abc"])
def test_non_positive_word_number_rejected(tmp_path, value):
    build_fixture(tmp_path)
    bad = [(100, 10, value, "tere")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, bad)
    with pytest.raises(DataError, match="word_number"):
        read_words("krl", tmp_path)


def test_word_number_zero_rejected(tmp_path):
    build_fixture(tmp_path)
    bad = [(100, 10, 0, "tere")]
    write_zst_csv(tmp_path / "corpus" / "words_krl.csv.zst", WORD_COLS, bad)
    with pytest.raises(DataError, match="word_number"):
        read_words("krl", tmp_path)


@pytest.mark.parametrize("value", ["3", "abc", ""])
def test_invalid_relevance_rejected(tmp_path, value):
    build_fixture(tmp_path)
    bad = [(100, 9001, "SG+NOM", value)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst", CAND_COLS, bad)
    with pytest.raises(DataError, match="relevance"):
        read_candidates("krl", tmp_path)


def test_invalid_year_rejected(tmp_path):
    build_fixture(tmp_path)
    bad = [(1, "3", "", "", "21")]
    write_zst_csv(tmp_path / "corpus" / "texts_krl.csv.zst", TEXT_COLS, bad)
    with pytest.raises(DataError, match="year_recorded"):
        read_texts("krl", tmp_path)


def test_empty_gramset_rejected(tmp_path):
    build_fixture(tmp_path)
    bad = [(100, 9001, "", 2)]
    write_zst_csv(tmp_path / "corpus" / "candidate_analyses_krl.csv.zst", CAND_COLS, bad)
    with pytest.raises(DataError, match="gramset"):
        read_candidates("krl", tmp_path)


def test_unsupported_language_rejected(tmp_path):
    with pytest.raises(DataError, match="Unsupported language"):
        read_corpus_tables("eng", tmp_path)


def test_missing_source_file_rejected(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "corpus" / "texts_krl.csv.zst").unlink()
    with pytest.raises(DataError, match="not found"):
        read_texts("krl", tmp_path)