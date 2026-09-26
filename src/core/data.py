"""Typed loading of the four dictorpus-data corpus tables into pandas DataFrames.

The local ``dictorpus-data`` checkout is always read through ``pandas.read_csv``
with Zstandard decompression straight from the ``.csv.zst`` stream; no
uncompressed copy is ever written next to the source files.  The checkout is
treated as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUPPORTED_LANGUAGES = ("vep", "krl", "olo", "lud")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dictorpus-data"

_TEXT_COLUMNS = ("text_id", "corpus_id", "dialect_code", "genre_id", "year_recorded")
_SENTENCE_COLUMNS = ("sentence_id", "text_id", "sentence_xml", "sentence_ru")
_WORD_COLUMNS = ("word_id", "sentence_id", "word_number", "word")
_CANDIDATE_COLUMNS = ("word_id", "wordform_id", "gramset", "relevance")

_TEXT_RESULT_COLUMNS = ("text_id", "corpus_ids", "dialect_code", "genre_ids", "year_recorded")

_REQUIRED_ID_COLUMNS = {
    "texts": ("text_id",),
    "sentences": ("sentence_id", "text_id"),
    "words": ("word_id", "sentence_id", "word_number"),
    "candidate_analyses": ("word_id", "wordform_id"),
}

_POSITIVE_INT = r"[1-9][0-9]*"
_META_PATTERN = rf"{_POSITIVE_INT}(\|{_POSITIVE_INT})*"


def _check_zstd_support() -> None:
    import importlib.util

    if importlib.util.find_spec("zstandard") is None:
        raise DataError(
            "Zstandard reading support is unavailable; install the 'zstandard' package"
        )


class DataError(Exception):
    """The corpus export could not be read or does not match the typed schema."""


@dataclass(frozen=True)
class CorpusTables:
    """Typed container for the four parsed corpus tables."""

    texts: pd.DataFrame
    sentences: pd.DataFrame
    words: pd.DataFrame
    candidates: pd.DataFrame


def resolve_data_dir(data_dir: Path | None) -> Path:
    """Return the root of the dictorpus-data checkout to read from."""
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir)


def _table_path(stem: str, data_dir: Path | None) -> Path:
    return resolve_data_dir(data_dir) / "corpus" / f"{stem}.csv.zst"


def _read_raw(stem: str, lang: str, data_dir: Path | None) -> pd.DataFrame:
    if lang not in SUPPORTED_LANGUAGES:
        raise DataError(
            f"Unsupported language {lang!r}; expected one of {', '.join(SUPPORTED_LANGUAGES)}"
        )
    _check_zstd_support()
    expected = {
        "texts": _TEXT_COLUMNS,
        "sentences": _SENTENCE_COLUMNS,
        "words": _WORD_COLUMNS,
        "candidate_analyses": _CANDIDATE_COLUMNS,
    }[stem]
    path = _table_path(f"{stem}_{lang}", data_dir)
    if not path.is_file():
        raise DataError(
            f"Expected corpus export not found: {path}; "
            "run 'python3 src/cli.py fetch-data <tag>' to create the checkout"
        )
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # malformed CSV, corrupt stream, unsupported engine
        raise DataError(f"Could not read {path}: {exc}") from exc
    actual = tuple(str(col) for col in frame.columns)
    if actual != expected:
        raise DataError(
            f"Schema mismatch in {path}: expected columns {list(expected)}, "
            f"got {list(actual)}"
        )
    return frame


def _fail_column(path: Path, column: str, count: int, examples: list[str]) -> None:
    sample = ", ".join(repr(v) for v in examples[:5])
    raise DataError(
        f"Invalid values in {path}, column {column!r}: {count} invalid row(s); "
        f"examples: {sample}"
    )


def _validate_required_ids(frame: pd.DataFrame, stem: str, path: Path) -> None:
    for column in _REQUIRED_ID_COLUMNS[stem]:
        raw = frame[column].astype(str)
        bad = ~raw.str.fullmatch(_POSITIVE_INT)
        if bad.any():
            examples = raw[bad].unique()[:5].tolist()
            _fail_column(path, column, int(bad.sum()), examples)
        frame[column] = raw.astype("Int64")


def _parse_pipe_int_list(frame: pd.DataFrame, column: str, path: Path) -> pd.Series:
    raw = frame[column].astype(str)
    valid = (raw == "") | raw.str.fullmatch(_META_PATTERN)
    bad = ~valid
    if bad.any():
        examples = raw[bad].unique()[:5].tolist()
        _fail_column(path, column, int(bad.sum()), examples)
    return raw.str.split("|").apply(lambda parts: tuple(int(p) for p in parts if p))


def _parse_year(frame: pd.DataFrame, path: Path) -> pd.Series:
    raw = frame["year_recorded"].astype(str)
    blank = raw == ""
    bad = ~blank & ~raw.str.fullmatch(r"[1-9][0-9]{3}")
    if bad.any():
        examples = raw[bad].unique()[:5].tolist()
        _fail_column(path, "year_recorded", int(bad.sum()), examples)
    years = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    years[~blank] = raw[~blank].astype("Int64")
    return years


def _check_gramset_nonempty(frame: pd.DataFrame, path: Path) -> None:
    empties = frame["gramset"].astype(str) == ""
    if empties.any():
        examples = frame.loc[empties, "word_id"].astype(str).unique()[:5].tolist()
        _fail_column(path, "gramset", int(empties.sum()), examples)


def read_texts(lang: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``texts_<lang>.csv.zst`` into a typed table with parsed metadata."""
    frame = _read_raw("texts", lang, data_dir)
    path = _table_path(f"texts_{lang}", data_dir)
    _validate_required_ids(frame, "texts", path)
    result = frame.assign(
        corpus_ids=_parse_pipe_int_list(frame, "corpus_id", path),
        dialect_code=frame["dialect_code"].astype("string"),
        genre_ids=_parse_pipe_int_list(frame, "genre_id", path),
        year_recorded=_parse_year(frame, path),
    )
    return result[list(_TEXT_RESULT_COLUMNS)]


def read_sentences(lang: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``sentences_<lang>.csv.zst`` into a typed table."""
    frame = _read_raw("sentences", lang, data_dir)
    path = _table_path(f"sentences_{lang}", data_dir)
    _validate_required_ids(frame, "sentences", path)
    frame["sentence_xml"] = frame["sentence_xml"].astype("string")
    frame["sentence_ru"] = frame["sentence_ru"].astype("string")
    return frame[list(_SENTENCE_COLUMNS)]


def read_words(lang: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``words_<lang>.csv.zst`` into a typed table."""
    frame = _read_raw("words", lang, data_dir)
    path = _table_path(f"words_{lang}", data_dir)
    _validate_required_ids(frame, "words", path)
    frame["word"] = frame["word"].astype("string")
    return frame[list(_WORD_COLUMNS)]


def read_candidates(lang: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read ``candidate_analyses_<lang>.csv.zst`` into a typed table."""
    frame = _read_raw("candidate_analyses", lang, data_dir)
    path = _table_path(f"candidate_analyses_{lang}", data_dir)
    _validate_required_ids(frame, "candidate_analyses", path)
    raw = frame["relevance"].astype(str)
    bad = ~raw.isin({"0", "1", "2"})
    if bad.any():
        examples = raw[bad].unique()[:5].tolist()
        _fail_column(path, "relevance", int(bad.sum()), examples)
    frame["relevance"] = raw.astype("Int64")
    _check_gramset_nonempty(frame, path)
    frame["gramset"] = frame["gramset"].astype("string")
    return frame[list(_CANDIDATE_COLUMNS)]


def read_corpus_tables(lang: str, data_dir: Path | None = None) -> CorpusTables:
    """Read and type-check all four tables of ``lang`` as one unit."""
    return CorpusTables(
        texts=read_texts(lang, data_dir),
        sentences=read_sentences(lang, data_dir),
        words=read_words(lang, data_dir),
        candidates=read_candidates(lang, data_dir),
    )