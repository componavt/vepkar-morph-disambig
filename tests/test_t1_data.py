"""Offline tests for the shared benchmark loader and T1 input preparation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import benchmark_context  # noqa: E402
from core.data import SUPPORTED_LANGUAGES, CorpusTables, DataError  # noqa: E402
from core.predictions import BenchmarkIntegrityError  # noqa: E402
from core.splits import SplitError, split_csv_bytes  # noqa: E402
from t1_features.data import T1Inputs, load_t1_inputs  # noqa: E402
from t1_features.features import ContextWord, build_candidate_examples  # noqa: E402

WORD_BASE = {"vep": 100, "krl": 200, "olo": 300, "lud": 400}

TRAIN_TARGETS = [("krl", 202, 1), ("lud", 402, 1), ("olo", 302, 1), ("vep", 102, 1)]
DEV_TARGETS = [("krl", 205, 2), ("lud", 405, 2), ("olo", 305, 2), ("vep", 105, 2)]


def make_texts(text_ids):
    return pd.DataFrame(
        {
            "text_id": pd.array(text_ids, dtype="Int64"),
            "corpus_ids": [tuple() for _ in text_ids],
            "dialect_code": pd.array([None] * len(text_ids), dtype="string"),
            "genre_ids": [tuple() for _ in text_ids],
            "year_recorded": pd.array([pd.NA] * len(text_ids), dtype="Int64"),
        }
    )


def make_sentences(rows):
    return pd.DataFrame(
        {
            "sentence_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "text_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "sentence_xml": pd.array([r[2] for r in rows], dtype="string"),
            "sentence_ru": pd.array([r[3] for r in rows], dtype="string"),
        }
    )


def make_words(rows):
    return pd.DataFrame(
        {
            "word_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "sentence_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "word_number": pd.array([r[2] for r in rows], dtype="Int64"),
            "word": pd.array([r[3] for r in rows], dtype="string"),
        }
    )


def make_candidates(rows):
    return pd.DataFrame(
        {
            "word_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "wordform_id": pd.array([r[1] for r in rows], dtype="Int64"),
            "gramset": pd.array([r[2] for r in rows], dtype="string"),
            "relevance": pd.array([r[3] for r in rows], dtype="Int64"),
        }
    )


def _shared_texts():
    return make_texts([1, 2, 3])


def _shared_sentences():
    return make_sentences(
        [
            (10, 1, '<s id="10"/>', ""),
            (20, 2, '<s id="20"/>', ""),
            (30, 3, '<s id="30"/>', ""),
        ]
    )


def language_tables(lang, *, include_train=True, word_base=None):
    """One small corpus per language; texts/sentences reuse shared numeric ids.

    Every target word gets two distinct nonempty analyses with exactly one
    relevance-2 gold.  Neighbor words have no candidate rows.  Words table rows
    are deliberately shuffled with nonconsecutive positions.
    """
    b = WORD_BASE[lang] if word_base is None else word_base
    word_rows = [
        (b + 3, 10, 5, "C"),
        (b + 1, 10, 1, "A"),
        (b + 4, 10, 7, "D"),
        (b + 2, 10, 2, " MiXeD "),
        (b + 6, 20, 2, "a\u0308"),
        (b + 5, 20, 1, "kissa"),
        (b + 7, 20, 3, " H\u00e4nt\u00e4"),
        (b + 9, 30, 2, "y"),
        (b + 8, 30, 1, "z"),
    ]
    candidate_rows = []
    if include_train:
        candidate_rows += [(b + 2, b + 50, "SG", 2), (b + 2, b + 51, "PL", 1)]
    candidate_rows += [
        (b + 5, b + 52, "N+SG", 2),
        (b + 5, b + 53, "N+PL", 1),
        (b + 8, b + 54, "A+GEN", 1),
        (b + 8, b + 55, "A+NOM", 2),
    ]
    return CorpusTables(
        texts=_shared_texts(),
        sentences=_shared_sentences(),
        words=make_words(word_rows),
        candidates=make_candidates(candidate_rows),
    )


def empty_language_tables():
    return CorpusTables(
        texts=make_texts([]),
        sentences=make_sentences([]),
        words=make_words([]),
        candidates=make_candidates([]),
    )


def full_split_rows():
    return tuple(
        (lang, text_id, split)
        for lang in SUPPORTED_LANGUAGES
        for text_id, split in ((1, "train"), (2, "dev"), (3, "test"))
    )


def standard_split_path(tmp_path):
    path = tmp_path / "splits_custom.csv"
    path.write_bytes(split_csv_bytes(full_split_rows()))
    return path


def install_tables(monkeypatch, tables):
    calls = []

    def spy(lang, data_dir):
        calls.append(lang)
        return tables[lang]

    monkeypatch.setattr(benchmark_context, "read_corpus_tables", spy)
    return calls


def expected_train_context():
    return tuple(
        context_word
        for lang in sorted(WORD_BASE)
        for context_word in [
            ContextWord(lang, 10, WORD_BASE[lang] + 1, 1, "A"),
            ContextWord(lang, 10, WORD_BASE[lang] + 2, 2, " MiXeD "),
            ContextWord(lang, 10, WORD_BASE[lang] + 3, 5, "C"),
            ContextWord(lang, 10, WORD_BASE[lang] + 4, 7, "D"),
        ]
    )


def expected_dev_context():
    return tuple(
        context_word
        for lang in sorted(WORD_BASE)
        for context_word in [
            ContextWord(lang, 20, WORD_BASE[lang] + 5, 1, "kissa"),
            ContextWord(lang, 20, WORD_BASE[lang] + 6, 2, "a\u0308"),
            ContextWord(lang, 20, WORD_BASE[lang] + 7, 3, " H\u00e4nt\u00e4"),
        ]
    )


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
    calls = install_tables(monkeypatch, tables)
    return tables, calls, standard_split_path(tmp_path)


class TestSharedLoaderBehavior:
    def test_default_context_unchanged(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        ctx = benchmark_context.load_benchmark_context(tmp_path, "fixture-tag", split_path)
        assert ctx.word_tables is None
        assert ctx.tag == "fixture-tag"
        assert ctx.split_path == split_path
        assert ctx.split_rows == full_split_rows()
        assert len(ctx.instances) == 12
        expected_ids = {
            (lang, base + offset)
            for lang, base in WORD_BASE.items()
            for offset in (2, 5, 8)
        }
        assert {(inst.language, inst.word_id) for inst in ctx.instances} == expected_ids
        assert calls == list(SUPPORTED_LANGUAGES)

    def test_legacy_four_field_construction(self, tmp_path):
        path = tmp_path / "legacy.csv"
        legacy = benchmark_context.BenchmarkContext("tag", path, (), ())
        assert legacy.word_tables is None
        assert (legacy.tag, legacy.split_path, legacy.instances, legacy.split_rows) == (
            "tag",
            path,
            (),
            (),
        )

    def test_opt_in_retains_original_words_tables(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        enabled = benchmark_context.load_benchmark_context(
            tmp_path, "fixture-tag", split_path, include_words=True
        )
        disabled = benchmark_context.load_benchmark_context(
            tmp_path, "fixture-tag", split_path
        )
        assert set(enabled.word_tables) == set(SUPPORTED_LANGUAGES)
        for lang in SUPPORTED_LANGUAGES:
            assert enabled.word_tables[lang] is tables[lang].words
        assert disabled.word_tables is None
        assert enabled.instances == disabled.instances
        assert calls == list(SUPPORTED_LANGUAGES) * 2

    def test_opt_in_retains_empty_tables(self, tmp_path, monkeypatch):
        tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        tables["lud"] = empty_language_tables()
        install_tables(monkeypatch, tables)
        ctx = benchmark_context.load_benchmark_context(
            tmp_path, "fixture-tag", standard_split_path(tmp_path), include_words=True
        )
        assert set(ctx.word_tables) == set(SUPPORTED_LANGUAGES)
        assert len(ctx.word_tables["lud"]) == 0
        assert len(ctx.instances) == 9


class TestT1Loading:
    def test_loads_each_language_once_and_preserves_inputs(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        inputs = load_t1_inputs(
            tmp_path, "custom-tag", split="train", split_file_override=split_path
        )
        assert isinstance(inputs, T1Inputs)
        assert calls == list(SUPPORTED_LANGUAGES)
        assert inputs.benchmark.tag == "custom-tag"
        assert inputs.benchmark.split_path == split_path
        assert inputs.split == "train"

    def test_invalid_split_rejected_before_io(self, tmp_path, monkeypatch):
        def spy(lang, data_dir):
            raise AssertionError("loader must not run")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", spy)
        with pytest.raises(ValueError) as exc:
            load_t1_inputs(tmp_path, "tag", split="test")
        message = str(exc.value)
        assert "test" in message and "train" in message and "dev" in message

    @pytest.mark.parametrize(
        ("split", "targets", "context"),
        [
            ("train", TRAIN_TARGETS, expected_train_context()),
            ("dev", DEV_TARGETS, expected_dev_context()),
        ],
    )
    def test_exact_targets_and_full_context(
        self, corpus, tmp_path, split, targets, context
    ):
        tables, calls, split_path = corpus
        inputs = load_t1_inputs(
            tmp_path, "tag", split=split, split_file_override=split_path
        )
        assert [(inst.language, inst.word_id, inst.text_id) for inst in inputs.instances] == targets
        assert all(inst.text_id != 3 for inst in inputs.instances)
        assert inputs.context_words == context

    def test_literal_preservation_and_deterministic_order(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        first = load_t1_inputs(tmp_path, "tag", split="train", split_file_override=split_path)
        second = load_t1_inputs(tmp_path, "tag", split="train", split_file_override=split_path)
        assert first.instances == second.instances
        assert first.context_words == second.context_words
        words = [candidate.word for candidate in first.context_words]
        assert " MiXeD " in words
        dev = load_t1_inputs(tmp_path, "tag", split="dev", split_file_override=split_path)
        dev_words = [candidate.word for candidate in dev.context_words]
        assert "a\u0308" in dev_words
        assert " H\u00e4nt\u00e4" in dev_words
        keys = [
            (candidate.language, candidate.sentence_id, candidate.word_number, candidate.word_id)
            for candidate in first.context_words
        ]
        assert keys == sorted(keys)

    def test_language_aware_shared_numeric_ids(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        inputs = load_t1_inputs(tmp_path, "tag", split="dev", split_file_override=split_path)
        per_language: dict[str, set[int]] = {}
        for candidate in inputs.context_words:
            per_language.setdefault(candidate.language, set()).add(candidate.word_id)
        assert set(per_language) == set(SUPPORTED_LANGUAGES)
        for lang in SUPPORTED_LANGUAGES:
            base = WORD_BASE[lang]
            assert per_language[lang] == {base + 5, base + 6, base + 7}

    def test_original_instances_and_full_benchmark_retained(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        inputs = load_t1_inputs(tmp_path, "tag", split="train", split_file_override=split_path)
        assert len(inputs.benchmark.instances) == 12
        assert inputs.benchmark.split_rows == full_split_rows()
        assert inputs.benchmark.word_tables is None
        benchmark_ids = {id(inst) for inst in inputs.benchmark.instances}
        assert all(id(inst) in benchmark_ids for inst in inputs.instances)
        assert any(inst.text_id == 3 for inst in inputs.benchmark.instances)
        for inst in inputs.instances:
            base = WORD_BASE[inst.language]
            assert inst.word_id == base + 2
            assert inst.word == " MiXeD "
            assert inst.word_number == 2
            assert len(inst.candidates) == 2
            assert inst.gold_analysis == inst.candidates[0]

    def test_build_candidate_examples_on_prepared_inputs(self, corpus, tmp_path):
        tables, calls, split_path = corpus
        train = load_t1_inputs(tmp_path, "tag", split="train", split_file_override=split_path)
        examples = build_candidate_examples(train.instances, train.context_words, mode="train")
        assert len(examples) == 8
        assert sum(1 for example in examples if example.label == 1) == 4
        for example in examples:
            if example.word_id == WORD_BASE[example.language] + 2:
                assert (example.left_word, example.right_word) == ("A", "C")
        dev = load_t1_inputs(tmp_path, "tag", split="dev", split_file_override=split_path)
        dev_examples = build_candidate_examples(
            dev.instances,
            dev.context_words,
            mode="predict",
        )
        assert all(
            e.label is None and e.sample_weight is None
            for e in dev_examples
        )
        assert len(dev_examples) == 8
        for example in dev_examples:
            if example.word_id == WORD_BASE[example.language] + 5:
                assert (example.left_word, example.right_word) == ("<BOS>", "a\u0308")


class TestIntegrityChecks:
    def install_standard(self, tmp_path, monkeypatch, tables=None):
        if tables is None:
            tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        install_tables(monkeypatch, tables)

    def test_unsupported_label_raises_split_error(self, tmp_path, monkeypatch):
        self.install_standard(tmp_path, monkeypatch)
        rows = list(full_split_rows())
        rows[0] = ("vep", 1, "val")
        path = tmp_path / "label.csv"
        path.write_bytes(split_csv_bytes(tuple(rows)))
        with pytest.raises(SplitError) as exc:
            load_t1_inputs(tmp_path, "tag", split="train", split_file_override=path)
        message = str(exc.value)
        assert "val" in message and "vep" in message

    def test_duplicate_assignment_raises_split_error(self, tmp_path, monkeypatch):
        self.install_standard(tmp_path, monkeypatch)
        path = tmp_path / "dup.csv"
        path.write_bytes(split_csv_bytes(full_split_rows() + (("krl", 1, "train"),)))
        with pytest.raises(SplitError) as exc:
            load_t1_inputs(tmp_path, "tag", split="train", split_file_override=path)
        message = str(exc.value)
        assert "Duplicate" in message and "('krl', 1)" in message

    def test_unassigned_instance_raises_split_error(self, tmp_path, monkeypatch):
        self.install_standard(tmp_path, monkeypatch)
        rows = tuple(row for row in full_split_rows() if row != ("vep", 1, "train"))
        path = tmp_path / "missing.csv"
        path.write_bytes(split_csv_bytes(rows))
        with pytest.raises(SplitError) as exc:
            load_t1_inputs(tmp_path, "tag", split="train", split_file_override=path)
        message = str(exc.value)
        assert "vep" in message and "102" in message and "no split assignment" in message

    def test_duplicate_word_ids_raise_integrity_error(self, tmp_path, monkeypatch):
        tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        tables["olo"] = language_tables("olo", word_base=100)
        install_tables(monkeypatch, tables)
        with pytest.raises(BenchmarkIntegrityError) as exc:
            load_t1_inputs(
                tmp_path, "tag", split="train",
                split_file_override=standard_split_path(tmp_path),
            )
        assert "duplicate word_id" in str(exc.value)


class TestEmptySelections:
    def test_empty_requested_split(self, tmp_path, monkeypatch):
        tables = {
            lang: language_tables(lang, include_train=False)
            for lang in SUPPORTED_LANGUAGES
        }
        install_tables(monkeypatch, tables)
        inputs = load_t1_inputs(
            tmp_path, "tag", split="train", split_file_override=standard_split_path(tmp_path)
        )
        assert inputs.instances == ()
        assert inputs.context_words == ()
        assert len(inputs.benchmark.instances) == 8

    def test_empty_language_tables_supported(self, tmp_path, monkeypatch):
        tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        tables["lud"] = empty_language_tables()
        install_tables(monkeypatch, tables)
        split_path = standard_split_path(tmp_path)
        ctx = benchmark_context.load_benchmark_context(
            tmp_path, "tag", split_path, include_words=True
        )
        assert len(ctx.word_tables["lud"]) == 0
        inputs = load_t1_inputs(tmp_path, "tag", split="train", split_file_override=split_path)
        assert {inst.language for inst in inputs.instances} == {"krl", "olo", "vep"}
        assert "lud" not in {candidate.language for candidate in inputs.context_words}


class TestErrorPropagation:
    def test_data_error_propagates(self, tmp_path, monkeypatch):
        def spy(lang, data_dir):
            raise DataError(f"corpus tables unavailable for {lang}")

        monkeypatch.setattr(benchmark_context, "read_corpus_tables", spy)
        with pytest.raises(DataError) as exc:
            load_t1_inputs(
                tmp_path, "tag", split="train", split_file_override=tmp_path / "nope.csv"
            )
        assert "unavailable" in str(exc.value)

    def test_split_read_failures_propagate(self, tmp_path, monkeypatch):
        tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        install_tables(monkeypatch, tables)
        with pytest.raises(SplitError) as exc:
            load_t1_inputs(
                tmp_path, "tag", split="train", split_file_override=tmp_path / "nope.csv"
            )
        assert "cannot read split file" in str(exc.value)
        bad = tmp_path / "bad.csv"
        bad.write_text("language,text_id\nvep,1\n", encoding="utf-8")
        with pytest.raises(SplitError) as exc:
            load_t1_inputs(tmp_path, "tag", split="train", split_file_override=bad)
        assert "header" in str(exc.value)


class TestNoSideEffects:
    def test_no_writes_and_unchanged_tables(self, tmp_path, monkeypatch):
        tables = {lang: language_tables(lang) for lang in SUPPORTED_LANGUAGES}
        snapshots = {
            lang: tables[lang].words.copy(deep=True) for lang in SUPPORTED_LANGUAGES
        }
        install_tables(monkeypatch, tables)
        split_path = standard_split_path(tmp_path)
        before_entries = sorted(entry.name for entry in tmp_path.iterdir())
        before_bytes = split_path.read_bytes()
        inputs = load_t1_inputs(tmp_path, "tag", split="dev", split_file_override=split_path)
        assert sorted(entry.name for entry in tmp_path.iterdir()) == before_entries
        assert split_path.read_bytes() == before_bytes
        assert inputs.context_words
        for lang in SUPPORTED_LANGUAGES:
            pd.testing.assert_frame_equal(tables[lang].words, snapshots[lang])