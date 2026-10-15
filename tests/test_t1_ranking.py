"""Offline tests for pure ranking of candidate examples into prediction rows."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.predictions import BenchmarkIntegrityError, PredictionRow  # noqa: E402
from t1_features.features import CandidateExample  # noqa: E402
from t1_features.ranking import rank_candidate_scores  # noqa: E402


def make_example(
    language,
    word_id,
    wordform_id,
    gramset,
    *,
    word="w",
    label=None,
    sample_weight=None,
    features=None,
):
    return CandidateExample(
        language=language,
        word_id=word_id,
        wordform_id=wordform_id,
        gramset=gramset,
        word=word,
        word_length=len(word),
        left_word="<BOS>",
        right_word="<EOS>",
        features={} if features is None else features,
        label=label,
        sample_weight=sample_weight,
    )


def word_500_pairs():
    return [
        (make_example("krl", 500, 9, "SG"), 0.75),
        (make_example("vep", 200, 5, "PL"), 0.90),
        (make_example("krl", 500, 7, "SG"), 0.75),
        (make_example("vep", 200, 2, "SG"), 0.90),
        (make_example("krl", 500, 7, "PL"), 0.75),
        (make_example("vep", 200, 5, "SG"), 0.10),
        (make_example("krl", 500, 3, "GEN"), -0.40),
    ]


def expected_multiword_rows():
    return (
        PredictionRow(200, 2, "SG", 1, 0.90),
        PredictionRow(200, 5, "PL", 2, 0.90),
        PredictionRow(200, 5, "SG", 3, 0.10),
        PredictionRow(500, 7, "PL", 1, 0.75),
        PredictionRow(500, 7, "SG", 2, 0.75),
        PredictionRow(500, 9, "SG", 3, 0.75),
        PredictionRow(500, 3, "GEN", 4, -0.40),
    )


def split_pairs(pairs):
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


class TestExplicitResult:
    def test_multiword_complete_result(self):
        pairs = word_500_pairs()
        examples, scores = split_pairs(pairs)
        rows = rank_candidate_scores(examples, scores)
        assert rows == expected_multiword_rows()

    def test_rank_completeness_per_word(self):
        pairs = word_500_pairs()
        examples, scores = split_pairs(pairs)
        rows = rank_candidate_scores(examples, scores)
        for word_id in (200, 500):
            ranks = sorted(row.rank for row in rows if row.word_id == word_id)
            assert ranks == list(range(1, len(ranks) + 1))

    def test_both_tie_breaks(self):
        pairs = [
            (make_example("krl", 500, 9, "SG"), 0.75),
            (make_example("krl", 500, 7, "SG"), 0.75),
            (make_example("krl", 500, 7, "PL"), 0.75),
            (make_example("krl", 500, 3, "GEN"), -0.40),
        ]
        examples, scores = split_pairs(pairs)
        rows = rank_candidate_scores(examples, scores)
        assert rows == (
            PredictionRow(500, 7, "PL", 1, 0.75),
            PredictionRow(500, 7, "SG", 2, 0.75),
            PredictionRow(500, 9, "SG", 3, 0.75),
            PredictionRow(500, 3, "GEN", 4, -0.40),
        )

    def test_scores_are_python_floats(self):
        examples = [
            make_example("krl", 1, 1, "SG"),
            make_example("krl", 1, 2, "PL"),
        ]
        rows = rank_candidate_scores(examples, [1, 0])
        assert all(type(row.score) is float for row in rows)
        assert [row.score for row in rows] == [1.0, 0.0]

    def test_fractional_and_negative_preserved(self):
        examples = [
            make_example("krl", 1, 1, "SG"),
            make_example("krl", 1, 2, "PL"),
        ]
        rows = rank_candidate_scores(examples, [0.125, -2.5])
        assert [row.score for row in rows] == [0.125, -2.5]


class TestReordering:
    def test_reversed_pairs_preserve_result(self):
        pairs = word_500_pairs()
        examples, scores = split_pairs(pairs)
        first = rank_candidate_scores(examples, scores)
        reversed_examples, reversed_scores = split_pairs(list(reversed(pairs)))
        second = rank_candidate_scores(reversed_examples, reversed_scores)
        assert first == expected_multiword_rows()
        assert second == first


class TestCandidateDistinctness:
    def test_identical_gramsets_retained(self):
        examples = [
            make_example("krl", 1, 1, "SG"),
            make_example("krl", 1, 2, "SG"),
        ]
        rows = rank_candidate_scores(examples, [0.5, 0.5])
        assert rows == (
            PredictionRow(1, 1, "SG", 1, 0.5),
            PredictionRow(1, 2, "SG", 2, 0.5),
        )


class TestLengthMismatch:
    @pytest.mark.parametrize(
        ("examples", "scores"),
        [
            ([make_example("krl", 1, 1, "SG")], []),
            ([], [0.5]),
            (
                [make_example("krl", 1, 1, "SG"), make_example("krl", 1, 2, "PL")],
                [0.5],
            ),
            ([make_example("krl", 1, 1, "SG")], [0.5, 0.1]),
        ],
    )
    def test_unequal_lengths(self, examples, scores):
        with pytest.raises(ValueError) as exc:
            rank_candidate_scores(examples, scores)
        message = str(exc.value)
        assert str(len(examples)) in message
        assert str(len(scores)) in message


class TestNonFiniteScores:
    @pytest.mark.parametrize(
        "bad_score",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_rejected(self, bad_score):
        examples = [make_example("krl", 1, 1, "SG")]
        with pytest.raises(ValueError):
            rank_candidate_scores(examples, [bad_score])


class TestIntegrityErrors:
    def test_duplicate_candidate_identity(self):
        examples = [
            make_example("krl", 1, 7, "SG"),
            make_example("krl", 1, 7, "SG"),
        ]
        with pytest.raises(BenchmarkIntegrityError) as exc:
            rank_candidate_scores(examples, [0.5, 0.6])
        assert "(1, 7, 'SG')" in str(exc.value)

    def test_cross_language_word_id_collision(self):
        examples = [
            make_example("krl", 1, 7, "SG"),
            make_example("vep", 1, 8, "PL"),
        ]
        with pytest.raises(BenchmarkIntegrityError) as exc:
            rank_candidate_scores(examples, [0.5, 0.6])
        message = str(exc.value)
        assert "krl" in message and "vep" in message


class TestGeneratorsAndEmpty:
    def test_generator_inputs(self):
        pairs = word_500_pairs()
        examples = (pair[0] for pair in pairs)
        scores = (pair[1] for pair in pairs)
        assert rank_candidate_scores(examples, scores) == expected_multiword_rows()

    def test_both_empty(self):
        assert rank_candidate_scores([], []) == ()
        assert rank_candidate_scores(iter([]), iter([])) == ()


class TestInputsUnchanged:
    def test_collections_metadata_and_features_unchanged(self):
        features_a = {'["language","krl"]': 1.0}
        features_b = {'["language","krl"]': 1.0}
        example_a = make_example(
            "krl", 1, 7, "SG", word="a", label=1, sample_weight=0.5,
            features=features_a,
        )
        example_b = make_example(
            "krl", 1, 8, "PL", word="b", label=0, sample_weight=0.5,
            features=features_b,
        )
        examples = [example_a, example_b]
        scores = [0.4, 0.9]
        original_examples = list(examples)
        original_scores = list(scores)

        rank_candidate_scores(examples, scores)

        assert examples == original_examples
        assert scores == original_scores
        assert example_a.features is features_a
        assert example_b.features is features_b
        assert example_a.features == {'["language","krl"]': 1.0}
        assert example_b.features == {'["language","krl"]': 1.0}
        assert example_a.features is not example_b.features
        assert (example_a.label, example_a.sample_weight) == (1, 0.5)
        assert (example_b.label, example_b.sample_weight) == (0, 0.5)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"label": 1},
            {"sample_weight": 0.25},
            {"features": {'["gramset","SG"]': 1.0}},
        ],
    )
    def test_metadata_changes_do_not_affect_ranking(self, overrides):
        base = [
            make_example("krl", 1, 7, "SG"),
            make_example("krl", 1, 8, "PL"),
        ]
        changed = [
            make_example("krl", 1, 7, "SG", **overrides),
            make_example("krl", 1, 8, "PL", **overrides),
        ]
        scores = [0.4, 0.9]
        assert rank_candidate_scores(changed, scores) == rank_candidate_scores(
            base, scores
        )


class GuardedExample:
    _ALLOWED = frozenset({"language", "word_id", "wordform_id", "gramset"})

    def __init__(self, example):
        self._example = example

    def __getattr__(self, name):
        if name in self._ALLOWED:
            return getattr(self._example, name)
        raise AssertionError(f"forbidden attribute access: {name}")


class TestGuardedIdentityAccess:
    def test_ranking_uses_only_identity_fields(self):
        examples = [
            GuardedExample(example) for example, _ in word_500_pairs()
        ]
        _, scores = split_pairs(word_500_pairs())
        rows = rank_candidate_scores(examples, scores)
        assert rows == expected_multiword_rows()

    @pytest.mark.parametrize(
        "forbidden",
        ["word", "word_length", "left_word", "right_word", "features", "label", "sample_weight"],
    )
    def test_forbidden_fields_raise_assertion_error(self, forbidden):
        guarded = GuardedExample(make_example("krl", 1, 7, "SG"))
        with pytest.raises(AssertionError):
            getattr(guarded, forbidden)
