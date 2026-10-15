import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from t1_features.features import CandidateExample  # noqa: E402
from t1_features.vectorizer import (  # noqa: E402
    fit_feature_vectorizer,
    transform_candidate_examples,
)

FIRST_FEATURES = {
    '["gramset","SG"]': 1.0,
    '["language","krl"]': 1.0,
}
SECOND_FEATURES = {
    '["gramset","PL"]': 1.0,
    '["language","krl"]': 1.0,
}
EXPECTED_NAMES = [
    '["gramset","PL"]',
    '["gramset","SG"]',
    '["language","krl"]',
]


def make_example(features, *, language="krl", gramset="SG", label=1, weight=0.5):
    return CandidateExample(
        language=language,
        word_id=1,
        wordform_id=2,
        gramset=gramset,
        word="in",
        word_length=2,
        left_word="<BOS>",
        right_word="<EOS>",
        features=features,
        label=label,
        sample_weight=weight,
    )


def other_fields(example):
    return tuple(
        getattr(example, field.name)
        for field in fields(CandidateExample)
        if field.name != "features"
    )


class TestExactHandBuiltExample:
    def test_matrix_and_names(self):
        first = make_example(FIRST_FEATURES)
        second = make_example(SECOND_FEATURES)
        vectorizer, matrix = fit_feature_vectorizer([first, second])
        assert isinstance(matrix, csr_matrix)
        assert matrix.dtype == np.float64
        assert matrix.shape == (2, 3)
        assert matrix.nnz == 4
        assert vectorizer.get_feature_names_out().tolist() == EXPECTED_NAMES
        assert matrix.toarray().tolist() == [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]

    def test_unknown_feature_transform(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        unknown = make_example(
            {'["gramset","UNKNOWN"]': 1.0, '["language","krl"]': 1.0}
        )
        transformed = transform_candidate_examples(vectorizer, [unknown])
        assert isinstance(transformed, csr_matrix)
        assert transformed.dtype == np.float64
        assert transformed.toarray().tolist() == [[0.0, 0.0, 1.0]]


class TestOrdering:
    def test_reversed_train_rows_preserve_names(self):
        forward_vec, forward_mat = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        reverse_vec, reverse_mat = fit_feature_vectorizer(
            [make_example(SECOND_FEATURES), make_example(FIRST_FEATURES)]
        )
        assert (
            reverse_vec.get_feature_names_out().tolist()
            == forward_vec.get_feature_names_out().tolist()
            == EXPECTED_NAMES
        )
        assert reverse_mat.toarray().tolist() == [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]

    def test_reversed_transform_rows_reverse(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        transformed = transform_candidate_examples(
            vectorizer,
            [make_example(SECOND_FEATURES), make_example(FIRST_FEATURES)],
        )
        assert transformed.toarray().tolist() == [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]


class TestUnknownFeatures:
    def test_vocabulary_and_names_unchanged(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        names_before = vectorizer.get_feature_names_out().tolist()
        vocab_before = dict(vectorizer.vocabulary_)
        unknown = make_example(
            {'["gramset","UNKNOWN"]': 1.0, '["language","krl"]': 1.0}
        )
        transform_candidate_examples(vectorizer, [unknown])
        assert vectorizer.get_feature_names_out().tolist() == names_before
        assert dict(vectorizer.vocabulary_) == vocab_before

    def test_unknown_only_and_empty_rows_zero(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        transformed = transform_candidate_examples(
            vectorizer,
            [
                make_example({'["gramset","UNKNOWN"]': 1.0}),
                make_example({}),
            ],
        )
        assert transformed.toarray().tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        assert transformed.nnz == 0


class TestEmptyTransform:
    def test_empty_examples_shape_dtype_nnz(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(FIRST_FEATURES), make_example(SECOND_FEATURES)]
        )
        transformed = transform_candidate_examples(vectorizer, [])
        assert isinstance(transformed, csr_matrix)
        assert transformed.shape == (0, len(vectorizer.vocabulary_))
        assert transformed.dtype == np.float64
        assert transformed.nnz == 0


class TestFitErrors:
    def test_empty_train_raises(self):
        with pytest.raises(ValueError):
            fit_feature_vectorizer([])
        with pytest.raises(ValueError):
            fit_feature_vectorizer(iter([]))

    def test_all_empty_dictionaries_raise(self):
        with pytest.raises(ValueError):
            fit_feature_vectorizer([make_example({}), make_example({})])

    def test_mixed_empty_and_nonempty_accepted(self):
        vectorizer, matrix = fit_feature_vectorizer(
            [make_example({}), make_example(FIRST_FEATURES)]
        )
        assert isinstance(matrix, csr_matrix)
        assert matrix.dtype == np.float64
        assert matrix.shape == (2, 2)
        assert matrix.nnz == 2
        assert vectorizer.get_feature_names_out().tolist() == [
            '["gramset","SG"]',
            '["language","krl"]',
        ]
        assert matrix.toarray().tolist() == [
            [0.0, 0.0],
            [1.0, 1.0],
        ]


class TestNotFitted:
    def test_unfitted_nonempty_input(self):
        vectorizer = DictVectorizer()
        with pytest.raises(NotFittedError):
            transform_candidate_examples(vectorizer, [make_example(FIRST_FEATURES)])

    def test_unfitted_empty_input(self):
        vectorizer = DictVectorizer()
        with pytest.raises(NotFittedError):
            transform_candidate_examples(vectorizer, [])

    def test_unfitted_check_precedes_consumption(self):
        def raising_examples():
            raise AssertionError("examples must not be consumed")
            yield  # pragma: no cover

        vectorizer = DictVectorizer()
        with pytest.raises(NotFittedError):
            transform_candidate_examples(vectorizer, raising_examples())


class TestGeneratorsAndImmutability:
    def test_generator_inputs_and_unchanged_objects(self):
        first = make_example(dict(FIRST_FEATURES))
        second = make_example(dict(SECOND_FEATURES))
        first_snapshot = dict(first.features)
        second_snapshot = dict(second.features)
        first_snapshot_id = id(first.features)
        second_snapshot_id = id(second.features)
        first_other_fields = other_fields(first)
        second_other_fields = other_fields(second)
        vectorizer, matrix = fit_feature_vectorizer(e for e in [first, second])
        assert matrix.toarray().tolist() == [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
        assert first.features == first_snapshot
        assert second.features == second_snapshot
        assert id(first.features) == first_snapshot_id
        assert id(second.features) == second_snapshot_id
        transformed = transform_candidate_examples(
            vectorizer, (e for e in [first, second])
        )
        assert transformed.toarray().tolist() == [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
        assert first.features == first_snapshot
        assert second.features == second_snapshot
        assert id(first.features) == first_snapshot_id
        assert id(second.features) == second_snapshot_id
        assert other_fields(first) == first_other_fields
        assert other_fields(second) == second_other_fields

    def test_metadata_label_weight_do_not_matter(self):
        plain = make_example(dict(FIRST_FEATURES), label=1, weight=0.5)
        variant = make_example(
            dict(FIRST_FEATURES),
            language="vep",
            gramset="PL+GEN",
            label=0,
            weight=1.0,
        )
        vectorizer_a, matrix_a = fit_feature_vectorizer([plain])
        vectorizer_b, matrix_b = fit_feature_vectorizer([variant])
        assert (
            vectorizer_a.get_feature_names_out().tolist()
            == vectorizer_b.get_feature_names_out().tolist()
        )
        assert matrix_a.toarray().tolist() == matrix_b.toarray().tolist()
        transformed = transform_candidate_examples(vectorizer_a, [variant])
        assert transformed.toarray().tolist() == matrix_a.toarray().tolist()


class FeaturesOnlyExample:
    def __init__(self, example):
        self._example = example

    @property
    def features(self):
        return self._example.features

    def __getattr__(self, name):
        raise AssertionError(f"forbidden attribute access: {name}")


class TestFeaturesOnlyAccess:
    def test_wrapper_rejects_other_fields(self):
        wrapped = FeaturesOnlyExample(make_example(FIRST_FEATURES))
        assert wrapped.features == FIRST_FEATURES
        with pytest.raises(AssertionError):
            wrapped.label
        with pytest.raises(AssertionError):
            wrapped.sample_weight
        with pytest.raises(AssertionError):
            wrapped.gramset

    def test_both_apis_use_only_features(self):
        vectorizer, matrix = fit_feature_vectorizer(
            [
                FeaturesOnlyExample(make_example(FIRST_FEATURES)),
                FeaturesOnlyExample(make_example(SECOND_FEATURES)),
            ]
        )
        assert matrix.toarray().tolist() == [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
        transformed = transform_candidate_examples(
            vectorizer,
            [FeaturesOnlyExample(make_example(FIRST_FEATURES))],
        )
        assert transformed.toarray().tolist() == [[0.0, 1.0, 1.0]]
