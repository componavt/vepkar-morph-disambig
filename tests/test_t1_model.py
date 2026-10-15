import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from t1_features.features import CandidateExample  # noqa: E402
from t1_features.model import (  # noqa: E402
    T1Model,
    score_candidate_examples,
    train_candidate_model,
)
from t1_features.vectorizer import fit_feature_vectorizer, transform_candidate_examples  # noqa: E402

POS_FEATURES = {
    '["gramset","SG"]': 1.0,
    '["language","krl"]': 1.0,
}
NEG_FEATURES = {
    '["gramset","PL"]': 1.0,
    '["language","krl"]': 1.0,
}


def make_example(features, *, label=1, weight=0.5):
    return CandidateExample(
        language="krl",
        word_id=1,
        wordform_id=2,
        gramset="SG",
        word="in",
        word_length=2,
        left_word="<BOS>",
        right_word="<EOS>",
        features=features,
        label=label,
        sample_weight=weight,
    )


class TestTraining:
    def test_real_two_class_training(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        assert isinstance(model, T1Model)
        assert isinstance(model.vectorizer, DictVectorizer)
        assert isinstance(model.classifier, LogisticRegression)
        with pytest.raises(AttributeError):
            model.vectorizer = DictVectorizer()
        assert model.classifier.C == 1.0
        assert model.classifier.solver == "lbfgs"
        assert model.classifier.class_weight is None
        assert model.classifier.max_iter == 1000
        assert model.classifier.fit_intercept is True
        assert model.classifier.classes_.tolist() == [0, 1]
        assert model.classifier.coef_.shape[1] == len(model.vectorizer.vocabulary_)

    def test_fit_spy_verifies_inputs(self, monkeypatch):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        real_fit = LogisticRegression.fit
        captured = {}

        def spy_fit(self, X, y, sample_weight=None):
            captured["X"] = X
            captured["y"] = y
            captured["sample_weight"] = sample_weight
            return real_fit(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(LogisticRegression, "fit", spy_fit)
        train_candidate_model([pos, neg])
        X = captured["X"]
        assert isinstance(X, csr_matrix)
        assert X.dtype == np.float64
        assert X.toarray().tolist() == [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
        y = captured["y"]
        sample_weight = captured["sample_weight"]
        assert y.dtype == np.int64
        assert y.shape == (2,)
        np.testing.assert_array_equal(y, np.array([1, 0], dtype=np.int64))
        assert sample_weight.dtype == np.float64
        assert sample_weight.shape == (2,)
        np.testing.assert_array_equal(
            sample_weight, np.array([0.5, 1.0], dtype=np.float64)
        )

    @pytest.mark.parametrize(
        "examples",
        [
            [],
            [
                make_example(dict(POS_FEATURES), label=1),
                make_example(dict(POS_FEATURES), label=1),
            ],
            [
                make_example(dict(POS_FEATURES), label=None),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=2),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=1, weight=None),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=1, weight=0.0),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=1, weight=-1.0),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=1, weight=np.nan),
                make_example(dict(NEG_FEATURES), label=0),
            ],
            [
                make_example(dict(POS_FEATURES), label=1, weight=np.inf),
                make_example(dict(NEG_FEATURES), label=0),
            ],
        ],
    )
    def test_invalid_train_raises_value_error(self, examples):
        with pytest.raises(ValueError):
            train_candidate_model(examples)

    def test_custom_C_is_retained(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg], C=2.5)
        assert model.classifier.C == 2.5

    @pytest.mark.parametrize("C", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_invalid_C_raises(self, C):
        with pytest.raises(ValueError):
            train_candidate_model(
                [
                    make_example(dict(POS_FEATURES), label=1),
                    make_example(dict(NEG_FEATURES), label=0),
                ],
                C=C,
            )

    def test_C_validation_precedes_consumption(self):
        def raising_examples():
            raise AssertionError("examples must not be consumed")
            yield  # pragma: no cover

        with pytest.raises(ValueError):
            train_candidate_model(raising_examples(), C=0.0)

    def test_convergence_warning_raises(self, monkeypatch):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        real_fit = LogisticRegression.fit

        def warning_fit(self, X, y, sample_weight=None):
            warnings.warn("did not converge", ConvergenceWarning)
            return real_fit(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(LogisticRegression, "fit", warning_fit)
        with pytest.raises(ConvergenceWarning):
            train_candidate_model([pos, neg])


class TestScoring:
    def test_scores_match_direct_decision_function(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        examples = [neg, pos, neg]
        scores = score_candidate_examples(model, examples)
        X = transform_candidate_examples(model.vectorizer, examples)
        expected = model.classifier.decision_function(X)
        assert scores.dtype == np.float64
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))
        np.testing.assert_allclose(scores, expected)

    def test_separable_fixture_ranks_positive_above_negative(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=1.0)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        pos_score = score_candidate_examples(model, [pos])[0]
        neg_score = score_candidate_examples(model, [neg])[0]
        assert pos_score > neg_score

    def test_reordering_examples_reorders_scores(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        forward = score_candidate_examples(model, [pos, neg])
        backward = score_candidate_examples(model, [neg, pos])
        np.testing.assert_allclose(forward, backward[::-1])

    def test_unknown_features_leave_vocabulary_unchanged(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        vocab_before = dict(model.vectorizer.vocabulary_)
        names_before = model.vectorizer.get_feature_names_out().tolist()
        unknown = make_example({'["gramset","UNKNOWN"]': 1.0})
        scores = score_candidate_examples(model, [unknown])
        assert dict(model.vectorizer.vocabulary_) == vocab_before
        assert model.vectorizer.get_feature_names_out().tolist() == names_before
        assert scores.shape == (1,)
        assert np.all(np.isfinite(scores))

    def test_empty_scoring_skips_decision_function(self, monkeypatch):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])

        def boom(X):
            raise AssertionError("decision_function must not be called")

        monkeypatch.setattr(model.classifier, "decision_function", boom)
        scores = score_candidate_examples(model, [])
        assert scores.dtype == np.float64
        assert scores.shape == (0,)
        np.testing.assert_array_equal(scores, np.empty(0, dtype=np.float64))

    def test_unfitted_vectorizer_raises_before_consumption(self):
        def raising_examples():
            raise AssertionError("examples must not be consumed")
            yield  # pragma: no cover

        model = T1Model(vectorizer=DictVectorizer(), classifier=LogisticRegression())
        with pytest.raises(NotFittedError):
            score_candidate_examples(model, raising_examples())

    def test_unfitted_classifier_raises_before_consumption(self):
        def raising_examples():
            raise AssertionError("examples must not be consumed")
            yield  # pragma: no cover

        vectorizer, _ = fit_feature_vectorizer(
            [make_example(dict(POS_FEATURES)), make_example(dict(NEG_FEATURES))]
        )
        model = T1Model(vectorizer=vectorizer, classifier=LogisticRegression())
        with pytest.raises(NotFittedError):
            score_candidate_examples(model, raising_examples())

    def test_unfitted_components_raise_for_empty_input(self):
        model = T1Model(vectorizer=DictVectorizer(), classifier=LogisticRegression())
        with pytest.raises(NotFittedError):
            score_candidate_examples(model, [])

    def test_unfitted_classifier_raises_for_empty_input(self):
        vectorizer, _ = fit_feature_vectorizer(
            [make_example(dict(POS_FEATURES)), make_example(dict(NEG_FEATURES))]
        )
        model = T1Model(vectorizer=vectorizer, classifier=LogisticRegression())
        with pytest.raises(NotFittedError):
            score_candidate_examples(model, [])

    def test_generator_inputs_and_unchanged_model_state(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        pos_snapshot = dict(pos.features)
        neg_snapshot = dict(neg.features)
        pos_id = id(pos.features)
        neg_id = id(neg.features)
        model = train_candidate_model(e for e in [pos, neg])
        assert pos.features == pos_snapshot
        assert neg.features == neg_snapshot
        assert id(pos.features) == pos_id
        assert id(neg.features) == neg_id
        vocab_before = dict(model.vectorizer.vocabulary_)
        names_before = model.vectorizer.get_feature_names_out().tolist()
        classes_before = model.classifier.classes_.copy()
        coef_before = model.classifier.coef_.copy()
        intercept_before = model.classifier.intercept_.copy()
        scores = score_candidate_examples(model, (e for e in [pos, neg]))
        assert scores.shape == (2,)
        assert pos.features == pos_snapshot
        assert neg.features == neg_snapshot
        assert id(pos.features) == pos_id
        assert id(neg.features) == neg_id
        assert dict(model.vectorizer.vocabulary_) == vocab_before
        assert model.vectorizer.get_feature_names_out().tolist() == names_before
        np.testing.assert_array_equal(model.classifier.classes_, classes_before)
        np.testing.assert_array_equal(model.classifier.coef_, coef_before)
        np.testing.assert_array_equal(model.classifier.intercept_, intercept_before)

    def test_nonfinite_scores_raise(self, monkeypatch):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])

        def bad_decision_function(X):
            return np.array([np.nan, np.inf], dtype=np.float64)

        monkeypatch.setattr(model.classifier, "decision_function", bad_decision_function)
        with pytest.raises(ValueError):
            score_candidate_examples(model, [pos, neg])


class TrainFieldsOnlyExample:
    def __init__(self, example):
        self._example = example

    @property
    def features(self):
        return self._example.features

    @property
    def label(self):
        return self._example.label

    @property
    def sample_weight(self):
        return self._example.sample_weight

    def __getattr__(self, name):
        raise AssertionError(f"forbidden attribute access: {name}")


class FeaturesOnlyExample:
    def __init__(self, example):
        self._example = example

    @property
    def features(self):
        return self._example.features

    def __getattr__(self, name):
        raise AssertionError(f"forbidden attribute access: {name}")


class TestNarrowAccess:
    def test_training_uses_only_required_fields(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model(
            [TrainFieldsOnlyExample(pos), TrainFieldsOnlyExample(neg)]
        )
        assert model.classifier.classes_.tolist() == [0, 1]

    def test_scoring_uses_only_features(self):
        pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
        neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        model = train_candidate_model([pos, neg])
        scores = score_candidate_examples(
            model, [FeaturesOnlyExample(pos), FeaturesOnlyExample(neg)]
        )
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
