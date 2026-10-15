import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from t1_features.features import CandidateExample
from t1_features.vectorizer import fit_feature_vectorizer, transform_candidate_examples


@dataclass(frozen=True)
class T1Model:
    vectorizer: DictVectorizer
    classifier: LogisticRegression


def train_candidate_model(
    train_examples: Iterable[CandidateExample],
    *,
    C: float = 1.0,
) -> T1Model:
    if not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    examples = list(train_examples)
    if not examples:
        raise ValueError("train_examples must not be empty")
    labels: list[int] = []
    sample_weights: list[float] = []
    for example in examples:
        if example.label is None or example.label not in (0, 1):
            raise ValueError("all train labels must be 0 or 1")
        if (
            example.sample_weight is None
            or not np.isfinite(example.sample_weight)
            or example.sample_weight <= 0
        ):
            raise ValueError("all train sample weights must be finite and positive")
        labels.append(example.label)
        sample_weights.append(example.sample_weight)
    if set(labels) != {0, 1}:
        raise ValueError("train data must contain both classes")
    vectorizer, X = fit_feature_vectorizer(examples)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    classifier = LogisticRegression(
        C=C,
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        classifier.fit(X, y, sample_weight=weights)
    return T1Model(vectorizer=vectorizer, classifier=classifier)


def score_candidate_examples(
    model: T1Model,
    examples: Iterable[CandidateExample],
) -> np.ndarray:
    check_is_fitted(model.vectorizer, ["vocabulary_"])
    check_is_fitted(model.classifier)
    materialized = list(examples)
    if not materialized:
        return np.empty(0, dtype=np.float64)
    X = transform_candidate_examples(model.vectorizer, materialized)
    scores = model.classifier.decision_function(X)
    if not np.all(np.isfinite(scores)):
        raise ValueError("candidate scores must be finite")
    return scores
