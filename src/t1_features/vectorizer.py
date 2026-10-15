from typing import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted

from t1_features.features import CandidateExample


def fit_feature_vectorizer(
    train_examples: Iterable[CandidateExample],
) -> tuple[DictVectorizer, csr_matrix]:
    examples = list(train_examples)
    if not examples:
        raise ValueError("train_examples must not be empty")
    feature_dicts = [example.features for example in examples]
    if not any(feature_dicts):
        raise ValueError("train_examples must contain at least one feature key")
    vectorizer = DictVectorizer(dtype=float, sparse=True, sort=True)
    matrix = vectorizer.fit_transform(feature_dicts)
    return vectorizer, matrix


def transform_candidate_examples(
    vectorizer: DictVectorizer,
    examples: Iterable[CandidateExample],
) -> csr_matrix:
    check_is_fitted(vectorizer, ["vocabulary_"])
    feature_dicts = [example.features for example in examples]
    if not feature_dicts:
        return csr_matrix((0, len(vectorizer.vocabulary_)), dtype=float)
    return vectorizer.transform(feature_dicts)