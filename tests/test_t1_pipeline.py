"""T1.1b4 acceptance: a real offline train/save/load/predict pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.instances import Candidate, Instance  # noqa: E402
from t1_features.features import ContextWord, build_candidate_examples  # noqa: E402
from t1_features.model import score_candidate_examples, train_candidate_model  # noqa: E402
from t1_features.model_io import load_model_bundle, save_model_bundle  # noqa: E402

UNKNOWN_KEY = '["suffix_gramset","SG","2","qz"]'

TRAIN_IDENTITIES = [
    ("krl", 10, 100, "SG"),
    ("krl", 10, 101, "PL"),
    ("krl", 20, 200, "SG"),
    ("krl", 20, 201, "PL"),
    ("krl", 20, 202, "GEN"),
]
TRAIN_LABELS = [1, 0, 0, 1, 0]
TRAIN_WEIGHTS = [1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3]


def _make_fixture():
    train_instances = (
        Instance(
            language="krl",
            word_id=10,
            sentence_id=100,
            text_id=1,
            word="a",
            word_number=2,
            sentence_xml="",
            candidates=(Candidate(100, "SG"), Candidate(101, "PL")),
            gold_analysis=Candidate(100, "SG"),
        ),
        Instance(
            language="krl",
            word_id=20,
            sentence_id=200,
            text_id=2,
            word="b",
            word_number=2,
            sentence_xml="",
            candidates=(
                Candidate(200, "SG"),
                Candidate(201, "PL"),
                Candidate(202, "GEN"),
            ),
            gold_analysis=Candidate(201, "PL"),
        ),
    )
    predict_instances = (
        Instance(
            language="krl",
            word_id=30,
            sentence_id=300,
            text_id=3,
            word="qz",
            word_number=2,
            sentence_xml="",
            candidates=(Candidate(100, "SG"), Candidate(101, "PL")),
            gold_analysis=Candidate(100, "SG"),
        ),
    )
    context_words = (
        ContextWord(language="krl", sentence_id=100, word_id=101, word_number=1, word="lw1"),
        ContextWord(language="krl", sentence_id=100, word_id=10, word_number=2, word="a"),
        ContextWord(language="krl", sentence_id=100, word_id=103, word_number=3, word="rw1"),
        ContextWord(language="krl", sentence_id=200, word_id=201, word_number=1, word="lw2"),
        ContextWord(language="krl", sentence_id=200, word_id=20, word_number=2, word="b"),
        ContextWord(language="krl", sentence_id=200, word_id=203, word_number=3, word="rw2"),
        ContextWord(language="krl", sentence_id=300, word_id=301, word_number=1, word="lw3"),
        ContextWord(language="krl", sentence_id=300, word_id=30, word_number=2, word="qz"),
        ContextWord(language="krl", sentence_id=300, word_id=303, word_number=3, word="rw3"),
    )
    return train_instances, predict_instances, context_words


def _identity(example):
    return (example.language, example.word_id, example.wordform_id, example.gramset)


class TestOfflinePipeline:
    def test_train_save_load_predict_round_trip(self, tmp_path):
        train_instances, predict_instances, context_words = _make_fixture()
        train_examples = build_candidate_examples(
            train_instances, context_words, mode="train"
        )
        predict_examples = build_candidate_examples(
            predict_instances, context_words, mode="predict"
        )

        assert [_identity(example) for example in train_examples] == TRAIN_IDENTITIES
        assert [example.label for example in train_examples] == TRAIN_LABELS
        np.testing.assert_allclose(
            [example.sample_weight for example in train_examples], TRAIN_WEIGHTS
        )
        np.testing.assert_allclose(
            sum(e.sample_weight for e in train_examples if e.word_id == 10), 1.0
        )
        np.testing.assert_allclose(
            sum(e.sample_weight for e in train_examples if e.word_id == 20), 1.0
        )
        assert {(e.word_id, e.left_word, e.right_word) for e in train_examples} == {
            (10, "lw1", "rw1"),
            (20, "lw2", "rw2"),
        }

        assert [_identity(example) for example in predict_examples] == [
            ("krl", 30, 100, "SG"),
            ("krl", 30, 101, "PL"),
        ]
        assert all(e.label is None and e.sample_weight is None for e in predict_examples)
        sg_example = next(e for e in predict_examples if e.gramset == "SG")
        assert UNKNOWN_KEY in sg_example.features
        assert {(e.left_word, e.right_word) for e in predict_examples} == {("lw3", "rw3")}

        model = train_candidate_model(train_examples)
        before = score_candidate_examples(model, predict_examples)
        assert before.shape == (2,)
        assert before.dtype == np.float64
        assert np.all(np.isfinite(before))
        assert UNKNOWN_KEY not in model.vectorizer.vocabulary_

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        path = bundle_dir / "model.bundle"
        save_model_bundle(model, path)
        assert sorted(p.name for p in bundle_dir.iterdir()) == ["model.bundle"]

        before_bytes = path.read_bytes()
        before_entries = sorted((p.name, p.stat().st_size) for p in bundle_dir.iterdir())

        loaded = load_model_bundle(path)
        after = score_candidate_examples(loaded, predict_examples)
        np.testing.assert_allclose(after, before)
        assert after.shape == (2,)
        assert after.dtype == np.float64
        assert np.all(np.isfinite(after))
        assert UNKNOWN_KEY not in loaded.vectorizer.vocabulary_

        assert path.read_bytes() == before_bytes
        assert sorted((p.name, p.stat().st_size) for p in bundle_dir.iterdir()) == (
            before_entries
        )

        assert dict(loaded.vectorizer.vocabulary_) == dict(
            model.vectorizer.vocabulary_
        )
        np.testing.assert_array_equal(
            loaded.vectorizer.get_feature_names_out(),
            model.vectorizer.get_feature_names_out(),
        )
        np.testing.assert_array_equal(
            loaded.classifier.classes_, model.classifier.classes_
        )
        np.testing.assert_allclose(loaded.classifier.coef_, model.classifier.coef_)
        np.testing.assert_allclose(
            loaded.classifier.intercept_, model.classifier.intercept_
        )

    def test_predict_order_gold_and_state_are_stable(self, tmp_path):
        train_instances, predict_instances, context_words = _make_fixture()
        train_examples = build_candidate_examples(
            train_instances, context_words, mode="train"
        )
        predict_examples = build_candidate_examples(
            predict_instances, context_words, mode="predict"
        )
        train_feature_copies = [dict(e.features) for e in train_examples]
        predict_feature_copies = [dict(e.features) for e in predict_examples]
        train_feature_ids = [id(e.features) for e in train_examples]
        predict_feature_ids = [id(e.features) for e in predict_examples]

        model = train_candidate_model(train_examples)
        path = tmp_path / "model.bundle"
        save_model_bundle(model, path)
        loaded = load_model_bundle(path)

        model_coef = model.classifier.coef_.copy()
        model_intercept = model.classifier.intercept_.copy()
        model_vocab = dict(model.vectorizer.vocabulary_)
        loaded_coef = loaded.classifier.coef_.copy()
        loaded_intercept = loaded.classifier.intercept_.copy()
        loaded_vocab = dict(loaded.vectorizer.vocabulary_)

        forward = score_candidate_examples(loaded, predict_examples)
        reversed_examples = predict_examples[::-1]
        assert [_identity(e) for e in reversed_examples] == [
            ("krl", 30, 101, "PL"),
            ("krl", 30, 100, "SG"),
        ]
        backward = score_candidate_examples(loaded, reversed_examples)
        np.testing.assert_allclose(backward, forward[::-1])

        altered_gold = Instance(
            language="krl",
            word_id=30,
            sentence_id=300,
            text_id=3,
            word="qz",
            word_number=2,
            sentence_xml="",
            candidates=(Candidate(100, "SG"), Candidate(101, "PL")),
            gold_analysis=Candidate(101, "PL"),
        )
        rebuilt = build_candidate_examples((altered_gold,), context_words, mode="predict")
        assert rebuilt == predict_examples
        np.testing.assert_allclose(score_candidate_examples(loaded, rebuilt), forward)

        train_instances_again, predict_instances_again, context_words_again = _make_fixture()
        assert train_instances == train_instances_again
        assert predict_instances == predict_instances_again
        assert context_words == context_words_again

        for example, copy, identity in zip(
            train_examples, train_feature_copies, train_feature_ids
        ):
            assert example.features == copy
            assert id(example.features) == identity
        for example, copy, identity in zip(
            predict_examples, predict_feature_copies, predict_feature_ids
        ):
            assert example.features == copy
            assert id(example.features) == identity

        np.testing.assert_array_equal(model.classifier.coef_, model_coef)
        np.testing.assert_array_equal(model.classifier.intercept_, model_intercept)
        assert dict(model.vectorizer.vocabulary_) == model_vocab
        np.testing.assert_array_equal(loaded.classifier.coef_, loaded_coef)
        np.testing.assert_array_equal(loaded.classifier.intercept_, loaded_intercept)
        assert dict(loaded.vectorizer.vocabulary_) == loaded_vocab