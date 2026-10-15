"""Offline tests for saving and loading T1Model bundles."""

import copy
import os
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
import scipy
import sklearn
from sklearn.exceptions import InconsistentVersionWarning, NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from t1_features.features import CandidateExample  # noqa: E402
from t1_features.model import T1Model, score_candidate_examples, train_candidate_model  # noqa: E402
import t1_features.model_io as model_io  # noqa: E402
from t1_features.model_io import (  # noqa: E402
    ModelBundleError,
    ModelBundlePublishedCleanupError,
    load_model_bundle,
    save_model_bundle,
)

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


@pytest.fixture
def trained_model():
    pos = make_example(dict(POS_FEATURES), label=1, weight=0.5)
    neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
    return train_candidate_model([pos, neg])


def dump_payload(tmp_path, name, payload):
    path = tmp_path / name
    with path.open("wb") as fh:
        joblib.dump(payload, fh, compress=0, protocol=5)
    return path


class ClosingFailureStream:
    """Wraps a real binary stream, delegating writes; ``close`` closes the
    underlying stream first and then raises a controlled ``OSError``."""

    def __init__(self, stream, close_error):
        self._stream = stream
        self._close_error = close_error

    def write(self, data):
        return self._stream.write(data)

    def close(self):
        self._stream.close()
        raise self._close_error

    def __getattr__(self, name):
        return getattr(self._stream, name)


class TestRoundTrip:
    def test_save_load_round_trip(self, tmp_path, trained_model):
        path = tmp_path / "model.bundle"
        save_model_bundle(trained_model, path)
        loaded = load_model_bundle(path)
        assert isinstance(loaded, T1Model)
        assert loaded is not trained_model
        assert dict(loaded.vectorizer.vocabulary_) == dict(
            trained_model.vectorizer.vocabulary_
        )
        np.testing.assert_array_equal(
            loaded.vectorizer.get_feature_names_out(),
            trained_model.vectorizer.get_feature_names_out(),
        )
        np.testing.assert_array_equal(
            loaded.classifier.classes_, trained_model.classifier.classes_
        )
        assert loaded.classifier.classes_.tolist() == [0, 1]
        assert loaded.classifier.C == trained_model.classifier.C
        np.testing.assert_allclose(
            loaded.classifier.coef_, trained_model.classifier.coef_
        )
        np.testing.assert_allclose(
            loaded.classifier.intercept_, trained_model.classifier.intercept_
        )
        examples = [
            make_example(dict(POS_FEATURES)),
            make_example({'["gramset","UNKNOWN"]': 1.0}),
            make_example(dict(NEG_FEATURES)),
        ]
        np.testing.assert_allclose(
            score_candidate_examples(loaded, examples),
            score_candidate_examples(trained_model, examples),
        )

    def test_independent_joblib_read_of_payload(self, tmp_path, trained_model):
        path = tmp_path / "model.bundle"
        save_model_bundle(trained_model, path)
        with path.open("rb") as fh:
            payload = joblib.load(fh)
        assert list(payload) == [
            "format",
            "format_version",
            "feature_schema_version",
            "environment",
            "model",
        ]
        assert payload["format"] == "vepkar-t1-model"
        assert payload["format_version"] == 1
        assert payload["feature_schema_version"] == 1
        environment = payload["environment"]
        assert list(environment) == ["python", "numpy", "scipy", "scikit-learn", "joblib"]
        assert environment["python"] == platform.python_version()
        assert environment["numpy"] == np.__version__
        assert environment["scipy"] == scipy.__version__
        assert environment["scikit-learn"] == sklearn.__version__
        assert environment["joblib"] == joblib.__version__
        assert all(isinstance(value, str) for value in environment.values())
        assert isinstance(payload["model"], T1Model)


class TestLoadDoesNotModify:
    def test_loading_never_fits_and_keeps_files_unchanged(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        save_model_bundle(trained_model, path)
        before_bytes = path.read_bytes()
        before_names = sorted(p.name for p in tmp_path.iterdir())

        def fail_fit(self, *args, **kwargs):
            raise AssertionError("fit must not be called during load")

        monkeypatch.setattr(LogisticRegression, "fit", fail_fit)
        loaded = load_model_bundle(path)
        assert isinstance(loaded, T1Model)
        assert path.read_bytes() == before_bytes
        assert sorted(p.name for p in tmp_path.iterdir()) == before_names


class TestSaveValidation:
    def test_unfitted_components_rejected_without_files(self, tmp_path):
        path = tmp_path / "model.bundle"
        model = T1Model(vectorizer=DictVectorizer(), classifier=LogisticRegression())
        with pytest.raises(ModelBundleError):
            save_model_bundle(model, path)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("which", ["vectorizer", "classifier"])
    def test_partially_unfitted_rejected(self, tmp_path, trained_model, which):
        if which == "vectorizer":
            model = T1Model(
                vectorizer=DictVectorizer(), classifier=trained_model.classifier
            )
        else:
            model = T1Model(
                vectorizer=trained_model.vectorizer, classifier=LogisticRegression()
            )
        path = tmp_path / "model.bundle"
        with pytest.raises(ModelBundleError):
            save_model_bundle(model, path)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_inconsistent_feature_count_rejected(self, tmp_path, trained_model):
        wider_pos = make_example(
            {**POS_FEATURES, '["extra","x"]': 1.0}, label=1, weight=0.5
        )
        wider_neg = make_example(dict(NEG_FEATURES), label=0, weight=1.0)
        wider = train_candidate_model([wider_pos, wider_neg])
        inconsistent = T1Model(
            vectorizer=trained_model.vectorizer, classifier=wider.classifier
        )
        path = tmp_path / "model.bundle"
        with pytest.raises(ModelBundleError):
            save_model_bundle(inconsistent, path)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_wrong_component_types_rejected(self, tmp_path):
        path = tmp_path / "model.bundle"
        with pytest.raises(ModelBundleError):
            save_model_bundle(object(), path)
        assert not path.exists()

    def test_reversed_class_order_rejected_at_save(self, tmp_path, trained_model):
        path = tmp_path / "model.bundle"
        model = T1Model(
            vectorizer=trained_model.vectorizer,
            classifier=copy.deepcopy(trained_model.classifier),
        )
        model.classifier.classes_ = np.array([1, 0])
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(model, path)
        assert "got [1, 0]" in str(exc_info.value)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


MISSING_FITTED_ATTRIBUTES = [
    ("vectorizer", "vocabulary_"),
    ("vectorizer", "feature_names_"),
    ("classifier", "classes_"),
    ("classifier", "n_features_in_"),
    ("classifier", "coef_"),
    ("classifier", "intercept_"),
]


class TestMissingFittedAttribute:
    @pytest.mark.parametrize("component,attribute", MISSING_FITTED_ATTRIBUTES)
    def test_save_rejects_missing_fitted_attribute(
        self, tmp_path, trained_model, component, attribute
    ):
        damaged = copy.deepcopy(trained_model)
        delattr(getattr(damaged, component), attribute)
        path = tmp_path / "model.bundle"
        with pytest.raises(ModelBundleError) as caught:
            save_model_bundle(damaged, path)
        assert isinstance(caught.value.__cause__, NotFittedError)
        assert "unfitted" in str(caught.value)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("component,attribute", MISSING_FITTED_ATTRIBUTES)
    def test_load_rejects_missing_fitted_attribute(
        self, tmp_path, trained_model, component, attribute
    ):
        saved = tmp_path / "saved.bundle"
        save_model_bundle(trained_model, saved)
        with saved.open("rb") as fh:
            base = joblib.load(fh)
        damaged = copy.deepcopy(trained_model)
        delattr(getattr(damaged, component), attribute)
        payload = dict(base)
        payload["model"] = damaged
        path = dump_payload(tmp_path, "damaged.bundle", payload)
        before_bytes = path.read_bytes()
        before_entries = sorted(p.name for p in tmp_path.iterdir())
        with pytest.raises(ModelBundleError) as caught:
            load_model_bundle(path)
        assert str(path) in str(caught.value)
        assert "unfitted" in str(caught.value)
        cause = caught.value.__cause__
        while cause is not None:
            if isinstance(cause, NotFittedError):
                break
            cause = cause.__cause__
        else:
            raise AssertionError("NotFittedError not in exception chain")
        assert path.read_bytes() == before_bytes
        assert sorted(p.name for p in tmp_path.iterdir()) == before_entries


def _without_key(base, key):
    payload = dict(base)
    del payload[key]
    return payload


def _with_extra_key(base):
    payload = dict(base)
    payload["extra"] = 1
    return payload


def _wrong_format(base):
    payload = dict(base)
    payload["format"] = "other-format"
    return payload


def _bad_version(base, key):
    payload = dict(base)
    payload[key] = 2
    return payload


def _float_version(base, key):
    payload = dict(base)
    payload[key] = 1.0
    return payload


def _env_missing_key(base):
    payload = dict(base)
    environment = dict(base["environment"])
    del environment["numpy"]
    payload["environment"] = environment
    return payload


def _env_extra_key(base):
    payload = dict(base)
    environment = dict(base["environment"])
    environment["extra"] = "x"
    payload["environment"] = environment
    return payload


def _env_non_string(base):
    payload = dict(base)
    environment = dict(base["environment"])
    environment["scipy"] = 1.18
    payload["environment"] = environment
    return payload


def _env_wrong_version(base):
    payload = dict(base)
    environment = dict(base["environment"])
    environment["scikit-learn"] = "0.0.0"
    payload["environment"] = environment
    return payload


def _not_a_dict(base):
    return ["not", "a", "dict"]


def _payload_with_numeric_key(base):
    payload = dict(base)
    payload[42] = "x"
    return payload


def _env_with_numeric_key(base):
    payload = dict(base)
    environment = dict(base["environment"])
    environment[42] = "x"
    payload["environment"] = environment
    return payload


class TestLoadValidation:
    @pytest.mark.parametrize(
        ("name", "mutator"),
        [
            ("not-a-dict", _not_a_dict),
            ("missing-model-key", lambda b: _without_key(b, "model")),
            ("missing-environment-key", lambda b: _without_key(b, "environment")),
            ("extra-top-level-key", _with_extra_key),
            ("wrong-format", _wrong_format),
            ("format-version-2", lambda b: _bad_version(b, "format_version")),
            ("feature-version-2", lambda b: _bad_version(b, "feature_schema_version")),
            ("format-version-float", lambda b: _float_version(b, "format_version")),
            ("feature-version-float", lambda b: _float_version(b, "feature_schema_version")),
            ("env-missing-key", _env_missing_key),
            ("env-extra-key", _env_extra_key),
            ("env-non-string", _env_non_string),
            ("env-wrong-version", _env_wrong_version),
            ("top-level-numeric-key", _payload_with_numeric_key),
            ("environment-numeric-key", _env_with_numeric_key),
        ],
    )
    def test_invalid_payload_rejected(self, tmp_path, trained_model, name, mutator):
        saved = tmp_path / "saved.bundle"
        save_model_bundle(trained_model, saved)
        with saved.open("rb") as fh:
            base = joblib.load(fh)
        payload = mutator(base)
        path = dump_payload(tmp_path, f"{name}.bundle", payload)
        with pytest.raises(ModelBundleError):
            load_model_bundle(path)

    def test_load_rejects_reversed_class_order(self, tmp_path, trained_model):
        model = T1Model(
            vectorizer=trained_model.vectorizer,
            classifier=copy.deepcopy(trained_model.classifier),
        )
        model.classifier.classes_ = np.array([1, 0])
        payload = {
            "format": "vepkar-t1-model",
            "format_version": 1,
            "feature_schema_version": 1,
            "environment": model_io._current_environment(),
            "model": model,
        }
        path = dump_payload(tmp_path, "reversed.bundle", payload)
        with pytest.raises(ModelBundleError) as exc_info:
            load_model_bundle(path)
        assert str(path) in str(exc_info.value)
        assert "got [1, 0]" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("name", "mutator"),
        [
            ("top-level-numeric-key", _payload_with_numeric_key),
            ("environment-numeric-key", _env_with_numeric_key),
        ],
    )
    def test_mixed_key_types_report_path_and_key(
        self, tmp_path, trained_model, name, mutator
    ):
        saved = tmp_path / "saved.bundle"
        save_model_bundle(trained_model, saved)
        with saved.open("rb") as fh:
            base = joblib.load(fh)
        payload = mutator(base)
        path = dump_payload(tmp_path, f"{name}.bundle", payload)
        with pytest.raises(ModelBundleError) as exc_info:
            load_model_bundle(path)
        message = str(exc_info.value)
        assert str(path) in message
        assert "42" in message


class TestLoadErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ModelBundleError):
            load_model_bundle(tmp_path / "missing.bundle")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.bundle"
        path.write_bytes(b"")
        with pytest.raises(ModelBundleError):
            load_model_bundle(path)

    def test_deterministic_invalid_pickle(self, tmp_path):
        path = tmp_path / "bad.bundle"
        path.write_bytes(b"\x80\x05")
        with pytest.raises(ModelBundleError):
            load_model_bundle(path)

    def test_controlled_inconsistent_version_warning(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        save_model_bundle(trained_model, path)

        def warning_load(stream):
            warnings.warn(
                InconsistentVersionWarning(
                    estimator_name="LogisticRegression",
                    current_sklearn_version=sklearn.__version__,
                    original_sklearn_version="0.0.0",
                )
            )
            return {"format": "unreachable"}

        monkeypatch.setattr(model_io.joblib, "load", warning_load)
        filters_before = list(warnings.filters)
        with pytest.raises(ModelBundleError) as exc_info:
            load_model_bundle(path)
        assert str(path) in str(exc_info.value)
        assert warnings.filters == filters_before


class TestDestinationProtection:
    def test_existing_file_unchanged(self, tmp_path, trained_model):
        path = tmp_path / "model.bundle"
        path.write_bytes(b"precious data")
        with pytest.raises(ModelBundleError):
            save_model_bundle(trained_model, path)
        assert path.read_bytes() == b"precious data"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.bundle"]

    def test_dangling_symlink_unchanged(self, tmp_path, trained_model):
        path = tmp_path / "model.bundle"
        path.symlink_to(tmp_path / "missing-target")
        with pytest.raises(ModelBundleError):
            save_model_bundle(trained_model, path)
        assert path.is_symlink()
        assert not (tmp_path / "missing-target").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.bundle"]

    def test_missing_parent_not_created(self, tmp_path, trained_model):
        path = tmp_path / "missing" / "model.bundle"
        with pytest.raises(ModelBundleError):
            save_model_bundle(trained_model, path)
        assert not (tmp_path / "missing").exists()


class TestSaveFailureCleanup:
    def test_dump_failure_cleans_temporary_and_final(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"

        def failing_dump(payload, stream, **kwargs):
            stream.write(b"partial bytes")
            raise OSError("simulated disk full")

        monkeypatch.setattr(model_io.joblib, "dump", failing_dump)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error)
        assert not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_competing_link_keeps_foreign_bytes(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        original_link = os.link

        def competing_link(source, destination):
            with open(destination, "wb") as fh:
                fh.write(b"foreign bytes")
            return original_link(source, destination)

        monkeypatch.setattr(model_io.os, "link", competing_link)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert path.read_bytes() == b"foreign bytes"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.bundle"]

    def test_temporary_creation_failure(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"

        def failing_mkstemp(**kwargs):
            raise OSError("simulated temp creation failure")

        monkeypatch.setattr(model_io.tempfile, "mkstemp", failing_mkstemp)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_open_failure_closes_descriptor_and_cleans_temp(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        real_mkstemp = model_io.tempfile.mkstemp
        real_close = os.close
        descriptor, name = real_mkstemp(
            prefix=".vepkar-t1-model-", suffix=".tmp", dir=str(tmp_path)
        )
        closed = []

        def failing_fdopen(*args, **kwargs):
            raise OSError("simulated fdopen failure")

        def record_close(fd):
            closed.append(fd)
            real_close(fd)

        monkeypatch.setattr(model_io.tempfile, "mkstemp", lambda **kwargs: (descriptor, name))
        monkeypatch.setattr(model_io.os, "fdopen", failing_fdopen)
        monkeypatch.setattr(model_io.os, "close", record_close)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert descriptor in closed
        assert not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_dump_error_close_error_preserves_dump_cause(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        real_fdopen = os.fdopen
        link_calls = []

        def failing_dump(payload, stream, **kwargs):
            stream.write(b"partial bytes")
            raise OSError("simulated disk full")

        def closing_fail_fdopen(descriptor, *args, **kwargs):
            return ClosingFailureStream(
                real_fdopen(descriptor, *args, **kwargs),
                OSError("simulated close failure"),
            )

        def recording_link(source, destination):
            link_calls.append(Path(source))
            return os.link(source, destination)

        monkeypatch.setattr(model_io.joblib, "dump", failing_dump)
        monkeypatch.setattr(model_io.os, "fdopen", closing_fail_fdopen)
        monkeypatch.setattr(model_io.os, "link", recording_link)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated disk full" in str(error.__cause__)
        assert any("simulated close failure" in note for note in error.__notes__)
        assert link_calls == []
        assert not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_close_error_after_successful_dump(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        real_fdopen = os.fdopen
        link_calls = []

        def closing_fail_fdopen(descriptor, *args, **kwargs):
            return ClosingFailureStream(
                real_fdopen(descriptor, *args, **kwargs),
                OSError("simulated close failure"),
            )

        def recording_link(source, destination):
            link_calls.append(Path(source))
            return os.link(source, destination)

        monkeypatch.setattr(model_io.os, "fdopen", closing_fail_fdopen)
        monkeypatch.setattr(model_io.os, "link", recording_link)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert "simulated close failure" in str(error.__cause__)
        assert link_calls == []
        assert not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_primary_error_with_unlink_failure(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        original_unlink = Path.unlink
        original_link = os.link
        link_calls = []

        def failing_dump(payload, stream, **kwargs):
            stream.write(b"partial bytes")
            raise OSError("simulated disk full")

        def recording_link(source, destination):
            link_calls.append(Path(source))
            return original_link(source, destination)

        def failing_unlink(self, *args, **kwargs):
            raise OSError("simulated unlink failure")

        monkeypatch.setattr(model_io.joblib, "dump", failing_dump)
        monkeypatch.setattr(model_io.os, "link", recording_link)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(ModelBundleError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert str(path) in str(error)
        assert isinstance(error.__cause__, OSError)
        assert any("simulated unlink failure" in note for note in error.__notes__)
        assert link_calls == []
        assert not path.exists()
        temp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert len(temp_files) == 1
        for temp in temp_files:
            original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_non_oserror_primary_survives_close_failure(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        real_fdopen = os.fdopen

        def failing_dump(payload, stream, **kwargs):
            raise ValueError("simulated serialization error")

        def closing_fail_fdopen(descriptor, *args, **kwargs):
            return ClosingFailureStream(
                real_fdopen(descriptor, *args, **kwargs),
                OSError("simulated close failure"),
            )

        monkeypatch.setattr(model_io.joblib, "dump", failing_dump)
        monkeypatch.setattr(model_io.os, "fdopen", closing_fail_fdopen)
        with pytest.raises(ValueError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert "simulated serialization error" in str(error)
        assert any("simulated close failure" in note for note in error.__notes__)
        assert not path.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_post_publication_unlink_failure(self, tmp_path, trained_model, monkeypatch):
        path = tmp_path / "model.bundle"
        original_link = os.link
        original_unlink = Path.unlink
        published_temps = []

        def recording_link(source, destination):
            published_temps.append(Path(source))
            return original_link(source, destination)

        def failing_unlink(self, *args, **kwargs):
            if self in published_temps:
                raise OSError("simulated permission denied")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(model_io.os, "link", recording_link)
        monkeypatch.setattr(Path, "unlink", failing_unlink)
        with pytest.raises(ModelBundlePublishedCleanupError) as exc_info:
            save_model_bundle(trained_model, path)
        error = exc_info.value
        assert "published" in str(error).lower()
        assert str(path) in str(error)
        assert len(published_temps) == 1
        assert str(published_temps[0]) in str(error)
        assert error.__cause__ is not None
        loaded = load_model_bundle(path)
        assert isinstance(loaded, T1Model)
        for temp in published_temps:
            if temp.exists():
                original_unlink(temp)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.bundle"]