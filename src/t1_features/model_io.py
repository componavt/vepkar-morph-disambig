"""Persist a trained :class:`t1_features.model.T1Model` as a single bundle.

The bundle is a single binary stream written with ``joblib.dump`` containing a
small fixed payload: a format identifier, integer schema versions, the current
runtime environment, and the model itself.  Classifier parameters are already
stored inside the model; nothing else (examples, matrices, labels, weights,
corpus, metrics, timestamps, or Git metadata) is recorded.

Saving validates the model and the destination, writes to a temporary sibling
file, and publishes it with a hard link so an existing destination is never
overwritten.  Loading is intended only for trusted, verified local artifacts.
"""

import os
import pickle
import platform
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import InconsistentVersionWarning, NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from t1_features.model import T1Model

_FORMAT_NAME = "vepkar-t1-model"
_FORMAT_VERSION = 1
_FEATURE_SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = ("format", "format_version", "feature_schema_version", "environment", "model")
_ENVIRONMENT_KEYS = ("python", "numpy", "scipy", "scikit-learn", "joblib")
_TEMP_PREFIX = ".vepkar-t1-model-"
_TEMP_SUFFIX = ".tmp"


class ModelBundleError(ValueError):
    """A model bundle could not be validated, saved, or loaded."""


class ModelBundlePublishedCleanupError(ModelBundleError):
    """The bundle was published, but removing its temporary name failed."""

    def __init__(self, path: Path, temp_path: Path):
        self.path = path
        self.temp_path = temp_path
        super().__init__(
            f"model bundle published at {path} but removal of temporary "
            f"{temp_path} failed"
        )


def _current_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _validate_model(model: T1Model) -> None:
    if not isinstance(model, T1Model):
        raise ModelBundleError(
            f"model must be a T1Model instance, got {type(model).__name__}"
        )
    vectorizer = model.vectorizer
    classifier = model.classifier
    if not isinstance(vectorizer, DictVectorizer):
        raise ModelBundleError(
            f"model.vectorizer must be a DictVectorizer, got {type(vectorizer).__name__}"
        )
    if not isinstance(classifier, LogisticRegression):
        raise ModelBundleError(
            f"model.classifier must be a LogisticRegression, got {type(classifier).__name__}"
        )
    try:
        check_is_fitted(vectorizer, ["vocabulary_", "feature_names_"])
        check_is_fitted(
            classifier,
            ["classes_", "n_features_in_", "coef_", "intercept_"],
        )
    except NotFittedError as exc:
        raise ModelBundleError("model contains unfitted components") from exc
    vocabulary = vectorizer.vocabulary_
    n_features = len(vocabulary)
    if n_features == 0:
        raise ModelBundleError("model has no features")
    try:
        feature_names = vectorizer.get_feature_names_out()
    except NotFittedError as exc:
        raise ModelBundleError("model vectorizer is unfitted") from exc
    if len(feature_names) != n_features:
        raise ModelBundleError(
            f"feature-name count {len(feature_names)} does not match "
            f"vocabulary size {n_features}"
        )
    classes = classifier.classes_
    if classes.shape != (2,) or classes.tolist() != [0, 1]:
        raise ModelBundleError(
            f"classifier classes must be exactly [0, 1], got {classes.tolist()}"
        )
    if classifier.n_features_in_ != n_features:
        raise ModelBundleError(
            f"classifier expects {classifier.n_features_in_} features but "
            f"vectorizer has {n_features}"
        )
    if classifier.coef_.shape != (1, n_features):
        raise ModelBundleError(
            f"classifier coefficient shape must be (1, {n_features}), "
            f"got {classifier.coef_.shape}"
        )
    if classifier.intercept_.shape != (1,):
        raise ModelBundleError(
            f"classifier intercept shape must be (1,), got {classifier.intercept_.shape}"
        )
    if not np.all(np.isfinite(classifier.coef_)) or not np.all(
        np.isfinite(classifier.intercept_)
    ):
        raise ModelBundleError("classifier coefficients and intercept must be finite")


def _validate_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ModelBundleError(
            f"payload must be a dictionary, got {type(payload).__name__}"
        )
    if set(payload) != set(_TOP_LEVEL_KEYS):
        raise ModelBundleError(
            f"payload keys must be exactly {sorted(_TOP_LEVEL_KEYS)}, "
            f"got {sorted(repr(key) for key in payload)}"
        )
    if payload["format"] != _FORMAT_NAME:
        raise ModelBundleError(f"unsupported format {payload['format']!r}")
    for key in ("format_version", "feature_schema_version"):
        value = payload[key]
        if type(value) is not int or value != 1:
            raise ModelBundleError(f"{key} must be integer 1, got {value!r}")
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ModelBundleError(
            f"environment must be a dictionary, got {type(environment).__name__}"
        )
    if set(environment) != set(_ENVIRONMENT_KEYS):
        raise ModelBundleError(
            f"environment keys must be exactly {sorted(_ENVIRONMENT_KEYS)}, "
            f"got {sorted(repr(key) for key in environment)}"
        )
    expected_versions = _current_environment()
    for key in _ENVIRONMENT_KEYS:
        value = environment[key]
        if not isinstance(value, str):
            raise ModelBundleError(
                f"environment {key!r} must be a string, got {type(value).__name__}"
            )
        if value != expected_versions[key]:
            raise ModelBundleError(
                f"environment {key!r} recorded {value!r} but current is "
                f"{expected_versions[key]!r}"
            )
    _validate_model(payload["model"])


def save_model_bundle(model: T1Model, path: Path) -> None:
    """Validate ``model`` and write it to ``path`` without overwriting anything.

    The destination must not already exist (files, directories, and symlinks
    including broken ones are all rejected) and its parent directory must
    already exist; nothing is created besides the bundle itself.  The bundle is
    written to a temporary sibling file that is then hard-linked into place, so
    an existing destination is never overwritten or replaced.  A successful
    hard link publishes the output; a later failure to remove the temporary
    file raises :class:`ModelBundlePublishedCleanupError` and never touches the
    published output.
    """
    if path.is_symlink() or path.exists():
        raise ModelBundleError(f"destination already exists: {path}")
    parent = path.parent
    if not parent.is_dir():
        raise ModelBundleError(f"parent directory does not exist: {parent}")
    _validate_model(model)
    payload = {
        "format": _FORMAT_NAME,
        "format_version": _FORMAT_VERSION,
        "feature_schema_version": _FEATURE_SCHEMA_VERSION,
        "environment": _current_environment(),
        "model": model,
    }
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=_TEMP_PREFIX, suffix=_TEMP_SUFFIX, dir=str(parent)
        )
    except OSError as exc:
        raise ModelBundleError(
            f"cannot save model bundle {path}: create temporary file: {exc}"
        ) from exc
    temp_path = Path(temp_name)
    try:
        try:
            stream = os.fdopen(descriptor, "wb")
        except OSError as exc:
            try:
                os.close(descriptor)
            except OSError as close_exc:
                wrapped = ModelBundleError(
                    f"cannot save model bundle {path}: open temporary stream "
                    f"{temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary model bundle descriptor: {close_exc}"
                )
                raise wrapped from exc
            raise ModelBundleError(
                f"cannot save model bundle {path}: open temporary stream "
                f"{temp_path}: {exc}"
            ) from exc
        try:
            joblib.dump(payload, stream, compress=0, protocol=5)
        except OSError as exc:
            try:
                stream.close()
            except OSError as close_exc:
                wrapped = ModelBundleError(
                    f"cannot save model bundle {path}: write temporary stream "
                    f"{temp_path}: {exc}"
                )
                wrapped.add_note(
                    f"failed to close temporary model bundle stream {temp_path}: "
                    f"{close_exc}"
                )
                raise wrapped from exc
            raise ModelBundleError(
                f"cannot save model bundle {path}: write temporary stream "
                f"{temp_path}: {exc}"
            ) from exc
        except BaseException as primary:
            try:
                stream.close()
            except OSError as close_exc:
                primary.add_note(
                    f"failed to close temporary model bundle stream {temp_path}: "
                    f"{close_exc}"
                )
            raise
        try:
            stream.close()
        except OSError as exc:
            raise ModelBundleError(
                f"cannot save model bundle {path}: close temporary stream "
                f"{temp_path}: {exc}"
            ) from exc
        try:
            os.link(temp_path, path)
        except OSError as exc:
            raise ModelBundleError(
                f"cannot save model bundle {path}: publish {temp_path}: {exc}"
            ) from exc
    except BaseException as exc:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            exc.add_note(
                f"failed to clean up temporary model bundle {temp_path}: "
                f"{cleanup_exc}"
            )
        raise
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ModelBundlePublishedCleanupError(path, temp_path) from exc


def load_model_bundle(path: Path) -> T1Model:
    """Load and validate a model bundle previously saved by
    :func:`save_model_bundle`.

    Only trusted, verified local artifacts should be loaded: joblib
    deserialization can execute arbitrary code, and the validation performed
    after loading does not protect against malicious artifacts or provide
    unsupported cross-version compatibility.  See
    https://scikit-learn.org/stable/model_persistence.html
    """
    try:
        with path.open("rb") as stream:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=InconsistentVersionWarning)
                payload = joblib.load(stream)
    except InconsistentVersionWarning as exc:
        raise ModelBundleError(
            f"cannot load model bundle {path}: incompatible scikit-learn "
            f"version: {exc}"
        ) from exc
    except (OSError, EOFError, pickle.UnpicklingError) as exc:
        raise ModelBundleError(f"cannot load model bundle {path}: {exc}") from exc
    try:
        _validate_payload(payload)
    except ModelBundleError as exc:
        raise ModelBundleError(f"invalid model bundle {path}: {exc}") from exc
    return payload["model"]