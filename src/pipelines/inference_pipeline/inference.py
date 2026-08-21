"""Inference helpers for the fitted admissions model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

FEATURE_COLUMNS: tuple[str, ...] = (
    "GRE Score",
    "TOEFL Score",
    "CGPA",
    "University Rating",
    "SOP",
    "LOR",
    "Research",
)
MODEL_FILENAME = "05_model_selection_pipeline.joblib"
INTEGER_FEATURES = frozenset({"GRE Score", "TOEFL Score"})
CONTINUOUS_RANGES: dict[str, tuple[float, float]] = {
    "GRE Score": (0.0, 340.0),
    "TOEFL Score": (0.0, 120.0),
    "CGPA": (0.0, 10.0),
}
CATEGORICAL_DOMAINS: dict[str, frozenset[float]] = {
    "University Rating": frozenset({1.0, 2.0, 3.0, 4.0, 5.0}),
    "SOP": frozenset({1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}),
    "LOR": frozenset({1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}),
    "Research": frozenset({0.0, 1.0}),
}


class PredictivePipeline(Protocol):
    """Protocol for the fitted Pipeline operation used by inference."""

    def predict(self, features: pd.DataFrame) -> Sequence[float]:
        """Return predictions for the provided feature rows."""


def find_repository_root(start: Path | None = None) -> Path:
    """Find the repository root by locating its project manifest."""
    path = (start or Path(__file__)).resolve()
    search_paths = (path, *path.parents)
    for candidate in search_paths:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repository root.")


def default_model_path() -> Path:
    """Return the repository-relative path of the immutable model artifact."""
    return find_repository_root() / "models" / MODEL_FILENAME


def load_model(model_path: Path | None = None) -> Pipeline:
    """Load and verify the fitted admissions Pipeline without modifying it."""
    path = model_path or default_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    loaded_model = joblib.load(path)
    if not isinstance(loaded_model, Pipeline):
        raise TypeError("The model artifact does not contain a scikit-learn Pipeline.")

    check_is_fitted(loaded_model)
    if tuple(loaded_model.named_steps) != ("preprocessor", "model"):
        raise TypeError("The model Pipeline does not have the expected fitted steps.")
    if not isinstance(loaded_model.named_steps["model"], LinearRegression):
        raise TypeError("The model Pipeline does not contain LinearRegression.")

    feature_names = tuple(loaded_model.feature_names_in_)
    if feature_names != FEATURE_COLUMNS:
        raise ValueError("The model Pipeline feature schema does not match the demo schema.")

    return loaded_model


def _validate_feature_value(feature_name: str, value: object) -> None:
    """Validate one feature value without applying model preprocessing."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{feature_name} must be numeric.")

    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{feature_name} must be finite.")
    if feature_name in INTEGER_FEATURES and not numeric_value.is_integer():
        raise ValueError(f"{feature_name} must be an integer.")
    if feature_name in CONTINUOUS_RANGES:
        lower_bound, upper_bound = CONTINUOUS_RANGES[feature_name]
        if not lower_bound <= numeric_value <= upper_bound:
            raise ValueError(f"{feature_name} must be between {lower_bound:g} and {upper_bound:g}.")
    if (
        feature_name in CATEGORICAL_DOMAINS
        and numeric_value not in CATEGORICAL_DOMAINS[feature_name]
    ):
        raise ValueError(f"{feature_name} is outside the supported categorical domain.")


def validate_features(features: Mapping[str, object]) -> None:
    """Validate the input schema without applying model preprocessing."""
    expected_fields = set(FEATURE_COLUMNS)
    received_fields = set(features)
    missing_fields = expected_fields - received_fields
    unexpected_fields = received_fields - expected_fields
    if missing_fields or unexpected_fields:
        details = []
        if missing_fields:
            details.append(f"missing fields: {sorted(missing_fields)}")
        if unexpected_fields:
            details.append(f"unexpected fields: {sorted(unexpected_fields)}")
        raise ValueError("Invalid feature fields (" + "; ".join(details) + ").")

    for feature_name in FEATURE_COLUMNS:
        _validate_feature_value(feature_name, features[feature_name])


def build_feature_frame(features: Mapping[str, object]) -> pd.DataFrame:
    """Build the one-row DataFrame in the fitted Pipeline's canonical order."""
    validate_features(features)
    ordered_features = {feature_name: features[feature_name] for feature_name in FEATURE_COLUMNS}
    return pd.DataFrame([ordered_features], columns=list(FEATURE_COLUMNS))


def predict_admission(
    model: PredictivePipeline,
    features: Mapping[str, object],
) -> float:
    """Return the raw prediction from the fitted Pipeline without clipping."""
    feature_frame = build_feature_frame(features)
    predictions = model.predict(feature_frame)
    if len(predictions) != 1:
        raise ValueError("The inference Pipeline returned an unexpected number of predictions.")
    return float(predictions[0])
