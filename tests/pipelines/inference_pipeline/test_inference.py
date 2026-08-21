from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from pipelines.inference_pipeline.inference import (
    FEATURE_COLUMNS,
    build_feature_frame,
    load_model,
    predict_admission,
)

EXPECTED_PREDICTION = 0.572
OUT_OF_RANGE_PREDICTION = 1.25


VALID_FEATURES: dict[str, object] = {
    "GRE Score": 316,
    "TOEFL Score": 107,
    "University Rating": 3,
    "SOP": 3.5,
    "LOR": 3.0,
    "CGPA": 8.62,
    "Research": 1,
}


class PredictOnlyModel:
    def __init__(self, prediction: float = 0.572) -> None:
        self.prediction = prediction
        self.predict_calls = 0

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.predict_calls += 1
        return [self.prediction]

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("Inference must never call fit().")


def test_build_feature_frame_uses_canonical_column_order() -> None:
    frame = build_feature_frame(VALID_FEATURES)

    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.shape == (1, len(FEATURE_COLUMNS))


@pytest.mark.parametrize(
    "features",
    [
        {key: value for key, value in VALID_FEATURES.items() if key != "CGPA"},
        {**VALID_FEATURES, "Unexpected": 1},
    ],
)
def test_missing_or_unexpected_fields_are_rejected(
    features: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError, match="Invalid feature fields"):
        build_feature_frame(features)


@pytest.mark.parametrize(
    "field,value",
    [
        ("University Rating", 6),
        ("SOP", 3.25),
        ("LOR", 0.0),
        ("Research", 2),
    ],
)
def test_invalid_categorical_values_are_rejected(field: str, value: object) -> None:
    features = {**VALID_FEATURES, field: value}

    with pytest.raises(ValueError, match="supported categorical domain"):
        build_feature_frame(features)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(value: float) -> None:
    features = {**VALID_FEATURES, "CGPA": value}

    with pytest.raises(ValueError, match="must be finite"):
        build_feature_frame(features)


def test_non_numeric_values_are_rejected() -> None:
    features = {**VALID_FEATURES, "CGPA": "8.62"}

    with pytest.raises(TypeError, match="must be numeric"):
        build_feature_frame(features)


def test_non_integral_score_is_rejected() -> None:
    features = {**VALID_FEATURES, "GRE Score": 316.5}

    with pytest.raises(ValueError, match="must be an integer"):
        build_feature_frame(features)


def test_valid_continuous_values_outside_observed_range_are_accepted() -> None:
    features = {**VALID_FEATURES, "GRE Score": 0, "TOEFL Score": 0, "CGPA": 0.0}

    frame = build_feature_frame(features)

    assert frame.loc[0, "GRE Score"] == 0
    assert frame.loc[0, "TOEFL Score"] == 0
    assert frame.loc[0, "CGPA"] == 0.0


def test_inference_calls_predict_but_never_fit() -> None:
    model = PredictOnlyModel(prediction=EXPECTED_PREDICTION)

    prediction = predict_admission(model, VALID_FEATURES)

    assert prediction == EXPECTED_PREDICTION
    assert model.predict_calls == 1


def test_prediction_is_not_clipped() -> None:
    model = PredictOnlyModel(prediction=OUT_OF_RANGE_PREDICTION)

    prediction = predict_admission(model, VALID_FEATURES)

    assert prediction == OUT_OF_RANGE_PREDICTION


def test_prediction_matches_the_serialized_pipeline() -> None:
    model = load_model()
    expected_frame = pd.DataFrame([VALID_FEATURES], columns=list(FEATURE_COLUMNS))

    expected_prediction = float(model.predict(expected_frame)[0])
    actual_prediction = predict_admission(model, VALID_FEATURES)

    assert actual_prediction == expected_prediction
