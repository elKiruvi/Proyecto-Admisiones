from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from pipelines.inference_pipeline.inference import (
    CATEGORICAL_DOMAINS,
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
        ("University Rating", -1),
        ("University Rating", 6),
        ("SOP", -0.5),
        ("SOP", 5.5),
        ("SOP", 3.25),
        ("LOR", -0.5),
        ("LOR", 5.5),
        ("LOR", 3.25),
        ("Research", -1),
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


@pytest.mark.parametrize(
    "field,values",
    [
        ("GRE Score", [0, 340]),
        ("TOEFL Score", [0, 120]),
        ("CGPA", [0.0, 10.0]),
        ("University Rating", [1, 5]),
        ("SOP", [1.0, 5.0]),
        ("LOR", [1.0, 5.0]),
        ("Research", [0, 1]),
    ],
)
def test_valid_boundary_values_are_accepted(field: str, values: list[object]) -> None:
    for value in values:
        features = {**VALID_FEATURES, field: value}
        build_feature_frame(features)


def test_categorical_domains_match_the_serialized_pipeline() -> None:
    model = load_model()
    ordinal_transformer = model.named_steps["preprocessor"].named_transformers_["ordinal"]
    encoder = ordinal_transformer.named_steps["encoder"]
    fitted_domains = dict(
        zip(
            ("University Rating", "SOP", "LOR"),
            (frozenset(map(float, categories)) for categories in encoder.categories_),
            strict=True,
        )
    )

    assert {field: CATEGORICAL_DOMAINS[field] for field in fitted_domains} == fitted_domains


@pytest.mark.parametrize(
    "field,value",
    [
        ("GRE Score", -1),
        ("GRE Score", 341),
        ("TOEFL Score", -1),
        ("TOEFL Score", 121),
        ("CGPA", -0.01),
        ("CGPA", 10.01),
    ],
)
def test_continuous_values_outside_project_range_are_rejected(field: str, value: object) -> None:
    features = {**VALID_FEATURES, field: value}

    with pytest.raises(ValueError, match="must be between"):
        build_feature_frame(features)


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
