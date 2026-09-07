"""Unit tests for the Graduate Admissions batch inference pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)
from pipelines.inference_pipeline.inference import load_model
from pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COLUMN,
    InferenceInputValidationError,
    default_new_data_path,
    default_predictions_output_path,
    generate_predictions,
    persist_predictions,
    read_new_data,
    run_inference_pipeline,
    validate_inference_input,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SYNTHETIC_ROW_COUNT = 3
FEATURE_COUNT = 7

SYNTHETIC_FEATURE_ROWS = [
    {
        "GRE Score": 316,
        "TOEFL Score": 107,
        "University Rating": 3,
        "SOP": 3.5,
        "LOR": 3.0,
        "CGPA": 8.62,
        "Research": 1,
    },
    {
        "GRE Score": 298,
        "TOEFL Score": 95,
        "University Rating": 2,
        "SOP": 2.0,
        "LOR": np.nan,
        "CGPA": 7.90,
        "Research": 0,
    },
    {
        "GRE Score": 330,
        "TOEFL Score": 118,
        "University Rating": 5,
        "SOP": 4.0,
        "LOR": 4.0,
        "CGPA": 9.40,
        "Research": 1,
    },
]

SHUFFLED_FEATURE_COLUMNS = [
    "Research",
    "CGPA",
    "GRE Score",
    "SOP",
    "LOR",
    "TOEFL Score",
    "University Rating",
]

SYNTHETIC_TARGETS = [0.72, 0.55, 0.91]


def build_synthetic_new_data() -> pd.DataFrame:
    """Return a features-only frame with one null in the LOR column."""
    return pd.DataFrame(SYNTHETIC_FEATURE_ROWS)


def write_synthetic_input_csv(path: Path, columns: list[str] | None = None) -> Path:
    """Write a synthetic new-applicants CSV and return its path."""
    frame = build_synthetic_new_data()
    if columns is not None:
        frame = frame.loc[:, columns]
    frame.to_csv(path, index=False)
    return path


def build_dummy_model() -> Pipeline:
    """Fit a preprocessor plus LinearRegression dummy on synthetic data."""
    frame = build_synthetic_new_data()
    target = pd.Series(SYNTHETIC_TARGETS, name=TARGET_COLUMN)
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )
    return model.fit(frame.loc[:, list(FEATURE_COLUMNS)], target)


def persist_dummy_model(path: Path) -> Path:
    """Serialize the fitted dummy model and return its path."""
    joblib.dump(build_dummy_model(), path)
    return path


class BatchPredictOnlyModel:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.predict_calls += 1
        return [0.0] * len(features)

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("Inference must never call fit().")


def test_default_paths_point_to_expected_locations() -> None:
    assert (
        default_new_data_path()
        == REPOSITORY_ROOT / "data" / "05_model_input" / "new_applicants.csv"
    )
    assert (
        default_predictions_output_path()
        == REPOSITORY_ROOT / "data" / "07_model_output" / "admission_predictions.csv"
    )


def test_read_new_data_loads_frame_and_preserves_nulls(tmp_path: Path) -> None:
    input_path = write_synthetic_input_csv(tmp_path / "new_applicants.csv")

    new_data = read_new_data(input_path)

    assert sorted(new_data.columns) == sorted(FEATURE_COLUMNS)
    assert new_data.shape == (SYNTHETIC_ROW_COUNT, FEATURE_COUNT)
    assert pd.isna(new_data.loc[1, "LOR"])


def test_read_new_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="New data file not found"):
        read_new_data(tmp_path / "missing.csv")


def test_validate_inference_input_accepts_valid_frame_with_nulls() -> None:
    new_data = build_synthetic_new_data()

    validate_inference_input(new_data)


def test_validate_inference_input_does_not_mutate_the_frame() -> None:
    new_data = build_synthetic_new_data()
    frame_before = new_data.copy(deep=True)

    validate_inference_input(new_data)

    assert_frame_equal(new_data, frame_before)


def test_missing_column_is_rejected() -> None:
    new_data = build_synthetic_new_data().drop(columns=["CGPA"])

    with pytest.raises(InferenceInputValidationError, match="missing columns"):
        validate_inference_input(new_data)


def test_unexpected_column_is_rejected() -> None:
    new_data = build_synthetic_new_data()
    new_data["Unexpected"] = 1

    with pytest.raises(InferenceInputValidationError, match="unexpected columns"):
        validate_inference_input(new_data)


def test_non_numeric_column_is_rejected() -> None:
    new_data = build_synthetic_new_data()
    new_data["CGPA"] = new_data["CGPA"].astype(str)

    with pytest.raises(InferenceInputValidationError, match="numeric dtype"):
        validate_inference_input(new_data)


def test_non_finite_values_are_rejected() -> None:
    new_data = build_synthetic_new_data()
    new_data.loc[0, "CGPA"] = np.inf

    with pytest.raises(InferenceInputValidationError, match="non-finite"):
        validate_inference_input(new_data)


@pytest.mark.parametrize(
    ("column", "value", "error_match"),
    [
        ("GRE Score", 341, "outside range"),
        ("CGPA", -0.01, "outside range"),
        ("SOP", 3.25, "outside allowed categories"),
        ("Research", 2, "outside allowed categories"),
    ],
)
def test_out_of_domain_values_are_rejected(
    column: str,
    value: float,
    error_match: str,
) -> None:
    new_data = build_synthetic_new_data()
    new_data.loc[0, column] = value

    with pytest.raises(InferenceInputValidationError, match=error_match):
        validate_inference_input(new_data)


def test_empty_frame_is_rejected() -> None:
    new_data = build_synthetic_new_data().iloc[:0]

    with pytest.raises(InferenceInputValidationError, match="empty"):
        validate_inference_input(new_data)


def test_generate_predictions_calls_predict_once_and_never_fit() -> None:
    model = BatchPredictOnlyModel()

    predictions = generate_predictions(model, build_synthetic_new_data())

    assert model.predict_calls == 1
    assert predictions.tolist() == [0.0] * SYNTHETIC_ROW_COUNT
    assert predictions.name == PREDICTION_COLUMN


def test_predictions_match_direct_artifact_predictions(tmp_path: Path) -> None:
    model = joblib.load(persist_dummy_model(tmp_path / "dummy_model.joblib"))
    new_data = build_synthetic_new_data()

    predictions = generate_predictions(model, new_data)
    expected = pd.Series(
        model.predict(new_data.loc[:, list(FEATURE_COLUMNS)]),
        index=new_data.index,
        name=PREDICTION_COLUMN,
    )

    assert_series_equal(predictions, expected)
    assert predictions.notna().all()


def test_predictions_are_identical_for_shuffled_column_order(tmp_path: Path) -> None:
    model = joblib.load(persist_dummy_model(tmp_path / "dummy_model.joblib"))
    canonical = build_synthetic_new_data()
    shuffled = build_synthetic_new_data().loc[:, SHUFFLED_FEATURE_COLUMNS]

    canonical_predictions = generate_predictions(model, canonical)
    shuffled_predictions = generate_predictions(model, shuffled)

    assert_series_equal(canonical_predictions, shuffled_predictions)


def test_persist_predictions_writes_csv(tmp_path: Path) -> None:
    new_data = build_synthetic_new_data()
    predictions = pd.Series([0.5, 0.6, 0.7], name=PREDICTION_COLUMN)
    output_path = tmp_path / "predictions.csv"

    returned_path = persist_predictions(new_data, predictions, output_path)
    output_frame = pd.read_csv(returned_path)

    assert returned_path == output_path
    assert sorted(output_frame.columns) == sorted([*FEATURE_COLUMNS, PREDICTION_COLUMN])
    assert output_frame.columns[-1] == PREDICTION_COLUMN
    assert output_frame[PREDICTION_COLUMN].tolist() == predictions.tolist()


def test_run_inference_pipeline_end_to_end(tmp_path: Path) -> None:
    input_path = write_synthetic_input_csv(tmp_path / "new_applicants.csv")
    model_path = persist_dummy_model(tmp_path / "dummy_model.joblib")
    output_path = tmp_path / "predictions.csv"
    input_bytes_before = input_path.read_bytes()

    returned_path = run_inference_pipeline(
        input_path=input_path,
        model_path=model_path,
        output_path=output_path,
    )

    assert returned_path == output_path
    assert input_path.read_bytes() == input_bytes_before
    output_frame = pd.read_csv(output_path)
    assert sorted(output_frame.columns) == sorted([*FEATURE_COLUMNS, PREDICTION_COLUMN])
    assert len(output_frame) == SYNTHETIC_ROW_COUNT
    model = joblib.load(model_path)
    expected = pd.Series(
        model.predict(read_new_data(input_path).loc[:, list(FEATURE_COLUMNS)]),
        name=PREDICTION_COLUMN,
    )
    assert_series_equal(output_frame[PREDICTION_COLUMN], expected)


def test_run_inference_pipeline_missing_input_raises(tmp_path: Path) -> None:
    model_path = persist_dummy_model(tmp_path / "dummy_model.joblib")

    with pytest.raises(FileNotFoundError, match="New data file not found"):
        run_inference_pipeline(
            input_path=tmp_path / "missing.csv",
            model_path=model_path,
            output_path=tmp_path / "predictions.csv",
        )


def test_run_inference_pipeline_missing_model_raises(tmp_path: Path) -> None:
    input_path = write_synthetic_input_csv(tmp_path / "new_applicants.csv")

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        run_inference_pipeline(
            input_path=input_path,
            model_path=tmp_path / "missing_model.joblib",
            output_path=tmp_path / "predictions.csv",
        )


def test_real_artifact_batch_flow_applies_training_transformations(tmp_path: Path) -> None:
    input_path = write_synthetic_input_csv(tmp_path / "new_applicants.csv")
    model = load_model()

    new_data = read_new_data(input_path)
    validate_inference_input(new_data)
    predictions = generate_predictions(model, new_data)
    expected = pd.Series(
        model.predict(new_data.loc[:, list(FEATURE_COLUMNS)]),
        index=new_data.index,
        name=PREDICTION_COLUMN,
    )

    assert_series_equal(predictions, expected)
    assert predictions.notna().all()
