"""Unit tests for the Streamlit batch helper layer."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from inference.batch_helpers import (
    parse_uploaded_csv,
    predict_batch,
    predictions_to_csv_bytes,
)
from pipelines.feature_pipeline.feature_pipeline import FEATURE_COLUMNS
from pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COLUMN,
    InferenceInputValidationError,
)

SAMPLE_CSV_BYTES = (
    b"GRE Score,TOEFL Score,University Rating,SOP,LOR,CGPA,Research\n"
    b"316,107,3,3.5,3.0,8.62,1\n"
    b"298,95,2,2.0,2.5,7.90,0\n"
)

WHITESPACED_HEADER_CSV_BYTES = (
    b" GRE Score ,TOEFL Score,University Rating,SOP,LOR,CGPA,Research\n316,107,3,3.5,3.0,8.62,1\n"
)

EXPECTED_PREDICTIONS = [0.5, 0.5]


class BatchPredictOnlyModel:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, features: pd.DataFrame) -> list[float]:
        self.predict_calls += 1
        return EXPECTED_PREDICTIONS

    def fit(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("Inference must never call fit().")


def build_synthetic_frame() -> pd.DataFrame:
    """Return a two-row frame matching the canonical feature columns."""
    return pd.DataFrame(
        [
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
        ]
    )


def test_parse_uploaded_csv_returns_expected_frame() -> None:
    frame = parse_uploaded_csv(SAMPLE_CSV_BYTES)

    assert sorted(frame.columns) == sorted(FEATURE_COLUMNS)
    assert frame.shape == (2, len(FEATURE_COLUMNS))


def test_parse_uploaded_csv_strips_header_whitespace() -> None:
    frame = parse_uploaded_csv(WHITESPACED_HEADER_CSV_BYTES)

    assert frame.columns[0] == "GRE Score"


def test_parse_uploaded_csv_rejects_malformed_csv() -> None:
    with pytest.raises(pd.errors.ParserError):
        parse_uploaded_csv(b'a,b\n"unclosed,2\n')


def test_parse_uploaded_csv_rejects_empty_csv() -> None:
    with pytest.raises(pd.errors.EmptyDataError):
        parse_uploaded_csv(b"")


def test_predict_batch_appends_prediction_column_last() -> None:
    model = BatchPredictOnlyModel()
    frame = build_synthetic_frame()

    result = predict_batch(model, frame)

    assert list(result.columns) == [*list(frame.columns), PREDICTION_COLUMN]
    assert result[PREDICTION_COLUMN].tolist() == EXPECTED_PREDICTIONS
    assert model.predict_calls == 1


def test_predict_batch_preserves_original_input_columns() -> None:
    model = BatchPredictOnlyModel()
    frame = build_synthetic_frame()

    result = predict_batch(model, frame)

    pd.testing.assert_frame_equal(result.loc[:, list(frame.columns)], frame)


def test_predict_batch_propagates_validation_errors() -> None:
    model = BatchPredictOnlyModel()
    frame = build_synthetic_frame().drop(columns=["CGPA"])

    with pytest.raises(InferenceInputValidationError, match="missing columns"):
        predict_batch(model, frame)


def test_predictions_to_csv_bytes_round_trips_with_nulls() -> None:
    model = BatchPredictOnlyModel()
    result = predict_batch(model, build_synthetic_frame())

    csv_bytes = predictions_to_csv_bytes(result)
    round_tripped = pd.read_csv(BytesIO(csv_bytes))

    assert sorted(round_tripped.columns) == sorted([*FEATURE_COLUMNS, PREDICTION_COLUMN])
    assert round_tripped.columns[-1] == PREDICTION_COLUMN
    assert round_tripped[PREDICTION_COLUMN].tolist() == EXPECTED_PREDICTIONS
    assert pd.isna(round_tripped.loc[1, "LOR"])
