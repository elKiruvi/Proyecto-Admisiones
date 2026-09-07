"""Presentation-oriented batch helpers for the Streamlit layer.

These functions orchestrate the existing batch inference functionality
only: no validation rule, preprocessing step, model loading or
prediction calculation is implemented here.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COLUMN,
    BatchPredictivePipeline,
    generate_predictions,
    parse_new_data_csv,
    validate_inference_input,
)


def parse_uploaded_csv(raw: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes into the new-data frame expected by the batch pipeline."""
    return parse_new_data_csv(BytesIO(raw))


def predict_batch(model: BatchPredictivePipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the frame and return the input rows with the prediction column appended.

    The original input columns are preserved and the prediction column is
    always appended last, matching the batch pipeline output contract.
    """
    validate_inference_input(frame)
    predictions = generate_predictions(model, frame)
    return frame.assign(**{PREDICTION_COLUMN: predictions})


def predictions_to_csv_bytes(result: pd.DataFrame) -> bytes:
    """Serialize the result frame to UTF-8 CSV bytes for the download button."""
    csv_payload = str(result.to_csv(index=False))
    return csv_payload.encode("utf-8")
