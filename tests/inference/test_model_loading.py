from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference import (
    FEATURE_COLUMNS,
    default_model_path,
    load_model,
)


def test_default_model_path_is_repository_relative() -> None:
    model_path = default_model_path()

    assert model_path == Path(__file__).parents[2] / "models" / "05_model_selection_pipeline.joblib"


def test_load_model_returns_expected_fitted_pipeline() -> None:
    model = load_model()

    assert isinstance(model, Pipeline)
    assert tuple(model.named_steps) == ("preprocessor", "model")
    assert isinstance(model.named_steps["model"], LinearRegression)
    np.testing.assert_array_equal(model.feature_names_in_, FEATURE_COLUMNS)
