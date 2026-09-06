"""Unit tests for the Graduate Admissions training pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference import load_model
from pipelines.training_pipeline.train_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CrossValidationMetrics,
    TrainTestValidationReport,
    build_metrics_report,
    build_model_pipeline,
    cross_validate_model,
    default_feature_input_path,
    default_metrics_output_path,
    default_model_output_path,
    evaluate_model,
    read_features,
    run_training_pipeline,
    save_metrics,
    save_model,
    split_features,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SYNTHETIC_ROW_COUNT = 60
TRAIN_ROW_COUNT = 48
TEST_ROW_COUNT = 12
FEATURE_COUNT = 7
COLUMN_COUNT = 8
CV_FOLD_COUNT = 5

HISTORICAL_TRAIN_ROWS = 376
HISTORICAL_TEST_ROWS = 95
HISTORICAL_TRAIN_RMSE = 0.062440
HISTORICAL_TRAIN_MAE = 0.045083
HISTORICAL_TRAIN_R2 = 0.807426
HISTORICAL_CV_RMSE_MEAN = 0.064463
HISTORICAL_CV_MAE_MEAN = 0.046825
HISTORICAL_CV_R2_MEAN = 0.786251
HISTORICAL_TEST_RMSE = 0.068500
HISTORICAL_TEST_MAE = 0.051192
HISTORICAL_TEST_R2 = 0.787195
METRIC_TOLERANCE = 1e-6


def build_synthetic_feature_frame() -> pd.DataFrame:
    """Return a typed frame in the feature schema with seeded nulls."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "GRE Score": rng.integers(260, 341, size=SYNTHETIC_ROW_COUNT),
            "TOEFL Score": rng.integers(85, 121, size=SYNTHETIC_ROW_COUNT),
            "CGPA": np.round(rng.uniform(7.0, 10.0, size=SYNTHETIC_ROW_COUNT), 2),
            "University Rating": rng.integers(1, 6, size=SYNTHETIC_ROW_COUNT),
            "SOP": np.round(rng.uniform(1.0, 5.0, size=SYNTHETIC_ROW_COUNT) * 2) / 2,
            "LOR": np.round(rng.uniform(1.0, 5.0, size=SYNTHETIC_ROW_COUNT) * 2) / 2,
            "Research": rng.integers(0, 2, size=SYNTHETIC_ROW_COUNT),
            TARGET_COLUMN: np.round(rng.uniform(0.3, 1.0, size=SYNTHETIC_ROW_COUNT), 3),
        }
    )
    frame.loc[frame.index[:3], "GRE Score"] = np.nan
    frame.loc[frame.index[3:5], "SOP"] = np.nan
    return frame


def persist_synthetic_features(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a synthetic frame as Parquet and return its path."""
    frame.to_parquet(path, index=False, engine="pyarrow")
    return path


def build_split_validation_report() -> TrainTestValidationReport:
    """Return a minimal valid split-validation report for report assembly."""
    return {
        "rows": {"train": TRAIN_ROW_COUNT, "test": TEST_ROW_COUNT},
        "ratio": {
            "test_to_train": TEST_ROW_COUNT / TRAIN_ROW_COUNT,
            "test_to_total": TEST_ROW_COUNT / SYNTHETIC_ROW_COUNT,
        },
        "checks": [
            {"name": "disjoint_indices", "status": "PASS", "detail": "ok"},
        ],
        "drift": {"GRE Score": 0.05, TARGET_COLUMN: 0.02},
        "null_rates": {"train": {}, "test": {}},
        "warnings": [],
    }


def test_default_paths_point_to_expected_locations() -> None:
    assert (
        default_model_output_path()
        == REPOSITORY_ROOT / "models" / "05_model_selection_pipeline.joblib"
    )
    assert (
        default_metrics_output_path()
        == REPOSITORY_ROOT / "data" / "08_reporting" / "training_metrics.json"
    )
    assert (
        default_feature_input_path()
        == REPOSITORY_ROOT / "data" / "04_feature" / "admission_features.parquet"
    )


def test_read_features_loads_persisted_frame(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame()
    features_path = persist_synthetic_features(frame, tmp_path / "features.parquet")

    loaded = read_features(features_path)

    assert loaded.shape == (SYNTHETIC_ROW_COUNT, COLUMN_COUNT)
    assert list(loaded.columns) == list(frame.columns)


def test_read_features_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_features(tmp_path / "missing.parquet")


def test_read_features_missing_column_raises_value_error(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame().drop(columns=["CGPA"])
    features_path = persist_synthetic_features(frame, tmp_path / "features.parquet")

    with pytest.raises(ValueError, match="missing columns"):
        read_features(features_path)


def test_read_features_unexpected_column_raises_value_error(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame()
    frame["Unexpected"] = 1
    features_path = persist_synthetic_features(frame, tmp_path / "features.parquet")

    with pytest.raises(ValueError, match="unexpected columns"):
        read_features(features_path)


def test_split_features_shapes_and_disjoint_indices() -> None:
    X_train, X_test, y_train, y_test = split_features(build_synthetic_feature_frame())

    assert X_train.shape == (TRAIN_ROW_COUNT, FEATURE_COUNT)
    assert X_test.shape == (TEST_ROW_COUNT, FEATURE_COUNT)
    assert y_train.shape == (TRAIN_ROW_COUNT,)
    assert y_test.shape == (TEST_ROW_COUNT,)
    assert list(X_train.columns) == list(FEATURE_COLUMNS)
    assert y_train.name == TARGET_COLUMN
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_features_is_deterministic() -> None:
    frame = build_synthetic_feature_frame()

    first = split_features(frame)
    second = split_features(frame)

    for first_part, second_part in zip(first, second, strict=True):
        assert list(first_part.index) == list(second_part.index)


def test_build_model_pipeline_has_unfitted_preprocessor_and_linear_model() -> None:
    model_pipeline = build_model_pipeline()

    assert isinstance(model_pipeline, Pipeline)
    assert tuple(model_pipeline.named_steps) == ("preprocessor", "model")
    assert isinstance(model_pipeline.named_steps["model"], LinearRegression)
    assert not hasattr(model_pipeline.named_steps["preprocessor"], "transformers_")


def test_fit_and_evaluate_model_on_synthetic_data() -> None:
    frame = build_synthetic_feature_frame()
    X_train, X_test, y_train, y_test = split_features(frame)
    model_pipeline = build_model_pipeline()

    model_pipeline.fit(X_train, y_train)
    train_metrics = evaluate_model(model_pipeline, X_train, y_train)
    test_metrics = evaluate_model(model_pipeline, X_test, y_test)

    for metrics in (train_metrics, test_metrics):
        assert set(metrics) == {"rmse", "mae", "r2"}
        for value in metrics.values():
            assert np.isfinite(value)


def test_evaluate_model_matches_manual_sklearn_metrics() -> None:
    frame = build_synthetic_feature_frame()
    X_train, X_test, y_train, y_test = split_features(frame)
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    metrics = evaluate_model(model_pipeline, X_test, y_test)
    predictions = model_pipeline.predict(X_test)

    assert metrics["rmse"] == pytest.approx(root_mean_squared_error(y_test, predictions))
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_test, predictions))
    assert metrics["r2"] == pytest.approx(r2_score(y_test, predictions))


def test_evaluate_model_does_not_refit_the_pipeline() -> None:
    frame = build_synthetic_feature_frame()
    X_train, X_test, y_train, y_test = split_features(frame)
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)

    numeric_imputer = (
        model_pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )
    assert isinstance(numeric_imputer, SimpleImputer)
    statistics_before = numeric_imputer.statistics_.copy()

    evaluate_model(model_pipeline, X_test, y_test)

    np.testing.assert_array_equal(numeric_imputer.statistics_, statistics_before)


def test_cross_validate_model_runs_cv_on_train_only() -> None:
    frame = build_synthetic_feature_frame()
    X_train, _, y_train, _ = split_features(frame)
    model_pipeline = build_model_pipeline()

    cv_metrics = cross_validate_model(model_pipeline, X_train, y_train)

    assert len(cv_metrics["folds"]) == CV_FOLD_COUNT
    for fold in cv_metrics["folds"]:
        assert set(fold) == {"rmse", "mae", "r2"}
        for value in fold.values():
            assert np.isfinite(value)
    for key in ("rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"):
        assert np.isfinite(cv_metrics[key])


def test_cross_validate_model_leaves_the_pipeline_unfitted() -> None:
    frame = build_synthetic_feature_frame()
    X_train, _, y_train, _ = split_features(frame)
    model_pipeline = build_model_pipeline()

    cross_validate_model(model_pipeline, X_train, y_train)

    assert not hasattr(model_pipeline.named_steps["preprocessor"], "transformers_")
    assert not hasattr(model_pipeline.named_steps["model"], "coef_")


def test_cross_validate_model_is_deterministic() -> None:
    frame = build_synthetic_feature_frame()
    X_train, _, y_train, _ = split_features(frame)

    first = cross_validate_model(build_model_pipeline(), X_train, y_train)
    second = cross_validate_model(build_model_pipeline(), X_train, y_train)

    assert first == second


def test_save_model_round_trip(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame()
    X_train, _, y_train, _ = split_features(frame)
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)
    artifact_path = tmp_path / "nested" / "model.joblib"

    saved_path = save_model(model_pipeline, artifact_path)
    reloaded = joblib.load(saved_path)

    assert isinstance(reloaded, Pipeline)
    np.testing.assert_allclose(model_pipeline.predict(X_train), reloaded.predict(X_train))


def test_saved_artifact_satisfies_the_inference_contract(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame()
    X_train, X_test, y_train, _ = split_features(frame)
    model_pipeline = build_model_pipeline()
    model_pipeline.fit(X_train, y_train)
    artifact_path = save_model(model_pipeline, tmp_path / "model.joblib")

    loaded = load_model(artifact_path)

    assert isinstance(loaded, Pipeline)
    assert tuple(loaded.named_steps) == ("preprocessor", "model")
    np.testing.assert_allclose(model_pipeline.predict(X_test), loaded.predict(X_test))


def test_save_metrics_writes_deterministic_json(tmp_path: Path) -> None:
    train_metrics = {"rmse": 0.1, "mae": 0.08, "r2": 0.9}
    cv_metrics: CrossValidationMetrics = {
        "folds": [{"rmse": 0.12, "mae": 0.09, "r2": 0.88}],
        "rmse_mean": 0.12,
        "rmse_std": 0.01,
        "mae_mean": 0.09,
        "mae_std": 0.005,
        "r2_mean": 0.88,
        "r2_std": 0.02,
    }
    test_metrics = {"rmse": 0.14, "mae": 0.10, "r2": 0.85}
    report = build_metrics_report(
        train_metrics,
        cv_metrics,
        test_metrics,
        build_split_validation_report(),
    )

    first_path = save_metrics(report, tmp_path / "first.json")
    second_path = save_metrics(report, tmp_path / "second.json")

    assert first_path.read_bytes() == second_path.read_bytes()
    with first_path.open(encoding="utf-8") as handle:
        assert json.load(handle) == report


def test_build_metrics_report_contains_expected_structure_and_gaps() -> None:
    train_metrics = {"rmse": 0.1, "mae": 0.08, "r2": 0.9}
    cv_metrics: CrossValidationMetrics = {
        "folds": [{"rmse": 0.12, "mae": 0.09, "r2": 0.88}],
        "rmse_mean": 0.12,
        "rmse_std": 0.01,
        "mae_mean": 0.09,
        "mae_std": 0.005,
        "r2_mean": 0.88,
        "r2_std": 0.02,
    }
    test_metrics = {"rmse": 0.14, "mae": 0.10, "r2": 0.85}
    split_validation = build_split_validation_report()

    report = build_metrics_report(
        train_metrics,
        cv_metrics,
        test_metrics,
        split_validation,
    )

    assert report["model"] == "LinearRegression"
    assert report["rows"] == {"train": TRAIN_ROW_COUNT, "test": TEST_ROW_COUNT}
    assert report["train"] == train_metrics
    assert report["cv"] == cv_metrics
    assert report["test"] == test_metrics
    assert report["split_validation"] == split_validation
    assert report["gaps"]["train_minus_cv_rmse"] == pytest.approx(0.1 - 0.12)
    assert report["gaps"]["test_minus_cv_rmse"] == pytest.approx(0.14 - 0.12)


def test_run_training_pipeline_end_to_end(tmp_path: Path) -> None:
    frame = build_synthetic_feature_frame()
    features_path = persist_synthetic_features(frame, tmp_path / "features.parquet")
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    output = run_training_pipeline(
        features_path=features_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
    )

    assert Path(output["model_path"]).is_file()
    assert Path(output["metrics_path"]).is_file()
    assert output["report"]["rows"] == {"train": TRAIN_ROW_COUNT, "test": TEST_ROW_COUNT}
    assert len(output["report"]["cv"]["folds"]) == CV_FOLD_COUNT
    assert output["report"]["split_validation"]["rows"] == {
        "train": TRAIN_ROW_COUNT,
        "test": TEST_ROW_COUNT,
    }

    reloaded = joblib.load(model_path)
    _, X_test, _, _ = split_features(frame)
    assert np.isfinite(reloaded.predict(X_test)).all()


def test_test_target_values_never_influence_training() -> None:
    frame = build_synthetic_feature_frame()
    X_train, X_test, y_train, _ = split_features(frame)

    baseline_pipeline = build_model_pipeline()
    baseline_pipeline.fit(X_train, y_train)
    baseline_test_predictions = baseline_pipeline.predict(X_test)

    tampered_frame = frame.copy()
    tampered_frame.loc[X_test.index, TARGET_COLUMN] = 1.0

    X_train_tampered, X_test_tampered, y_train_tampered, _ = split_features(tampered_frame)
    assert list(X_train_tampered.index) == list(X_train.index)

    tampered_pipeline = build_model_pipeline()
    tampered_pipeline.fit(X_train_tampered, y_train_tampered)
    tampered_test_predictions = tampered_pipeline.predict(X_test_tampered)

    np.testing.assert_allclose(baseline_test_predictions, tampered_test_predictions)


@pytest.mark.skipif(
    not default_feature_input_path().is_file(),
    reason="Feature set not available in this environment (synthetic fixtures cover CI).",
)
def test_real_feature_set_reproduces_historical_metrics() -> None:
    features_df = read_features()
    X_train, X_test, y_train, y_test = split_features(features_df)

    assert X_train.shape == (HISTORICAL_TRAIN_ROWS, FEATURE_COUNT)
    assert X_test.shape == (HISTORICAL_TEST_ROWS, FEATURE_COUNT)

    model_pipeline = build_model_pipeline()
    cv_metrics = cross_validate_model(model_pipeline, X_train, y_train)
    model_pipeline.fit(X_train, y_train)
    train_metrics = evaluate_model(model_pipeline, X_train, y_train)
    test_metrics = evaluate_model(model_pipeline, X_test, y_test)

    assert train_metrics["rmse"] == pytest.approx(HISTORICAL_TRAIN_RMSE, abs=METRIC_TOLERANCE)
    assert train_metrics["mae"] == pytest.approx(HISTORICAL_TRAIN_MAE, abs=METRIC_TOLERANCE)
    assert train_metrics["r2"] == pytest.approx(HISTORICAL_TRAIN_R2, abs=METRIC_TOLERANCE)
    assert cv_metrics["rmse_mean"] == pytest.approx(HISTORICAL_CV_RMSE_MEAN, abs=METRIC_TOLERANCE)
    assert cv_metrics["mae_mean"] == pytest.approx(HISTORICAL_CV_MAE_MEAN, abs=METRIC_TOLERANCE)
    assert cv_metrics["r2_mean"] == pytest.approx(HISTORICAL_CV_R2_MEAN, abs=METRIC_TOLERANCE)
    assert test_metrics["rmse"] == pytest.approx(HISTORICAL_TEST_RMSE, abs=METRIC_TOLERANCE)
    assert test_metrics["mae"] == pytest.approx(HISTORICAL_TEST_MAE, abs=METRIC_TOLERANCE)
    assert test_metrics["r2"] == pytest.approx(HISTORICAL_TEST_R2, abs=METRIC_TOLERANCE)
