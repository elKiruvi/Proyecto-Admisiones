"""Unit tests for the train/test split validation (Issue #32)."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.training_pipeline import train_pipeline
from pipelines.training_pipeline.train_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TrainTestValidationError,
    _ks_statistic,
    default_feature_input_path,
    read_features,
    run_training_pipeline,
    split_features,
    validate_train_test_split,
)

FRAME_ROW_COUNT = 120
TRAIN_ROW_COUNT = 96
TEST_ROW_COUNT = 24

STRUCTURAL_CHECK_NAMES = {
    "disjoint_indices",
    "unique_indices",
    "cross_partition_duplicates",
    "size_ratio",
    "schema_parity",
    "alignment",
    "target_completeness",
    "no_new_categories",
}


def build_balanced_frame(row_count: int = FRAME_ROW_COUNT) -> pd.DataFrame:
    """Return a low-entropy frame whose canonical split is drift-free.

    Values are periodic with few levels so that any deterministic 80/20
    split keeps nearly identical empirical distributions in both
    partitions, and CGPA is unique per row so no two rows are identical.
    """
    index = np.arange(row_count)
    return pd.DataFrame(
        {
            "GRE Score": 260 + (index % 4) * 20,
            "TOEFL Score": 85 + (index % 4) * 8,
            "CGPA": 7.0 + (index % 4) * 0.5 + (index // 4) * 0.01,
            "University Rating": 1 + (index % 4),
            "SOP": 1.0 + (index % 4) * 0.5,
            "LOR": 1.0 + (index % 4) * 0.5,
            "Research": index % 2,
            TARGET_COLUMN: 0.4 + (index % 4) * 0.1,
        }
    )


def test_ks_statistic_identical_samples_is_zero() -> None:
    result = _ks_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

    assert result == 0.0


def test_ks_statistic_handles_ties() -> None:
    result = _ks_statistic(
        np.array([1.0, 1.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 2.0, 2.0]),
    )

    assert result == pytest.approx(0.5)


def test_ks_statistic_ignores_nans() -> None:
    result = _ks_statistic(np.array([1.0, 2.0, np.nan]), np.array([2.0, 3.0]))

    assert result == pytest.approx(0.5)


def test_ks_statistic_empty_sample_returns_nan() -> None:
    result = _ks_statistic(np.array([np.nan]), np.array([1.0]))

    assert math.isnan(result)


def test_valid_split_passes_all_structural_checks() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert report["rows"] == {"train": TRAIN_ROW_COUNT, "test": TEST_ROW_COUNT}
    assert report["ratio"]["test_to_total"] == pytest.approx(0.2)
    assert report["warnings"] == []
    assert caught == []
    assert {check["name"] for check in report["checks"]} == STRUCTURAL_CHECK_NAMES
    assert all(check["status"] == "PASS" for check in report["checks"])
    assert set(report["drift"]) == {*FEATURE_COLUMNS, TARGET_COLUMN}
    assert all(value is None or 0.0 <= value <= 1.0 for value in report["drift"].values())
    assert set(report["null_rates"]) == {"train", "test"}
    assert set(report["null_rates"]["train"]) == {*FEATURE_COLUMNS, TARGET_COLUMN}


def test_overlapping_indices_raise_controlled_error() -> None:
    frame = build_balanced_frame()
    X_train = frame.iloc[:48][list(FEATURE_COLUMNS)]
    X_test = frame.iloc[46:58][list(FEATURE_COLUMNS)]
    y_train = frame.iloc[:48][TARGET_COLUMN]
    y_test = frame.iloc[46:58][TARGET_COLUMN]

    with pytest.raises(TrainTestValidationError, match="overlap"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_duplicate_feature_row_across_partitions_raises() -> None:
    frame = build_balanced_frame()
    X_train = frame.iloc[:48][list(FEATURE_COLUMNS)]
    y_train = frame.iloc[:48][TARGET_COLUMN]

    test_features = frame.iloc[48:59][list(FEATURE_COLUMNS)]
    copied_row = X_train.iloc[[0]].copy()
    copied_row.index = pd.Index([999])
    X_test = pd.concat([test_features, copied_row])

    test_target = frame.iloc[48:59][TARGET_COLUMN]
    copied_target = y_train.iloc[[0]].copy()
    copied_target.index = pd.Index([999])
    y_test = pd.concat([test_target, copied_target])

    with pytest.raises(TrainTestValidationError, match="identical feature rows"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_invalid_size_ratio_raises() -> None:
    frame = build_balanced_frame()
    X_train = frame.iloc[:60][list(FEATURE_COLUMNS)]
    X_test = frame.iloc[60:][list(FEATURE_COLUMNS)]
    y_train = frame.iloc[:60][TARGET_COLUMN]
    y_test = frame.iloc[60:][TARGET_COLUMN]

    with pytest.raises(TrainTestValidationError, match="size"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_schema_mismatch_missing_column_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())

    with pytest.raises(TrainTestValidationError, match="columns differ"):
        validate_train_test_split(X_train, X_test.drop(columns=["CGPA"]), y_train, y_test)


def test_schema_mismatch_dtype_divergence_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_test_dtype = X_test.copy()
    X_test_dtype["Research"] = X_test_dtype["Research"].astype("float64")

    with pytest.raises(TrainTestValidationError, match="dtypes"):
        validate_train_test_split(X_train, X_test_dtype, y_train, y_test)


def test_missing_categorical_column_raises_schema_error_not_key_error() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_test_missing = X_test.drop(columns=["University Rating"])

    with pytest.raises(TrainTestValidationError, match="columns differ"):
        validate_train_test_split(X_train, X_test_missing, y_train, y_test)


def test_duplicated_categorical_column_raises_schema_error_not_incidental() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_test_duplicated = pd.concat([X_test, X_test[["University Rating"]]], axis=1)

    with pytest.raises(TrainTestValidationError, match="columns differ"):
        validate_train_test_split(X_train, X_test_duplicated, y_train, y_test)


def test_duplicated_column_in_both_partitions_raises_schema_error_not_attribute_error() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_train_duplicated = pd.concat([X_train, X_train[["University Rating"]]], axis=1)
    X_test_duplicated = pd.concat([X_test, X_test[["University Rating"]]], axis=1)

    with pytest.raises(TrainTestValidationError, match="duplicate feature column"):
        validate_train_test_split(X_train_duplicated, X_test_duplicated, y_train, y_test)


def test_feature_target_length_misalignment_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())

    with pytest.raises(TrainTestValidationError, match="lengths differ"):
        validate_train_test_split(X_train, X_test, y_train.iloc[:-1], y_test)


def test_feature_target_index_misalignment_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    y_test_shifted = y_test.copy()
    y_test_shifted.index = y_test_shifted.index + 1000

    with pytest.raises(TrainTestValidationError, match="indices differ"):
        validate_train_test_split(X_train, X_test, y_train, y_test_shifted)


def test_missing_target_value_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    y_test.iloc[0] = np.nan

    with pytest.raises(TrainTestValidationError, match="missing values"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_new_category_in_test_raises() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["University Rating"] = 1
    X_test["University Rating"] = 2

    with pytest.raises(TrainTestValidationError, match="new categor"):
        validate_train_test_split(X_train, X_test, y_train, y_test)


def test_feature_drift_emits_warning_not_error() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["GRE Score"] = 260
    X_test["GRE Score"] = 340

    with pytest.warns(UserWarning, match="GRE Score"):
        report = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert report["drift"]["GRE Score"] == pytest.approx(1.0)
    assert any("GRE Score" in message for message in report["warnings"])
    assert all(check["status"] == "PASS" for check in report["checks"])


def test_target_drift_emits_warning_not_error() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    y_train = y_train.copy()
    y_test = y_test.copy()
    y_train.iloc[:] = 0.3
    y_test.iloc[:] = 0.9

    with pytest.warns(UserWarning, match="Target drift"):
        report = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert report["drift"][TARGET_COLUMN] == pytest.approx(1.0)
    assert any("Target drift" in message for message in report["warnings"])


def test_validation_does_not_mutate_inputs() -> None:
    X_train, X_test, y_train, y_test = split_features(build_balanced_frame())
    X_train_before = X_train.copy()
    X_test_before = X_test.copy()
    y_train_before = y_train.copy()
    y_test_before = y_test.copy()

    validate_train_test_split(X_train, X_test, y_train, y_test)

    pd.testing.assert_frame_equal(X_train, X_train_before)
    pd.testing.assert_frame_equal(X_test, X_test_before)
    pd.testing.assert_series_equal(y_train, y_train_before)
    pd.testing.assert_series_equal(y_test, y_test_before)


def test_repeated_validation_is_deterministic() -> None:
    first = validate_train_test_split(*split_features(build_balanced_frame()))
    second = validate_train_test_split(*split_features(build_balanced_frame()))

    assert first == second


def test_null_rates_reported_per_partition() -> None:
    frame = build_balanced_frame()
    frame.loc[frame.index[:3], "GRE Score"] = np.nan
    X_train, X_test, y_train, y_test = split_features(frame)

    report = validate_train_test_split(X_train, X_test, y_train, y_test)

    train_missing = report["null_rates"]["train"]["GRE Score"] * report["rows"]["train"]
    test_missing = report["null_rates"]["test"]["GRE Score"] * report["rows"]["test"]
    assert train_missing + test_missing == pytest.approx(3, abs=0.01)
    assert report["null_rates"]["train"][TARGET_COLUMN] == 0.0
    assert report["null_rates"]["test"][TARGET_COLUMN] == 0.0


def test_structural_failure_stops_pipeline_before_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = build_balanced_frame()
    frame.loc[frame.index[0], TARGET_COLUMN] = np.nan
    features_path = tmp_path / "features.parquet"
    frame.to_parquet(features_path, index=False, engine="pyarrow")
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("CV must not run when split validation fails")

    monkeypatch.setattr(train_pipeline, "cross_validate_model", fail_if_called)

    with pytest.raises(TrainTestValidationError, match="missing values"):
        run_training_pipeline(
            features_path=features_path,
            model_output_path=model_path,
            metrics_output_path=metrics_path,
        )

    assert not model_path.exists()
    assert not metrics_path.exists()


def test_run_training_pipeline_persists_split_validation(tmp_path: Path) -> None:
    frame = build_balanced_frame()
    features_path = tmp_path / "features.parquet"
    frame.to_parquet(features_path, index=False, engine="pyarrow")
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    output = run_training_pipeline(
        features_path=features_path,
        model_output_path=model_path,
        metrics_output_path=metrics_path,
    )

    split_validation = output["report"]["split_validation"]
    assert split_validation["rows"] == {"train": TRAIN_ROW_COUNT, "test": TEST_ROW_COUNT}
    assert all(check["status"] == "PASS" for check in split_validation["checks"])
    assert split_validation["warnings"] == []

    with metrics_path.open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["split_validation"] == split_validation


@pytest.mark.skipif(
    not default_feature_input_path().is_file(),
    reason="Feature set not available in this environment (synthetic fixtures cover CI).",
)
def test_real_feature_dataset_canonical_split_passes() -> None:
    X_train, X_test, y_train, y_test = split_features(read_features())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_train_test_split(X_train, X_test, y_train, y_test)

    assert report["rows"] == {"train": 376, "test": 95}
    assert report["warnings"] == []
    assert caught == []
    assert all(check["status"] == "PASS" for check in report["checks"])
