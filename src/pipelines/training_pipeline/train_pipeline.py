"""Training pipeline for the Graduate Admissions project.

The pipeline consumes the curated feature set persisted by the feature
pipeline and produces the fitted model artifact consumed by the inference
pipeline, following the FTI training pipeline contract: select features,
apply model dependent transformations, train and validate.

Data-leakage contract (Issue #31 acceptance criteria)
-----------------------------------------------------

- The train/test split is defined once, before any fitting, and is never
  redrawn.
- The preprocessor is reused unfitted from the feature pipeline and is
  fitted only inside the scikit-learn Pipeline: independently within every
  cross-validation fold and once more in the fresh final fit.
- Cross-validation operates exclusively on the training set; the test set
  is never passed to :func:`cross_validate_model`.
- The test set participates only in the final evaluation of the refit
  model.
- The serialized artifact comes from a fresh fit on the complete training
  set, never from a cross-validation fold estimator.

Validation scope (Issue #31)
----------------------------

- K-Fold cross-validation (5 folds, shuffled, seed 42) on the training
  set only, reproducing the historical outer CV of Issue #6.
- Train, CV and Test metrics: RMSE, MAE and R2.
- Minimal over/underfitting signal: train-CV and test-CV RMSE gaps
  compared against the fold-level standard deviation.

Split validation (Issue #32)
----------------------------

:func:`validate_train_test_split` runs immediately after the split and
before any fitting:

- structural checks (disjointness, cross-partition duplicates, canonical
  size ratio, schema parity, feature/target alignment, target
  completeness, no new ordinal/binary category in test) raise
  :class:`TrainTestValidationError` and stop training before CV;
- distribution checks (two-sample Kolmogorov-Smirnov drift per feature
  and for the target) only emit :class:`UserWarning` and are recorded in
  the returned :class:`TrainTestValidationReport`;
- the report is persisted inside the metrics report so the split
  evidence survives next to the artifact.

Deep validation analysis (learning curves, formal diagnostics,
visualizations, improvement actions) belongs to Issue #33. Train/test
split checks belong to Issue #32.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_REPOSITORY_ROOT = _SRC_DIR.parent

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    BINARY_FEATURES,
    FEATURE_COLUMNS,
    ORDINAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)

FEATURE_INPUT_FILENAME = "admission_features.parquet"
MODEL_OUTPUT_FILENAME = "05_model_selection_pipeline.joblib"
METRICS_OUTPUT_FILENAME = "training_metrics.json"

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
CV_N_SPLITS = 5
CV_RANDOM_STATE = 42

SIZE_RATIO_TOLERANCE = 1
FEATURE_DRIFT_THRESHOLD = 0.20
TARGET_DRIFT_THRESHOLD = 0.15

CATEGORICAL_SPLIT_FEATURES: tuple[str, ...] = ORDINAL_FEATURES + BINARY_FEATURES

SCORING_METRICS: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


class SplitCheckResult(TypedDict):
    """One split-validation check with an explicit outcome status."""

    name: str
    status: str
    detail: str


class TrainTestValidationReport(TypedDict):
    """Structured, deterministic split-validation evidence.

    Structural failures raise :class:`TrainTestValidationError` instead of
    producing a report, so every entry in ``checks`` has status ``PASS``;
    drift findings are recorded as ``WARNING`` entries in ``warnings`` and
    keep their measured values in ``drift``.
    """

    rows: dict[str, int]
    ratio: dict[str, float]
    checks: list[SplitCheckResult]
    drift: dict[str, float | None]
    null_rates: dict[str, dict[str, float]]
    warnings: list[str]


class CrossValidationMetrics(TypedDict):
    """Fold-level and summary metrics produced by K-Fold cross-validation."""

    folds: list[dict[str, float]]
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float


class MetricsReport(TypedDict):
    """Deterministic training evidence report persisted next to the artifact."""

    model: str
    test_size: float
    random_state: int
    cv_config: dict[str, object]
    rows: dict[str, int]
    train: dict[str, float]
    cv: CrossValidationMetrics
    test: dict[str, float]
    gaps: dict[str, float]
    split_validation: TrainTestValidationReport


class TrainingRunOutput(TypedDict):
    """Paths and evidence returned by one end-to-end pipeline execution."""

    model_path: str
    metrics_path: str
    report: MetricsReport


def default_feature_input_path() -> Path:
    """Return the repository-relative path of the persisted feature set."""
    return _REPOSITORY_ROOT / "data" / "04_feature" / FEATURE_INPUT_FILENAME


def default_model_output_path() -> Path:
    """Return the repository-relative path of the serialized model artifact."""
    return _REPOSITORY_ROOT / "models" / MODEL_OUTPUT_FILENAME


def default_metrics_output_path() -> Path:
    """Return the repository-relative path of the evaluation report."""
    return _REPOSITORY_ROOT / "data" / "08_reporting" / METRICS_OUTPUT_FILENAME


def read_features(features_path: Path | None = None) -> pd.DataFrame:
    """Read the persisted feature set and verify its schema."""
    path = features_path or default_feature_input_path()
    if not path.is_file():
        raise FileNotFoundError(f"Feature set not found: {path}")

    features_df = pd.read_parquet(path)
    expected_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    actual_columns = list(features_df.columns)
    missing_columns = [column for column in expected_columns if column not in actual_columns]
    unexpected_columns = [column for column in actual_columns if column not in expected_columns]
    if not features_df.columns.is_unique or missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if unexpected_columns:
            details.append(f"unexpected columns: {unexpected_columns}")
        raise ValueError("Invalid feature schema (" + "; ".join(details) + ").")
    return features_df


def split_features(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the feature set into deterministic train/test sets.

    The split is defined once, before any fitting. Downstream Issues
    (split checks, deeper validation) consume this exact contract.
    """
    features = features_df[list(FEATURE_COLUMNS)]
    target = features_df[TARGET_COLUMN]
    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=SPLIT_RANDOM_STATE,
        ),
    )


class TrainTestValidationError(ValueError):
    """Raised when the train/test split fails structural validation rules."""


def _ks_statistic(train_values: np.ndarray, test_values: np.ndarray) -> float:
    """Return the two-sample Kolmogorov-Smirnov distance between two samples.

    Missing values are ignored. Ties are handled exactly: the distance is
    the maximum absolute difference between the empirical CDFs evaluated
    on the sorted union of observed values, which matches the KS definition
    for discrete distributions.
    """
    train_values = pd.Series(train_values, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)
    test_values = pd.Series(test_values, dtype="Float64").to_numpy(dtype=float, na_value=np.nan)
    train_values = train_values[~np.isnan(train_values)]
    test_values = test_values[~np.isnan(test_values)]
    n_train, n_test = len(train_values), len(test_values)
    if n_train == 0 or n_test == 0:
        return float("nan")

    points = np.sort(np.unique(np.concatenate([train_values, test_values])))
    train_cdf = np.searchsorted(np.sort(train_values), points, side="right") / n_train
    test_cdf = np.searchsorted(np.sort(test_values), points, side="right") / n_test
    return float(np.max(np.abs(train_cdf - test_cdf)))


def _check_partition_sizes(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Require both partitions to contain rows."""
    n_train, n_test = len(X_train), len(X_test)
    if n_train == 0 or n_test == 0:
        return [f"empty partition (train {n_train} rows, test {n_test} rows)"]
    return []


def _check_index_integrity(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Require unique indices within partitions and disjointness across them."""
    failures: list[str] = []
    if not X_train.index.is_unique or not X_test.index.is_unique:
        failures.append("duplicate index values within a partition")

    index_overlap = X_train.index.intersection(X_test.index)
    if not index_overlap.empty:
        failures.append(f"train and test indices overlap ({len(index_overlap)} shared value(s))")
    return failures


def _columns_are_unique_and_equal(X_train: pd.DataFrame, X_test: pd.DataFrame) -> bool:
    """Return whether both partitions share the same unique column labels."""
    return (
        X_train.columns.is_unique
        and X_test.columns.is_unique
        and list(X_train.columns) == list(X_test.columns)
    )


def _column_is_indexable(X_train: pd.DataFrame, X_test: pd.DataFrame, column: str) -> bool:
    """Return whether ``column`` yields a Series in both partitions.

    A missing label raises ``KeyError`` and a duplicated label returns a
    DataFrame; both conditions are reported by the schema-parity check
    instead of being re-indexed here.
    """
    return (
        column in X_train.columns
        and column in X_test.columns
        and list(X_train.columns).count(column) == 1
        and list(X_test.columns).count(column) == 1
    )


def _check_cross_partition_duplicates(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Require that no identical feature row appears in both partitions.

    Skipped when the column labels differ or repeat between partitions:
    the schema-parity check already reports that condition and row-wise
    comparison is only defined for identical unique schemas.
    """
    if not _columns_are_unique_and_equal(X_train, X_test):
        return []

    combined = pd.concat([X_train, X_test], keys=["train", "test"], names=["partition", None])
    duplicate_mask = combined.duplicated(keep=False)
    if not duplicate_mask.any():
        return []

    duplicated_partitions = set(combined.loc[duplicate_mask].index.get_level_values("partition"))
    if {"train", "test"} <= duplicated_partitions:
        return ["identical feature rows appear in both train and test partitions"]
    return []


def _check_size_ratio(n_test: int, total_rows: int) -> list[str]:
    """Require the canonical test_size ratio with a small row tolerance."""
    expected_test_rows = round(total_rows * TEST_SIZE)
    if abs(n_test - expected_test_rows) > SIZE_RATIO_TOLERANCE:
        return [
            f"test size {n_test} deviates from the canonical {TEST_SIZE:.0%} split "
            f"of {total_rows} rows (expected ~{expected_test_rows}, "
            f"tolerance ±{SIZE_RATIO_TOLERANCE})"
        ]
    return []


def _check_schema_parity(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Require identical feature columns, order and dtypes in both partitions.

    Duplicate labels are detected before any column access: indexing a
    duplicated label returns a DataFrame, so ``.dtype`` would fail with an
    incidental exception instead of the aggregated
    :class:`TrainTestValidationError`.
    """
    failures: list[str] = []
    if not X_train.columns.is_unique:
        failures.append("train has duplicate feature column(s)")
    if not X_test.columns.is_unique:
        failures.append("test has duplicate feature column(s)")
    if list(X_train.columns) != list(X_test.columns):
        failures.append("train and test feature columns differ")
    if failures:
        return failures

    dtype_mismatches = [
        column
        for column in X_train.columns
        if str(X_train[column].dtype) != str(X_test[column].dtype)
    ]
    if dtype_mismatches:
        return [f"train and test dtypes differ on column(s): {dtype_mismatches}"]
    return []


def _check_feature_target_alignment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[str]:
    """Require feature and target lengths and indices to match per partition."""
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        return ["feature and target lengths differ within a partition"]
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        return ["feature and target indices differ within a partition"]
    return []


def _check_target_completeness(y_train: pd.Series, y_test: pd.Series) -> list[str]:
    """Require the target to have no missing values in either partition."""
    if y_train.isna().any() or y_test.isna().any():
        return ["target contains missing values in a partition"]
    return []


def _check_no_new_categories(X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
    """Require ordinal/binary test values to be a subset of train values.

    Columns that are missing or duplicated in either partition are
    skipped: the schema-parity check already reported them, so indexing
    them here must never raise an incidental exception instead of the
    aggregated :class:`TrainTestValidationError`.
    """
    failures: list[str] = []
    for column in CATEGORICAL_SPLIT_FEATURES:
        if not _column_is_indexable(X_train, X_test, column):
            continue
        train_values = set(X_train[column].dropna().unique())
        test_values = set(X_test[column].dropna().unique())
        new_values = sorted(test_values - train_values)
        if new_values:
            failures.append(f"new category(ies) in test for '{column}': {new_values}")
    return failures


def _feature_drift(
    X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float
) -> tuple[dict[str, float | None], list[str]]:
    """Measure KS drift per feature and build the warning messages."""
    drift: dict[str, float | None] = {}
    messages: list[str] = []
    for column in FEATURE_COLUMNS:
        ks = _ks_statistic(X_train[column].to_numpy(), X_test[column].to_numpy())
        drift[column] = None if math.isnan(ks) else float(ks)
        if ks > threshold:
            messages.append(
                f"Feature drift detected in '{column}': KS {ks:.4f} exceeds threshold {threshold}"
            )
    return drift, messages


def _target_drift(
    y_train: pd.Series, y_test: pd.Series, threshold: float
) -> tuple[float | None, list[str]]:
    """Measure KS drift for the target and build the warning message."""
    ks = _ks_statistic(y_train.to_numpy(), y_test.to_numpy())
    drift_value = None if math.isnan(ks) else float(ks)
    if ks > threshold:
        return drift_value, [f"Target drift detected: KS {ks:.4f} exceeds threshold {threshold}"]
    return drift_value, []


def _build_split_checks(n_test: int, total_rows: int) -> list[SplitCheckResult]:
    """Assemble the PASS entries of the structural checks that just ran."""
    return [
        {
            "name": "disjoint_indices",
            "status": "PASS",
            "detail": "train and test indices are disjoint",
        },
        {
            "name": "unique_indices",
            "status": "PASS",
            "detail": "indices are unique within each partition",
        },
        {
            "name": "cross_partition_duplicates",
            "status": "PASS",
            "detail": "no identical feature row appears in both partitions",
        },
        {
            "name": "size_ratio",
            "status": "PASS",
            "detail": (
                f"test ratio {n_test / total_rows:.4f} matches "
                f"test_size {TEST_SIZE} (tolerance ±{SIZE_RATIO_TOLERANCE} row)"
            ),
        },
        {
            "name": "schema_parity",
            "status": "PASS",
            "detail": (
                f"train and test share {len(FEATURE_COLUMNS)} feature columns with matching dtypes"
            ),
        },
        {
            "name": "alignment",
            "status": "PASS",
            "detail": "features and targets are aligned within each partition",
        },
        {
            "name": "target_completeness",
            "status": "PASS",
            "detail": "target has no missing values in either partition",
        },
        {
            "name": "no_new_categories",
            "status": "PASS",
            "detail": "ordinal/binary test values are a subset of train values",
        },
    ]


def _build_null_rates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Report the null fraction per column for each partition separately."""
    null_rates: dict[str, dict[str, float]] = {"train": {}, "test": {}}
    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        train_column = X_train[column] if column != TARGET_COLUMN else y_train
        test_column = X_test[column] if column != TARGET_COLUMN else y_test
        null_rates["train"][column] = round(float(train_column.isna().mean()), 6)
        null_rates["test"][column] = round(float(test_column.isna().mean()), 6)
    return null_rates


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> TrainTestValidationReport:
    """Validate the train/test separation before any model fitting.

    Structural failures (overlapping indices, cross-partition duplicate
    rows, non-canonical size ratio, schema divergence, misaligned
    features/targets, missing target values, new ordinal/binary category
    in test) are collected and raise :class:`TrainTestValidationError`.
    Distribution drift (KS above the fixed Issue #32 thresholds) only
    emits :class:`UserWarning` and is recorded in the returned report,
    which also carries train/test sizes, ratio, check outcomes and
    per-partition null rates. The function never mutates its inputs.
    """
    failures = [
        *_check_partition_sizes(X_train, X_test),
        *_check_index_integrity(X_train, X_test),
        *_check_cross_partition_duplicates(X_train, X_test),
        *_check_size_ratio(len(X_test), len(X_train) + len(X_test)),
        *_check_schema_parity(X_train, X_test),
        *_check_feature_target_alignment(X_train, X_test, y_train, y_test),
        *_check_target_completeness(y_train, y_test),
        *_check_no_new_categories(X_train, X_test),
    ]
    if failures:
        header = f"Train/test validation failed with {len(failures)} issue(s):"
        details = "\n".join(f"- {failure}" for failure in failures)
        raise TrainTestValidationError(f"{header}\n{details}")

    feature_drift, feature_warnings = _feature_drift(X_train, X_test, FEATURE_DRIFT_THRESHOLD)
    target_drift_value, target_warnings = _target_drift(y_train, y_test, TARGET_DRIFT_THRESHOLD)
    drift_warnings = [*feature_warnings, *target_warnings]
    for message in drift_warnings:
        warnings.warn(message, UserWarning, stacklevel=2)

    return {
        "rows": {"train": len(X_train), "test": len(X_test)},
        "ratio": {
            "test_to_train": len(X_test) / len(X_train),
            "test_to_total": len(X_test) / (len(X_train) + len(X_test)),
        },
        "checks": _build_split_checks(len(X_test), len(X_train) + len(X_test)),
        "drift": {**feature_drift, TARGET_COLUMN: target_drift_value},
        "null_rates": _build_null_rates(X_train, X_test, y_train, y_test),
        "warnings": drift_warnings,
    }


def build_model_pipeline() -> Pipeline:
    """Build the unfitted preprocessor + LinearRegression Pipeline.

    The preprocessor is reused from the feature pipeline and is never
    fitted here: fitting happens inside every cross-validation fold and in
    the fresh final fit, always through this complete Pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def evaluate_model(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Score a fitted Pipeline on one dataset without refitting it."""
    predictions = model_pipeline.predict(X)
    return {
        "rmse": float(root_mean_squared_error(y, predictions)),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }


def cross_validate_model(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> CrossValidationMetrics:
    """Validate the Pipeline with K-Fold CV on the training set only.

    Every fold fits its own complete Pipeline clone (preprocessor plus
    model), so imputation, encoding and scaling statistics are learned
    independently within each training fold. Fold estimators are discarded
    after scoring: the serialized artifact never comes from this step. The
    caller's Pipeline is not fitted by this function.
    """
    cv = KFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    results = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        scoring=SCORING_METRICS,
        cv=cv,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )

    fold_metrics: list[dict[str, float]] = []
    rmse_values = -results["test_rmse"]
    mae_values = -results["test_mae"]
    r2_values = results["test_r2"]
    for rmse, mae, r2 in zip(rmse_values, mae_values, r2_values, strict=True):
        fold_metrics.append({"rmse": float(rmse), "mae": float(mae), "r2": float(r2)})

    return {
        "folds": fold_metrics,
        "rmse_mean": float(rmse_values.mean()),
        "rmse_std": float(rmse_values.std(ddof=1)),
        "mae_mean": float(mae_values.mean()),
        "mae_std": float(mae_values.std(ddof=1)),
        "r2_mean": float(r2_values.mean()),
        "r2_std": float(r2_values.std(ddof=1)),
    }


def build_metrics_report(
    train_metrics: dict[str, float],
    cv_metrics: CrossValidationMetrics,
    test_metrics: dict[str, float],
    split_validation: TrainTestValidationReport,
) -> MetricsReport:
    """Assemble the deterministic training evidence report."""
    return {
        "model": "LinearRegression",
        "test_size": TEST_SIZE,
        "random_state": SPLIT_RANDOM_STATE,
        "cv_config": {
            "n_splits": CV_N_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
        },
        "rows": dict(split_validation["rows"]),
        "train": train_metrics,
        "cv": cv_metrics,
        "test": test_metrics,
        "gaps": {
            "train_minus_cv_rmse": train_metrics["rmse"] - cv_metrics["rmse_mean"],
            "test_minus_cv_rmse": test_metrics["rmse"] - cv_metrics["rmse_mean"],
        },
        "split_validation": split_validation,
    }


def save_model(model_pipeline: Pipeline, output_path: Path) -> Path:
    """Serialize the fitted complete Pipeline and return its path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, output_path)
    return output_path


def save_metrics(report: MetricsReport, output_path: Path) -> Path:
    """Persist the deterministic evaluation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def run_training_pipeline(
    features_path: Path | None = None,
    model_output_path: Path | None = None,
    metrics_output_path: Path | None = None,
) -> TrainingRunOutput:
    """Run the training pipeline end to end and return its outputs.

    Data flow (leakage-safe by construction):

    1. read the feature set;
    2. split once into train/test;
    3. validate the split (structural failures stop training here);
    4. K-Fold cross-validation on the training set only;
    5. fresh final fit of the complete Pipeline on the full training set;
    6. evaluate train and test with the final refit model;
    7. serialize the artifact and the metrics report.

    The test set is inspected by the pre-fit split validation but is
    excluded from fitting, model selection and cross-validation; it
    participates only in the final evaluation.
    """
    features_df = read_features(features_path)
    X_train, X_test, y_train, y_test = split_features(features_df)
    split_validation = validate_train_test_split(X_train, X_test, y_train, y_test)

    model_pipeline = build_model_pipeline()
    cv_metrics = cross_validate_model(model_pipeline, X_train, y_train)

    model_pipeline.fit(X_train, y_train)
    train_metrics = evaluate_model(model_pipeline, X_train, y_train)
    test_metrics = evaluate_model(model_pipeline, X_test, y_test)

    report = build_metrics_report(
        train_metrics,
        cv_metrics,
        test_metrics,
        split_validation,
    )

    model_path = save_model(model_pipeline, model_output_path or default_model_output_path())
    metrics_path = save_metrics(report, metrics_output_path or default_metrics_output_path())
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report": report,
    }


def main() -> None:
    """Run the training pipeline with the repository defaults from the CLI."""
    output = run_training_pipeline()
    report = output["report"]
    train_metrics = report["train"]
    cv_metrics = report["cv"]
    test_metrics = report["test"]
    gaps = report["gaps"]

    print(f"Model: {report['model']}")
    print(
        f"Train/test split: {report['rows']['train']} train rows, "
        f"{report['rows']['test']} test rows "
        f"(test_size={report['test_size']}, random_state={report['random_state']})"
    )
    split_validation = report["split_validation"]
    print(f"Split validation: {len(split_validation['checks'])} structural check(s) passed")
    for warning in split_validation["warnings"]:
        print(f"  WARNING: {warning}")
    print(
        f"Cross-validation: {report['cv_config']['n_splits']}-fold KFold on train only "
        f"(shuffle=True, random_state={report['cv_config']['random_state']})"
    )
    print(
        f"Train metrics: RMSE {train_metrics['rmse']:.6f} | "
        f"MAE {train_metrics['mae']:.6f} | R2 {train_metrics['r2']:.6f}"
    )
    print(
        f"CV metrics: RMSE {cv_metrics['rmse_mean']:.6f} ± {cv_metrics['rmse_std']:.6f} | "
        f"MAE {cv_metrics['mae_mean']:.6f} ± {cv_metrics['mae_std']:.6f} | "
        f"R2 {cv_metrics['r2_mean']:.6f} ± {cv_metrics['r2_std']:.6f}"
    )
    print(
        f"Test metrics: RMSE {test_metrics['rmse']:.6f} | "
        f"MAE {test_metrics['mae']:.6f} | R2 {test_metrics['r2']:.6f}"
    )
    print(
        f"RMSE gaps: train-CV {gaps['train_minus_cv_rmse']:+.6f} | "
        f"test-CV {gaps['test_minus_cv_rmse']:+.6f} "
        f"(fold RMSE std {cv_metrics['rmse_std']:.6f})"
    )
    print(f"Model artifact: {output['model_path']}")
    print(f"Metrics report: {output['metrics_path']}")


if __name__ == "__main__":
    main()
