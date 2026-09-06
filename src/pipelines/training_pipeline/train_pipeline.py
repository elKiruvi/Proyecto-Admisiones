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

Deep validation analysis (learning curves, formal diagnostics,
visualizations, improvement actions) belongs to Issue #33. Train/test
split checks belong to Issue #32.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict, cast

import joblib
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
    FEATURE_COLUMNS,
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

SCORING_METRICS: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


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
    if missing_columns or unexpected_columns:
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
    train_rows: int,
    test_rows: int,
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
        "rows": {"train": train_rows, "test": test_rows},
        "train": train_metrics,
        "cv": cv_metrics,
        "test": test_metrics,
        "gaps": {
            "train_minus_cv_rmse": train_metrics["rmse"] - cv_metrics["rmse_mean"],
            "test_minus_cv_rmse": test_metrics["rmse"] - cv_metrics["rmse_mean"],
        },
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
    3. K-Fold cross-validation on the training set only;
    4. fresh final fit of the complete Pipeline on the full training set;
    5. evaluate train and test with the final refit model;
    6. serialize the artifact and the metrics report.

    The test set participates exclusively in step 5.
    """
    features_df = read_features(features_path)
    X_train, X_test, y_train, y_test = split_features(features_df)

    model_pipeline = build_model_pipeline()
    cv_metrics = cross_validate_model(model_pipeline, X_train, y_train)

    model_pipeline.fit(X_train, y_train)
    train_metrics = evaluate_model(model_pipeline, X_train, y_train)
    test_metrics = evaluate_model(model_pipeline, X_test, y_test)

    report = build_metrics_report(
        train_metrics,
        cv_metrics,
        test_metrics,
        train_rows=X_train.shape[0],
        test_rows=X_test.shape[0],
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
