"""MLflow experiment tracking practice - Unit 11 academic evidence.

Academic practice / evidence for Unit 11 (Experiment Tracking). This does not
modify the production system: it reads the official training report and the
canonical model artifact read-only and registers them in a local MLflow
tracking store. No re-training, no model selection, no metric recalculation:
MLflow only records evidence produced by the existing, methodologically valid
process, and the Test set never participates in any decision here.

Local configuration (no remote server, no credentials):

- Tracking backend : SQLite database ``mlflow.db`` at the repository root
  (``sqlite:///<repo>/mlflow.db``).
- Artifact store   : local directory ``mlruns/`` at the repository root
  (pinned explicitly when the experiment is created).
- UI               : ``MLFLOW_TRACKING_URI=sqlite:///<repo>/mlflow.db mlflow ui``

The canonical artifact ``models/05_model_selection_pipeline.joblib`` is
registered via ``mlflow.log_artifact`` (an exact copy, never re-serialized).

Usage:

    uv run python scripts/mlflow_experiment_tracking.py

Exit code: 0 when the experiment/run is created and the official parameters,
metrics and model artifact are registered; non-zero otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

REPORT_PATH_PARTS: tuple[str, ...] = ("data", "08_reporting", "training_metrics.json")
MODEL_PATH_PARTS: tuple[str, ...] = ("models", "05_model_selection_pipeline.joblib")

EXPERIMENT_NAME = "admissions_experiment_tracking"
ARTIFACT_DIR_NAME = "mlruns"
DB_FILENAME = "mlflow.db"

REQUIRED_REPORT_KEYS: dict[str, tuple[str, ...]] = {
    "model": (),
    "test_size": (),
    "random_state": (),
    "cv_config": ("n_splits", "shuffle", "random_state"),
    "train": ("rmse", "mae", "r2"),
    "cv": ("rmse_mean", "rmse_std", "mae_mean", "r2_mean"),
    "test": ("rmse", "mae", "r2"),
    "gaps": ("train_minus_cv_rmse", "test_minus_cv_rmse"),
}


def find_repository_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for pyproject.toml."""
    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repository root.")


def load_official_report(report_path: Path) -> dict[str, Any]:
    """Read the official training report and validate its required keys."""
    if not report_path.is_file():
        raise FileNotFoundError(f"Training report not found: {report_path}")
    with report_path.open(encoding="utf-8") as handle:
        report: dict[str, Any] = json.load(handle)

    missing_keys = [
        f"{section}.{key}"
        for section, keys in REQUIRED_REPORT_KEYS.items()
        for key in keys
        if section not in report or (keys and key not in report[section])
    ]
    if missing_keys:
        raise KeyError(f"Training report is missing required keys: {missing_keys}")
    return report


def extract_params(report: dict[str, Any]) -> dict[str, str]:
    """Return the experiment parameters available in the official report."""
    cv_config = report["cv_config"]
    return {
        "model": str(report["model"]),
        "test_size": str(report["test_size"]),
        "random_state": str(report["random_state"]),
        "cv_n_splits": str(cv_config["n_splits"]),
        "cv_shuffle": str(cv_config["shuffle"]),
        "cv_random_state": str(cv_config["random_state"]),
    }


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Return the official metrics available in the training report."""
    metrics: dict[str, float] = {
        "train_rmse": float(report["train"]["rmse"]),
        "train_mae": float(report["train"]["mae"]),
        "train_r2": float(report["train"]["r2"]),
        "cv_rmse_mean": float(report["cv"]["rmse_mean"]),
        "cv_rmse_std": float(report["cv"]["rmse_std"]),
        "cv_mae_mean": float(report["cv"]["mae_mean"]),
        "cv_r2_mean": float(report["cv"]["r2_mean"]),
        "test_rmse": float(report["test"]["rmse"]),
        "test_mae": float(report["test"]["mae"]),
        "test_r2": float(report["test"]["r2"]),
        "gap_train_minus_cv_rmse": float(report["gaps"]["train_minus_cv_rmse"]),
        "gap_test_minus_cv_rmse": float(report["gaps"]["test_minus_cv_rmse"]),
    }
    gap_diagnostics = report.get("gap_diagnostics")
    if isinstance(gap_diagnostics, dict):
        reference_std = gap_diagnostics.get("reference_rmse_std")
        if isinstance(reference_std, (int, float)):
            metrics["gap_reference_rmse_std"] = float(reference_std)
        for key in ("train_minus_cv_rmse", "test_minus_cv_rmse"):
            entry = gap_diagnostics.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("gap"), (int, float)):
                metrics[f"{key}_gap"] = float(entry["gap"])
    return metrics


def extract_tags(report: dict[str, Any]) -> dict[str, str]:
    """Return the diagnostic verdict tags available in the official report.

    The ``within_fold_variability`` verdict is a classification of the gap
    diagnosis, so it is registered as a run tag (not as a metric).
    """
    tags: dict[str, str] = {}
    gap_diagnostics = report.get("gap_diagnostics")
    if isinstance(gap_diagnostics, dict):
        for key in ("train_minus_cv_rmse", "test_minus_cv_rmse"):
            entry = gap_diagnostics.get(key)
            if isinstance(entry, dict) and "within_fold_variability" in entry:
                tags[f"{key}_within_fold_variability"] = str(
                    entry["within_fold_variability"]
                ).lower()
    return tags


def _get_or_create_experiment_id(
    client: MlflowClient,
    experiment_name: str,
    artifact_location: str,
) -> str:
    """Return the experiment id, creating the experiment if it does not exist.

    The create call is retried against a race condition: if another process
    created the experiment concurrently, MLflow raises
    ``RESOURCE_ALREADY_EXISTS`` and the experiment is fetched again instead.
    """
    existing_experiment = client.get_experiment_by_name(experiment_name)
    if existing_experiment is not None:
        return str(existing_experiment.experiment_id)

    try:
        return str(
            client.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
            )
        )
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        concurrent_experiment = client.get_experiment_by_name(experiment_name)
        if concurrent_experiment is None:
            raise
        return str(concurrent_experiment.experiment_id)


def track_experiment(
    report: dict[str, Any],
    model_path: Path,
    tracking_uri: str,
    experiment_name: str,
    artifact_location: str | None = None,
) -> str:
    """Register the official experiment evidence in a local MLflow store.

    Logs the reported parameters, metrics and diagnostic tags and registers
    the canonical model artifact as an exact copy (never re-serialized).
    Returns the run id.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _get_or_create_experiment_id(
        client,
        experiment_name,
        artifact_location or str(Path.cwd() / ARTIFACT_DIR_NAME),
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run() as run:
        mlflow.log_params(extract_params(report))
        mlflow.log_metrics(extract_metrics(report))
        mlflow.set_tags(extract_tags(report))
        mlflow.log_artifact(str(model_path))
        return str(run.info.run_id)


def main() -> int:
    """Run the tracking practice with the repository defaults and return the exit code."""
    root = find_repository_root()
    report = load_official_report(root / Path(*REPORT_PATH_PARTS))
    model_path = root / Path(*MODEL_PATH_PARTS)
    tracking_uri = f"sqlite:///{root / DB_FILENAME}"
    artifact_location = str(root / ARTIFACT_DIR_NAME)

    run_id = track_experiment(
        report,
        model_path,
        tracking_uri,
        EXPERIMENT_NAME,
        artifact_location,
    )

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run id: {run_id}")
    print("Parameters registered:")
    for key, value in extract_params(report).items():
        print(f"  - {key}: {value}")
    print("Metrics registered (official values from training_metrics.json):")
    for metric_name, metric_value in extract_metrics(report).items():
        print(f"  - {metric_name}: {metric_value}")
    tags = extract_tags(report)
    if tags:
        print("Diagnostic tags registered:")
        for tag_name, tag_value in tags.items():
            print(f"  - {tag_name}: {tag_value}")
    print(f"Model artifact registered: {model_path.name} (exact copy, read-only source)")
    print(f"Tracking backend: {tracking_uri}")
    print(f"Artifact store: {artifact_location}")
    print(f"Local UI: MLFLOW_TRACKING_URI={tracking_uri} mlflow ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
