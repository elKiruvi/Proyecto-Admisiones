"""Tests for the Unit 11 MLflow experiment tracking practice (academic evidence).

These tests validate the practice script's behavior, not production logic.
They are hermetic: every tracking run uses an isolated temporary store, and
the canonical model artifact is only read (never modified). The official
``training_metrics.json`` is not tracked in the repository, so parameter and
metric extraction is verified against a synthetic report that mirrors its
structure; the real-data run remains the script's own demonstration.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from scripts.mlflow_experiment_tracking import (
    _get_or_create_experiment_id,
    extract_metrics,
    extract_params,
    extract_tags,
    load_official_report,
    track_experiment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MODEL_PATH = REPOSITORY_ROOT / "models" / "05_model_selection_pipeline.joblib"

SYNTHETIC_REPORT: dict = {
    "model": "LinearRegression",
    "test_size": 0.2,
    "random_state": 42,
    "cv_config": {"n_splits": 5, "shuffle": True, "random_state": 42},
    "train": {"rmse": 0.06244016435488824, "mae": 0.04508260919933707, "r2": 0.8074258357800073},
    "cv": {
        "rmse_mean": 0.06446251670104981,
        "rmse_std": 0.0073488227452346115,
        "mae_mean": 0.046824762054105594,
        "r2_mean": 0.7862508205727947,
    },
    "test": {"rmse": 0.06849967630119724, "mae": 0.051191608746781844, "r2": 0.787195301893142},
    "gaps": {
        "train_minus_cv_rmse": -0.0020223523461615697,
        "test_minus_cv_rmse": 0.00403715960014743,
    },
    "gap_diagnostics": {
        "rule": "abs(gap) <= rmse_std",
        "reference_rmse_std": 0.0073488227452346115,
        "train_minus_cv_rmse": {"gap": -0.0020223523461615697, "within_fold_variability": True},
        "test_minus_cv_rmse": {"gap": 0.00403715960014743, "within_fold_variability": True},
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_official_report_validates_required_keys(tmp_path: Path) -> None:
    report_path = tmp_path / "training_metrics.json"
    report_path.write_text(json.dumps(SYNTHETIC_REPORT), encoding="utf-8")

    loaded = load_official_report(report_path)
    assert loaded["model"] == "LinearRegression"

    incomplete = {key: value for key, value in SYNTHETIC_REPORT.items() if key != "cv"}
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")

    with pytest.raises(KeyError, match="cv"):
        load_official_report(incomplete_path)


def test_extract_params_and_metrics_match_report_values() -> None:
    params = extract_params(SYNTHETIC_REPORT)
    assert params == {
        "model": "LinearRegression",
        "test_size": "0.2",
        "random_state": "42",
        "cv_n_splits": "5",
        "cv_shuffle": "True",
        "cv_random_state": "42",
    }

    metrics = extract_metrics(SYNTHETIC_REPORT)
    assert metrics["train_rmse"] == pytest.approx(SYNTHETIC_REPORT["train"]["rmse"])
    assert metrics["test_rmse"] == pytest.approx(SYNTHETIC_REPORT["test"]["rmse"])
    assert metrics["cv_rmse_std"] == pytest.approx(SYNTHETIC_REPORT["cv"]["rmse_std"])
    assert metrics["gap_train_minus_cv_rmse"] == pytest.approx(
        SYNTHETIC_REPORT["gaps"]["train_minus_cv_rmse"]
    )


def test_extract_metrics_registers_gap_diagnostics_as_metrics_and_tags() -> None:
    metrics = extract_metrics(SYNTHETIC_REPORT)

    assert metrics["gap_reference_rmse_std"] == pytest.approx(
        SYNTHETIC_REPORT["gap_diagnostics"]["reference_rmse_std"]
    )
    assert metrics["train_minus_cv_rmse_gap"] == pytest.approx(
        SYNTHETIC_REPORT["gap_diagnostics"]["train_minus_cv_rmse"]["gap"]
    )
    assert metrics["test_minus_cv_rmse_gap"] == pytest.approx(
        SYNTHETIC_REPORT["gap_diagnostics"]["test_minus_cv_rmse"]["gap"]
    )
    assert not any("within_fold_variability" in key for key in metrics)

    tags = extract_tags(SYNTHETIC_REPORT)
    assert tags == {
        "train_minus_cv_rmse_within_fold_variability": "true",
        "test_minus_cv_rmse_within_fold_variability": "true",
    }


def test_extract_metrics_without_gap_diagnostics_is_defensive() -> None:
    report = {key: value for key, value in SYNTHETIC_REPORT.items() if key != "gap_diagnostics"}

    metrics = extract_metrics(report)
    tags = extract_tags(report)

    assert "gap_reference_rmse_std" not in metrics
    assert "train_minus_cv_rmse_gap" not in metrics
    assert not any("within_fold_variability" in key for key in metrics)
    assert tags == {}
    assert "gap_train_minus_cv_rmse" in metrics


def test_track_experiment_logs_params_metrics_and_artifact(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlruns")

    run_id = track_experiment(
        SYNTHETIC_REPORT,
        CANONICAL_MODEL_PATH,
        tracking_uri,
        "ci_experiment",
        artifact_location,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["model"] == "LinearRegression"
    assert run.data.params["cv_n_splits"] == "5"
    assert run.data.metrics["test_rmse"] == pytest.approx(SYNTHETIC_REPORT["test"]["rmse"])
    assert run.data.metrics["cv_rmse_std"] == pytest.approx(SYNTHETIC_REPORT["cv"]["rmse_std"])
    assert run.data.metrics["train_minus_cv_rmse_gap"] == pytest.approx(
        SYNTHETIC_REPORT["gap_diagnostics"]["train_minus_cv_rmse"]["gap"]
    )
    assert run.data.metrics["gap_reference_rmse_std"] == pytest.approx(
        SYNTHETIC_REPORT["gap_diagnostics"]["reference_rmse_std"]
    )
    assert "train_minus_cv_rmse_within_fold_variability" not in run.data.metrics
    assert run.data.tags["train_minus_cv_rmse_within_fold_variability"] == "true"
    artifact_paths = [artifact.path for artifact in client.list_artifacts(run_id)]
    assert CANONICAL_MODEL_PATH.name in artifact_paths


def test_get_or_create_experiment_recovers_from_concurrent_creation(tmp_path: Path) -> None:
    concurrent_experiment_id = "42"
    expected_create_calls = 1
    expected_lookup_calls = 2
    duplicate_error = MlflowException("Experiment 'race_experiment' already exists")
    duplicate_error.error_code = "RESOURCE_ALREADY_EXISTS"

    client = Mock()
    client.get_experiment_by_name.side_effect = [
        None,
        Mock(experiment_id=concurrent_experiment_id),
    ]
    client.create_experiment.side_effect = duplicate_error

    experiment_id = _get_or_create_experiment_id(
        client, "race_experiment", str(tmp_path / "mlruns")
    )

    assert experiment_id == concurrent_experiment_id
    assert client.create_experiment.call_count == expected_create_calls
    assert client.get_experiment_by_name.call_count == expected_lookup_calls


def test_get_or_create_experiment_reraises_unexpected_errors(tmp_path: Path) -> None:
    client = Mock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.side_effect = MlflowException(
        "backend unavailable",
        error_code=2,
    )

    with pytest.raises(MlflowException, match="backend unavailable"):
        _get_or_create_experiment_id(client, "race_experiment", str(tmp_path / "mlruns"))


def test_tracking_preserves_canonical_artifact(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = str(tmp_path / "mlruns")
    hash_before = _sha256(CANONICAL_MODEL_PATH)

    track_experiment(
        SYNTHETIC_REPORT,
        CANONICAL_MODEL_PATH,
        tracking_uri,
        "ci_experiment",
        artifact_location,
    )

    assert _sha256(CANONICAL_MODEL_PATH) == hash_before
