"""AppTest coverage for the Streamlit online and batch prediction modes."""

import re
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from pipelines.feature_pipeline.feature_pipeline import FEATURE_COLUMNS
from pipelines.inference_pipeline.inference_pipeline import PREDICTION_COLUMN

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPOSITORY_ROOT / "src" / "inference" / "streamlit_app.py"

SAMPLE_BATCH_CSV = (
    b"GRE Score,TOEFL Score,University Rating,SOP,LOR,CGPA,Research\n"
    b"316,107,3,3.5,3.0,8.62,1\n"
    b"298,95,2,2.0,2.5,7.90,0\n"
    b"330,118,5,4.0,4.0,9.40,1\n"
)

MISSING_CGPA_CSV = (
    b"GRE Score,TOEFL Score,University Rating,SOP,LOR,Research\n316,107,3,3.5,3.0,1\n"
)

MALFORMED_CSV = b'a,b\n"unclosed,2\n'

EXPECTED_BATCH_ROW_COUNT = 3


def run_app() -> AppTest:
    """Run the Streamlit entrypoint headlessly."""
    return AppTest.from_file(str(ENTRYPOINT), default_timeout=30).run()


def test_app_renders_both_modes_without_exception() -> None:
    at = run_app()

    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Online prediction", "Batch prediction"]


def test_online_mode_submits_and_displays_a_prediction() -> None:
    at = run_app()
    at.tabs[0].button[0].click().run()

    assert not at.exception
    assert len(at.metric) == 1
    assert re.fullmatch(r"-?\d+\.\d{3}", at.metric[0].value)


def test_batch_mode_uploads_predicts_and_offers_download() -> None:
    at = run_app()
    at.tabs[1].file_uploader[0].set_value(("applicants.csv", SAMPLE_BATCH_CSV, "text/csv")).run()
    at.tabs[1].button[0].click().run()

    assert not at.exception
    assert len(at.success) == 1
    assert len(at.tabs[1].dataframe) == 1
    assert len(at.tabs[1].download_button) == 1

    result_frame = at.tabs[1].dataframe[0].value
    assert sorted(result_frame.columns) == sorted([*FEATURE_COLUMNS, PREDICTION_COLUMN])
    assert result_frame.columns[-1] == PREDICTION_COLUMN
    assert len(result_frame) == EXPECTED_BATCH_ROW_COUNT
    assert pd.to_numeric(result_frame[PREDICTION_COLUMN]).notna().all()


def test_batch_result_survives_streamlit_rerun() -> None:
    at = run_app()
    at.tabs[1].file_uploader[0].set_value(("applicants.csv", SAMPLE_BATCH_CSV, "text/csv")).run()
    at.tabs[1].button[0].click().run()

    assert len(at.tabs[1].dataframe) == 1
    assert len(at.tabs[1].download_button) == 1

    at.tabs[1].download_button[0].click().run()

    assert not at.exception
    result_frame = at.tabs[1].dataframe[0].value
    assert result_frame.columns[-1] == PREDICTION_COLUMN
    assert len(result_frame) == EXPECTED_BATCH_ROW_COUNT
    assert len(at.tabs[1].download_button) == 1


def test_batch_result_survives_new_upload_before_run() -> None:
    at = run_app()
    at.tabs[1].file_uploader[0].set_value(("applicants.csv", SAMPLE_BATCH_CSV, "text/csv")).run()
    at.tabs[1].button[0].click().run()

    at.tabs[1].file_uploader[0].set_value(("other.csv", MISSING_CGPA_CSV, "text/csv")).run()

    assert not at.exception
    assert len(at.tabs[1].dataframe) == 1
    assert at.tabs[1].dataframe[0].value.columns[-1] == PREDICTION_COLUMN
    assert len(at.tabs[1].download_button) == 1


def test_batch_mode_reports_missing_column_error() -> None:
    at = run_app()
    at.tabs[1].file_uploader[0].set_value(("invalid.csv", MISSING_CGPA_CSV, "text/csv")).run()
    at.tabs[1].button[0].click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "missing columns" in at.error[0].value
    assert len(at.tabs[1].dataframe) == 0
    assert len(at.tabs[1].download_button) == 0


def test_batch_mode_reports_malformed_file_error() -> None:
    at = run_app()
    at.tabs[1].file_uploader[0].set_value(("broken.csv", MALFORMED_CSV, "text/csv")).run()
    at.tabs[1].button[0].click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "could not be parsed as a CSV file" in at.error[0].value
