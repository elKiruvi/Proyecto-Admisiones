"""Unit tests for the data validation rules of the feature pipeline (Issue #30)."""

from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    CATEGORICAL_DOMAINS,
    CONTINUOUS_RANGES,
    FEATURE_COLUMNS,
    FEATURE_MAX_NULL_FRACTION,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    MAX_NULL_FRACTION,
    TARGET_COLUMN,
    TARGET_MAX_NULL_FRACTION,
    FeatureValidationError,
    build_features,
    read_raw_data,
    run_feature_pipeline,
    validate_features,
)

VALID_ROW_COUNT = 12
EXPECTED_FEATURE_NULL_FRACTION = 0.10
EXPECTED_TARGET_NULL_FRACTION = 0.0

SOP_CATEGORY_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
OUT_OF_RANGE_CGPA = 12.5
OUT_OF_DOMAIN_RATING = 6
OUT_OF_DOMAIN_SOP = 2.7


def build_valid_feature_frame(row_count: int = VALID_ROW_COUNT) -> pd.DataFrame:
    """Return a typed frame that satisfies every validation rule."""
    rows = [
        {
            "GRE Score": 300 + index,
            "TOEFL Score": 100 + index % 20,
            "University Rating": 1 + index % 5,
            "SOP": SOP_CATEGORY_VALUES[index % len(SOP_CATEGORY_VALUES)],
            "LOR": SOP_CATEGORY_VALUES[(index + 1) % len(SOP_CATEGORY_VALUES)],
            "CGPA": 7.0 + (index % 25) * 0.1,
            "Research": index % 2,
            "Chance of Admit": 0.35 + (index % 12) * 0.04,
        }
        for index in range(row_count)
    ]
    frame = pd.DataFrame(rows)
    frame[list(INTEGER_COLUMNS)] = frame[list(INTEGER_COLUMNS)].astype("Int64")
    frame[list(FLOAT_COLUMNS)] = frame[list(FLOAT_COLUMNS)].astype("float64")
    return frame


def write_valid_raw_csv(path: Path) -> Path:
    """Write a CSV that passes every validation rule end to end."""
    build_valid_feature_frame().to_csv(path, index=False)
    return path


def write_invalid_raw_csv(path: Path, offending_column: str, offending_value: float) -> Path:
    """Write a CSV with one offending value that fails validation."""
    write_valid_raw_csv(path)
    raw_df = pd.read_csv(path)
    raw_df.loc[0, offending_column] = offending_value
    raw_df.to_csv(path, index=False)
    return path


def test_valid_frame_passes_validation() -> None:
    validate_features(build_valid_feature_frame())


def test_real_dataset_passes_validation() -> None:
    features_df = build_features(read_raw_data())

    validate_features(features_df)


def test_invalid_dtype_raises() -> None:
    frame = build_valid_feature_frame()
    frame["GRE Score"] = frame["GRE Score"].astype(object)

    with pytest.raises(FeatureValidationError, match="GRE Score"):
        validate_features(frame)


def test_range_violation_raises() -> None:
    frame = build_valid_feature_frame()
    frame.loc[0, "CGPA"] = OUT_OF_RANGE_CGPA

    with pytest.raises(FeatureValidationError, match="CGPA"):
        validate_features(frame)


def test_out_of_domain_university_rating_raises() -> None:
    frame = build_valid_feature_frame()
    frame.loc[0, "University Rating"] = OUT_OF_DOMAIN_RATING

    with pytest.raises(FeatureValidationError, match="University Rating"):
        validate_features(frame)


def test_out_of_domain_sop_raises() -> None:
    frame = build_valid_feature_frame()
    frame.loc[0, "SOP"] = OUT_OF_DOMAIN_SOP

    with pytest.raises(FeatureValidationError, match="SOP"):
        validate_features(frame)


def test_excessive_null_fraction_raises() -> None:
    frame = build_valid_feature_frame(row_count=10)
    frame.loc[0:1, "GRE Score"] = pd.NA

    with pytest.raises(FeatureValidationError, match="GRE Score"):
        validate_features(frame)


def test_null_target_raises() -> None:
    frame = build_valid_feature_frame()
    frame.loc[0, TARGET_COLUMN] = pd.NA

    with pytest.raises(FeatureValidationError, match=TARGET_COLUMN):
        validate_features(frame)


def test_duplicate_rows_raise_integrity_error() -> None:
    frame = build_valid_feature_frame()
    duplicated_frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(FeatureValidationError, match="duplicated"):
        validate_features(duplicated_frame)


def test_missing_column_raises() -> None:
    frame = build_valid_feature_frame().drop(columns=["CGPA"])

    with pytest.raises(FeatureValidationError, match="CGPA"):
        validate_features(frame)


def test_unexpected_column_raises() -> None:
    frame = build_valid_feature_frame()
    frame["Extra Column"] = 1

    with pytest.raises(FeatureValidationError, match="Extra Column"):
        validate_features(frame)


def test_empty_frame_raises() -> None:
    with pytest.raises(FeatureValidationError, match="empty"):
        validate_features(pd.DataFrame())


def test_error_message_reports_column_condition_and_evidence() -> None:
    frame = build_valid_feature_frame()
    frame.loc[0, "CGPA"] = OUT_OF_RANGE_CGPA

    with pytest.raises(FeatureValidationError) as exc_info:
        validate_features(frame)

    message = str(exc_info.value)
    assert "CGPA" in message
    assert "outside range" in message
    assert str(OUT_OF_RANGE_CGPA) in message


def test_validation_failure_blocks_persistence(tmp_path: Path) -> None:
    raw_path = write_invalid_raw_csv(tmp_path / "raw.csv", "CGPA", OUT_OF_RANGE_CGPA)
    output_path = tmp_path / "features.parquet"

    with pytest.raises(FeatureValidationError):
        run_feature_pipeline(raw_path=raw_path, output_path=output_path)

    assert not output_path.exists()


def test_validation_failure_preserves_existing_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "features.parquet"
    build_valid_feature_frame().to_parquet(output_path, index=False)
    original_bytes = output_path.read_bytes()

    raw_path = write_invalid_raw_csv(tmp_path / "raw.csv", "CGPA", OUT_OF_RANGE_CGPA)

    with pytest.raises(FeatureValidationError):
        run_feature_pipeline(raw_path=raw_path, output_path=output_path)

    assert output_path.read_bytes() == original_bytes


def test_null_thresholds_are_fixed_and_cover_every_column() -> None:
    expected_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}

    assert set(MAX_NULL_FRACTION) == expected_columns
    assert FEATURE_MAX_NULL_FRACTION == EXPECTED_FEATURE_NULL_FRACTION
    assert TARGET_MAX_NULL_FRACTION == EXPECTED_TARGET_NULL_FRACTION
    for column in FEATURE_COLUMNS:
        assert MAX_NULL_FRACTION[column] == EXPECTED_FEATURE_NULL_FRACTION
    assert MAX_NULL_FRACTION[TARGET_COLUMN] == EXPECTED_TARGET_NULL_FRACTION


def test_domain_constants_cover_all_columns() -> None:
    assert set(CONTINUOUS_RANGES) | set(CATEGORICAL_DOMAINS) == {
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    }
