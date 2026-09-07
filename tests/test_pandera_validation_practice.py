"""Tests for the Unit 9 Pandera validation practice (academic evidence).

These tests validate the practice script's behavior, not production logic:
the production data validation lives in the feature pipeline and is covered
by its own test modules. A small synthetic fixture is used because the
persisted feature set (``data/04_feature/admission_features.parquet``) is not
tracked in the repository and is unavailable in CI. The real-data valid case
is demonstrated by ``scripts/pandera_validation_practice.py`` itself.
"""

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from scripts.pandera_validation_practice import (
    build_invalid_frame,
    build_schema,
    validate_valid_case,
)


def build_synthetic_valid_frame() -> pd.DataFrame:
    """Return a small deterministic frame that satisfies the practice schema."""
    frame = pd.DataFrame(
        {
            "GRE Score": [320, 330, 300],
            "TOEFL Score": [110, 115, 105],
            "University Rating": [4, 5, 3],
            "SOP": [4.5, 4.0, 3.5],
            "LOR": [4.0, 4.5, 3.5],
            "CGPA": [9.5, 8.8, 9.1],
            "Research": [1, 0, 1],
            "Chance of Admit": [0.9, 0.8, 0.7],
        }
    )
    integer_columns = ["GRE Score", "TOEFL Score", "University Rating", "Research"]
    float_columns = ["SOP", "LOR", "CGPA", "Chance of Admit"]
    frame[integer_columns] = frame[integer_columns].astype("Int64")
    frame[float_columns] = frame[float_columns].astype("float64")
    return frame


@pytest.fixture()
def feature_frame() -> pd.DataFrame:
    return build_synthetic_valid_frame()


def test_valid_feature_frame_passes_schema(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()

    validate_valid_case(feature_frame, schema)


def test_invalid_feature_frame_raises_schema_error(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()
    invalid_frame = build_invalid_frame(feature_frame)

    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(invalid_frame, lazy=True)


def test_unexpected_column_fails_validation(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()
    frame_with_extra_column = feature_frame.copy()
    frame_with_extra_column["Extra Column"] = 1

    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(frame_with_extra_column, lazy=True)


def test_feature_null_fraction_over_limit_fails_validation(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()
    frame_with_excess_nulls = feature_frame.copy()
    frame_with_excess_nulls.loc[0, "GRE Score"] = pd.NA

    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(frame_with_excess_nulls, lazy=True)


def test_target_null_fails_validation(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()
    frame_with_null_target = feature_frame.copy()
    frame_with_null_target.loc[0, "Chance of Admit"] = np.nan

    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(frame_with_null_target, lazy=True)
