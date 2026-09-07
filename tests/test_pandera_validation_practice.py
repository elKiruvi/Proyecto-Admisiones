"""Tests for the Unit 9 Pandera validation practice (academic evidence).

These tests validate the practice script's behavior, not production logic:
the production data validation lives in the feature pipeline and is covered
by its own test modules.
"""

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from scripts.pandera_validation_practice import (
    build_invalid_frame,
    build_schema,
    load_feature_frame,
    validate_valid_case,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = REPOSITORY_ROOT / "data" / "04_feature" / "admission_features.parquet"


@pytest.fixture(scope="module")
def feature_frame() -> pd.DataFrame:
    """Load the real persisted feature set once (read-only)."""
    return load_feature_frame(FEATURE_PATH)


def test_valid_feature_frame_passes_schema(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()

    validate_valid_case(feature_frame, schema)


def test_invalid_feature_frame_raises_schema_error(feature_frame: pd.DataFrame) -> None:
    schema = build_schema()
    invalid_frame = build_invalid_frame(feature_frame)

    with pytest.raises((SchemaError, SchemaErrors)):
        schema.validate(invalid_frame, lazy=True)
