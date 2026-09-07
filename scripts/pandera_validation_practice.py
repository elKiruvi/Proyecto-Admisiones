"""Pandera data validation practice - Unit 9 academic evidence.

Academic practice / evidence for Unit 9 (data validation). This does not replace
the production data validation implemented in
``src/pipelines/feature_pipeline/feature_pipeline.py``. Pandera is a dev-only
dependency and is never imported by the production pipelines.

The script demonstrates the four course practice points:

1. define a validation schema with Pandera;
2. validate a correct case (real feature data, read-only);
3. trigger a controlled failure (in-memory copy only);
4. interpret the resulting ``SchemaError`` report.

Usage:

    uv run python scripts/pandera_validation_practice.py

Exit code: 0 when the valid case passes AND the controlled failure is raised and
captured as expected; non-zero otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

FEATURE_PATH_PARTS: tuple[str, ...] = ("data", "04_feature", "admission_features.parquet")

RATING_CATEGORIES: list[int] = [1, 2, 3, 4, 5]
HALF_STEP_CATEGORIES: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
BINARY_CATEGORIES: list[int] = [0, 1]

TARGET_COLUMN = "Chance of Admit"
FEATURE_MAX_NULL_FRACTION = 0.10

INVALID_CGPA = 12.5


def find_repository_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching for pyproject.toml."""
    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repository root.")


def _feature_null_fractions_within_limit(dataframe: pd.DataFrame) -> bool:
    """Return whether every feature column has at most 10% nulls."""
    feature_columns = [column for column in dataframe.columns if column != TARGET_COLUMN]
    return bool(
        all(
            dataframe[column].isna().mean() <= FEATURE_MAX_NULL_FRACTION
            for column in feature_columns
        )
    )


def _target_has_no_nulls(dataframe: pd.DataFrame) -> bool:
    """Return whether the target column has zero missing values."""
    return bool(not dataframe[TARGET_COLUMN].isna().any())


def build_schema() -> pa.DataFrameSchema:
    """Return the Pandera schema that describes the curated feature set.

    The rules mirror the production data-validation contract (types, ranges,
    categories, nullability, maximum null fraction and strict columns) without
    importing or depending on the production validation module.
    """
    return pa.DataFrameSchema(
        {
            "GRE Score": pa.Column(int, pa.Check.in_range(0, 340), nullable=True),
            "TOEFL Score": pa.Column(int, pa.Check.in_range(0, 120), nullable=True),
            "University Rating": pa.Column(int, pa.Check.isin(RATING_CATEGORIES), nullable=True),
            "SOP": pa.Column(float, pa.Check.isin(HALF_STEP_CATEGORIES), nullable=True),
            "LOR": pa.Column(float, pa.Check.isin(HALF_STEP_CATEGORIES), nullable=True),
            "CGPA": pa.Column(float, pa.Check.in_range(0.0, 10.0), nullable=True),
            "Research": pa.Column(int, pa.Check.isin(BINARY_CATEGORIES), nullable=True),
            TARGET_COLUMN: pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
        },
        checks=[
            pa.Check(
                _feature_null_fractions_within_limit,
                name="feature_max_null_fraction_10pct",
            ),
            pa.Check(_target_has_no_nulls, name="target_no_missing_values"),
        ],
        coerce=False,
        strict=True,
    )


def load_feature_frame(feature_path: Path | None = None) -> pd.DataFrame:
    """Read the persisted feature set (read-only) and return it."""
    path = feature_path or find_repository_root() / Path(*FEATURE_PATH_PARTS)
    if not path.is_file():
        raise FileNotFoundError(f"Feature set not found: {path}")
    return pd.read_parquet(path)


def build_invalid_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Return an in-memory copy with one representative violation (CGPA=12.5)."""
    invalid_frame = feature_frame.copy()
    invalid_frame.loc[0, "CGPA"] = INVALID_CGPA
    return invalid_frame


def capture_schema_error(
    feature_frame: pd.DataFrame,
    schema: pa.DataFrameSchema,
) -> SchemaError | SchemaErrors:
    """Validate with lazy error collection and return the captured error.

    The invalid frame must fail validation; any other outcome raises so the
    script reports an unexpected behavior instead of hiding it. With lazy=True,
    pandera raises ``SchemaErrors`` (a sibling of ``SchemaError``).
    """
    try:
        schema.validate(feature_frame, lazy=True)
    except (SchemaError, SchemaErrors) as error:
        return error
    raise RuntimeError("Expected SchemaError was not raised for the invalid frame.")


def validate_valid_case(
    feature_frame: pd.DataFrame,
    schema: pa.DataFrameSchema,
) -> None:
    """Validate the correct case; an invalid frame raises SchemaError here."""
    schema.validate(feature_frame, lazy=True)


def print_failure_interpretation(failure_cases: pd.DataFrame) -> None:
    """Print a readable interpretation of the captured failure cases."""
    print("Failure cases collected by Pandera (lazy=True):")
    columns = [
        column
        for column in ("schema_context", "column", "check", "failure_case", "index")
        if column in failure_cases.columns
    ]
    print(failure_cases[columns].to_string(index=False))
    print()
    print("Interpretation:")
    print(
        "- Rule violated: CGPA must stay within the documented domain [0.0, 10.0] "
        "(dataset specification)."
    )
    print(f"- Injected value: {INVALID_CGPA} in row 0 of an in-memory copy.")
    print("- Consequence: the row is outside the specification domain and would be")
    print("  rejected by validation before persistence in a production pipeline.")
    print("- Demonstration: Pandera reports every failing check with its column,")
    print("  check description, offending value and row index in failure_cases.")


def main() -> int:
    """Run the practice flow and return the process exit code."""
    root = find_repository_root()
    schema = build_schema()
    print("Pandera schema defined for the curated feature set:")
    print(f"- Columns: {list(schema.columns)}")
    print(
        "- Checks: types, ranges, categories, nullability, "
        f"max {FEATURE_MAX_NULL_FRACTION:.0%} nulls per feature, target complete "
        "(coerce=False, strict=True)."
    )
    print()

    feature_frame = load_feature_frame(root / Path(*FEATURE_PATH_PARTS))
    print(
        f"Valid case: {FEATURE_PATH_PARTS[-1]} "
        f"({feature_frame.shape[0]} rows, {feature_frame.shape[1]} columns, read-only)."
    )
    validate_valid_case(feature_frame, schema)
    print("Valid case: PASS - the real feature data satisfies the schema.")
    print()

    invalid_frame = build_invalid_frame(feature_frame)
    error = capture_schema_error(invalid_frame, schema)
    print("Controlled failure: PASS - SchemaError raised and captured (not an")
    print("uncontrolled traceback) for the mutated in-memory copy.")
    print_failure_interpretation(pd.DataFrame(error.failure_cases))

    return 0


if __name__ == "__main__":
    sys.exit(main())
