"""Feature pipeline for the Graduate Admissions project.

The pipeline reads the immutable RAW dataset, normalizes its representation,
removes exact duplicates, validates data quality and integrity, and persists
the curated feature set consumed by the downstream training pipeline.

No transformation is fitted in this module. Imputation, encoding and scaling
are defined by :func:`build_preprocessor` but fitted only inside the training
pipeline, within each cross-validation fold, to prevent data leakage.

Validation contract (Issue #30)
-------------------------------

:func:`validate_features` runs before persistence and enforces fixed rules:

- schema: exactly the feature columns plus the target, with no extras;
- dtypes: integer columns use an integer dtype, float columns a float dtype;
- ranges (documented specification domains): GRE Score 0-340, TOEFL Score
  0-120, CGPA 0-10, Chance of Admit 0-1;
- categories (effective pipeline contract, matching the OrdinalEncoder
  categories): University Rating {1..5}, SOP/LOR {1.0..5.0 in 0.5 steps},
  Research {0, 1};
- null tolerance: at most 10% nulls per feature column and 0% for the target;
- integrity: a non-empty frame without duplicate rows after deduplication.

The 10% null threshold is a fixed, reproducible quality rule defined from the
project contract and the historically observed data (maximum observed null
fraction after deduplication is ~6.2% for Research); it is NOT a parameter
learned from the data at runtime. The target must be complete because the
pipeline never drops rows and supervised training requires a label per row.

Rules deliberately NOT applicable to this dataset:

- date formats: there is no date column;
- key uniqueness: rows are anonymous candidates with no natural primary key;
- cross-dataset relations: the pipeline consumes a single dataset.

A validation failure raises :class:`FeatureValidationError` and blocks
persistence: the output artifact is neither created nor overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

RAW_FILENAME = "Admission_Predict.csv"
FEATURE_OUTPUT_FILENAME = "admission_features.parquet"

TARGET_COLUMN = "Chance of Admit"
NUMERIC_FEATURES: tuple[str, ...] = ("GRE Score", "TOEFL Score", "CGPA")
ORDINAL_FEATURES: tuple[str, ...] = ("University Rating", "SOP", "LOR")
BINARY_FEATURES: tuple[str, ...] = ("Research",)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES

INTEGER_COLUMNS: tuple[str, ...] = (
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "Research",
)
FLOAT_COLUMNS: tuple[str, ...] = ("SOP", "LOR", "CGPA", TARGET_COLUMN)
EXPECTED_RAW_COLUMNS: tuple[str, ...] = (
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    TARGET_COLUMN,
)

ORDINAL_CATEGORIES: list[list[float]] = [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
]

FEATURE_MAX_NULL_FRACTION = 0.10
TARGET_MAX_NULL_FRACTION = 0.0

MAX_NULL_FRACTION: dict[str, float] = {
    **{column: FEATURE_MAX_NULL_FRACTION for column in FEATURE_COLUMNS},
    TARGET_COLUMN: TARGET_MAX_NULL_FRACTION,
}

CONTINUOUS_RANGES: dict[str, tuple[float, float]] = {
    "GRE Score": (0.0, 340.0),
    "TOEFL Score": (0.0, 120.0),
    "CGPA": (0.0, 10.0),
    TARGET_COLUMN: (0.0, 1.0),
}

UNIVERSITY_RATING_CATEGORIES = frozenset({1.0, 2.0, 3.0, 4.0, 5.0})
RATING_HALF_STEP_CATEGORIES = frozenset({1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0})
BINARY_CATEGORIES = frozenset({0.0, 1.0})

CATEGORICAL_DOMAINS: dict[str, frozenset[float]] = {
    "University Rating": UNIVERSITY_RATING_CATEGORIES,
    "SOP": RATING_HALF_STEP_CATEGORIES,
    "LOR": RATING_HALF_STEP_CATEGORIES,
    "Research": BINARY_CATEGORIES,
}

OFFENDING_VALUE_SAMPLE_LIMIT = 5


class FeatureValidationError(ValueError):
    """Raised when the curated feature set fails data quality or integrity rules."""


def find_repository_root(start: Path | None = None) -> Path:
    """Find the repository root by locating its project manifest."""
    path = (start or Path(__file__)).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find the repository root.")


def default_raw_path() -> Path:
    """Return the repository-relative path of the immutable RAW dataset."""
    return find_repository_root() / "data" / "01_raw" / RAW_FILENAME


def default_feature_output_path() -> Path:
    """Return the repository-relative path of the persisted feature set."""
    return find_repository_root() / "data" / "04_feature" / FEATURE_OUTPUT_FILENAME


def read_raw_data(raw_path: Path | None = None) -> pd.DataFrame:
    """Read the RAW dataset and normalize its representation.

    Column names are stripped, empty and whitespace-only cells become NaN and
    dtypes are coerced to nullable integers or floats. Missing values are
    preserved: no imputation happens in this layer.
    """
    path = raw_path or default_raw_path()
    raw_df = pd.read_csv(path)
    raw_df.columns = raw_df.columns.str.strip()

    missing_columns = [column for column in EXPECTED_RAW_COLUMNS if column not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"RAW dataset is missing required columns: {missing_columns}")

    raw_df = raw_df.replace(r"^\s*$", np.nan, regex=True)
    raw_df[list(INTEGER_COLUMNS)] = raw_df[list(INTEGER_COLUMNS)].astype("Int64")
    raw_df[list(FLOAT_COLUMNS)] = raw_df[list(FLOAT_COLUMNS)].astype("float64")
    return raw_df


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the curated feature set from a normalized RAW frame.

    Exact duplicate rows are removed and the canonical columns (features plus
    target) are selected in model order. Missing values are preserved:
    imputation belongs to the training pipeline.
    """
    missing_columns = [
        column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    return raw_df.drop_duplicates().loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]].reset_index(drop=True)


def _check_schema(features_df: pd.DataFrame) -> list[str]:
    """Return failures about unexpected columns and column dtypes."""
    failures: list[str] = []
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
        failures.append("Schema: " + "; ".join(details) + ".")

    for column in INTEGER_COLUMNS:
        if column in features_df.columns and not pd.api.types.is_integer_dtype(
            features_df[column].dtype
        ):
            failures.append(f"{column}: expected integer dtype, found {features_df[column].dtype}.")
    for column in FLOAT_COLUMNS:
        if column in features_df.columns and not pd.api.types.is_float_dtype(
            features_df[column].dtype
        ):
            failures.append(f"{column}: expected float dtype, found {features_df[column].dtype}.")
    return failures


def _check_null_fractions(features_df: pd.DataFrame) -> list[str]:
    """Return failures about columns whose null fraction exceeds its contract."""
    failures: list[str] = []
    for column, max_fraction in MAX_NULL_FRACTION.items():
        if column not in features_df.columns:
            continue
        null_fraction = float(features_df[column].isna().mean())
        if null_fraction > max_fraction:
            failures.append(
                f"{column}: null fraction {null_fraction:.2%} exceeds maximum {max_fraction:.0%}."
            )
    return failures


def _check_ranges(features_df: pd.DataFrame) -> list[str]:
    """Return failures about values outside their documented specification range."""
    failures: list[str] = []
    for column, (lower_bound, upper_bound) in CONTINUOUS_RANGES.items():
        if column not in features_df.columns:
            continue
        values = features_df[column].dropna()
        outside = values[(values < lower_bound) | (values > upper_bound)]
        if outside.empty:
            continue
        sample = sorted(set(outside.tolist()))[:OFFENDING_VALUE_SAMPLE_LIMIT]
        failures.append(
            f"{column}: values {sample} outside range [{lower_bound:g}, {upper_bound:g}]."
        )
    return failures


def _check_categories(features_df: pd.DataFrame) -> list[str]:
    """Return failures about values outside their effective categorical domain."""
    failures: list[str] = []
    for column, allowed_categories in CATEGORICAL_DOMAINS.items():
        if column not in features_df.columns:
            continue
        values = features_df[column].dropna()
        outside = values[~values.isin(allowed_categories)]
        if outside.empty:
            continue
        sample = sorted(set(outside.tolist()))[:OFFENDING_VALUE_SAMPLE_LIMIT]
        failures.append(
            f"{column}: values {sample} outside allowed categories {sorted(allowed_categories)}."
        )
    return failures


def _check_integrity(features_df: pd.DataFrame) -> list[str]:
    """Return failures about the structural invariants of the curated frame."""
    failures: list[str] = []
    if features_df.empty:
        failures.append("Integrity: feature frame is empty.")
    duplicate_count = int(features_df.duplicated().sum())
    if duplicate_count:
        failures.append(f"Integrity: found {duplicate_count} duplicated rows.")
    return failures


def validate_features(features_df: pd.DataFrame) -> None:
    """Validate data quality and integrity rules; raise on any failure.

    Validation runs after structural cleaning and deduplication and before
    persistence. All failures are collected so one call reports every issue.
    """
    failures = [
        *_check_schema(features_df),
        *_check_null_fractions(features_df),
        *_check_ranges(features_df),
        *_check_categories(features_df),
        *_check_integrity(features_df),
    ]
    if failures:
        header = f"Feature validation failed with {len(failures)} issue(s):"
        details = "\n".join(f"- {failure}" for failure in failures)
        raise FeatureValidationError(f"{header}\n{details}")


def build_preprocessor() -> ColumnTransformer:
    """Build the unfitted Issue #4 preprocessing transformer.

    The returned ColumnTransformer must never be fitted here: fitting happens
    inside the training pipeline, within each cross-validation fold.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=ORDINAL_CATEGORIES)),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("ordinal", ordinal_pipeline, list(ORDINAL_FEATURES)),
            ("binary", binary_pipeline, list(BINARY_FEATURES)),
        ]
    )


def persist_features(features_df: pd.DataFrame, output_path: Path) -> Path:
    """Persist the feature set as a typed Parquet file and return its path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False, engine="pyarrow")
    return output_path


def run_feature_pipeline(
    raw_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Run the feature pipeline end to end and return the output path."""
    raw_df = read_raw_data(raw_path)
    features_df = build_features(raw_df)
    validate_features(features_df)
    return persist_features(features_df, output_path or default_feature_output_path())


def main() -> None:
    """Run the feature pipeline with the repository defaults from the CLI."""
    raw_path = default_raw_path()
    raw_df = read_raw_data(raw_path)
    features_df = build_features(raw_df)
    validate_features(features_df)
    output_path = persist_features(features_df, default_feature_output_path())

    print(f"RAW dataset: {raw_path} ({raw_df.shape[0]} rows)")
    print(f"Duplicate rows removed: {raw_df.shape[0] - features_df.shape[0]}")
    print(f"Validation passed: {features_df.shape[0]} valid rows")
    print(f"Feature set persisted to: {output_path} ({features_df.shape})")


if __name__ == "__main__":
    main()
