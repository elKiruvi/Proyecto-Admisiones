"""Feature pipeline for the Graduate Admissions project.

The pipeline reads the immutable RAW dataset, normalizes its representation,
removes exact duplicates and persists the curated feature set consumed by the
downstream training pipeline.

No transformation is fitted in this module. Imputation, encoding and scaling
are defined by :func:`build_preprocessor` but fitted only inside the training
pipeline, within each cross-validation fold, to prevent data leakage.
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
    return persist_features(features_df, output_path or default_feature_output_path())


def main() -> None:
    """Run the feature pipeline with the repository defaults from the CLI."""
    raw_path = default_raw_path()
    raw_df = read_raw_data(raw_path)
    features_df = build_features(raw_df)
    output_path = persist_features(features_df, default_feature_output_path())

    print(f"RAW dataset: {raw_path} ({raw_df.shape[0]} rows)")
    print(f"Duplicate rows removed: {raw_df.shape[0] - features_df.shape[0]}")
    print(f"Feature set persisted to: {output_path} ({features_df.shape})")


if __name__ == "__main__":
    main()
