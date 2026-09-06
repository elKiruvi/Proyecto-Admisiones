"""Unit tests for the Graduate Admissions feature pipeline."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from pipelines.feature_pipeline.feature_pipeline import (
    EXPECTED_RAW_COLUMNS,
    FEATURE_COLUMNS,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    ORDINAL_CATEGORIES,
    TARGET_COLUMN,
    build_features,
    build_preprocessor,
    default_feature_output_path,
    default_raw_path,
    find_repository_root,
    persist_features,
    read_raw_data,
    run_feature_pipeline,
)

RAW_ROW_COUNT = 623
UNIQUE_ROW_COUNT = 471
COLUMN_COUNT = 8
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

SYNTHETIC_CSV_HEADER_ROW_COUNT = 1
SYNTHETIC_DUPLICATE_ROW_COUNT = 1

SYNTHETIC_CSV_ROWS = [
    "GRE Score,TOEFL Score,University Rating,SOP,LOR ,CGPA,Research,Chance of Admit ",
    "316,107,3,3.5,3.0,8.62,1,0.72",
    "316,107,3,3.5,3.0,8.62,1,0.72",
    "298,,2,2.0,2.5,7.90,0,0.55",
    "   ,95,1,1.5,2.0,7.20,1,0.48",
    "322,115,4,4.5,4.5,9.10,1,0.85",
    "305,100,2,2.5,3.0,7.95,0,0.62",
    "330,118,5,4.0,4.0,9.40,1,0.91",
    "312,103,3,3.0,3.5,8.45,1,0.71",
    "295,98,2,2.0,2.0,7.60,0,0.52",
    "318,110,4,3.5,4.0,8.80,1,0.79",
    "302,97,1,2.0,2.5,7.45,0,0.49",
    "326,116,5,4.5,5.0,9.25,1,0.88",
    "310,105,3,3.0,3.0,8.30,1,0.68",
]

SYNTHETIC_ROW_COUNT = len(SYNTHETIC_CSV_ROWS) - SYNTHETIC_CSV_HEADER_ROW_COUNT
SYNTHETIC_UNIQUE_ROW_COUNT = SYNTHETIC_ROW_COUNT - SYNTHETIC_DUPLICATE_ROW_COUNT


def write_synthetic_raw_csv(path: Path) -> Path:
    """Write a small RAW-like CSV with dirty columns, nulls and a duplicate."""
    path.write_text("\n".join(SYNTHETIC_CSV_ROWS), encoding="utf-8")
    return path


def build_synthetic_feature_frame() -> pd.DataFrame:
    """Return a typed frame in the normalized RAW schema derived from the CSV fixture."""
    frame = pd.read_csv(StringIO("\n".join(SYNTHETIC_CSV_ROWS)))
    frame.columns = frame.columns.str.strip()
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    frame[list(INTEGER_COLUMNS)] = frame[list(INTEGER_COLUMNS)].astype("Int64")
    frame[list(FLOAT_COLUMNS)] = frame[list(FLOAT_COLUMNS)].astype("float64")
    return frame.drop_duplicates().reset_index(drop=True)


def test_find_repository_root_resolves_from_module_location() -> None:
    assert find_repository_root() == REPOSITORY_ROOT


def test_default_raw_path_points_to_tracked_raw_dataset() -> None:
    assert default_raw_path() == REPOSITORY_ROOT / "data" / "01_raw" / "Admission_Predict.csv"


def test_default_feature_output_path_points_to_feature_layer() -> None:
    assert (
        default_feature_output_path()
        == REPOSITORY_ROOT / "data" / "04_feature" / "admission_features.parquet"
    )


def test_read_raw_data_normalizes_columns_dtypes_and_nulls(tmp_path: Path) -> None:
    raw_df = read_raw_data(write_synthetic_raw_csv(tmp_path / "synthetic.csv"))

    assert list(raw_df.columns) == list(EXPECTED_RAW_COLUMNS)
    for column in INTEGER_COLUMNS:
        assert str(raw_df[column].dtype) == "Int64"
    for column in FLOAT_COLUMNS:
        assert str(raw_df[column].dtype) == "float64"
    assert raw_df.shape == (SYNTHETIC_ROW_COUNT, COLUMN_COUNT)
    assert raw_df["GRE Score"].isna().sum() == 1
    assert raw_df["TOEFL Score"].isna().sum() == 1


def test_read_raw_data_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_raw_data(tmp_path / "does_not_exist.csv")


def test_read_raw_data_missing_column_raises_value_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column.csv"
    csv_path.write_text("GRE Score,TOEFL Score\n316,107\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        read_raw_data(csv_path)


def test_build_features_drops_exact_duplicates() -> None:
    features_df = build_features(build_synthetic_feature_frame())

    assert features_df.shape[0] == SYNTHETIC_UNIQUE_ROW_COUNT
    assert not features_df.duplicated().any()


def test_build_features_selects_canonical_columns_in_model_order() -> None:
    features_df = build_features(build_synthetic_feature_frame())

    assert list(features_df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]


def test_build_features_preserves_missing_values() -> None:
    features_df = build_features(build_synthetic_feature_frame())

    assert features_df["GRE Score"].isna().sum() == 1
    assert features_df["TOEFL Score"].isna().sum() == 1


def test_build_features_missing_column_raises_value_error() -> None:
    frame = build_synthetic_feature_frame().drop(columns=["CGPA"])

    with pytest.raises(ValueError, match="missing required columns"):
        build_features(frame)


def test_build_features_is_deterministic() -> None:
    frame = build_synthetic_feature_frame()

    assert_frame_equal(build_features(frame), build_features(frame))


def test_build_preprocessor_is_unfitted_and_matches_poc_spec() -> None:
    preprocessor = build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
    assert not hasattr(preprocessor, "transformers_")

    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert transformer_names == ["numeric", "ordinal", "binary"]

    numeric_pipeline = preprocessor.transformers[0][1]
    assert isinstance(numeric_pipeline, Pipeline)
    numeric_imputer = numeric_pipeline.named_steps["imputer"]
    assert isinstance(numeric_imputer, SimpleImputer)
    assert numeric_imputer.strategy == "median"
    assert isinstance(numeric_pipeline.named_steps["scaler"], StandardScaler)

    ordinal_pipeline = preprocessor.transformers[1][1]
    assert isinstance(ordinal_pipeline, Pipeline)
    ordinal_imputer = ordinal_pipeline.named_steps["imputer"]
    assert isinstance(ordinal_imputer, SimpleImputer)
    assert ordinal_imputer.strategy == "most_frequent"
    ordinal_encoder = ordinal_pipeline.named_steps["encoder"]
    assert isinstance(ordinal_encoder, OrdinalEncoder)
    assert ordinal_encoder.categories == ORDINAL_CATEGORIES
    assert isinstance(ordinal_pipeline.named_steps["scaler"], StandardScaler)

    binary_pipeline = preprocessor.transformers[2][1]
    assert isinstance(binary_pipeline, Pipeline)
    binary_imputer = binary_pipeline.named_steps["imputer"]
    assert isinstance(binary_imputer, SimpleImputer)
    assert binary_imputer.strategy == "most_frequent"
    assert tuple(binary_pipeline.named_steps) == ("imputer",)


def test_build_preprocessor_transforms_synthetic_frame_without_remaining_nulls() -> None:
    preprocessor = build_preprocessor()
    feature_frame = build_synthetic_feature_frame()[[*FEATURE_COLUMNS]]

    preprocessor.fit(feature_frame)
    transformed = np.asarray(preprocessor.transform(feature_frame))

    assert list(preprocessor.get_feature_names_out()) == [
        "numeric__GRE Score",
        "numeric__TOEFL Score",
        "numeric__CGPA",
        "ordinal__University Rating",
        "ordinal__SOP",
        "ordinal__LOR",
        "binary__Research",
    ]
    assert transformed.shape == (SYNTHETIC_UNIQUE_ROW_COUNT, len(FEATURE_COLUMNS))
    assert not np.isnan(transformed).any()


def test_persist_features_writes_readable_parquet(tmp_path: Path) -> None:
    features_df = build_features(build_synthetic_feature_frame())
    output_path = tmp_path / "features.parquet"

    persisted_path = persist_features(features_df, output_path)
    reloaded_df = pd.read_parquet(persisted_path)

    assert persisted_path.is_file()
    assert list(reloaded_df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert reloaded_df.shape == features_df.shape
    assert reloaded_df.isna().sum().to_dict() == features_df.isna().sum().to_dict()
    assert_frame_equal(reloaded_df.astype("float64"), features_df.astype("float64"))


def test_run_feature_pipeline_end_to_end_with_tmp_paths(tmp_path: Path) -> None:
    raw_path = write_synthetic_raw_csv(tmp_path / "synthetic.csv")
    output_path = tmp_path / "nested" / "features.parquet"

    result_path = run_feature_pipeline(raw_path=raw_path, output_path=output_path)
    reloaded_df = pd.read_parquet(result_path)

    assert result_path == output_path
    assert reloaded_df.shape == (SYNTHETIC_UNIQUE_ROW_COUNT, COLUMN_COUNT)
    assert list(reloaded_df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert not reloaded_df.duplicated().any()
    assert reloaded_df["GRE Score"].isna().sum() == 1


def test_real_raw_data_contract() -> None:
    raw_df = read_raw_data()

    assert raw_df.shape == (RAW_ROW_COUNT, COLUMN_COUNT)
    assert set(raw_df.columns) == set(EXPECTED_RAW_COLUMNS)

    features_df = build_features(raw_df)

    assert features_df.shape == (UNIQUE_ROW_COUNT, COLUMN_COUNT)
    assert list(features_df.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert not features_df.duplicated().any()
    assert features_df[TARGET_COLUMN].notna().all()
