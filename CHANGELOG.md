# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Issue #1: problem-understanding notebook and RAW admissions dataset.
- Issue #2: typed intermediate Parquet with normalized column names and null representation.
- Issue #3: univariate, bivariate and multivariate EDA notebooks.
- Issue #4: feature-engineering notebook with deduplication, train/test split and scikit-learn preprocessing.
- Issue #5: baseline notebook (DummyRegressor mean + CGPA band heuristic).
- Issue #6: model-selection notebook with nested CV, statistical tests and the serialized LinearRegression Pipeline artifact.
- Issue #7: model-interpretation notebook (coefficients, VIF, permutation importance, error analysis).
- Issue #8: Streamlit admissions demo using the fitted Pipeline with local execution instructions.
- Issue #29: standalone feature pipeline script (RAW to deduplicated typed feature set persisted as Parquet) with unit tests.
- Issue #30: data validation and integrity rules in the feature pipeline (schema, dtypes, ranges, categories, null tolerance, structural integrity) with unit tests.
- Issue #31: standalone training pipeline script (feature Parquet → train/test split → LinearRegression Pipeline training with K-Fold validation → model artifact and evaluation report persistence) with unit tests.
- Issue #32: train/test split validation in the training pipeline (structural split checks with controlled errors, KS distribution drift warnings, split-validation report persisted in the metrics report) with unit tests.

### Fixed

- Issue #30: hardened feature validation so invalid dtypes skip dependent range/category checks and report a single `FeatureValidationError` instead of incidental `TypeError` exceptions.
