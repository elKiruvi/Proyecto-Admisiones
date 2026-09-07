# Proyecto-Admisiones

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

## Overview

`Proyecto-Admisiones` is a supervised regression system that estimates the
`Chance of Admit` score of a graduate-school applicant from seven academic
profile attributes: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA,
and Research.

The project is built as a reproducible, pipeline-oriented machine learning
system rather than a set of ad-hoc scripts:

- layered data structure with an immutable raw layer;
- deterministic feature and training pipelines with fixed random seeds;
- strict data-quality and input validation at every boundary;
- explicit data-leakage prevention (canonical split, in-Pipeline preprocessing,
  nested cross-validation);
- a serialized scikit-learn `Pipeline` containing the complete preprocessing
  plus the final estimator;
- clean separation between training and inference code;
- online and batch inference through a Streamlit application;
- automated testing, code-quality gates, and a blocking dependency audit in CI;
- containerization with Docker for reproducible execution.

The target is the `Chance of Admit` score stored in the dataset (0–1 scale).
The model output is a raw regression estimate: it is not a calibrated
probability and is not clipped to `[0, 1]`. This is a predictive modeling
project, not a decision system for real admissions.

## Engineering value

This repository demonstrates a complete, disciplined machine learning
lifecycle:

- **Reproducibility** — locked dependencies (`uv.lock`), deterministic
  pipelines with documented random seeds, and a model artifact that can be
  regenerated bit-for-bit (identical SHA-256) from a fresh clone.
- **Production-oriented structure** — exploratory notebooks are separated from
  reusable pipeline code under `src/`.
- **Data quality** — schema, dtype, domain, and integrity validation with
  aggregated, blocking error reports.
- **Leakage prevention** — train/test isolation, preprocessing fitted only
  inside cross-validation folds, and nested CV for model selection.
- **End-to-end artifact** — one fitted `Pipeline` covers imputation, encoding,
  scaling, and prediction for both training and inference.
- **Serving** — online and batch prediction share the same serialized
  artifact and validation contract.
- **Engineering discipline** — 90%+ branch coverage, pre-commit quality gates,
  CI/CD, and a blocking dependency-vulnerability audit.

## Problem and dataset

| Property | Value |
|---|---|
| Dataset | `Admission_Predict.csv` (Graduate Admissions) |
| Original rows | 623 |
| Exact duplicate rows removed | 152 |
| Unique observations | 471 |
| Target | `Chance of Admit` (0–1 scale) |
| Problem type | Supervised regression |
| Missing values | Present in several predictors; tolerated up to 10% per feature column |

| Feature | Type | Domain |
|---|---|---|
| GRE Score | Numeric (integer-valued) | 0–340 |
| TOEFL Score | Numeric (integer-valued) | 0–120 |
| CGPA | Numeric (continuous) | 0–10 |
| University Rating | Ordinal | {1, 2, 3, 4, 5} |
| SOP | Ordinal | {1.0, 1.5, …, 5.0} |
| LOR | Ordinal | {1.0, 1.5, …, 5.0} |
| Research | Binary | {0, 1} |

## Architecture

```text
data/01_raw (immutable CSV)
      ↓
Feature Pipeline  ──── data-quality validation ────→  data/04_feature (Parquet)
      ↓
Training Pipeline ──── split validation + 5-fold CV + diagnosis ────→  data/08_reporting (JSON)
      ↓
Serialized Pipeline  →  models/05_model_selection_pipeline.joblib  (preprocessing + estimator)
      ↓
Inference Pipeline ──── input validation → artifact predict() ────→  data/07_model_output
      ↓
Streamlit application
      ├── Online prediction (single applicant)
      └── Batch prediction (CSV upload → predictions → CSV download)
      ↓
Docker image (multi-stage, non-root)
```

Exploratory analysis and model-selection experiments live in `notebooks/` as
documented evidence; the reusable implementation lives in `src/` pipelines.

## Data science lifecycle

| Stage | Artifact |
|---|---|
| Data acquisition | `data/01_raw/` (immutable) |
| Data preparation | `notebooks/1-data/`, `notebooks/2-exploration/` → typed Parquet in `data/02_intermediate/` |
| Exploratory analysis | `notebooks/3-analysis/` (univariate, bivariate, multivariate) |
| Feature engineering | `notebooks/4-feat_eng/` + `src/pipelines/feature_pipeline/` |
| Baseline modeling | `notebooks/5-models/05.1-SCO_baseline_model-*.ipynb` |
| Model selection | `notebooks/5-models/05.2-SCO_model_selection-*.ipynb` + `src/pipelines/training_pipeline/` |
| Model interpretation | `notebooks/6-interpretation/` |
| Inference | `src/pipelines/inference_pipeline/` |
| Application | `src/inference/streamlit_app.py` (online + batch) |
| Containerization | `Dockerfile`, `.dockerignore` |

Notebooks preserve the exploratory history of the project. `src/` contains the
reproducible, tested pipeline implementation that produces and consumes the
official artifacts.

## Data quality

Validation is implemented in `src/pipelines/feature_pipeline/feature_pipeline.py`
(feature layer) and in `src/pipelines/inference_pipeline/` (inference boundary).

- **Schema** — exactly the expected columns; duplicates, missing columns, and
  unexpected columns are rejected. Duplicate headers are detected even after
  pandas mangles repeated names (`GRE Score` → `GRE Score.1`).
- **Dtypes** — integer columns must be integer-typed; float columns float-typed.
- **Domains** — GRE Score 0–340, TOEFL Score 0–120, CGPA 0–10; University
  Rating {1..5}; SOP/LOR {1.0..5.0 in 0.5 steps}; Research {0, 1}.
- **Integers** — GRE Score and TOEFL Score must be integer-valued.
- **Finiteness** — non-finite values are rejected.
- **Missing values** — allowed (up to 10% per feature column; the target must
  be complete) and handled by the imputers inside the model `Pipeline`.
- **Duplicates** — exact duplicate rows are removed before any split.
- **Errors** — all failures are collected and reported together in one
  descriptive message; validation failures block further processing.

Structural validation (schema/dtypes/integrity) is separated from domain
validation (ranges/categories). Imputation, encoding, and scaling are not part
of validation: they are modeling-time transformations that live inside the
model `Pipeline` and are fitted only on training data.

## Data leakage prevention

Leakage prevention is a first-class design constraint:

- **Canonical split** — the train/test split (80/20, `random_state=42`) is
  defined once in the feature-engineering stage and consumed downstream; the
  test set is never used for selection, tuning, or modification.
- **Duplicates before split** — exact duplicates are removed before splitting
  so that no identical row can appear in both partitions.
- **Preprocessing inside the Pipeline** — imputation, encoding, and scaling
  live inside the serialized `Pipeline` and are re-fitted within every
  cross-validation fold; no preprocessing statistics are ever learned globally.
- **Nested CV** — model selection uses inner CV for hyperparameter tuning and
  outer CV for comparison; only outer-CV scores drive the final decision.
- **Independent screening folds** — the candidate-screening folds are drawn
  independently from the outer-CV folds (`random_state=123` vs `random_state=42`).

Historical notes (recorded for transparency):

- The model-selection notebook originally reused its screening folds for the
  outer-CV evaluation; this was corrected by separating screening and
  outer-CV folds, so selection and evaluation no longer share partitions.
- Early exploratory notebooks analyzed the complete dataset before the
  canonical train/test split was defined. That analysis is retained as
  historical exploration; it did not drive the model-selection or evaluation
  decisions, which use the canonical split and per-fold preprocessing.

## Feature engineering and preprocessing

The preprocessing contract is defined in
`src/pipelines/feature_pipeline/feature_pipeline.py` and applied inside the
model `Pipeline` as a `ColumnTransformer`:

| Group | Columns | Pipeline |
|---|---|---|
| Numeric | GRE Score, TOEFL Score, CGPA | `SimpleImputer(median)` → `StandardScaler` |
| Ordinal | University Rating, SOP, LOR | `SimpleImputer(most_frequent)` → `OrdinalEncoder` → `StandardScaler` |
| Binary | Research | `SimpleImputer(most_frequent)` |

Transformations belong inside the model `Pipeline` so that everything learned
from the data (medians, modes, categories, scaling statistics) is fitted only
on the training portion of each fold and applied consistently at inference
time.

## Baseline models

Two baselines were established in `notebooks/5-models/05.1-SCO_baseline_model-*.ipynb`
and re-evaluated with the canonical outer CV:

| Baseline | Outer CV RMSE | Test RMSE |
|---|---|---|
| `DummyRegressor(strategy="mean")` | 0.1423 ± 0.0122 | ≈ 0.1485 |
| CGPA band heuristic | 0.0902 ± 0.0083 | 0.0919 |

Baselines define the reference performance: the CGPA band heuristic (three
fixed CGPA bands with their historical mean targets) is the more demanding
primary baseline, while the mean predictor is the non-informed reference.

## Model selection methodology

`notebooks/5-models/05.2-SCO_model_selection-*.ipynb` documents the selection
process; the standalone training pipeline re-implements it as reproducible
code.

- **Candidates** — 8 candidate models across 6 model families: ordinary
  linear regression, regularized linear regression (Ridge, Lasso, ElasticNet),
  instance-based (k-NN), decision trees, bagging (Random Forest), and boosting
  (Gradient Boosting).
- **Screening** — all candidates evaluated with 5-fold CV on the training
  partition only; candidates with CV RMSE above the average of the eight
  candidate means are eliminated.
- **Nested CV** — finalists are tuned with `GridSearchCV` inside inner CV
  (5-fold, `random_state=7`) and compared with outer CV (5-fold,
  `random_state=42`); screening folds use an independent seed (`123`).
- **Statistical comparison** — paired outer-CV RMSE values per finalist with a
  Friedman test; with only five folds, statistical power is low and p-values
  are interpreted conservatively.
- **Final decision** — Ridge achieved the lowest observed outer-CV RMSE
  (0.064376 ± 0.007775) but the difference from LinearRegression
  (0.064463 ± 0.007349) was minimal and the tests did not establish
  statistical superiority. `LinearRegression` was selected for parsimony,
  direct coefficient interpretability, and lower tuning cost, with practically
  equivalent predictive performance.

All finalists beat the primary baseline by a wide margin. The test set was
used exactly once, for the final evaluation of the selected model.

## Model performance

Official metrics from `data/08_reporting/training_metrics.json`
(produced by the training pipeline, `random_state=42`, 5-fold CV):

| Partition | RMSE | MAE | R² |
|---|---|---|---|
| Train (376 rows) | 0.062440 | 0.045083 | 0.807426 |
| CV (5-fold, mean ± std) | 0.064463 ± 0.007349 | 0.046825 ± 0.005633 | 0.786251 ± 0.053931 |
| Test (95 rows) | 0.068500 | 0.051192 | 0.787195 |

Interpretation: RMSE (primary metric) keeps the target's units and penalizes
large errors; MAE is the average absolute deviation; R² measures explained
variance. The train, CV, and test RMSE values are close, indicating stable
generalization on this dataset. The output is a regression estimate on the
0–1 target scale — not a calibrated probability, and R² is not "accuracy".

## Model validation and diagnosis

The training pipeline persists a validation report alongside the metrics:

- **Split validation** — 8 structural checks on the train/test split
  (disjoint and unique indices, no cross-partition duplicate rows, size
  ratio, schema parity, feature/target alignment, target completeness, no
  new categories in test), plus KS-based drift diagnostics on features and
  target. All checks pass.
- **Gap diagnostics** — the train↔CV and test↔CV RMSE gaps are compared
  against the fold-level RMSE standard deviation (rule: `abs(gap) <= rmse_std`);
  both gaps are within fold variability.
- **Diagnosis verdicts** — overfitting, underfitting, and generalization
  signals are evaluated against evidence thresholds: all report `no_signal`
  with an overall `acceptable` status.
- **Baseline grounding** — the CV RMSE improves the non-informed baseline
  (0.1423) by ≈ 54.7%.

These diagnostics are descriptive evidence, not statistical proof of
generalization beyond this dataset.

## Model interpretation

`notebooks/6-interpretation/06.1-SCO_model_interpretation-*.ipynb` interprets
the fixed artifact (never retraining or re-selecting):

- **Coefficients** — LinearRegression is directly interpretable: standardized
  features give comparable coefficient magnitudes. CGPA, GRE, and TOEFL are
  the strongest positive drivers.
- **Multicollinearity** — VIF diagnostics describe correlation among
  predictors; coefficients under correlated predictors must be interpreted
  with caution.
- **Permutation importance** — a post-selection diagnostic on the test
  partition confirming the coefficient-based ranking.
- **Error analysis** — residual distribution, outliers, and the practical
  consequences of prediction errors.

## Final model artifact

```text
models/05_model_selection_pipeline.joblib
SHA-256: 8cb82a4a8d417ee5f0e19c3010295de74fe85ebd3f9ef5a41e746de5af978a71
```

- A fitted scikit-learn `Pipeline` with steps `preprocessor` (the complete
  `ColumnTransformer`) and `model` (`LinearRegression`).
- Expected feature schema (canonical order): `GRE Score`, `TOEFL Score`,
  `CGPA`, `University Rating`, `SOP`, `LOR`, `Research`.
- Immutable: inference loads it read-only, verifies it is fitted and
  schema-compatible, and never modifies, refits, or re-serializes it.
- Reproducible: the training pipeline regenerates it deterministically — the
  regenerated file has the same SHA-256 as the tracked artifact.

## Inference architecture

```text
Input (single record or CSV frame)
      ↓
Validation (schema, dtypes, domains — no preprocessing applied here)
      ↓
Serialized Pipeline.predict()
      ↓
Raw regression estimate (input rows + predicted_chance_of_admit)
```

Inference never retrains, never refits preprocessing, never reproduces
training transformations by hand, and never modifies the artifact. Online
and batch inference use the same validation contract and the same serialized
pipeline, which guarantees consistency between training and inference.

- **Single record** — `src/pipelines/inference_pipeline/inference.py`
  (`validate_features` → `build_feature_frame` → `predict_admission`).
- **Batch** — `src/pipelines/inference_pipeline/inference_pipeline.py`
  (`parse_new_data_csv` → `validate_inference_input` → `generate_predictions`
  → `persist_predictions`). Missing values are allowed and handled by the
  artifact's imputers.

## Running the application

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

### Environment

```bash
uv sync --locked
```

### Launch Streamlit locally

```bash
uv run streamlit run src/inference/streamlit_app.py
```

Open the printed URL (normally `http://localhost:8501`). The application has
two tabs: **Online prediction** and **Batch prediction**.

### Online prediction

The Online tab accepts a single applicant profile (GRE Score 0–340, TOEFL
Score 0–120, University Rating 1–5, SOP and LOR 1.0–5.0 in 0.5 steps, CGPA
0–10, Research yes/no) and returns the raw `LinearRegression` estimate after
submitting the form. Continuous values outside the observed training range
are accepted and may represent extrapolation.

### Batch prediction

The Batch tab processes a CSV upload through the same validation and
inference contract as the batch pipeline:

```text
CSV upload
    ↓
Schema validation (required/unexpected/duplicated columns)
    ↓
Domain validation (dtypes, finiteness, integer-valued, ranges, categories)
    ↓
Model inference (serialized Pipeline)
    ↓
Results preview
    ↓
CSV download (input columns + predicted_chance_of_admit appended last)
```

Expected schema — one applicant per row, exactly these columns:

```text
GRE Score,TOEFL Score,University Rating,SOP,LOR,CGPA,Research
```

A representative sample input is versioned at
`data/05_model_input/new_applicants.csv` with its sample output at
`data/07_model_output/admission_predictions.csv`. Invalid files (malformed
CSV, missing or unexpected columns, out-of-range or out-of-domain values)
produce a descriptive, aggregated validation error before any prediction is
generated.

The batch mode was validated with a 10,000-record scenario: the complete
batch workflow processed 10,000 rows and the downloaded prediction file
contained 10,000 output rows.

### Deployment

The application is publicly accessible at:

```text
https://proyecto-admisiones-samuelc.streamlit.app/
```

Anyone with the URL can access it without logging in. It is served from the
`main` branch of this repository using `src/inference/streamlit_app.py` as the
main file and Python 3.12; no Streamlit secrets are required.

### Docker

Docker is an additional reproducibility layer for the same application: it
does not change the model, the inference pipeline, or the app behavior. The
image follows the UV multi-stage approach with a minimal Python runtime:

- **Builder stage** — `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
  installs only the locked runtime dependencies
  (`uv sync --frozen --no-default-groups`, dropping the dev group).
- **Runtime stage** — `python:3.12-slim-bookworm` carries the locked virtual
  environment, the inference code (`src/`), the immutable artifact
  (`models/05_model_selection_pipeline.joblib`), and the versioned sample
  batch input; it runs as a non-root user.

```bash
docker build -t proyecto-admisiones .
docker run --rm -p 8501:8501 --name admissions-docker proyecto-admisiones
```

Open `http://localhost:8501`. Verify the health endpoint (the image declares a
`HEALTHCHECK` against it):

```bash
curl -fsS http://localhost:8501/_stcore/health   # expected: ok
```

Stop the container with `docker stop admissions-docker`.

## Reproducibility from a fresh clone

The complete pipeline is reproducible from a clean environment. RAW data, the
intermediate dataset, the model artifact, and the lockfile are tracked in the
repository.

**Primary path** (application and training):

```text
git clone https://github.com/elKiruvi/Proyecto-Admisiones.git
      ↓
cd Proyecto-Admisiones
      ↓
uv sync --locked
      ↓
uv run pytest
      ↓
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
      ↓
uv run python src/pipelines/training_pipeline/train_pipeline.py
      ↓
uv run python src/pipelines/inference_pipeline/inference_pipeline.py
      ↓
uv run streamlit run src/inference/streamlit_app.py     # local app
docker build -t proyecto-admisiones .                   # or the containerized app
```

```bash
git clone https://github.com/elKiruvi/Proyecto-Admisiones.git
cd Proyecto-Admisiones

uv sync --locked              # locked, deterministic environment
uv run pytest                 # test suite

# training path
uv run python src/pipelines/feature_pipeline/feature_pipeline.py   # RAW → validated features
uv run python src/pipelines/training_pipeline/train_pipeline.py    # features → artifact + report

# inference
uv run python src/pipelines/inference_pipeline/inference_pipeline.py

# application
uv run streamlit run src/inference/streamlit_app.py
```

Determinism: the split uses `test_size=0.2, random_state=42`; cross-validation
is 5-fold KFold with `shuffle=True, random_state=42`. The model-selection
notebook uses outer CV `random_state=42`, inner CV `random_state=7`, and
independent screening folds `random_state=123`. Re-running the training
pipeline on a fresh clone reproduces the official metrics exactly and
regenerates the artifact with the same SHA-256.

The MLflow and Pandera workflows are supporting tools, separate from this
primary path; see [Supporting workflows](#supporting-workflows).

## Generated artifacts

| File | Status | Produced by |
|---|---|---|
| `data/01_raw/Admission_Predict.csv` | Versioned (input) | — |
| `data/02_intermediate/Admission_Predict.parquet` | Versioned (input) | Data preparation notebook |
| `models/05_model_selection_pipeline.joblib` | Versioned | Training pipeline / model-selection notebook |
| `data/05_model_input/new_applicants.csv` | Versioned (sample input) | — |
| `data/07_model_output/admission_predictions.csv` | Versioned (sample output) | Inference pipeline |
| `data/04_feature/admission_features.parquet` | Generated (gitignored) | Feature pipeline |
| `data/08_reporting/training_metrics.json` | Generated (gitignored) | Training pipeline |
| `mlruns/`, `mlflow.db` | Generated (gitignored) | MLflow tracking workflow |

Required generation order: feature pipeline → training pipeline → (optional)
MLflow/Pandera workflows → inference pipeline.

## Testing

```bash
uv run pytest
uv run pytest --cov --cov-branch
```

The suite covers the feature, training, and inference pipelines, validation
rules, model loading, and the Streamlit application (via Streamlit's AppTest,
including online prediction, batch upload, validation errors, and download).
Current coverage: ≈ 90.5% branch coverage (CI gate: 60%).

Two training-pipeline tests run only when the generated feature set is present
and are skipped on a fresh clone; synthetic fixtures cover the same behavior
in CI, so the suite passes without any prior generation step.

## Code quality

Quality gates run through pre-commit (Ruff lint/format, mypy, commitizen,
generic checks) using the configuration in `.code_quality/`:

```bash
uvx pre-commit run --all-files --show-diff-on-failure --color=always
```

Ruff and mypy are executed through pre-commit/CI rather than as direct
project dependencies.

## CI/CD workflow

```text
Issue
  ↓
Feature branch (feature/*, fix/*)
  ↓
Implementation + Conventional Commits
  ↓
Pull Request
  ↓
CI: actionlint · pre-commit · tests with coverage gate · dependency audit
  ↓
Review
  ↓
Merge to main
```

The CI pipeline (`.github/workflows/ci.yml`) runs actionlint, template
integrity checks, pre-commit, the test suite with a 60% branch-coverage
gate, and a blocking `uv audit --locked` dependency-vulnerability audit
(UV pinned to 0.12.2). Additional workflows cover dependency review,
automerge for dependency updates, labels, and pre-commit hook updates.

## Security and dependency management

- `uv.lock` is committed; environments are installed with
  `uv sync --locked` (Docker: `--frozen`), so resolutions are deterministic.
- A blocking dependency audit runs in CI; `uv audit --locked` currently
  reports zero known vulnerabilities.
- No secrets or credentials are stored in the repository; workflows use
  GitHub's secret references only.
- Inference validates all input (schema, dtypes, domains) before any
  prediction.
- The Docker runtime runs as a non-root user and carries only runtime
  dependencies.
- The model artifact is immutable and verified at load time (fitted
  `Pipeline`, expected steps, expected estimator, expected feature schema).

## Supporting workflows

Two standalone scripts demonstrate validation and experiment tracking with
external tools. Both are dev-only workflows: production validation and
modeling behavior are implemented in the pipelines, not in these scripts.

**MLflow experiment tracking** (`scripts/mlflow_experiment_tracking.py`) reads
the official training report and the canonical artifact read-only and
registers parameters, metrics, and an exact artifact copy in a local tracking
store (SQLite `mlflow.db` + `mlruns/`). It does not re-train or re-select
anything.

```bash
uv run python scripts/mlflow_experiment_tracking.py    # requires data/08_reporting/training_metrics.json
MLFLOW_TRACKING_URI=sqlite:///<repository-root>/mlflow.db mlflow ui
```

**Pandera validation** (`scripts/pandera_validation_practice.py`) defines a
Pandera schema, validates the real feature set read-only, triggers a
controlled failure in memory, and interprets the resulting `SchemaError`
report. Pandera is a dev-only dependency and is never imported by the
production pipelines.

```bash
uv run python scripts/pandera_validation_practice.py   # requires data/04_feature/admission_features.parquet
```

Execution order: feature pipeline → training pipeline → MLflow workflow;
feature pipeline → Pandera workflow.

## Project structure

```text
.
├── .code_quality/                # Ruff and mypy configuration
├── .github/
│   └── workflows/                # CI/CD (tests, pre-commit, audit, dependency review, ...)
├── conf/                         # template config scaffolding (not used by the pipelines)
├── data/
│   ├── 01_raw/                   # immutable raw data
│   ├── 02_intermediate/          # typed intermediate data
│   ├── 03_primary/               # domain model data
│   ├── 04_feature/               # generated feature set (gitignored)
│   ├── 05_model_input/           # sample batch input
│   ├── 06_models/                # serialized models
│   ├── 07_model_output/          # sample batch output
│   └── 08_reporting/             # generated training report (gitignored)
├── models/
│   └── 05_model_selection_pipeline.joblib   # fitted preprocessing + estimator
├── notebooks/
│   ├── 1-data/ 2-exploration/ 3-analysis/   # data prep and EDA
│   ├── 4-feat_eng/ 5-models/ 6-interpretation/   # modeling stages
│   └── 7-deploy/streamlit_deployment.md     # deployment evidence
├── scripts/                      # MLflow and Pandera supporting workflows
├── src/
│   ├── inference/                # Streamlit application and batch helpers
│   └── pipelines/
│       ├── feature_pipeline/     # RAW → validated features
│       ├── training_pipeline/    # features → artifact + report
│       └── inference_pipeline/   # artifact → online/batch predictions
├── tests/                        # pytest suite (pipelines, inference, Streamlit)
├── CHANGELOG.md
├── Dockerfile                    # multi-stage UV + Python 3.12 image
├── Makefile
├── pyproject.toml                # runtime + dev dependencies (UV)
└── uv.lock                       # locked dependency resolution
```

## Important methodological decisions

| Decision | Rationale |
|---|---|
| Exact duplicates removed before split | Prevents identical rows in both train and test |
| Target kept on its 0–1 scale | No transformation; metrics keep target units |
| Preprocessing inside the model `Pipeline` | Fit only within CV folds; consistent at inference |
| Canonical 80/20 split (`random_state=42`) defined once | Single train/test source of truth downstream |
| Nested CV for model selection | Tuning and comparison on separate validation partitions |
| Screening folds independent of outer CV | Selection and evaluation never share partitions |
| `LinearRegression` as final model | Parsimony, interpretability, lower tuning cost; performance equivalent to Ridge |
| Predictions not clipped or converted to percentages | Output stays a raw regression estimate |
| Immutable serialized artifact | Inference loads and verifies; never modifies |
| Inference shares the training artifact | Guarantees training/serving consistency |

## Known limitations

- **Dataset size** — 471 unique observations limit the power of statistical
  model comparisons and generalization claims.
- **Representativeness** — the dataset's population and collection process
  are not documented; findings should not be extrapolated to other applicant
  populations.
- **Correlated predictors** — GRE, TOEFL, and CGPA correlate; individual
  coefficient magnitudes should be interpreted cautiously.
- **Output interpretation** — predictions are raw regression estimates, not
  calibrated probabilities; LinearRegression can produce values outside
  `[0, 1]`.
- **No causal interpretation** — the model captures predictive associations
  only.
- **Not a decision system** — no governance, fairness assessment, or
  monitoring exists for real admissions decisions.

## Future improvements

- Repeated CV or bootstrap confidence intervals for metric uncertainty.
- Probability calibration or a constrained model if a true admission
  probability is required.
- Model monitoring and data-drift detection in serving.
- Validation on external applicant datasets.
- A model registry for versioned promotion of artifacts.
- API serving (e.g., FastAPI) alongside the Streamlit interface.
- Observability (structured logs, metrics) for the deployed application.

## References

- [uv documentation](https://docs.astral.sh/uv/)
- [scikit-learn: Pipeline and ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Docker documentation](https://docs.docker.com/)
- [pytest](https://docs.pytest.org/en/latest/)
- [Ruff](https://docs.astral.sh/ruff/) · [mypy](https://mypy-lang.org/) · [pre-commit](https://pre-commit.com/)
- [Layered data engineering convention](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
- [ML pipelines with FTI architecture](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

---

Project scaffolded from [@JoseRZapata]'s [data science project template].

[@JoseRZapata]: https://github.com/JoseRZapata
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
