# Streamlit Community Cloud deployment evidence — Task 3

This document records the deployment and functional evidence for Task 3
(Work 3) of the Graduate Admissions project. The Streamlit application itself
(online and batch modes) was implemented in earlier issues: the online demo
in Issue #8/#14 and the batch mode in Issue #56 / PR #57. Issue #58 documents
and evidences the public deployment of the complete Task 3 without changing
any application, inference, training, or model code.

## Deployment configuration

- Public URL: https://proyecto-admisiones-samuelc.streamlit.app/
- Repository: elKiruvi/Proyecto-Admisiones
- Branch: `main`
- Main file: `src/inference/streamlit_app.py`
- Python: 3.12
- Streamlit secrets: none required

The deployment serves the fitted admissions regression model through the
existing inference pipeline. It does not retrain, refit, or regenerate
anything: preprocessing (imputation, encoding, scaling) remains inside the
serialized model artifact.

## Model artifact

The deployed application consumes:

```text
models/05_model_selection_pipeline.joblib
```

SHA-256: `8cb82a4a8d417ee5f0e19c3010295de74fe85ebd3f9ef5a41e746de5af978a71`

This hash applies to the model artifact only.

## Functional verification

The following verifications were performed manually against the deployed
application (2026-09-07).

### Online prediction

The Online prediction tab was tested with a valid applicant profile. The
application returned:

```text
Raw LinearRegression estimate: 0.810
```

The interface states that this is a raw regression estimate, not a calibrated
probability or percentage.

### Batch prediction

The Batch prediction tab was tested with the versioned sample input:

```text
data/05_model_input/new_applicants.csv
```

The deployed application accepted the CSV, generated predictions, displayed
the resulting dataframe, preserved the seven input features, appended
`predicted_chance_of_admit` as the final column, and handled existing null
values through the inference pipeline. The observed predictions were
approximately:

```text
0.725993
0.572708
0.905749
0.575465
```

matching the versioned sample output `data/07_model_output/admission_predictions.csv`,
which was generated/reproduced using the current model artifact. The
predictions were downloaded as CSV and verified to contain the seven input
columns plus the prediction column.

### Validation and error handling

Two invalid CSV scenarios were tested manually against the deployed
application.

Missing column (`invalid_missing_cgpa.csv`):

```text
Inference input validation failed with 1 issue(s):
Schema: missing columns: ['CGPA'].
```

Invalid domain values (`invalid_admission_values.csv`): the application
reported seven validation issues covering GRE Score outside `[0, 340]`,
TOEFL Score outside `[0, 120]`, CGPA outside `[0, 10]`, University Rating
outside the allowed categories, SOP outside the allowed categories, LOR
outside the allowed categories, and Research outside `{0, 1}`.

Both cases produced clear validation errors and no unhandled exceptions.

## Task 3 traceability matrix

| Requirement | Evidence |
|---|---|
| Online interface | Streamlit Online prediction tab |
| Public URL | https://proyecto-admisiones-samuelc.streamlit.app/ |
| Online prediction | Verified result: `Raw LinearRegression estimate: 0.810` |
| Batch interface | Streamlit Batch prediction tab |
| Multiple records | `data/05_model_input/new_applicants.csv` |
| Batch predictions | `data/07_model_output/admission_predictions.csv` (≈ 0.725993 / 0.572708 / 0.905749 / 0.575465) |
| Visualization | Batch results dataframe (seven input columns + `predicted_chance_of_admit`) |
| Download | CSV download functionality (downloaded and verified) |
| Validation | Missing-column test and invalid-domain tests described above |
| Instructions | `README.md` (Admissions Demo section) and this document |

## Evidence retention

Screenshots captured during the manual verification are intentionally not
versioned in this repository. They are retained externally by the repository
owner as academic evidence for the final Task 3 submission, alongside this
document, which describes exactly which evidence was obtained for each
requirement.
