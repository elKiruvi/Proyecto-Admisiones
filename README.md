# Proyecto-Admisiones

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Supervised regression project that estimates the `Chance of Admit` (0-1) of a
graduate-school applicant from their academic profile (GRE Score, TOEFL Score,
CGPA, University Rating, SOP, LOR, Research). The repository was scaffolded
with the data science project template below and keeps its layered data
structure, tooling, and quality gates.

Using the data science project template <https://github.com/JoseRZapata/data-science-project-template>

- `Python` = `3.12`
- `devcontainer` to work in `VSCode` or [GitHub Codespaces](https://github.com/features/codespaces) using the same environment as in production.

## ✨ Features and Tools

Details about the features and tools provided by the template: <https://joserzapata.github.io/data-science-project-template/#features-and-tools>

Features                                     | Package  | Why?
 ---                                         | ---      | ---
Dependencies and env                         | [UV] | [article](https://astral.sh/blog/uv)
Lint - Format, sort imports  (Code Quality)  | [Ruff] | [article](https://www.sicara.fr/blog-technique/boost-code-quality-ruff-linter)
Static type checking                         | [Mypy] | [article](https://python.plainenglish.io/does-python-need-types-79753b88f521)
Code quality & security each commit          | [pre-commit] | [article](https://dev.to/techishdeep/maximize-your-python-efficiency-with-pre-commit-a-complete-but-concise-guide-39a5)
Test code                                    | [Pytest] | [article](https://realpython.com/pytest-python-testing/)
Test coverage                                | [coverage.py] | [article](https://martinxpn.medium.com/test-coverage-in-python-with-pytest-86-100-days-of-python-a3205c77296)
Project Template                             | [Cruft] or [Cookiecutter] | [article](https://medium.com/@bctello8/standardizing-dbt-projects-at-scale-with-cookiecutter-and-cruft-20acc4dc3f74)
Folder structure for data science projects   | [Data structure] | [article](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)
Template for pull requests                   | [Pull Request template] | [article](https://www.awesomecodereviews.com/pull-request-template/)
Template for notebooks                       | [Notebook template] |

## Set up the environment

1. Initialize git in local:

    ```bash
    make init_git
    ```

1. Set up the environment:

    ```bash
    make install_env
    ```

    This installs the runtime dependencies, the development tooling and the
    notebook libraries (Jupyter, matplotlib, seaborn, scipy, pyarrow), all
    declared in `pyproject.toml`.

1. Activate virtual environment:

    ```bash
    source .venv/bin/activate
    ```

1. Launch Jupyter to open the project notebooks:

    ```bash
    uv run jupyter lab
    ```

## Install dependencies

After init the environment to install a new package, run:

```bash
uv add <package-name>
```

Example to install [plotly](https://plotly.com/python/) in dev group:

```bash
uv add --group dev plotly
```

## Admissions Demo

The Issue #8/#14 demo requirement provides a local Streamlit form for the
fitted admissions model. The application uses the existing Pipeline artifact
without retraining or changing the model:

```text
models/05_model_selection_pipeline.joblib
```

The artifact-compatible runtime uses Python 3.12 with NumPy 2.5.2, pandas
3.0.5, scikit-learn 1.9.0, and joblib 1.5.3.

From the repository root, install the locked environment and launch the local
form:

```bash
uv sync --locked
uv run streamlit run src/inference/streamlit_app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. The form accepts GRE Score, TOEFL Score, University
Rating, SOP, LOR, CGPA, and Research. Prediction occurs only after submitting
the form.

The displayed result is a raw `LinearRegression` estimate. It is not a
calibrated probability, is not converted to a percentage, and is not clipped
to `[0, 1]`. Continuous values outside the observed training range are
accepted and may represent extrapolation.

For the assignment evidence, run the local application and attach a
screenshot of the completed form and prediction to the pull request or issue.

## 🗃️ Project structure

- [Data structure]
- [Pipelines based on Feature/Training/Inference Pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

```bash
.
├── .code_quality
│   ├── mypy.ini                        # mypy configuration
│   └── ruff.toml                       # ruff configuration
├── data
│   ├── 01_raw                          # raw immutable data
│   ├── 02_intermediate                 # typed data
│   ├── 03_primary                      # domain model data
│   ├── 04_feature                      # model features
│   ├── 05_model_input                  # often called 'master tables'
│   ├── 06_models                       # serialized models
│   ├── 07_model_output                 # data generated by model runs
│   ├── 08_reporting                    # reports, results, etc
│   └── README.md                       # description of the data structure
├── .editorconfig                       # editor configuration
├── .github                             # github configuration
│   ├── dependabot.yml                   # github action to update dependencies
│   ├── pull_request_template.md        # template for pull requests
│   └── workflows                       # github actions workflows
│       ├── automerge.yml               # merge labeled dependency PRs
│       ├── ci.yml                      # run continuous integration (tests, pre-commit, etc.)
│       ├── dependency-review.yml      # review dependencies
│       ├── labels.yml                  # manage repository labels
│       └── pre-commit_autoupdate.yml   # update pre-commit hooks
├── .gitignore                          # files to ignore in git
├── Makefile                            # useful commands to setup environment, run tests, etc.
├── models                              # store final models
├── notebooks
│   ├── 1-data                          # data extraction and cleaning
│   ├── 2-exploration                   # exploratory data analysis (EDA)
│   ├── 3-analysis                      # Statistical analysis, hypothesis testing.
│   ├── 4-feat_eng                      # feature engineering (creation, selection, and transformation.)
│   ├── 5-models                        # model training, evaluation, and hyperparameter tuning.
│   ├── 6-interpretation                # model interpretation
│   ├── 7-deploy                        # model packaging, deployment strategies.
│   ├── 8-reports                       # story telling, summaries and analysis conclusions.
│   ├── notebook_template.ipynb         # template for notebooks
│   └── README.md                       # information about the notebooks
├── .pre-commit-config.yaml             # configuration for pre-commit hooks
├── pyproject.toml                      # dependencies for the python project
├── README.md                           # description of your project
├── src                                 # source code for use in this project
│   ├── README.md                       # description of src structure
│   ├── tmp_mock.py                     # example python file
│   ├── data                            # data extraction, validation, processing, transformation
│   ├── model                           # model training, evaluation, validation, export
│   ├── inference                       # model prediction, serving, monitoring
│   └── pipelines                       # orchestration of pipelines
│       ├── feature_pipeline            # transforms raw data into features and labels
│       ├── training_pipeline           # transforms features and labels into a model
│       └── inference_pipeline          # takes features and a trained model for predictions
├── tests                               # test code for your project
│   ├── test_mock.py                    # example test file
│   ├── data                            # tests for data module
│   ├── model                           # tests for model module
│   ├── inference                       # tests for inference module
│   └── pipelines                       # tests for pipelines module
└── .vscode                             # vscode configuration
    ├── extensions.json                 # list of recommended extensions
    ├── launch.json                     # vscode launch configuration
    └── settings.json                   # vscode settings
```

## Credits

This project was generated from [@JoseRZapata]'s [data science project template] template.

## References

- [Config devcontainer with python and UV](https://tech.dentsusoken.com/entry/2023/05/02/Dev_Container%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E3%83%90%E3%82%A4%E3%82%B9%E3%83%86%E3%83%83%E3%83%97%E3%81%A7%E4%BD%9C%E3%82%8BPython%E3%82%A2%E3%83%97%E3%83%AA%E3%82%B1)

---
[@JoseRZapata]: https://github.com/JoseRZapata

[Cookiecutter]:https://cookiecutter.readthedocs.io/en/stable/
[coverage.py]: https://coverage.readthedocs.io/
[Cruft]: https://cruft.github.io/cruft/
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
[Data structure]: data/README.md
[Mypy]: http://mypy-lang.org/
[Notebook template]: notebooks/notebook_template.ipynb
[pre-commit]: https://pre-commit.com/
[Pull Request template]: .github/pull_request_template.md
[Pytest]: https://docs.pytest.org/en/latest/
[Ruff]: https://docs.astral.sh/ruff/
[UV]: https://docs.astral.sh/uv/
