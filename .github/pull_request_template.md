## 🔗 Issue

Closes #<issue-number>

<!-- One PR per Issue. Link the Issue this PR implements. -->

## 📌 Summary

<!-- 1–3 sentences: what this PR does and why (context and rationale). -->

## Type of change

<!-- Mark all that apply; remove unused options. -->

- [ ] ✨ Feature
- [ ] 🐛 Bugfix
- [ ] 🔥 Improvement / refactor
- [ ] 📝 Documentation
- [ ] ✅ Tests
- [ ] 👷 CI / configuration
- [ ] ⚗️ Notebook(s)
- [ ] 📦 Data
- [ ] 🤖 Model / serialized artifact
- [ ] Other (describe)

## Areas affected

<!-- Mark all that apply. -->

- [ ] `src/`
- [ ] `tests/`
- [ ] `notebooks/`
- [ ] `data/`
- [ ] Docs (`README.md`, `CHANGELOG.md`, `AGENTS.md`)
- [ ] `.github/` workflows / configuration
- [ ] `pyproject.toml` / `uv.lock`
- [ ] `models/`

## 🛠 What does this PR implement

<!-- Itemized list of the changes. -->

## 🚧 Scope and non-scope

<!-- What this PR deliberately does NOT change (e.g., ML methodology, model
artifact, historical notebooks) and what the Issue asked for that was not
implemented, if any. -->

## 🧪 Validation

<!-- Run locally before requesting review. CI must be green before merge. -->

- [ ] `uvx pre-commit run --all-files --show-diff-on-failure --color=always`
- [ ] `uv run pytest --cov --cov-branch --cov-fail-under=60`
- [ ] CI green on this PR

## 📓 Notebooks

<!-- Only when notebooks changed; otherwise delete this section.
     Re-run downstream notebooks only when their inputs, assumptions, or
     outputs are actually affected; avoid unnecessary cascades. -->

- [ ] Notebook(s) executed from top to bottom with no errors
- [ ] Downstream notebooks reviewed (`04.1` → `05.x`/`06.1`; `05.2` → serialized artifact and demo)
- [ ] Affected downstream notebooks re-executed and outputs verified, or justified why not

## 🛡 ML / data safety

<!-- Only for ML, data, or pipeline changes; otherwise delete this section. -->

- [ ] Preprocessing that learns parameters stays inside the `scikit-learn` `Pipeline`/`ColumnTransformer`
- [ ] Preprocessing fitted independently inside each CV fold (no full-train fit before CV)
- [ ] Test set isolated until final evaluation
- [ ] `models/05_model_selection_pipeline.joblib` not manually modified or regenerated outside its producing notebook
- [ ] `data/01_raw/` unchanged
- [ ] No `pyproject.toml` / `uv.lock` / workflow changes beyond what this Issue requires

## ✅ Checklist

### Contributor

- [ ] PR is scoped to one Issue
- [ ] Branch follows `feature/issue-N-*` or `fix/issue-N-*`
- [ ] Commits use Conventional Commits
- [ ] Diff reviewed: no unrelated files, secrets, caches, or `.atl/`
- [ ] `CHANGELOG.md` updated when the Issue adds/changes user-visible artifacts

### Reviewer

- [ ] Issue reference present and correct
- [ ] Scope matches the Issue (no scope creep)
- [ ] No data leakage introduced
- [ ] Notebook downstream dependencies addressed
- [ ] Validation reported and CI green
