---
name: 🎯 Task / Issue
about: A course Issue or repository task (analysis, model, docs, CI, fix)
title: "Issue #N: "
labels: ''
assignees: ''
---

## 🎯 Objective

<!-- What this Issue is about, in one or two sentences. -->

## 📦 Deliverables

<!-- Concrete artifacts that must exist when this is done:
     notebook(s), data files, src/ code, tests, docs, reports. -->

## ✅ Acceptance criteria

- [ ] ...

## 🚧 Out of scope

<!-- What this Issue explicitly does NOT include (later Issues, retraining, etc.). -->

## 🧪 Evidence / validation

<!-- How to verify completion: notebook runs end to end, uv run pytest,
     specific outputs or figures. CI must be green before merge. -->

## 🔗 Dependencies

<!-- Upstream artifacts consumed and downstream artifacts affected
     (e.g., 04.1 split feeds 05.x; 05.2 produces the serialized artifact).
     Delete this section when the Issue has no such dependencies. -->

## 🌿 Suggested branch (optional)

<!-- feature/issue-N-* or fix/issue-N-* -->

## 🛡 Data / ML planning

<!-- Delete this section when the Issue does not involve data, ML, or pipelines. -->

- [ ] Preprocessing strategy identified (inside scikit-learn Pipeline, fitted per CV fold)
- [ ] Test set isolation planned until final evaluation
- [ ] Downstream notebook impact identified (`04.1` → `05.x`/`06.1`; `05.2` → artifact/demo)
- [ ] `models/05_model_selection_pipeline.joblib` left untouched by this Issue

## 📎 Related (optional)

<!-- Related to #N / Depends on #N — only when genuinely useful. -->
