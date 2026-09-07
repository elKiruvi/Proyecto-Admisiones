# Builder stage: resolve and install the locked runtime dependencies with UV.
#
# Multi-stage layout follows the "Despliegue de Modelos con FastAPI y Docker"
# course approach: a UV builder stage prepares the locked virtual environment
# and a minimal Python 3.12 runtime stage only carries what the Streamlit
# application needs (the model artifact travels with the application; the
# course's FastAPI entrypoint is replaced by the existing Streamlit app).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Resolve from the repository's existing lockfile; --no-default-groups drops
# the default dev group (pytest, jupyter, mlflow, ...) so the image only
# carries the runtime dependencies declared in pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# Runtime stage: Python 3.12 with the locked virtual environment, the
# reusable inference code and the immutable model artifact.
FROM python:3.12-slim-bookworm

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
# pyproject.toml doubles as the repository-root marker used by the inference
# code (find_repository_root) to locate the model artifact.
COPY pyproject.toml ./
COPY src/ src/
COPY models/ models/
COPY data/05_model_input/new_applicants.csv data/05_model_input/new_applicants.csv

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Reliability check against Streamlit's health endpoint; stdlib only, so the
# runtime image needs no extra packages.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"]

CMD ["streamlit", "run", "src/inference/streamlit_app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.port=8501"]
