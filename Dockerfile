# # Stage 1: Build dependencies & pre-download Hugging Face model
# FROM python:3.11-slim AS builder

# WORKDIR /app

# # Install uv for extremely fast package installation
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# # Copy dependency definition
# COPY pyproject.toml .

# # Create virtual environment using system python
# RUN uv venv --python python /opt/venv

# # Set virtual environment environment variable for uv
# ENV VIRTUAL_ENV=/opt/venv

# # Install dependencies from pyproject.toml
# RUN uv pip install -r pyproject.toml


# # Pre-download the Hugging Face model to bake it into the image.
# # This prevents downloading it during runtime container boot.
# ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
# RUN /opt/venv/bin/python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

# # Stage 2: Final lean runtime image
# FROM python:3.11-slim AS runner

# WORKDIR /app

# # Install system dependencies (if any are needed at runtime)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy virtual environment and Hugging Face cache from the builder stage
# COPY --from=builder /opt/venv /opt/venv
# COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# # Copy codebase
# COPY main.py services.py custom_types.py data_loader.py vector_db.py ./
# COPY static/ ./static/

# # Environment configuration
# ENV PATH="/opt/venv/bin:$PATH"
# ENV PYTHONUNBUFFERED=1
# ENV PORT=8000

# EXPOSE 8000

# # Run uvicorn server with dynamic port allocation (defaulting to 8000)
# CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# ── Stage 1: Model cache (almost never re-runs) ──────────────────────────
FROM python:3.11-slim AS model-cache

RUN pip install huggingface_hub --quiet
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"


# ── Stage 2: Dependency install (re-runs only when requirements.lock changes) ─
FROM python:3.11-slim AS builder

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy lockfile ONLY — not pyproject.toml — so source changes don't bust this layer
COPY requirements.txt .

RUN uv venv --python python /opt/venv
ENV VIRTUAL_ENV=/opt/venv
RUN uv pip install -r requirements.txt


# ── Stage 3: Lean runtime image ───────────────────────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy model cache from dedicated stage (not from builder)
COPY --from=model-cache /root/.cache/huggingface /root/.cache/huggingface

# COPY order: stable → volatile
# static/ changes rarely — copy first so it stays cached across code edits
COPY static/ ./static/
# Python files change often — copy last so only this layer re-runs on edits
COPY main.py services.py custom_types.py data_loader.py vector_db.py ./

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]