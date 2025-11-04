ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/root/.local/bin:/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       build-essential git curl libglib2.0-0 libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project metadata and lockfile first (cache layer)
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Create project venv outside bind mount and install dependencies from lock
RUN mkdir -p /opt/venv \
    && cd /app \
    && uv sync --frozen --extra dev

# Copy app source
COPY src/ /app/src/
COPY api/ /app/api/
COPY application.py /app/application.py

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/api/health || exit 1

CMD ["uvicorn", "application:get_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--reload"]
