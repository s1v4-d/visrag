ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Install minimal runtime dependencies + pip/uv
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       libglib2.0-0 libgl1 curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy only metadata for dependency installation
COPY pyproject.toml uv.lock /app/

# Create venv directory (dependencies will be installed on first run via volume mount)
RUN python -m venv /opt/venv

# Copy app source
COPY src/ /app/src/
COPY api/ /app/api/
COPY application.py /app/application.py

# Install dependencies at container startup (via entrypoint)
COPY <<'EOF' /entrypoint.sh
#!/bin/sh
set -e
if [ ! -f /opt/venv/.installed ]; then
  echo "Installing dependencies on first run with uv sync..."
  uv sync --frozen --extra dev
  touch /opt/venv/.installed
fi
exec "$@"
EOF

RUN chmod +x /entrypoint.sh

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/api/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "application:get_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--reload"]
