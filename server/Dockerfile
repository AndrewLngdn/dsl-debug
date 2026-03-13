# =============================================================================
# DSL Debug Environment — Server Image
# =============================================================================
#
# Lightweight HTTP server for the DSL debug environment.
# No GPU required — runs the environment + test sets only.
#
# Build:
#   cd server && docker compose up --build
#
# Or standalone:
#   docker build -t dsl-debug-server -f server/Dockerfile .
#   docker run -p 8080:8080 -p 2222:22 dsl-debug-server

FROM python:3.12-slim

RUN apt-get update -qq && \
    apt-get install -y -qq openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /run/sshd /root/.ssh && \
    chmod 700 /root/.ssh

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY data/test_sets/ data/test_sets/
COPY server/ server/

RUN pip install --no-cache-dir ".[server]"

EXPOSE 8080 22

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
