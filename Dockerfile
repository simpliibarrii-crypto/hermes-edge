FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Runtime deps only
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY hermes/  hermes/
COPY scripts/ scripts/
COPY dist/    dist/

# Install hermes as package
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# litert-lm CLI (optional — falls back to simulated responses)
RUN pip install --no-cache-dir litert-lm 2>/dev/null || true

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "from hermes.router import get_intent; assert get_intent('test') in ('chat','reasoning','tools')" || exit 1

# Default: run the agent CLI
ENTRYPOINT ["python3", "-m", "hermes.cli"]
CMD ["--help"]
