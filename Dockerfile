# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================
# Production stage
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Add local packages to PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application
COPY --chown=app:app . .

# Download model at build time (optional - for Docker)
ARG MODEL_URL
ENV MODEL_URL=${MODEL_URL}
RUN if [ -n "$MODEL_URL" ]; then \
      mkdir -p models && \
      wget -q "$MODEL_URL" -O models/ensemble_final.pkl && \
      echo "Model downloaded at build time"; \
    fi

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "--timeout", "120", "--workers", "1"]