# alphaML 2.0 web application
#   docker compose up -d --build      then open http://localhost:8765
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba \
    ALPHAML_HOME=/data

# OpenMP runtime used by LightGBM and XGBoost; tini as PID 1 reaps the processes of finished or cancelled jobs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY setup.py README.md LICENSE ./
COPY alphaml ./alphaml
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 alphaml \
    && mkdir -p /data \
    && chown alphaml:alphaml /data
USER alphaml

# Data sets and job results
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/options', timeout=4)"

# The container listens on all its interfaces; docker-compose.yml publishes the port on localhost only
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["alphaml-web", "--host", "0.0.0.0", "--port", "8000"]
