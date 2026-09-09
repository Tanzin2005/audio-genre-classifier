FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt
COPY --chown=appuser:appuser genre_cnn ./genre_cnn
COPY --chown=appuser:appuser web ./web
COPY --chown=appuser:appuser models ./models
USER appuser
ENV PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib
# Fail the build if trained weights are missing or incompatible.
RUN python -c "from genre_cnn.predict import Predictor; Predictor('models/cnn.pt')"
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
