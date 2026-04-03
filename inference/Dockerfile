FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY inference/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY inference /app

# non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV MODEL_ID=distilgpt2 \
    TORCH_THREADS=4

EXPOSE 8000
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]