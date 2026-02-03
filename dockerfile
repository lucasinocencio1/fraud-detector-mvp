FROM python:3.11-slim

# dependências do xgboost / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
