FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir minio>=7.2.0

COPY test_minio.py /app/test_minio.py

CMD ["python", "/app/test_minio.py"]