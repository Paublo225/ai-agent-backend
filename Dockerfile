FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

COPY . ./backend/

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
