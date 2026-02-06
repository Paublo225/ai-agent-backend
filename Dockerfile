FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Install runtime dependencies only (remove build-essential after build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./backend/

# Install requirements first so numpy+scipy+scikit-learn are ABI-consistent
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Install PyTorch CPU-only AFTER so it uses the already-installed numpy
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remove build dependencies to save space
RUN apt-get purge -y --auto-remove build-essential git \
    && rm -rf /root/.cache

COPY . ./backend/

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
