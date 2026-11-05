# မြန်မာစာတန်းထိုး Video Editor
#
# Debian ffmpeg is built with --enable-libass --enable-libfreetype
# --enable-libharfbuzz, which is what Myanmar caption burn-in needs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HOME=/data/hf \
    MSE_UPLOADS_DIR=/data/uploads \
    MSE_PROJECTS_DIR=/data/projects

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        fonts-noto-core \
        libass9 \
        libharfbuzz0b \
        libfreetype6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

RUN mkdir -p /data/uploads /data/projects /data/hf

EXPOSE 8000

# ffmpeg capability check appears in the logs and at /api/capabilities
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
