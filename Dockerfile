# Backend image: FastAPI + WebSocket server (pose, DTW, scoring, coach).
# The frontend is not in this image; run it with Vite or build it separately.
#
# Build from the repo root:
#   docker build -t breakdown-backend .
# Run (secrets come from your .env at run time, never baked into the image):
#   docker run --rm -p 8000:8000 --env-file .env \
#     -v "$PWD/backend/kokoro-v1.0.onnx:/app/backend/kokoro-v1.0.onnx:ro" \
#     -v "$PWD/backend/voices-v1.0.bin:/app/backend/voices-v1.0.bin:ro" \
#     breakdown-backend
# Without the two Kokoro files the coach still runs, with captions and no voice.
# requirements.txt is a pip freeze made on Python 3.12, so use 3.12 here.
FROM python:3.12-slim

# libgl1 and libglib2.0-0: OpenCV. libgomp1: onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

COPY backend/requirements.txt backend/requirements-tts.txt ./
# Kokoro goes in last with --no-deps (see the comment in requirements-tts.txt).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps -r requirements-tts.txt

COPY backend/ ./

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 0.0.0.0 so the port mapping works. Set ACCESS_TOKEN in .env before exposing this.
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
