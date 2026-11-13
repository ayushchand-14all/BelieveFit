FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    BELIEVFIT_DATA_DIR=/var/data \
    BELIEVFIT_DB_PATH=/var/data/gym_coach.db \
    BELIEVFIT_MODEL_PATH=/var/data/models/pose_landmarker_full.task

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /var/data/videos /var/data/output /var/data/models

EXPOSE 10000
CMD ["python", "app.py"]
