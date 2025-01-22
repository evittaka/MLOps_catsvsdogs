FROM python:3.11-slim

EXPOSE 8080

WORKDIR /app

RUN git clone https://github.com/evittaka/MLOps_catsvsdogs.git /workspace/MLOps_catsvsdogs

# Move MLOps_catsvsdogs/src/catsvsdogs/api.py to /app
RUN cp /workspace/MLOps_catsvsdogs/src/catsvsdogs/api.py /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision timm \
    fastapi \
    pillow \
    google-cloud-storage \
    uvicorn \
    python-multipart


CMD exec uvicorn api:app --port 8080 --host 0.0.0.0 --workers 1
