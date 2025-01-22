FROM python:3.11-slim

EXPOSE 8080
WORKDIR /app

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

COPY ./src/catsvsdogs/api.py api.py

CMD exec uvicorn api:app --port 8080 --host 0.0.0.0 --workers 1
