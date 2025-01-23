# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libjpeg-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/evittaka/MLOps_catsvsdogs.git /workspace/MLOps_catsvsdogs
COPY requirements_frontend.txt requirements_frontend.txt

RUN pip install -r requirements_frontend.txt --no-cache-dir --verbose

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/catsvsdogs/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
