# Use the NVIDIA PyTorch container as the base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Install essential tools and dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN pip install --upgrade setuptools wheel pip

# Clone the target repository
# ARG REPO_URL
RUN git clone https://github.com/evittaka/MLOps_catsvsdogs.git /workspace/MLOps_catsvsdogs
RUN cd /workspace/MLOps_catsvsdogs && git checkout docker_kagglehub

# Set the working directory to the cloned repository
WORKDIR /workspace/MLOps_catsvsdogs

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install -r requirements_dev.txt && pip install -e .

RUN python src/catsvsdogs/data.py data/raw/ data/processed/

# Run the data.py script to download the dataset
CMD ["/bin/bash", "-c", "invoke train"]
