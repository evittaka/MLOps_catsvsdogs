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
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set Python default version
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip and install Python tools
RUN pip install --upgrade setuptools wheel pip

# Install Google Cloud SDK and gsutil
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    rm -rf /var/lib/apt/lists/*


# Install dvc
RUN pip install dvc[gs]

# Clone the target repository
RUN git clone https://github.com/evittaka/MLOps_catsvsdogs.git /workspace/MLOps_catsvsdogs

# Set the working directory to the cloned repository
WORKDIR /workspace/MLOps_catsvsdogs

# Install Python dependencies from requirements.txt and requirements_dev.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install -r requirements_dev.txt && pip install -e .

RUN mkdir -p data/processed

# Copy data from Google Cloud Storage
RUN gsutil -m cp -r gs://mlops_catsvsdogs/data/processed/* ./data/processed/

RUN wandb login --relogin ae15357077efc1030be7897029c1176a93502df0

RUN mkdir -p models/

# Add entrypoint script
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["invoke train"]
