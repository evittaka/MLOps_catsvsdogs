# Use the NVIDIA PyTorch container as the base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set the working directory
WORKDIR /workspace

# Install essential tools and dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Kaggle Python package
RUN pip install --no-cache-dir kaggle

# Clone the target repository
ARG REPO_URL
RUN git clone https://github.com/evittaka/MLOps_catsvsdogs.git /workspace/MLOps_catsvsdogs
RUN cd /workspace/MLOps_catsvsdogs && git checkout docker_kagglehub

# Set the working directory to the cloned repository
WORKDIR /workspace/MLOps_catsvsdogs

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install kagglehub

RUN python src/catsvsdogs/data.py data/raw/ data/processed/

# Run the data.py script to download the dataset
CMD ["/bin/bash"]
