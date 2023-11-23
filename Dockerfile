# Use NVIDIA CUDA 11.8.0 development environment for Ubuntu 22.04 as base image
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory to /workspace
WORKDIR /workspace

# Update system packages and install necessary dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-packaging git ninja-build && \
    # Install Accelerate and bitsandbytes
    pip3 install --upgrade pip && \
    pip3 install accelerate && \
    pip3 install -i https://test.pypi.org/simple/ bitsandbytes && \
    rm -rf /var/lib/apt/lists/*

# Specify supported CUDA architectures for PyTorch (adjust if necessary)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.2;7.5;8.0;8.6;8.9;9.0"

# Install PyTorch 
RUN pip3 install "torch>=2.0.0" 

# Copy and install Python requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy the default accelerate config and training script
COPY default_accelerate_config.yaml /root/.cache/torch/accelerate/default_config.yaml
COPY train.py .

# Use accelerate launch to run the training script
CMD ["accelerate", "launch", "train.py"]
