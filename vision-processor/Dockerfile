# Use an official NVIDIA CUDA runtime image as a parent image
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# Set up the working environment
WORKDIR /app

# Prevent tzdata from prompting for a timezone during build
ENV TZ=Etc/UTC

# Install system dependencies including Python and Pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy the dependencies file to the container
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run the application
CMD ["python3", "vision.py"]