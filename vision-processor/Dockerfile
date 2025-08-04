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
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces layer size
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Command to run the application
CMD ["python3", "main.py"]