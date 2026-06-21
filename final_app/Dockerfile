FROM python:3.10-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, PyTorch, and PaddleOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /code

# Copy requirements and install dependencies.
# Install CPU-only torch first (HF free tier is CPU) so we don't pull the huge
# CUDA build, then the remaining requirements.
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the application
COPY . /code

# Create necessary directories and ensure correct permissions
RUN mkdir -p /code/uploads /code/results /code/jobs && \
    chmod -R 777 /code

# Writable cache dirs + paddle/matplotlib settings (non-root user on Spaces).
ENV HF_HOME=/tmp/hf \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl \
    MPLBACKEND=Agg \
    FLAGS_use_mkldnn=0
RUN mkdir -p /tmp/hf /tmp/ultralytics /tmp/mpl && chmod -R 777 /tmp/hf /tmp/ultralytics /tmp/mpl

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
