FROM python:3.11-slim

WORKDIR /app

# Use Aliyun mirror for apt (system packages)
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies for OpenCV and MuJoCo
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
# Install torch CPU-only from PyTorch wheel index (avoids downloading large CUDA variant)
RUN pip install --no-cache-dir torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu
# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY assets/ ./assets/

# Create data directories
RUN mkdir -p data/raw_videos data/processed_videos data/metadata \
    data/pose_data data/analysis_results data/visualizations

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
