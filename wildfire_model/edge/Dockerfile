FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV and networking dependencies
    libglib2.0-0 \
    libgomp1 \
    libgl1-mesa-glx \
    # Basic utilities
    curl \
    wget \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
#COPY --chown=appuser:appuser . .
COPY  . .
# Create user and set permissions
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser && \
    mkdir -p /app/videos /app/results /app/models && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/models /app/videos /app/results

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Skip startup script - run app directly
CMD ["python3", "app.py"]
