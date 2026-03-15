FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless requires libglib2.0-0; PyTorch requires libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch/torchvision first (CPU-only wheel) so this expensive layer
# is cached independently of application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source (includes deepfake_resnet18_best.pt, loaded relative to __file__)
COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
