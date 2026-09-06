# Container image of the XAI chest radiology service.
# Build:  docker build -t xai-radiology .
# Run:    docker run -p 8000:8000 -v $(pwd)/models:/app/models xai-radiology
FROM python:3.11-slim

# libgl and libglib are required by the image processing stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# The checkpoint is mounted at runtime rather than baked into the image,
# so the image stays small and the weights stay out of version control.
ENV CHECKPOINT_PATH=/app/models/densenet121_rsna_best.pth

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
