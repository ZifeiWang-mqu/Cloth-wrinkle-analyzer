# Backend image. SAM deps are optional and gated by a build arg to keep the
# default image light.  Build the SAM-MVP with:  --build-arg INSTALL_SAM=true
FROM python:3.11-slim

# OpenCV (headless) needs a couple of shared libs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base (light) deps first for layer caching.
COPY requirements.txt requirements-sam.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Optionally install SAM (torch/torchvision/segment-anything). CPU wheels.
ARG INSTALL_SAM=false
RUN if [ "$INSTALL_SAM" = "true" ]; then \
        pip install --no-cache-dir \
          --extra-index-url https://download.pytorch.org/whl/cpu \
          -r requirements-sam.txt ; \
    fi

COPY . .

# Model files and input data are NOT baked into the image — they live on a
# mounted volume / persistent disk (see README).  WRINKLE_DATA_DIR can point at
# that volume, and WRINKLE_SAM_CHECKPOINT_PATH at the checkpoint.
ENV WRINKLE_DATA_DIR=/var/data \
    WRINKLE_SAM_DEVICE=cpu \
    WRINKLE_SAM_LAZY_LOAD=true \
    PORT=8000

# Honour $PORT (Render / Railway / Cloud Run inject it).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
