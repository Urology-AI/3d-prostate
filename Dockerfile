# ============================================================
#  Prostate3D — unified image (web server + segmentation)
#
#  Role A — web server (interactive compute node, no GPU):
#    singularity run --contain
#      --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp
#      --bind /tmp:/tmp
#      pipeline.sif
#
#  Role B — GPU segmentation batch job:
#    singularity exec --nv --contain
#      --bind /sc/arion/projects/video_rarp:/sc/arion/projects/video_rarp
#      --bind /sc/arion/projects/video_rarp/3dprostate/radiology_model:/app/radiology/model
#      --bind /sc/arion/projects/video_rarp/3dprostate/radiology_logs:/app/radiology/logs
#      --bind /tmp:/tmp
#      pipeline.sif python3 /app/segment_prostate.py --job-id <id>
#
#  The /app/radiology/model bind lets model weights be downloaded
#  once to shared storage instead of being re-downloaded per job.
#
#  Build for Minerva (from Mac M-chip or any machine):
#    docker buildx build --platform linux/amd64 \
#      -t adidix/prostate3d:latest --push .
#
#  Build for local dev (native arch, fast):
#    docker build -t prostate3d:dev .
# ============================================================

FROM python:3.10-slim

ARG TARGETARCH

LABEL org.opencontainers.image.title="Prostate3D Pipeline"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONWARNINGS="ignore" \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget gcc g++ \
        libglib2.0-0 libgomp1 \
        libsm6 libxext6 libxrender-dev \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch ───────────────────────────────────────────────────────────────────
# amd64 → CUDA 12.1 wheel  (Minerva GPU nodes; --nv provides the driver)
# arm64 → CPU-only wheel   (Mac M-chip local dev)
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install --no-cache-dir \
            torch==2.2.2 torchvision==0.17.2 \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir \
            torch==2.2.2 torchvision==0.17.2; \
    fi

# ── Version pins that must be set before anything else resolves them ──────────
RUN pip install --no-cache-dir "numpy<2.0" "pydicom>=2.3.0,<2.4.0"

# ── SAM2 stub — installed BEFORE monailabel so pip sees it as satisfied ───────
# This lets us install monailabel WITH --deps (the normal way), so all real
# transitive deps (pytorch-ignite, etc.) are resolved automatically by pip.
COPY docker/scripts/create_sam2_stub.py /tmp/create_sam2_stub.py
RUN python3 /tmp/create_sam2_stub.py

# ── MONAI with extras (pre-satisfy monailabel's monai dep with extras) ─────────
RUN pip install --no-cache-dir \
    "monai[fire,nibabel,pillow,psutil,skimage,tqdm,torchvision]>=1.4.0"

# ── MONAILabel with full deps (sam2 already stubbed; pytorch-ignite etc. auto-installed) ──
RUN pip install --no-cache-dir monailabel && \
    pip install --no-cache-dir girder-client

# ── App-specific deps not pulled by monailabel ────────────────────────────────
RUN pip install --no-cache-dir \
    "flask>=2.3.0" \
    "flask-cors>=4.0.0" \
    "gunicorn>=21.2.0" \
    "SimpleITK>=2.3.0" \
    "scikit-image>=0.21.0"

RUN pip install --no-cache-dir numpymaxflow 2>/dev/null || echo "numpymaxflow skipped (optional)"

# ── Patch monailabel/config.py (None name bug in distributions()) ────────────
COPY docker/scripts/patch_monailabel.py /tmp/patch_monailabel.py
RUN python3 /tmp/patch_monailabel.py

# ── Download MONAILabel radiology app (includes prostate_mri_anatomy model) ──
# The model WEIGHTS are downloaded at runtime on first job run (to /app/radiology/model,
# which is bind-mounted to /sc/arion/.../radiology_model so they're cached).
RUN monailabel apps --download --name radiology --output /tmp/radiology_dl && \
    if [ -d "/tmp/radiology_dl/radiology" ]; then \
        mv /tmp/radiology_dl/radiology /app/radiology; \
        rm -rf /tmp/radiology_dl; \
    else \
        mv /tmp/radiology_dl /app/radiology; \
    fi && \
    ls /app/radiology/main.py && \
    echo "Radiology app downloaded OK"

# ── Create writable placeholder dirs (will be bind-mounted on Minerva) ───────
RUN mkdir -p /app/radiology/model \
             /app/radiology/logs \
             /app/radiology/lib \
             /app/radiology/bin \
             /sc/arion/projects/video_rarp/3dprostate/seg_jobs

# ── Smoke-test imports ────────────────────────────────────────────────────────
RUN python3 -c "\
import torch; print('torch', torch.__version__); \
import monai; print('monai', monai.__version__); \
import monailabel; print('monailabel OK'); \
import ignite; print('pytorch-ignite', ignite.__version__); \
import girder_client; print('girder-client OK'); \
from monailabel.tasks.train.basic_train import BasicTrainTask; print('basic_train OK'); \
import SimpleITK; print('SimpleITK OK'); \
import nibabel; print('nibabel OK'); \
from skimage import measure; print('scikit-image OK'); \
from flask import Flask; print('flask OK')"

# ── Application code ──────────────────────────────────────────────────────────
COPY backend/app.py              ./
COPY backend/segment_prostate.py ./
COPY backend/pipeline/           ./pipeline/
COPY frontend/templates/         ./templates/

EXPOSE 5000

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "300", \
     "--max-requests", "200", \
     "app:app"]
