# Prostate3D — DICOM to 3D Anatomical Visualization

```
Upload DICOM → HPC GPU → MONAILabel segmentation → 3D render → Browser
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Your Mac                                                     │
│  ┌──────────────┐    HTTP     ┌──────────────────────────┐   │
│  │   Browser    │────────────▶│  Web App (Flask :5000)   │   │
│  └──────────────┘◀────────────└────────────┬─────────────┘   │
│                                            │ SSH/SFTP         │
└────────────────────────────────────────────┼─────────────────┘
                                             │
                    ┌────────────────────────▼──────────────────┐
                    │  HPC Cluster                              │
                    │  Singularity ← Docker image              │
                    │  prostate3d-monai:latest                  │
                    │  MONAILabel + prostate_mri_anatomy        │
                    │  NVIDIA GPU  ~15-30 sec/case              │
                    └───────────────────────────────────────────┘
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/prostate3d.git
cd prostate3d
cp .env.example .env          # edit with your HPC credentials
pip install -r backend/requirements.txt
python backend/app.py         # opens http://localhost:5000
```

Build HPC image (from Mac):
```bash
docker buildx build \
  --platform linux/amd64 \
  -t adidix/prostate3d-monai:latest \
  -f docker/Dockerfile.monai \
  --push \
  .
```

---

## Run Local + Tunnel

Start local web server:
```bash
cd 3d-prostate
pip install -r backend/requirements.txt
python backend/app.py
```

Create SSH tunnel from Mac:
```bash
ssh -L 8000:localhost:8000 your_hpc_username@your_hpc_login_host
```

On your HPC, load this image via Singularity:
```bash
singularity pull --force monai.sif docker://adidix/prostate3d-monai:latest
```

Use `prostate3d-monai:latest` (GPU image) on HPC.

---

## Docker Images

| Image | Purpose | Platform |
|---|---|---|
| `prostate3d-monai:latest` | MONAILabel GPU server | linux/amd64 + CUDA 11.8 |
| `prostate3d-monai-cpu:latest` | MONAILabel CPU server | linux/amd64 (HPC-compatible build) |
| `prostate3d-web:latest` | Flask web orchestrator | linux/amd64 |

### Pull
```bash
docker pull adidix/prostate3d-monai:latest       # GPU
docker pull adidix/prostate3d-monai-cpu:latest   # CPU / Mac
docker pull adidix/prostate3d-web:latest         # Web app
```

### Run with Docker Compose
```bash
# CPU only (Mac / testing)
docker compose -f docker/docker-compose.yml --profile cpu up monai-cpu

# GPU (Linux + NVIDIA)
docker compose -f docker/docker-compose.yml up monai-gpu

# Full stack
docker compose -f docker/docker-compose.yml up
```

### Build and push
```bash
docker buildx build \
  --platform linux/amd64 \
  -t your-dockerhub-username/prostate3d-monai:latest \
  -f docker/Dockerfile.monai \
  --push \
  .
```

### Push to GitHub + Docker Hub
```bash
# 1) Initialize and push repository
cd 3d-prostate
git init
git add .
git commit -m "Initial commit"
gh repo create prostate3d --public --push

# 2) Add GitHub Secrets for Docker Hub
# DOCKERHUB_USERNAME
# DOCKERHUB_TOKEN
```

With `.github/workflows/docker.yml` in place, pushes to `main` auto-build and push all 3 images.

### Use on HPC as Singularity
```bash
singularity pull --force monai.sif docker://adidix/prostate3d-monai:latest
```

---

## Pipeline Flow

```
1.  Upload DICOM via browser
2.  Auto-detect T2-weighted series
3.  Convert DICOM → NIfTI
4.  SFTP upload to HPC
5.  Submit LSF GPU job (bsub)
6.  Singularity starts MONAILabel on GPU
7.  prostate_mri_anatomy model runs (~15-30s)
8.  SFTP download segmentation
9.  VTK renders 4 anatomical views
10. Browser shows 3D model + volume stats
```

## Structure

```
prostate3d/
├── backend/
│   ├── app.py                  Flask orchestrator
│   └── pipeline/
│       ├── hpc_client.py       SSH/SFTP/LSF
│       ├── dicom_utils.py      DICOM → NIfTI
│       └── renderer.py         VTK 3D render
├── frontend/
│   └── templates/index.html   Web UI
├── docker/
│   ├── Dockerfile.monai        GPU server
│   ├── Dockerfile.cpu          CPU server
│   ├── Dockerfile.web          Web app
│   ├── docker-compose.yml
│   └── scripts/
├── scripts/
│   ├── start.sh
│   ├── setup_cluster.sh
│   ├── setup_hpc.sh          compatibility wrapper
│   └── docker_build.sh
└── .github/workflows/docker.yml  CI/CD
```

## License
MIT
