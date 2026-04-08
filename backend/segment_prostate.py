#!/usr/bin/env python3
"""
Prostate MRI Segmentation Script
==================================
Runs inside Singularity on a Minerva GPU node as a bsub job.

Pipeline
--------
  1. Find T2 DICOM series  →  NIfTI  (SimpleITK)
  2. Start MONAILabel radiology server  (prostate_mri_anatomy model)
  3. POST /infer/<model>?output=image  →  segmentation.nii.gz
  4. Per-label marching-cubes  →  binary STL files
  5. Write result.json + status.txt "done"

Model weight caching
--------------------
  The radiology app writes weights to /app/radiology/model/.
  On Minerva, bind-mount that to /sc/arion/.../radiology_model so
  weights are downloaded once and reused across jobs:
    singularity exec --nv --contain
      --bind .../radiology_model:/app/radiology/model
      --bind .../radiology_logs:/app/radiology/logs
      ...

Usage
-----
  python3 segment_prostate.py
      --job-id  <8-char hex>
      --jobs-base /sc/arion/projects/video_rarp/3dprostate/seg_jobs
      [--model prostate_mri_anatomy]
      [--port  8765]
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import measure


# ── Anatomy config ────────────────────────────────────────────────────────────
# Labels produced by the prostate_mri_anatomy model (radiology app)
LABEL_CONFIG = {
    1: {"name": "Whole Gland",     "color": "#d6ac78", "opacity": 0.35},
    2: {"name": "Transition Zone", "color": "#c73030", "opacity": 0.92},
    3: {"name": "Peripheral Zone", "color": "#e8b030", "opacity": 0.80},
}


# ── Status helpers ────────────────────────────────────────────────────────────

def write_status(job_dir: Path, text: str) -> None:
    tmp = job_dir / "status.tmp"
    tmp.write_text(text)
    tmp.replace(job_dir / "status.txt")
    print(f"[status] {text}", flush=True)


# ── DICOM → NIfTI ─────────────────────────────────────────────────────────────

def _find_dicom_dir(root: Path) -> str:
    best_dir, best_count = str(root), 0
    for dirpath, _, files in os.walk(str(root)):
        visible = [f for f in files if not f.startswith(".")]
        if len(visible) > best_count:
            best_count, best_dir = len(visible), dirpath
    if best_count == 0:
        raise RuntimeError(f"No DICOM files found under {root}")
    return best_dir


def convert_dicom_to_nifti(input_dir: Path, output_path: Path) -> sitk.Image:
    series_dir = _find_dicom_dir(input_dir)
    reader     = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)

    if series_ids:
        best  = max(series_ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(series_dir, s)))
        files = reader.GetGDCMSeriesFileNames(series_dir, best)
    else:
        files = sorted(
            str(Path(series_dir) / f)
            for f in os.listdir(series_dir)
            if not f.startswith(".")
        )

    reader.SetFileNames(files)
    img = reader.Execute()

    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    if img.GetDimension() == 4:
        sz = list(img.GetSize()); sz[3] = 0
        img = sitk.Extract(img, sz, [0, 0, 0, 0])

    sitk.WriteImage(img, str(output_path))
    print(f"[DICOM] NIfTI: size={img.GetSize()} "
          f"spacing={[round(s, 2) for s in img.GetSpacing()]}", flush=True)
    return img


# ── MONAILabel inference ───────────────────────────────────────────────────────

def _wait_for_server(port: int, timeout: int = 180) -> None:
    url      = f"http://127.0.0.1:{port}/info"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=4) as r:
                if r.status == 200:
                    print(f"[seg] Server ready on port {port}", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"MONAILabel server did not become ready within {timeout}s")


def _multipart_body(nifti_path: Path, boundary: str) -> bytes:
    """Build a multipart/form-data body with the NIfTI file + output=image."""
    with open(str(nifti_path), "rb") as f:
        file_data = f.read()

    # Part 1: the output format selector
    part_output = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="output"\r\n\r\n'
        f"image\r\n"
    ).encode()

    # Part 2: the NIfTI file
    part_file = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="mri.nii.gz"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + file_data + b"\r\n"

    tail = f"--{boundary}--\r\n".encode()
    return part_output + part_file + tail


def _extract_nifti_from_response(raw: bytes) -> bytes:
    """Extract the gzip/NIfTI payload from a multipart inference response."""
    gz_start = raw.find(b"\x1f\x8b")
    if gz_start < 0:
        # Response might be the NIfTI directly
        if len(raw) > 100:
            return raw
        raise RuntimeError(f"Inference returned no gzip payload ({len(raw)} bytes). "
                           "Check MONAILabel server logs in /app/radiology/logs.")

    gz_data = raw[gz_start:]
    # Trim trailing multipart boundary
    for marker in (b"\r\n--", b"\n--"):
        idx = gz_data.rfind(marker)
        if 0 < idx < len(gz_data):
            gz_data = gz_data[:idx]
            break
    return gz_data


def run_monailabel_inference(
    nifti_path: Path,
    output_dir: Path,
    model: str,
    port: int,
    job_dir: Path,
) -> Path:
    # The radiology app is baked into the container at /app/radiology
    # (downloaded during Docker build via `monailabel apps --download --name radiology`)
    radiology_app = Path("/app/radiology")
    if not radiology_app.is_dir() or not (radiology_app / "main.py").exists():
        raise RuntimeError(
            f"Radiology app not found at {radiology_app}. "
            "This should have been downloaded during the Docker build."
        )

    studies_dir = output_dir / "studies"
    studies_dir.mkdir(exist_ok=True)

    import shutil
    shutil.copy2(str(nifti_path), str(studies_dir / "mri.nii.gz"))

    # Proxy settings — needed for first-run model weight download on Minerva
    # compute nodes that don't have direct internet access.
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "all_proxy"):
        if k not in env and k.upper() not in env:
            env[k]          = "http://172.28.7.1:3128"
            env[k.upper()]  = "http://172.28.7.1:3128"
    env.setdefault("no_proxy",  "localhost,*.chimera.hpc.mssm.edu,172.28.0.0/16")
    env.setdefault("NO_PROXY",  "localhost,*.chimera.hpc.mssm.edu,172.28.0.0/16")

    # The monailabel console script is at /usr/local/bin/monailabel (installed by pip)
    monailabel_cmd = shutil.which("monailabel") or "/usr/local/bin/monailabel"

    print(f"[seg] Starting MONAILabel ({monailabel_cmd}) "
          f"app={radiology_app} port={port} model={model}", flush=True)

    server = subprocess.Popen(
        [
            monailabel_cmd, "start_server",
            "--app",     str(radiology_app),
            "--studies", str(studies_dir),
            "--conf",    "models", model,
            "--host",    "127.0.0.1",
            "--port",    str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        write_status(job_dir, "running|40|Waiting for inference server to start…")
        _wait_for_server(port, timeout=240)   # first start downloads weights → longer timeout

        write_status(job_dir, f"running|55|Running {model} inference…")
        boundary  = "ProstBoundary007"
        body      = _multipart_body(studies_dir / "mri.nii.gz", boundary)
        infer_url = (
            f"http://127.0.0.1:{port}/infer/"
            f"{urllib.parse.quote(model, safe='')}"
        )

        req = urllib.request.Request(
            infer_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        print(f"[seg] POST {infer_url} ({len(body)//1024} KB)", flush=True)
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
        print(f"[seg] Response: {len(raw)//1024} KB", flush=True)

        seg_path = output_dir / "segmentation.nii.gz"
        seg_path.write_bytes(_extract_nifti_from_response(raw))

        size = seg_path.stat().st_size
        print(f"[seg] Segmentation: {size // 1024} KB → {seg_path}", flush=True)
        if size < 512:
            raise RuntimeError(
                f"Segmentation file too small ({size} bytes) — inference likely failed. "
                "Check /app/radiology/logs for details."
            )
        return seg_path

    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        print("[seg] MONAILabel server stopped", flush=True)


# ── Segmentation NIfTI → STL meshes ──────────────────────────────────────────

def _write_stl_binary(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    with open(str(path), "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(faces)))
        for tri in faces:
            v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
            n  = np.cross(v1 - v0, v2 - v0)
            nl = np.linalg.norm(n)
            n  = (n / nl) if nl > 0 else n
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))


def _normalize_labels(arr: np.ndarray) -> np.ndarray:
    """
    Normalise MONAILabel output to {0,1,2,3} expected by LABEL_CONFIG.
    Handles binary {0,1} or {0,255} (whole-gland-only models).
    """
    labels = set(np.unique(arr).tolist())
    if labels.issubset({0, 1, 2, 3}):
        return arr
    if labels.issubset({0, 255}):
        return (arr > 0).astype(np.uint8)
    if labels.issubset({0, 1}):
        return arr
    # Unknown labels — try remapping to consecutive integers
    print(f"[STL] Unexpected labels {sorted(labels)} — using as-is", flush=True)
    return arr


def seg_to_stl_files(seg_path: Path, output_dir: Path) -> list:
    img   = nib.load(str(seg_path))
    arr   = np.round(np.asarray(img.dataobj)).astype(np.uint8)
    arr   = _normalize_labels(arr)
    zooms = np.abs(img.header.get_zooms()[:3])

    print(f"[STL] shape={arr.shape}  spacing={np.round(zooms,2)}"
          f"  labels={sorted(np.unique(arr).tolist())}", flush=True)

    # If the model returned only binary output, treat label 1 as whole gland
    unique = sorted(np.unique(arr).tolist())
    if unique == [0, 1]:
        print("[STL] Binary segmentation detected → treating as Whole Gland only", flush=True)
        effective_config = {1: LABEL_CONFIG[1]}
    else:
        effective_config = LABEL_CONFIG

    results = []
    for label_id, cfg in effective_config.items():
        mask      = arr == label_id
        vox_count = int(mask.sum())
        if vox_count < 50:
            print(f"[STL] label {label_id} ({cfg['name']}): {vox_count} voxels — skipped", flush=True)
            continue

        padded = np.pad(mask.astype(np.float32), 1)
        try:
            verts, faces, _, _ = measure.marching_cubes(
                padded, level=0.5, spacing=tuple(float(z) for z in zooms)
            )
        except Exception as exc:
            print(f"[STL] label {label_id} marching_cubes failed: {exc}", flush=True)
            continue

        stl_name = f"label_{label_id}.stl"
        _write_stl_binary(verts, faces, output_dir / stl_name)

        volume_cc = round(vox_count * float(np.prod(zooms)) / 1000.0, 1)
        print(f"[STL] label {label_id} ({cfg['name']}): "
              f"{vox_count:,} vox  {volume_cc} cc  {len(faces):,} triangles", flush=True)

        results.append({
            "id":        label_id,
            "name":      cfg["name"],
            "file":      stl_name,
            "volume_cc": volume_cc,
            "color":     cfg["color"],
            "opacity":   cfg["opacity"],
        })

    if not results:
        raise RuntimeError(
            "No anatomical structures found in segmentation output.\n"
            f"Labels in output: {sorted(np.unique(arr).tolist())}\n"
            "Possible causes:\n"
            "  - Input is not a T2-weighted axial MRI\n"
            "  - Wrong DICOM series selected\n"
            "  - Model weights not downloaded correctly"
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prostate MRI segmentation")
    parser.add_argument("--job-id",    required=True)
    parser.add_argument("--jobs-base", default="/sc/arion/projects/video_rarp/3dprostate/seg_jobs")
    parser.add_argument("--model",     default="prostate_mri_anatomy")
    parser.add_argument("--port",      type=int, default=8765)
    args = parser.parse_args()

    job_dir    = Path(args.jobs_base) / args.job_id
    input_dir  = job_dir / "input"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Job    : {args.job_id}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Model  : {args.model}")
    print(f"  Port   : {args.port}")
    print(f"{'='*55}\n", flush=True)

    try:
        # Step 1: DICOM → NIfTI
        write_status(job_dir, "running|15|Converting DICOM to NIfTI…")
        nifti_path = job_dir / "mri.nii.gz"
        convert_dicom_to_nifti(input_dir, nifti_path)

        # Step 2: MONAILabel inference
        write_status(job_dir, "running|30|Starting MONAILabel server…")
        seg_path = run_monailabel_inference(
            nifti_path=nifti_path,
            output_dir=output_dir,
            model=args.model,
            port=args.port,
            job_dir=job_dir,
        )

        # Step 3: STL generation
        write_status(job_dir, "running|80|Generating 3D meshes…")
        labels = seg_to_stl_files(seg_path, output_dir)

        # Step 4: Write result
        result = {"job_id": args.job_id, "labels": labels}
        (job_dir / "result.json").write_text(json.dumps(result, indent=2))
        write_status(job_dir, "done")

        print(f"\n[done] {len(labels)} structure(s)  →  {output_dir}", flush=True)

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        write_status(job_dir, f"error|{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
