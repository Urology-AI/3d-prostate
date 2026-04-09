"""
Prostate3D Backend — Minerva Edition
======================================
Runs INSIDE Singularity on a Minerva compute node.
No SSH, no paramiko.  The /sc/arion filesystem is mounted directly
via   singularity run --bind /sc/arion:/sc/arion pipeline.sif

Jobs lifecycle
--------------
  POST /api/upload   → save DICOMs → bsub GPU job → return job_id
  GET  /api/status/<job_id>   → read <job_dir>/status.txt
  GET  /api/output/<job_id>/<file>  → serve STL / result files
"""

import json
import os
import re
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, template_folder="templates")
CORS(app)

# ── Config (all overridable via env vars) ────────────────────────────────────

PROJ_DIR     = os.getenv("PROJ_DIR",
                  "/sc/arion/projects/video_rarp/3dprostate")
JOBS_DIR    = Path(os.getenv("JOBS_DIR",    f"{PROJ_DIR}/seg_jobs"))
SIF_PATH    = os.getenv("SIF_PATH",         f"{PROJ_DIR}/pipeline.sif")
LSF_PROJECT  = os.getenv("LSF_PROJECT",  "acc_video_rarp")
LSF_QUEUE    = os.getenv("LSF_QUEUE",    "gpu")
LSF_WALLTIME = os.getenv("LSF_WALLTIME", "4:00")
LSF_MEMORY   = os.getenv("LSF_MEMORY",  "32000")
LSF_CPUS     = os.getenv("LSF_CPUS",    "4")
# Set MOCK_BSUB=true in docker-compose to run segmentation inline (no HPC needed)
MOCK_BSUB    = os.getenv("MOCK_BSUB", "false").lower() == "true"

# bsub is on the host, not in the container — find it
import shutil as _shutil
BSUB_PATH = os.getenv("BSUB_PATH") or _shutil.which("bsub") or next(
    (p for p in [
        "/hpc/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub",  # Minerva
        "/usr/lsf/bin/bsub",
        "/opt/lsf/bin/bsub",
        "/opt/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub",
        "/usr/lsf/10.1/linux3.10-glibc2.17-x86_64/bin/bsub",
    ] if os.path.exists(p)),
    "bsub"  # fall back and let it fail with a clear error
)

JOBS_DIR.mkdir(parents=True, exist_ok=True)


# ── Status file helpers ──────────────────────────────────────────────────────

def _write_status(job_dir: Path, text: str) -> None:
    (job_dir / "status.txt").write_text(text)


def _read_job_state(job_id: str) -> dict:
    job_dir     = JOBS_DIR / job_id
    status_file = job_dir / "status.txt"

    if not status_file.exists():
        return {"status": "unknown", "progress": 0, "message": "Job not found"}

    raw = status_file.read_text().strip()

    if raw == "pending":
        return {"status": "pending", "progress": 5, "message": "Queued for GPU…"}

    if raw.startswith("running|"):
        parts = raw.split("|", 2)
        pct   = int(parts[1]) if len(parts) > 1 else 30
        msg   = parts[2]      if len(parts) > 2 else "Processing…"
        return {"status": "running", "progress": pct, "message": msg}

    if raw == "done":
        result_path = job_dir / "result.json"
        result = (json.loads(result_path.read_text())
                  if result_path.exists() else {})
        result["job_id"] = job_id
        return {"status": "done", "progress": 100,
                "message": "Segmentation complete!", "result": result}

    if raw.startswith("error|"):
        return {"status": "error", "progress": 0, "message": raw[6:]}

    return {"status": "unknown", "progress": 0, "message": raw}


# ── Job submission ────────────────────────────────────────────────────────────

def _submit_bsub(job_id: str) -> str:
    """Write an LSF job script and signal the host-side watcher to submit it.

    bsub cannot run from inside the Singularity container because it needs
    the full CentOS 7 LSF environment (Perl XS modules, eauth, etc.) which
    is incompatible with the Debian container.

    Instead we write job.sh + submit.pending to shared storage.
    The host-side lsf_watcher.sh picks it up and runs bsub on the host.
    """
    import time
    job_dir = JOBS_DIR / job_id

    script = f"""#!/bin/bash
#BSUB -J prostate3d_{job_id}
#BSUB -P {LSF_PROJECT}
#BSUB -q {LSF_QUEUE}
#BSUB -n {LSF_CPUS}
#BSUB -W {LSF_WALLTIME}
#BSUB -R "rusage[mem={LSF_MEMORY}]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no"
#BSUB -L /bin/bash
#BSUB -o {job_dir}/job.log
#BSUB -e {job_dir}/job.err

echo "[$(date)] Job started on $(hostname)"
echo "[$(date)] GPU: $CUDA_VISIBLE_DEVICES"

ml singularity || true

export http_proxy=http://172.28.7.1:3128
export https_proxy=http://172.28.7.1:3128
export https_proxy=http://172.28.7.1:3128
export no_proxy=localhost,*.chimera.hpc.mssm.edu,172.28.0.0/16

singularity exec --nv --contain \\
    --bind /sc/arion/projects/video_rarp/3dprostate:/sc/arion/projects/video_rarp/3dprostate \\
    --bind {PROJ_DIR}/radiology_model:/app/radiology/model \\
    --bind {PROJ_DIR}/radiology_logs:/app/radiology/logs \\
    --bind /tmp:/tmp \\
    {SIF_PATH} \\
    python3 /app/segment_prostate.py \\
    --job-id  {job_id} \\
    --jobs-base {JOBS_DIR}

echo "[$(date)] Job finished (exit $?)"
"""

    (job_dir / "job.sh").write_text(script)
    (job_dir / "submit.pending").write_text("pending")
    print(f"[{job_id}] Wrote job.sh + submit.pending, waiting for watcher...")

    # Wait up to 30s for host watcher to submit the job
    lsf_id_file = job_dir / "lsf_job_id.txt"
    for _ in range(30):
        time.sleep(1)
        if lsf_id_file.exists():
            content = lsf_id_file.read_text().strip()
            if content.startswith("ERROR:"):
                raise RuntimeError(content[6:])
            print(f"[{job_id}] LSF job submitted: {content}")
            return content

    raise RuntimeError(
        "Timeout waiting for job submission. "
        "Is lsf_watcher.sh running on the host? "
        f"Run: bash {PROJ_DIR}/lsf_watcher.sh &"
    )


def _run_mock_seg(job_id: str) -> None:
    """Local-dev fallback: run segment_prostate.py directly (no bsub)."""
    import threading

    def _worker():
        cmd = [
            "python3", "/app/segment_prostate.py",
            "--job-id", job_id,
            "--jobs-base", str(JOBS_DIR),
        ]
        subprocess.run(cmd, check=False)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    print(f"[{job_id}] Mock segmentation started (MOCK_BSUB=true)")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    job_id    = str(uuid.uuid4())[:8]
    job_dir   = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    filenames = []
    for f in files:
        if not f.filename:
            continue
        safe = f.filename.replace("/", "_").replace("\\", "_")
        f.save(str(input_dir / safe))
        filenames.append(safe)
        saved += 1

    # Expand any zip archives in-place
    for zf_path in list(input_dir.glob("*.zip")):
        with zipfile.ZipFile(zf_path) as zf:
            zf.extractall(input_dir)
        zf_path.unlink()

    # Save session metadata for history
    import datetime
    file_hash = request.form.get("file_hash", "")
    meta = {
        "job_id":    job_id,
        "created":   datetime.datetime.now().isoformat(),
        "files":     filenames,
        "n_files":   saved,
        "file_hash": file_hash,
    }
    (job_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    _write_status(job_dir, "pending")

    try:
        if MOCK_BSUB:
            _run_mock_seg(job_id)
        else:
            lsf_id = _submit_bsub(job_id)
            (job_dir / "lsf_job_id.txt").write_text(lsf_id)
    except Exception as exc:
        _write_status(job_dir, f"error|{exc}")
        return jsonify({"error": str(exc)}), 500

    print(f"[{job_id}] Uploaded {saved} file(s), job submitted")
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        return jsonify({"status": "unknown"}), 400
    return jsonify(_read_job_state(job_id))


@app.route("/api/jobs")
def list_jobs():
    """Return all previous sessions sorted by newest first."""
    jobs = []
    for meta_path in sorted(JOBS_DIR.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text())
            job_id = meta["job_id"]
            state  = _read_job_state(job_id)
            jobs.append({
                "job_id":    job_id,
                "created":   meta.get("created", ""),
                "n_files":   meta.get("n_files", 0),
                "files":     meta.get("files", []),
                "file_hash": meta.get("file_hash", ""),
                "status":    state["status"],
                "message":   state.get("message", ""),
            })
        except Exception:
            continue
    return jsonify(jobs)


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    """Delete a job and all its files."""
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        return jsonify({"error": "invalid job_id"}), 400
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(str(job_dir))
    return jsonify({"deleted": job_id})


@app.route("/api/load/<job_id>")
def load_job(job_id):
    """Return full state for a previous session — skips re-running if already done."""
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        return jsonify({"error": "invalid job_id"}), 400
    state = _read_job_state(job_id)
    state["job_id"] = job_id
    return jsonify(state)


@app.route("/api/logs/<job_id>")
def job_logs(job_id):
    """Return combined log output for a job (job.log + job.err + monailabel_server.log)."""
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        return jsonify({"error": "invalid job_id"}), 400
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        return jsonify({"logs": ""}), 404

    MAX_BYTES = 150_000
    parts = []
    for fname in ("job.log", "job.err", "monailabel_server.log"):
        fpath = job_dir / fname
        if fpath.exists():
            try:
                content = fpath.read_text(errors="replace")
                if content.strip():
                    parts.append(f"=== {fname} ===\n{content.rstrip()}")
            except Exception:
                pass

    combined = "\n\n".join(parts)
    if len(combined) > MAX_BYTES:
        combined = "[…truncated — showing last 150 KB…]\n" + combined[-MAX_BYTES:]

    return jsonify({"logs": combined})


@app.route("/api/output/<job_id>/<filename>")
def serve_output(job_id, filename):
    if not re.fullmatch(r"[a-f0-9]{8}", job_id):
        return jsonify({"error": "invalid job_id"}), 400
    if not re.fullmatch(r"[A-Za-z0-9_\-\.]+", filename):
        return jsonify({"error": "invalid filename"}), 400
    return send_from_directory(str(JOBS_DIR / job_id / "output"), filename)


# ── Dev server entry-point ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Prostate3D — Minerva Edition")
    print(f"  Jobs dir : {JOBS_DIR}")
    print(f"  SIF path : {SIF_PATH}")
    print(f"  Mock bsub: {MOCK_BSUB}")
    print("  http://0.0.0.0:5000")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0",
            port=int(os.getenv("PORT", 5000)), threaded=True)
