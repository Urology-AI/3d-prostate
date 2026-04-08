"""
HPC Client
===========
Manages SSH connection to Minerva HPC:
- Upload DICOM/NIfTI via SFTP
- Submit LSF jobs
- Poll job status
- Download results
"""

import os
import time
import paramiko
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class HPCClient:
    def __init__(self):
        self.host    = os.getenv("HPC_HOST", "login.hpc.mssm.edu")
        self.user    = os.getenv("HPC_USER", "dixita06")
        self.key     = os.path.expanduser(os.getenv("HPC_SSH_KEY", "~/.ssh/id_rsa"))
        self.project = os.getenv("HPC_PROJECT_DIR", "/sc/arion/projects/video_rarp")
        self.sif     = os.getenv("HPC_SINGULARITY_IMG",
                                  "/sc/arion/projects/video_rarp/monai.sif")
        self.work    = os.getenv("HPC_WORK_DIR",
                                  "/sc/arion/projects/video_rarp/prostate3d_jobs")
        self.model   = os.getenv("MONAI_MODEL", "prostate_mri_anatomy")

        self.ssh  = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.host, username=self.user,
                         key_filename=self.key, timeout=30)
        self.sftp = self.ssh.open_sftp()

    def close(self):
        try:
            self.sftp.close()
            self.ssh.close()
        except Exception:
            pass

    def test_connection(self):
        _, stdout, _ = self.ssh.exec_command("hostname")
        return stdout.read().decode().strip() != ""

    def _run(self, cmd):
        """Run command on HPC, return (stdout, stderr, exit_code)"""
        _, stdout, stderr = self.ssh.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        code = stdout.channel.recv_exit_status()
        return out, err, code

    def _mkdir(self, path):
        try:
            self.sftp.mkdir(path)
        except IOError:
            pass  # already exists

    # ── Upload ──────────────────────────────────────────────

    def upload_input(self, job_id, nifti_path):
        """Upload NIfTI to HPC job directory"""
        hpc_job_dir = f"{self.work}/{job_id}"
        hpc_input   = f"{hpc_job_dir}/input"
        hpc_output  = f"{hpc_job_dir}/output"

        self._run(f"mkdir -p {hpc_input} {hpc_output}")

        remote_nifti = f"{hpc_input}/mri.nii.gz"
        self.sftp.put(nifti_path, remote_nifti)
        print(f"[HPC] Uploaded {nifti_path} → {remote_nifti}")
        return hpc_job_dir

    # ── Job submission ──────────────────────────────────────

    def submit_job(self, job_id, hpc_job_dir):
        """Submit LSF job that runs MONAILabel inference in Singularity"""
        queue    = os.getenv("LSF_QUEUE",   "gpu")
        project  = os.getenv("LSF_PROJECT", "acc_video_rarp")
        mem      = os.getenv("LSF_MEMORY",  "32000")
        walltime = os.getenv("LSF_WALLTIME","02:00")
        gpus     = os.getenv("LSF_GPU_COUNT","1")
        cpus     = os.getenv("LSF_CPUS",    "4")
        model    = self.model
        sif      = self.sif

        # Write LSF script to HPC
        script = f"""#!/bin/bash
#BSUB -J prostate3d_{job_id}
#BSUB -P {project}
#BSUB -q {queue}
#BSUB -n {cpus}
#BSUB -R "rusage[mem={mem},ngpus_excl_p={gpus}]"
#BSUB -W {walltime}
#BSUB -o {hpc_job_dir}/job_%J.log
#BSUB -e {hpc_job_dir}/job_%J.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

INPUT_DIR={hpc_job_dir}/input
OUTPUT_DIR={hpc_job_dir}/output
STUDIES_DIR={hpc_job_dir}/studies

mkdir -p $STUDIES_DIR
cp $INPUT_DIR/mri.nii.gz $STUDIES_DIR/mri.nii.gz

# Copy patched config
CONFIG_PATH={self.project}/config.py
if [ -f "$CONFIG_PATH" ]; then
    BIND_CONFIG="--bind $CONFIG_PATH:/usr/local/lib/python3.10/dist-packages/monailabel/config.py"
else
    BIND_CONFIG=""
fi

# Start MONAILabel server in background
singularity exec \\
    --nv \\
    $BIND_CONFIG \\
    --bind $STUDIES_DIR:/data/studies \\
    --bind {hpc_job_dir}/model:/app/radiology/model \\
    --bind {hpc_job_dir}/lib:/app/radiology/lib \\
    --bind {hpc_job_dir}/logs:/app/radiology/logs \\
    --bind {hpc_job_dir}/bin:/app/radiology/bin \\
    {sif} \\
    monailabel start_server \\
    --app /app/radiology \\
    --studies /data/studies \\
    --conf models {model} \\
    --host 0.0.0.0 \\
    --port 8000 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/info > /dev/null 2>&1; then
        echo "Server ready after ${{i}}s"
        break
    fi
    sleep 2
done

# Validate model availability from MONAILabel /info response.
curl -sf http://localhost:8000/info -o $OUTPUT_DIR/info.json
if ! singularity exec {sif} python3 -c "
import json, sys
model = '{model}'
with open('{hpc_job_dir}/output/info.json', 'r') as f:
    info = json.load(f)

def contains_model(obj):
    if isinstance(obj, dict):
        return any(k == model or contains_model(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(contains_model(v) for v in obj)
    if isinstance(obj, str):
        return obj == model
    return False

if not contains_model(info):
    print(f'Model not found in /info: {model}')
    sys.exit(1)
print(f'Model available: {model}')
"; then
    echo "Model validation failed"
    kill $SERVER_PID 2>/dev/null
    exit 2
fi

# Run inference
echo "Running inference..."
curl -s -X POST http://localhost:8000/infer/{model} \\
    -F "file=@/data/studies/mri.nii.gz" \\
    -F "output=image" \\
    -o $OUTPUT_DIR/raw_response.bin \\
    --max-time 600

echo "Inference complete"

# Parse multipart response with Python
singularity exec {sif} python3 -c "
import os, email
path = '{hpc_job_dir}/output/raw_response.bin'
out  = '{hpc_job_dir}/output/segmentation.nii.gz'
with open(path, 'rb') as f:
    data = f.read()
# Try multipart parse
try:
    ct_line = b'Content-Type: multipart/form-data; boundary=xx\\r\\n\\r\\n'
    # Find gzip magic bytes
    idx = data.find(b'\\x1f\\x8b')
    if idx >= 0:
        end = len(data)
        # Find boundary after content
        seg_data = data[idx:]
        # Trim trailing boundary
        for i in range(len(seg_data)-1, 0, -1):
            if seg_data[i] == 0x8b and seg_data[i-1] == 0x1f:
                break
            if seg_data[i:i+2] == b'\\r\\n':
                seg_data = seg_data[:i]
                break
        with open(out, 'wb') as f:
            f.write(seg_data)
        print('Saved segmentation:', os.path.getsize(out), 'bytes')
    else:
        # Save raw
        with open(out, 'wb') as f:
            f.write(data)
        print('Saved raw response:', len(data), 'bytes')
except Exception as e:
    print('Parse error:', e)
    with open(out, 'wb') as f:
        f.write(data)
"

# Stop server
kill $SERVER_PID 2>/dev/null

echo "Job complete: $(date)"
ls -lh $OUTPUT_DIR/
"""

        # Create required dirs
        for d in ["model", "lib", "logs", "bin"]:
            self._run(f"mkdir -p {hpc_job_dir}/{d}")

        # Write script
        script_path = f"{hpc_job_dir}/run.sh"
        with self.sftp.open(script_path, "w") as f:
            f.write(script)
        self._run(f"chmod +x {script_path}")

        # Submit
        out, err, code = self._run(f"bsub < {script_path}")
        print(f"[HPC] bsub output: {out}")
        if code != 0:
            raise RuntimeError(f"bsub failed: {err}")

        # Parse job ID from "Job <12345> is submitted..."
        import re
        m = re.search(r"<(\d+)>", out)
        if not m:
            raise RuntimeError(f"Could not parse LSF job ID from: {out}")

        lsf_job_id = m.group(1)
        print(f"[HPC] LSF job ID: {lsf_job_id}")
        return lsf_job_id

    # ── Polling ─────────────────────────────────────────────

    def wait_for_job(self, lsf_job_id, on_progress=None, timeout=1800, poll_interval=15):
        """Poll bjobs until job finishes"""
        start = time.time()
        last_status = None

        while time.time() - start < timeout:
            out, _, _ = self._run(f"bjobs {lsf_job_id} 2>/dev/null | tail -1")
            parts = out.split()

            if len(parts) >= 3:
                status = parts[2]  # RUN, PEND, DONE, EXIT
            else:
                status = "UNKNOWN"

            elapsed = int(time.time() - start)
            pct = min(0.95, elapsed / timeout)

            if status != last_status:
                print(f"[HPC] Job {lsf_job_id} status: {status}")
                last_status = status

            if status == "DONE":
                if on_progress:
                    on_progress(1.0, "Segmentation complete on HPC")
                return True
            elif status == "EXIT":
                raise RuntimeError(f"LSF job {lsf_job_id} failed (EXIT status)")
            elif status in ("RUN", "PEND"):
                msg = f"HPC job {status.lower()}ning... ({elapsed}s)"
                if on_progress:
                    on_progress(pct, msg)
            else:
                # Job may have finished and left bjobs
                # Check if output exists
                out2, _, _ = self._run(
                    f"test -f {self._get_job_dir(lsf_job_id)}/output/segmentation.nii.gz "
                    f"&& echo exists || echo missing"
                )
                if "exists" in out2:
                    return True

            time.sleep(poll_interval)

        raise TimeoutError(f"HPC job {lsf_job_id} timed out after {timeout}s")

    def _get_job_dir(self, lsf_job_id):
        # We don't know job_id from lsf_job_id directly in this method
        # This is a fallback — in practice the caller passes job_id
        return self.work

    # ── Download ────────────────────────────────────────────

    def download_output(self, job_id, hpc_job_dir, local_seg_path):
        """Download segmentation.nii.gz from HPC"""
        remote = f"{hpc_job_dir}/output/segmentation.nii.gz"

        # Check it exists
        out, _, _ = self._run(f"test -f {remote} && echo exists || echo missing")
        if "missing" in out:
            # Try raw response
            remote = f"{hpc_job_dir}/output/raw_response.bin"
            out2, _, _ = self._run(f"test -f {remote} && echo exists || echo missing")
            if "missing" in out2:
                raise RuntimeError(f"No segmentation output found at {hpc_job_dir}/output/")

        self.sftp.get(remote, local_seg_path)
        size = os.path.getsize(local_seg_path)
        print(f"[HPC] Downloaded {remote} → {local_seg_path} ({size//1024}KB)")

        if size < 1000:
            raise RuntimeError(f"Downloaded file too small ({size} bytes) — segmentation likely failed")
