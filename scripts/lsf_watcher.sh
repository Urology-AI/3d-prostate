#!/bin/bash
# ============================================================
#  lsf_watcher.sh — run on the HOST (outside the container)
#
#  Polls seg_jobs/*/submit.pending and submits them via bsub.
#  The Flask app inside the container can't call bsub directly
#  (LSF needs CentOS 7 Perl/eauth, container is Debian).
#
#  Usage — run this BEFORE starting the container:
#    bash /sc/arion/projects/video_rarp/3dprostate/lsf_watcher.sh &
#
#  To stop:
#    kill $(cat /sc/arion/projects/video_rarp/3dprostate/watcher.pid)
# ============================================================

JOBS_DIR="/sc/arion/projects/video_rarp/3dprostate/seg_jobs"
PID_FILE="/sc/arion/projects/video_rarp/3dprostate/watcher.pid"

echo $$ > "${PID_FILE}"
echo "==> LSF watcher started (PID: $$)"
echo "    Watching: ${JOBS_DIR}"
echo "    Stop with: kill \$(cat ${PID_FILE})"
echo ""

mkdir -p "${JOBS_DIR}"

while true; do
    for pending_file in "${JOBS_DIR}"/*/submit.pending; do
        [ -f "$pending_file" ] || continue

        job_dir="$(dirname "$pending_file")"
        job_id="$(basename "$job_dir")"
        job_sh="${job_dir}/job.sh"
        lsf_id_file="${job_dir}/lsf_job_id.txt"

        [ -f "$job_sh" ]         || continue
        [ -f "$lsf_id_file" ]    && continue  # already submitted

        # Rename to .submitting to prevent double-submission
        mv "$pending_file" "${job_dir}/submit.submitting" 2>/dev/null || continue

        echo "[$(date +%H:%M:%S)] Submitting job ${job_id}..."
        lsf_output=$(bsub < "$job_sh" 2>&1)
        lsf_id=$(echo "$lsf_output" | grep -oP '(?<=<)\d+(?=>)' | head -1)

        if [ -n "$lsf_id" ]; then
            echo "$lsf_id" > "$lsf_id_file"
            echo "[$(date +%H:%M:%S)] Job ${job_id} → LSF ${lsf_id}"
        else
            echo "ERROR:bsub failed: $lsf_output" > "$lsf_id_file"
            echo "error|bsub failed: $lsf_output" > "${job_dir}/status.txt"
            echo "[$(date +%H:%M:%S)] Job ${job_id} FAILED: $lsf_output"
        fi

        rm -f "${job_dir}/submit.submitting"
    done
    sleep 2
done
