"""
GPU Worker — Pod HTTP Server

Minimal Flask server that exposes the same handler as the Runpod
Serverless worker, but via direct HTTP. Used when running as a
GPU Pod instead of Serverless.

Endpoints:
    POST /run        — Execute an operation synchronously (same payload as Serverless)
    POST /run-async  — Execute an operation asynchronously (returns job_id for polling)
    GET  /status/<id> — Poll async job status (pending/running/completed/failed)
    POST /cancel/<id> — Cancel a running async job
    GET  /health     — Quick health check

The handler logic is identical — only the transport layer changes.
"""

import os
import time
import uuid
import threading

# Must be set BEFORE importing handler to prevent runpod.serverless.start()
os.environ["POD_MODE"] = "1"

from flask import Flask, request, jsonify

from handler import handler as worker_handler

app = Flask(__name__)
app.url_map.strict_slashes = False


# ── Async Job Store ──────────────────────────────────────────────────
# In-memory store for async jobs. Each entry:
#   {
#     "status": "pending" | "running" | "completed" | "failed" | "cancelled",
#     "result": <dict or None>,
#     "error": <str or None>,
#     "created_at": <float>,
#     "started_at": <float or None>,
#     "completed_at": <float or None>,
#     "cancelled": <threading.Event>,
#   }

_async_jobs: dict = {}
_jobs_lock = threading.Lock()

# Max completed jobs to keep in memory (auto-cleanup oldest)
MAX_COMPLETED_JOBS = 100


def _cleanup_old_jobs() -> None:
    """Remove oldest completed/failed/cancelled jobs if over limit."""
    finished = [
        (jid, j) for jid, j in _async_jobs.items()
        if j["status"] in ("completed", "failed", "cancelled")
    ]
    if len(finished) <= MAX_COMPLETED_JOBS:
        return
    # Sort by completed_at, remove oldest
    finished.sort(key=lambda x: x[1].get("completed_at", 0))
    to_remove = len(finished) - MAX_COMPLETED_JOBS
    for jid, _ in finished[:to_remove]:
        _async_jobs.pop(jid, None)


def _run_job(job_id: str, event: dict) -> None:
    """Execute a job in a background thread."""
    with _jobs_lock:
        job = _async_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = time.time()

    try:
        # Check cancellation before starting
        if job["cancelled"].is_set():
            with _jobs_lock:
                job["status"] = "cancelled"
                job["completed_at"] = time.time()
            return

        result = worker_handler(event)

        # Check cancellation after completion
        if job["cancelled"].is_set():
            with _jobs_lock:
                job["status"] = "cancelled"
                job["completed_at"] = time.time()
            return

        with _jobs_lock:
            if isinstance(result, dict) and result.get("error"):
                job["status"] = "failed"
                job["error"] = result["error"]
            else:
                job["status"] = "completed"
                job["result"] = result
            job["completed_at"] = time.time()
            _cleanup_old_jobs()

    except Exception as e:
        with _jobs_lock:
            job["status"] = "failed"
            job["error"] = f"Server error: {str(e)[:500]}"
            job["completed_at"] = time.time()


# ── Sync Endpoint ────────────────────────────────────────────────────

@app.route("/run", methods=["POST"])
def run():
    """
    Execute a GPU operation synchronously.

    Expects JSON body with same structure as Runpod Serverless input:
    {"input": {"operation": "...", ...}}

    Returns the handler output directly (no Runpod wrapper).
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Build event in Runpod format
    event = {"input": data.get("input", data)}

    try:
        result = worker_handler(event)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)[:500]}"}), 500


# ── Async Endpoints ──────────────────────────────────────────────────

@app.route("/run-async", methods=["POST"])
def run_async():
    """
    Execute a GPU operation asynchronously.

    Accepts same payload as /run. Returns immediately with a job_id
    that can be polled via GET /status/<job_id>.

    Returns:
        {"job_id": "<uuid>"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    event = {"input": data.get("input", data)}
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _async_jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "cancelled": threading.Event(),
        }

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, event),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    """
    Poll async job status.

    Returns:
        {
            "status": "pending" | "running" | "completed" | "failed" | "cancelled",
            "result": <dict or null>,
            "error": <str or null>,
            "elapsed_seconds": <float>
        }
    """
    with _jobs_lock:
        job = _async_jobs.get(job_id)

    if not job:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404

    elapsed = 0.0
    if job["started_at"]:
        end = job["completed_at"] or time.time()
        elapsed = round(end - job["started_at"], 2)

    response = {
        "status": job["status"],
        "elapsed_seconds": elapsed,
    }

    if job["status"] == "completed":
        response["result"] = job["result"]
    elif job["status"] == "failed":
        response["error"] = job["error"]

    return jsonify(response)


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id: str):
    """
    Cancel a running or pending async job.

    Sets the cancelled flag. The job thread checks this flag
    and stops processing. Note: if the worker handler is already
    mid-execution (e.g. Ollama call), it will complete the current
    item before checking cancellation.

    Returns:
        {"cancelled": true/false}
    """
    with _jobs_lock:
        job = _async_jobs.get(job_id)

    if not job:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404

    if job["status"] in ("completed", "failed", "cancelled"):
        return jsonify({"cancelled": False, "reason": f"Job already {job['status']}"})

    job["cancelled"].set()

    # Give the thread a moment to notice
    time.sleep(0.1)

    with _jobs_lock:
        if job["status"] not in ("completed", "failed"):
            job["status"] = "cancelled"
            job["completed_at"] = time.time()

    return jsonify({"cancelled": True})


# ── Health & Root ────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Quick health check — runs the health operation."""
    event = {"input": {"operation": "health"}}
    try:
        result = worker_handler(event)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Health check failed: {str(e)[:500]}"}), 500


@app.route("/", methods=["GET"])
def index():
    """Root endpoint — confirms server is running."""
    return jsonify({
        "status": "ok",
        "service": "gpu-worker-pod",
        "endpoints": ["/run", "/run-async", "/status/<job_id>", "/cancel/<job_id>", "/health"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("POD_SERVER_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
