"""
GPU Worker — Pod HTTP Server

Minimal Flask server that exposes the same handler as the Runpod
Serverless worker, but via direct HTTP. Used when running as a
GPU Pod instead of Serverless.

Endpoints:
    POST /run   — Execute an operation (same payload as Serverless)
    GET  /health — Quick health check

The handler logic is identical — only the transport layer changes.
"""

import os

# Must be set BEFORE importing handler to prevent runpod.serverless.start()
os.environ["POD_MODE"] = "1"

from flask import Flask, request, jsonify

from handler import handler as worker_handler

app = Flask(__name__)


@app.route("/run", methods=["POST"])
def run():
    """
    Execute a GPU operation.

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


@app.route("/health", methods=["GET"])
def health():
    """Quick health check — runs the health operation."""
    event = {"input": {"operation": "health"}}
    try:
        result = worker_handler(event)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Health check failed: {str(e)[:500]}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("POD_SERVER_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
