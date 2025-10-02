"""Flask entrypoint for the waterproject unification service.

This module exposes a small API around :mod:`ai_unify_48` so that Render can
start the service with ``gunicorn main:app``.  The previous deployment failed
because ``main.py`` still contained merge-conflict markers.  The implementation
below provides a clean Flask application with a couple of convenience
endpoints:

* ``GET /`` – basic service description and the most recent run status.
* ``GET /status`` – detailed information about generated output artefacts.
* ``POST /unify`` – execute the unification pipeline (optionally forcing a rerun).
* ``GET /download/<name>`` – download generated files by logical name.

The heavy lifting continues to live inside :func:`ai_unify_48.main`.  Requests
serialize calls to that function via a threading lock to avoid concurrent runs
clobbering each other's output, and the stdout/stderr streams are captured so
callers receive useful diagnostics.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, Response, abort, jsonify, request, send_file

import ai_unify_48

app = Flask(__name__)

# Thread-safety primitives -------------------------------------------------
_PIPELINE_LOCK = threading.Lock()
_LAST_RESULT: Dict[str, Optional[object]] = {
    "timestamp": None,
    "duration_sec": None,
    "success": False,
    "error": None,
    "stdout": "",
    "stderr": "",
}

_OUTPUT_MAP = {
    "unified_csv": Path(ai_unify_48.OUT_UNIFIED),
    "mapping_json": Path(ai_unify_48.OUT_MAP),
    "conflicts_csv": Path(ai_unify_48.OUT_CONFLICTS),
    "ai_validation_md": Path(ai_unify_48.OUT_AI_MD),
}


def _path_details(path: Path) -> Optional[Dict[str, object]]:
    """Return metadata for *path* if it exists, otherwise ``None``."""

    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def _collect_outputs() -> Dict[str, Optional[Dict[str, object]]]:
    return {name: _path_details(path) for name, path in _OUTPUT_MAP.items()}


def _run_pipeline(force: bool = False) -> Dict[str, object]:
    """Execute ``ai_unify_48.main`` and capture stdout/stderr.

    If ``force`` is false and the unified CSV already exists, the previous
    result metadata is returned without re-running the pipeline.
    """

    with _PIPELINE_LOCK:
        if not force and Path(ai_unify_48.OUT_UNIFIED).exists() and _LAST_RESULT["timestamp"]:
            return {
                "reused": True,
                **_LAST_RESULT,
                "outputs": _collect_outputs(),
            }

        start = time.time()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result: Dict[str, object] = {}

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                ai_unify_48.main()
        except Exception as exc:  # pragma: no cover - defensive, kept for observability
            result.update(
                success=False,
                error=str(exc),
            )
        else:
            result.update(success=True, error=None)
        finally:
            duration = time.time() - start
            result.update(
                duration_sec=round(duration, 3),
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
            )
            _LAST_RESULT.update(result)

        result.update(reused=False, outputs=_collect_outputs())
        return result


@app.get("/")
def index() -> Response:
    """Basic health endpoint with the latest pipeline status."""

    return jsonify(
        {
            "service": "waterproject-unifier",
            "description": "Expose ai_unify_48 via a minimal Flask API.",
            "status": _LAST_RESULT,
            "outputs": _collect_outputs(),
            "endpoints": {
                "status": "/status",
                "unify": "/unify",
                "download": "/download/<name>",
            },
        }
    )


@app.get("/status")
def status() -> Response:
    """Return details about previously generated artefacts."""

    payload = {
        "last_run": _LAST_RESULT,
        "outputs": _collect_outputs(),
    }
    return jsonify(payload)


@app.post("/unify")
def unify() -> Response:
    """Trigger the unification pipeline.

    Accepts optional JSON payload ``{"force": true}`` to rerun even if the
    unified CSV already exists.
    """

    data = request.get_json(silent=True) or {}
    force = bool(data.get("force", False))
    result = _run_pipeline(force=force)
    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


@app.get("/download/<string:name>")
def download(name: str):
    """Send one of the generated files to the client."""

    if name not in _OUTPUT_MAP:
        abort(404, description=f"Unknown artefact '{name}'.")
    path = _OUTPUT_MAP[name]
    if not path.exists():
        abort(404, description=f"Artefact '{name}' not found on disk.")
    return send_file(path, as_attachment=True)


@app.get("/raw-last-result")
def raw_last_result() -> Response:
    """Return the last pipeline result without Flask's dict-to-JSON conversion.

    This is mostly useful for debugging in environments where JSON pretty
    printing is not desired; it mirrors the legacy behaviour of returning a raw
    JSON string.
    """

    return Response(json.dumps(_LAST_RESULT, default=str), mimetype="application/json")


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
