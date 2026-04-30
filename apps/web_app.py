"""
SWE-Vision Web App — ChatGPT-style interface for the VLM Tool Call Agent.

Upload images, enter prompts, and watch the agent reason step-by-step
with real-time streaming of tool calls, code execution, and results.

Usage:
    python apps/web_app.py [--port 8080] [--host 0.0.0.0]

Then open http://localhost:8080 in your browser.
"""

import argparse
import asyncio
import datetime
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``import swe_vision`` works
# when running this script directly (e.g. ``python apps/web_app.py``).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from swe_vision.agent import VLMToolCallAgent
    from swe_vision.trajectory import TrajectoryRecorder
    AGENT_AVAILABLE = True
    AGENT_IMPORT_ERROR = ""
except ImportError as e:
    AGENT_AVAILABLE = False
    AGENT_IMPORT_ERROR = str(e)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSION_BASE = os.path.join(
    os.environ.get("VLM_WEB_SESSION_DIR", "/tmp"),
    "vlm_web_sessions",
)
os.makedirs(SESSION_BASE, exist_ok=True)
FRONTEND_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_WEB_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
DEFAULT_WEB_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
DEFAULT_WEB_MAX_ITERATIONS = int(os.environ.get("VLM_MAX_ITERATIONS", "30"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# In-memory session store  {session_id: {queue, thread, ...}}
sessions: dict = {}


# ═══════════════════════════════════════════════════════════════════
# Streaming Trajectory Recorder
# ═══════════════════════════════════════════════════════════════════

class StreamingTrajectoryRecorder(TrajectoryRecorder if AGENT_AVAILABLE else object):
    """Extends TrajectoryRecorder to push SSE events to a queue."""

    def __init__(self, save_dir, event_queue, session_id):
        super().__init__(save_dir)
        self._eq = event_queue
        self._sid = session_id

    def _emit(self, event):
        self._eq.put(event)

    def _img_url(self, rel_path):
        return f"/api/files/{self._sid}/trajectory/{rel_path}"

    # -- overrides -------------------------------------------------------

    def record_user_step(self, query, image_paths=None):
        super().record_user_step(query, image_paths)
        step = self.steps[-1]
        images = [self._img_url(p) for p in step.get("images", [])]
        self._emit({"type": "user_message", "data": {"text": query, "images": images}})

    def record_assistant_step(self, content_text, tool_calls=None, reasoning_details=None):
        super().record_assistant_step(content_text, tool_calls, reasoning_details)
        step = self.steps[-1]

        if reasoning_details:
            if isinstance(reasoning_details, str):
                rd = reasoning_details
            else:
                rd = json.dumps(reasoning_details, ensure_ascii=False, indent=2)
            self._emit({"type": "thinking", "data": {"content": rd}})

        if content_text:
            self._emit({"type": "assistant_text", "data": {"content": content_text}})

        for tc in (step.get("tool_calls") or []):
            args_str = tc.get("arguments", "{}")
            try:
                parsed = json.loads(args_str) if isinstance(args_str, str) else args_str
            except Exception:
                parsed = {}
            self._emit({
                "type": "tool_call",
                "data": {
                    "name": tc.get("name", ""),
                    "id": tc.get("id", ""),
                    "code": parsed.get("code", ""),
                    "answer": parsed.get("answer", ""),
                    "arguments": args_str,
                },
            })

    def record_tool_step(self, tool_call_id, tool_name, code, text_output, base64_images=None):
        super().record_tool_step(tool_call_id, tool_name, code, text_output, base64_images)
        step = self.steps[-1]
        images = [self._img_url(p) for p in step.get("images", [])]
        self._emit({
            "type": "tool_result",
            "data": {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "output": text_output,
                "images": images,
                "is_error": "[Error]" in (text_output or "") or "[Execution Error]" in (text_output or ""),
            },
        })

    def record_finish(self, answer):
        super().record_finish(answer)
        self._emit({"type": "finish", "data": {"answer": answer}})


# ═══════════════════════════════════════════════════════════════════
# Web VLM Agent (subclass)
# ═══════════════════════════════════════════════════════════════════

if AGENT_AVAILABLE:
    class WebVLMAgent(VLMToolCallAgent):
        """VLMToolCallAgent that streams steps to an SSE queue."""

        def __init__(self, event_queue, session_id, **kwargs):
            super().__init__(**kwargs)
            self._event_queue = event_queue
            self._session_id = session_id

        def _init_trajectory(self, query, image_paths):
            save_dir = os.path.join(SESSION_BASE, self._session_id, "trajectory")
            recorder = StreamingTrajectoryRecorder(save_dir, self._event_queue, self._session_id)
            recorder.set_metadata(
                model=self.model,
                start_time=TrajectoryRecorder._now_iso(),
                query=query,
                image_paths=image_paths or [],
                max_iterations=self.max_iterations,
                system_prompt=self.system_prompt,
            )
            return recorder


# ═══════════════════════════════════════════════════════════════════
# Agent Thread
# ═══════════════════════════════════════════════════════════════════

def run_agent_thread(event_queue, session_id, prompt, image_paths, config):
    """Run the VLM agent in a background thread with its own asyncio loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    agent = WebVLMAgent(
        event_queue=event_queue,
        session_id=session_id,
        model=config.get("model", DEFAULT_WEB_MODEL),
        api_key=config.get("api_key") or None,
        base_url=config.get("base_url") or None,
        max_iterations=config.get("max_iterations", DEFAULT_WEB_MAX_ITERATIONS),
        verbose=True,
        reasoning=config.get("reasoning", True),
    )

    try:
        event_queue.put({"type": "status", "data": {"message": "Starting agent and Docker kernel..."}})
        answer = loop.run_until_complete(agent.run(prompt, image_paths if image_paths else None))
    except Exception as e:
        import traceback
        event_queue.put({"type": "error", "data": {"message": f"{e}\n{traceback.format_exc()}"}})
    finally:
        try:
            loop.run_until_complete(agent.cleanup())
        except Exception:
            pass
        loop.close()
        event_queue.put(None)  # sentinel


# ═══════════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/web/<path:filename>")
def web_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/config")
def api_config():
    return jsonify({
        "agent_available": AGENT_AVAILABLE,
        "agent_error": AGENT_IMPORT_ERROR,
        "default_model": DEFAULT_WEB_MODEL,
        "default_base_url": DEFAULT_WEB_BASE_URL,
        "default_max_iterations": DEFAULT_WEB_MAX_ITERATIONS,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not AGENT_AVAILABLE:
        return jsonify({"error": f"Agent not available: {AGENT_IMPORT_ERROR}"}), 500

    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    model = request.form.get("model", "").strip() or DEFAULT_WEB_MODEL
    api_key = request.form.get("api_key", "").strip() or os.environ.get("OPENAI_API_KEY", "")
    base_url = request.form.get("base_url", "").strip() or DEFAULT_WEB_BASE_URL
    reasoning = request.form.get("reasoning", "true") == "true"
    max_iterations = int(request.form.get("max_iterations", str(DEFAULT_WEB_MAX_ITERATIONS)))

    session_id = uuid.uuid4().hex[:12]
    upload_dir = os.path.join(SESSION_BASE, session_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    image_paths = []
    for f in request.files.getlist("images"):
        if f.filename:
            safe = secure_filename(f.filename)
            path = os.path.join(upload_dir, safe)
            f.save(path)
            image_paths.append(path)

    eq = queue.Queue()
    sessions[session_id] = {"queue": eq}

    t = threading.Thread(
        target=run_agent_thread,
        args=(eq, session_id, prompt, image_paths, {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "reasoning": reasoning,
            "max_iterations": max_iterations,
        }),
        daemon=True,
    )
    t.start()
    sessions[session_id]["thread"] = t

    image_urls = [
        f"/api/files/{session_id}/uploads/{os.path.basename(p)}" for p in image_paths
    ]
    return jsonify({"session_id": session_id, "image_urls": image_urls})


@app.route("/api/stream/<session_id>")
def api_stream(session_id):
    if session_id not in sessions:
        abort(404)

    eq = sessions[session_id]["queue"]

    def generate():
        while True:
            try:
                event = eq.get(timeout=300)
                if event is None:
                    yield "data: {\"type\":\"done\"}\n\n"
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/files/<session_id>/<path:filepath>")
def serve_file(session_id, filepath):
    full = os.path.realpath(os.path.join(SESSION_BASE, session_id, filepath))
    base = os.path.realpath(os.path.join(SESSION_BASE, session_id))
    if not full.startswith(base):
        abort(403)
    if not os.path.isfile(full):
        abort(404)
    return send_file(full)




# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SWE-Vision Web App")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  SWE-Vision Web App")
    print(f"  Agent available: {AGENT_AVAILABLE}")
    if not AGENT_AVAILABLE:
        print(f"  Agent import error: {AGENT_IMPORT_ERROR}")
    print(f"  Sessions dir: {SESSION_BASE}")
    print(f"  Open: http://localhost:{args.port}\n")

    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
