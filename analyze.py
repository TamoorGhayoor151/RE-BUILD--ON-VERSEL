"""
Vercel Python Serverless Function
----------------------------------
Endpoint: POST /api/analyze
Body: {"description": "18 tons of mixed C&D debris..."}

Wraps the existing 4-agent LangGraph pipeline (graph.run_pipeline) so it
can run as a stateless serverless function instead of inside Streamlit.

NOTE: Vercel's default Hobby-plan function timeout is 10s. This pipeline
makes 3 sequential Groq LLM calls (classification, recoverability, plan)
plus 1 deterministic calculation. Groq is fast, but if you hit timeouts,
upgrade to a Pro plan (60s limit) or set `maxDuration` in vercel.json.
"""

import json
import os
from http.server import BaseHTTPRequestHandler
from graph import run_pipeline


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body) if body else {}
            description = payload.get("description", "").strip()

            if not description:
                self._send_json(400, {"error": "Missing 'description' in request body."})
                return

            result = run_pipeline(description)
            self._send_json(200, result)

        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        # Diagnostic health check: confirms whether Vercel is passing the
        # GROQ_API_KEY env var to this function, WITHOUT exposing the key
        # itself (only its presence and length are shown).
        key = os.environ.get("GROQ_API_KEY")
        self._send_json(200, {
            "status": "ok",
            "message": "POST a waste description to this endpoint.",
            "groq_api_key_present": bool(key),
            "groq_api_key_length": len(key) if key else 0,
        })

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
