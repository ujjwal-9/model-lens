"""Local chat UI for poking at steered Qwen3.5-4B (MLX).

The model loads once at startup. Every question is answered from scratch as a
single-turn prompt (no history), steered with the specs given in the UI and
optionally next to an unsteered baseline. Tokens stream back as NDJSON.

    .venv/bin/python chat_server.py                # http://127.0.0.1:8000
    .venv/bin/python chat_server.py --port 8080 --spec L14:N8949:+1

The server is single-threaded on purpose: MLX work stays on the thread that
loaded the model, and requests are naturally serialized.
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from mlx_lm import load, stream_generate

import steer_mlx as S

PAGE = Path(__file__).with_name("chat.html")


def answer(model, tok, layers, req, emit):
    specs = [S.parse_spec(s) for s in re.split(r"[,\s]+", req.get("specs", "").strip()) if s]
    input_ids = S.encode(tok, req["prompt"], bool(req.get("think")))
    rows = S.plan(model, layers, input_ids, specs) if specs else []
    emit(type="plan", rows=rows, prompt_tokens=len(input_ids))

    runs = (["steered"] if specs else []) + (["baseline"] if req.get("baseline", True) or not specs else [])
    for run in runs:
        emit(type="start", run=run)
        resp = None
        with S.hooks(*(S.steering_installs(layers, rows) if run == "steered" else [])):
            for resp in stream_generate(model, tok, input_ids, max_tokens=int(req.get("max_tokens", 300))):
                emit(type="token", run=run, text=resp.text)
        emit(type="done", run=run, tps=resp.generation_tps if resp else 0,
             tokens=resp.generation_tokens if resp else 0, finish=resp.finish_reason if resp else None)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())  # re-read so edits show on refresh
        elif self.path == "/api/config":
            cfg = dict(model=self.server.model_name, specs=", ".join(self.server.default_specs),
                       layers=len(self.server.layers), neurons=self.server.layers[0].mlp.down_proj.weight.shape[1])
            self._send(200, "application/json", json.dumps(cfg).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send_error(404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(**event):
            self.wfile.write((json.dumps(event) + "\n").encode())

        try:
            answer(self.server.model, self.server.tok, self.server.layers, req, emit)
        except (BrokenPipeError, ConnectionResetError):
            print("  client disconnected, generation stopped")
        except Exception as e:
            try:
                emit(type="error", message=f"{type(e).__name__}: {e}")
            except OSError:
                pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if self.command == "POST":
            print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--spec", action="append", help="default steering specs shown in the UI (default: %s)" % " ".join(S.DEFAULT_SPECS))
    ap.add_argument("--model", default=S.MODEL)
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    model, tok = load(args.model)
    server = HTTPServer((args.host, args.port), Handler)
    server.model, server.tok, server.model_name = model, tok, args.model
    server.layers = model.model.layers  # the real list; model.layers returns a copy
    server.default_specs = args.spec or S.DEFAULT_SPECS
    print(f"ready: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
