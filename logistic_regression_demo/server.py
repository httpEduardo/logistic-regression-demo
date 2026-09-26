import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import predict, train

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"

MODEL = {"weights": {}, "bias": 0.0, "features": [], "label": "label"}


def _safe_json(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_POST(self):
        global MODEL
        if self.path == "/api/train":
            payload = self._read_json()
            rows = _safe_json(payload.get("rows", []), [])
            label = payload.get("label", "label")
            epochs = int(payload.get("epochs", 400))
            lr = float(payload.get("lr", 0.1))
            result = train(rows, label, epochs=epochs, lr=lr)
            MODEL = {
                "weights": result["weights"],
                "bias": result["bias"],
                "features": result["features"],
                "label": label,
            }
            self._send_json(result)
            return
        if self.path == "/api/predict":
            payload = self._read_json()
            row = _safe_json(payload.get("row", {}), {})
            prob = predict(row, MODEL["weights"], MODEL["bias"], MODEL["features"])
            self._send_json({"probability": round(prob, 4), "label": 1 if prob >= 0.5 else 0})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        super().do_GET()


def run(host="127.0.0.1", port=5173):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Logistic Regression Demo running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Logistic Regression Demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()

    run(host=args.host, port=args.port)
