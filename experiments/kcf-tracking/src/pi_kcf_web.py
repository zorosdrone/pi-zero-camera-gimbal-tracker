"""ブラウザー画面とHTTP APIを担当するモジュール。"""

from __future__ import annotations

import json
import logging
import socketserver
from http import server
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


MAX_REQUEST_BYTES = 4096
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _load_asset(name: str) -> bytes:
    return (WEB_ROOT / name).read_bytes()


def _render_asset(name: str, width: int, height: int) -> bytes:
    text = _load_asset(name).decode("utf-8")
    return (
        text.replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(height))
        .encode("utf-8")
    )


class TrackingHandler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "PiCameraKcfTracking/0.1"

    def _send_bytes(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        self._send_bytes(
            status,
            "application/json; charset=utf-8",
            (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"),
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", self.server.index_page)
            return
        if path == "/style.css":
            self._send_bytes(200, "text/css; charset=utf-8", self.server.style_sheet)
            return
        if path == "/app.js":
            self._send_bytes(
                200,
                "application/javascript; charset=utf-8",
                self.server.app_script,
            )
            return
        if path == "/health":
            self._send_bytes(200, "text/plain; charset=utf-8", b"ok\n")
            return
        if path == "/api/status":
            self._send_json(200, self.server.status())
            return
        if path == "/target.jpg":
            _, frame = self.server.target_output.snapshot()
            if frame is None:
                self.send_error(404, "追跡対象が未選択です")
                return
            self._send_bytes(200, "image/jpeg", frame)
            return
        if path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        sequence = 0
        try:
            while True:
                sequence, frame, closed = self.server.output.wait_next(sequence)
                if closed or frame is None:
                    break
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            logging.info("ストリームクライアントが切断しました")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("リクエストが大きすぎます")
            body = self.rfile.read(length) if length else b"{}"
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSONオブジェクトを指定してください")

            if path == "/api/roi":
                self.server.target_output.clear()
                self.server.tracking_state.request_roi(data)
            elif path == "/api/stop":
                self.server.tracking_state.stop()
                self.server.controller.stop_motion()
                self.server.target_output.clear()
            elif path == "/api/center":
                self.server.tracking_state.stop()
                self.server.controller.center()
                self.server.target_output.clear()
            elif path == "/api/release":
                self.server.tracking_state.stop()
                self.server.controller.release()
                self.server.target_output.clear()
            else:
                self.send_error(404)
                return
            self._send_json(200, self.server.status())
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            logging.exception("HTTP処理に失敗")
            self._send_json(500, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        path = urlparse(self.path).path
        if path in {"/stream.mjpg", "/api/status", "/target.jpg"}:
            return
        logging.info("HTTP %s - " + format, self.address_string(), *args)


class TrackingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """静的Web画面、MJPEG、追跡操作APIを同じポートで提供する。"""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, output, target_output, tracking_state, controller, width, height):
        super().__init__(address, TrackingHandler)
        self.output = output
        self.target_output = target_output
        self.tracking_state = tracking_state
        self.controller = controller
        self.stream_width = width
        self.stream_height = height
        self.index_page = _render_asset("index.html", width, height)
        self.style_sheet = _render_asset("style.css", width, height)
        self.app_script = _render_asset("app.js", width, height)

    def status(self) -> dict[str, Any]:
        result = self.tracking_state.status()
        result.update(self.controller.status())
        result["target_sequence"] = self.target_output.snapshot()[0]
        return result
