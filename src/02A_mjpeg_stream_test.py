#!/usr/bin/env python3
"""Phase 2A: ハードウェアMJPEGをHTTPで配信するヘッドレス試験。

PCのブラウザーで http://<pi-host>:8000/ を開いて映像を確認する。

参考:
  https://github.com/raspberrypi/picamera2/blob/main/examples/mjpeg_server_2.py
  Picamera2 is licensed under the BSD 2-Clause License.
"""

import argparse
import io
import logging
import socketserver
import threading
import time
from http import server

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput


DEFAULT_SIZE = (640, 480)
DEFAULT_FPS = 10.0
DEFAULT_PORT = 8000


PAGE = """\
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi Camera Gimbal MJPEG</title>
</head>
<body>
<h1>Pi Camera Gimbal MJPEG</h1>
<p>Phase 2A: hardware MJPEG stream</p>
<img src="/stream.mjpg" width="{width}" height="{height}" alt="Pi camera stream">
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    """エンコーダーから届いた最新JPEGをHTTPクライアントへ渡す。"""

    def __init__(self) -> None:
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    """トップページ、ヘルスチェック、MJPEGストリームを提供する。"""

    protocol_version = "HTTP/1.0"
    server_version = "PiCameraGimbalMJPEG/0.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/":
            content = PAGE.format(width=self.server.stream_width, height=self.server.stream_height).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
            return

        if self.path == "/health":
            content = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while True:
                    with self.server.output.condition:
                        self.server.output.condition.wait_for(
                            lambda: self.server.output.frame is not None
                        )
                        frame = self.server.output.frame

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
                logging.info("Streaming client disconnected: %s", exc)
            return

        self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        logging.info("HTTP %s - %s", self.address_string(), format % args)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """複数ブラウザー接続を許可するHTTPサーバー。"""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, handler, output, width: int, height: int):
        super().__init__(address, handler)
        self.output = output
        self.stream_width = width
        self.stream_height = height


def parse_size(value: str) -> tuple[int, int]:
    """WIDTHxHEIGHT形式を正の整数2個へ変換する。"""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("サイズは640x480の形式で指定してください") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")

    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Picamera2のハードウェアMJPEGをHTTPで配信します。"
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=DEFAULT_SIZE,
        metavar="WIDTHxHEIGHT",
        help="配信サイズ。既定値: 640x480",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="カメラの設定fps。既定値: 10",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="待受アドレス。既定値: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="HTTPポート。既定値: 8000",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="自動停止までの秒数。0はCtrl+Cまで継続。既定値: 0",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fpsには0より大きい値を指定してください")
    if not 1 <= args.port <= 65535:
        parser.error("--portには1〜65535を指定してください")
    if args.duration < 0:
        parser.error("--durationには0以上を指定してください")

    return args


def main() -> int:
    args = parse_args()
    width, height = args.size
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    picam2 = Picamera2()
    model = picam2.camera_properties.get("Model", "unknown")
    # RGB変換を指定せず、ハードウェアMJPEGエンコーダーへ渡す形式を使う。
    config = picam2.create_video_configuration(
        main={"size": args.size},
        controls={"FrameRate": args.fps},
        buffer_count=4,
    )
    picam2.configure(config)

    output = StreamingOutput()
    http_server = StreamingServer((args.host, args.port), StreamingHandler, output, width, height)
    encoder = MJPEGEncoder()

    stop_timer = None
    try:
        picam2.start_recording(encoder, FileOutput(output))
        print(f"Camera model : {model}", flush=True)
        print(f"Stream size  : {width}x{height}", flush=True)
        print(f"Target fps   : {args.fps:.2f}", flush=True)
        print(f"Open in PC   : http://raspberrypi.local:{args.port}/", flush=True)
        print(f"Health check : http://raspberrypi.local:{args.port}/health", flush=True)
        if args.duration > 0:
            print(f"Duration     : {args.duration:.1f} s", flush=True)

            def stop_later() -> None:
                time.sleep(args.duration)
                http_server.shutdown()

            stop_timer = threading.Thread(target=stop_later, daemon=True)
            stop_timer.start()
        else:
            print("Stop        : Ctrl+C", flush=True)

        http_server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        http_server.server_close()
        picam2.stop_recording()
        picam2.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
