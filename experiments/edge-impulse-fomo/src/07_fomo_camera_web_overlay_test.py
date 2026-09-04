#!/usr/bin/env python3
"""Phase 6B: FOMO検出枠を重ねたカメラ映像をPCへMJPEG配信する。

検出結果が実際の人物に重なっているかを確認するためのサンプル。
サーボ制御は行わず、Picamera2、FOMO推論、検出枠描画、HTTP配信だけを行う。
"""

from __future__ import annotations

import argparse
import threading
import time
from http import server
from pathlib import Path
from typing import Any

import cv2
from edge_impulse_linux.image import ImageImpulseRunner
from picamera2 import Picamera2


DEFAULT_MODEL = Path("~/src/fomo/models/person_fomo_pizero2_int8_v5.eim")
DEFAULT_SIZE = (640, 480)
DEFAULT_CAMERA_FPS = 15.0
DEFAULT_DURATION = 0.0
DEFAULT_WARMUP = 2.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_PORT = 8001
DEFAULT_JPEG_QUALITY = 80


PAGE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FOMO camera overlay</title>
<style>
body { margin: 0; background: #111827; color: #f9fafb; font-family: system-ui, sans-serif; }
main { width: min(960px, 100%); margin: auto; padding: 16px; box-sizing: border-box; }
h1 { font-size: 1.25rem; margin: 0 0 8px; }
p { color: #cbd5e1; }
img { display: block; width: 100%; max-width: 640px; height: auto; background: #000; border: 2px solid #334155; }
</style>
</head>
<body>
<main>
<h1>Pi Camera / FOMO検出枠</h1>
<p>緑・橙の枠がFOMOの検出セルです。Ctrl+Cで停止します。</p>
<img src="/stream.mjpg" alt="FOMO detection stream">
</main>
</body>
</html>
""".encode("utf-8")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "サイズは640x480形式で指定してください"
        ) from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--size", type=parse_size, default=DEFAULT_SIZE, metavar="WIDTHxHEIGHT")
    parser.add_argument("--camera-fps", type=float, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    args = parser.parse_args()

    if args.camera_fps <= 0:
        parser.error("--camera-fpsは0より大きくしてください")
    if args.duration < 0:
        parser.error("--durationは0以上で指定してください")
    if args.warmup < 0:
        parser.error("--warmupは0以上で指定してください")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--thresholdは0以上1以下で指定してください")
    if not 1 <= args.port <= 65535:
        parser.error("--portは1から65535で指定してください")
    if not 20 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-qualityは20から100で指定してください")
    return args


def get_model_features(runner: Any, frame: Any) -> tuple[Any, Any]:
    """Edge Impulse Studioのリサイズ設定に合わせて入力画像を作る。"""
    studio_method = getattr(runner, "get_features_from_image_auto_studio_settings", None)
    if studio_method is not None:
        return studio_method(frame)
    return runner.get_features_from_image(frame)


def result_detections(result: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    detections = result.get("result", {}).get("bounding_boxes", [])
    return [
        item
        for item in detections
        if float(item.get("value", 0.0)) >= threshold
    ]


def map_box_to_frame(
    detection: dict[str, Any],
    frame_width: int,
    frame_height: int,
    model_width: int,
    model_height: int,
    resize_mode: str,
) -> tuple[int, int, int, int] | None:
    """モデル入力座標を元のカメラ画像座標へ戻す。"""
    x = float(detection.get("x", 0))
    y = float(detection.get("y", 0))
    width = float(detection.get("width", 0))
    height = float(detection.get("height", 0))

    if resize_mode == "fit-longest":
        scale = min(model_width / frame_width, model_height / frame_height)
        resized_width = frame_width * scale
        resized_height = frame_height * scale
        pad_x = (model_width - resized_width) / 2.0
        pad_y = (model_height - resized_height) / 2.0
        left = (x - pad_x) / scale
        top = (y - pad_y) / scale
        right = (x + width - pad_x) / scale
        bottom = (y + height - pad_y) / scale
    else:
        # 現在のモデルはfit-longestだが、互換用にsquashも扱う。
        scale_x = frame_width / model_width
        scale_y = frame_height / model_height
        left = x * scale_x
        top = y * scale_y
        right = (x + width) * scale_x
        bottom = (y + height) * scale_y

    left = max(0, min(frame_width - 1, round(left)))
    top = max(0, min(frame_height - 1, round(top)))
    right = max(0, min(frame_width - 1, round(right)))
    bottom = max(0, min(frame_height - 1, round(bottom)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def draw_overlay(
    frame_rgb: Any,
    detections: list[dict[str, Any]],
    frame_count: int,
    inference_ms: float,
    model_width: int,
    model_height: int,
    resize_mode: str,
    jpeg_quality: int,
) -> bytes:
    frame_height, frame_width = frame_rgb.shape[:2]
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    for detection in detections:
        box = map_box_to_frame(
            detection,
            frame_width,
            frame_height,
            model_width,
            model_height,
            resize_mode,
        )
        if box is None:
            continue
        left, top, right, bottom = box
        score = float(detection.get("value", 0.0))
        color = (0, 255, 0) if score >= 0.75 else (0, 165, 255)
        label = f"{detection.get('label', 'unknown')} {score:.2f}"
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text_y = max(18, top - 5)
        cv2.putText(
            image,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    center_x = frame_width // 2
    center_y = frame_height // 2
    cv2.line(image, (center_x, 0), (center_x, frame_height), (0, 215, 255), 1)
    cv2.line(image, (0, center_y), (frame_width, center_y), (0, 215, 255), 1)
    status = f"FOMO {model_width}x{model_height}  detections={len(detections)}  inference={inference_ms:.1f}ms"
    cv2.putText(
        image,
        status,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError("JPEGエンコードに失敗しました")
    return encoded.tobytes()


class FrameBuffer:
    """最新のJPEGフレームだけをHTTPクライアントへ渡す。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.sequence = 0
        self.closed = False

    def update(self, frame: bytes) -> None:
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def wait_next(self, previous_sequence: int) -> tuple[int, bytes | None, bool]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.closed or self.sequence > previous_sequence
            )
            return self.sequence, self.frame, self.closed


class OverlayHandler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "PiCameraGimbalFomo/0.1"

    def send_bytes(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE)
            return
        if self.path == "/health":
            self.send_bytes(200, "text/plain; charset=utf-8", b"ok\n")
            return
        if self.path != "/stream.mjpg":
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
                sequence, frame, closed = self.server.buffer.wait_next(sequence)
                if closed or frame is None:
                    break
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        if self.path != "/stream.mjpg":
            super().log_message(format, *args)


class OverlayServer(server.ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], buffer: FrameBuffer):
        super().__init__(address, OverlayHandler)
        self.buffer = buffer


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    picam2: Picamera2 | None = None
    http_server: OverlayServer | None = None
    server_thread: threading.Thread | None = None
    camera_started = False
    buffer = FrameBuffer()

    with ImageImpulseRunner(str(model_path)) as runner:
        model_info = runner.init()
        parameters = model_info.get("model_parameters", {})
        model_width = int(parameters.get("image_input_width", 64))
        model_height = int(parameters.get("image_input_height", 64))
        resize_mode = str(parameters.get("image_resize_mode", "squash"))

        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": args.size, "format": "RGB888"},
            controls={"FrameRate": args.camera_fps},
            buffer_count=4,
        )
        picam2.configure(camera_config)

        http_server = OverlayServer(("0.0.0.0", args.port), buffer)
        server_thread = threading.Thread(
            target=http_server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
        )
        server_thread.start()

        print(f"Model        : {model_path}", flush=True)
        print(f"Model input  : {model_width}x{model_height} RGB", flush=True)
        print(f"Resize mode  : {resize_mode}", flush=True)
        print(f"Camera input : {args.size[0]}x{args.size[1]} RGB888", flush=True)
        print(f"Open in PC   : http://raspberrypi.local:{args.port}/", flush=True)

        frame_count = 0
        inference_count = 0
        total_inference_ms = 0.0
        started_at = 0.0
        next_report_at = 0.0

        try:
            picam2.start()
            camera_started = True
            time.sleep(args.warmup)
            started_at = time.monotonic()
            next_report_at = started_at + 1.0
            deadline = started_at + args.duration if args.duration > 0 else None
            print("Overlay stream started. Stop with Ctrl+C.", flush=True)

            while deadline is None or time.monotonic() < deadline:
                frame = picam2.capture_array("main")
                features, _model_image = get_model_features(runner, frame)
                result = runner.classify(features)
                frame_count += 1
                inference_count += 1

                timing = result.get("timing", {})
                inference_ms = sum(
                    float(timing.get(name, 0.0))
                    for name in ("dsp", "classification", "anomaly")
                )
                total_inference_ms += inference_ms
                detections = result_detections(result, args.threshold)
                buffer.update(
                    draw_overlay(
                        frame,
                        detections,
                        frame_count,
                        inference_ms,
                        model_width,
                        model_height,
                        resize_mode,
                        args.jpeg_quality,
                    )
                )

                now = time.monotonic()
                if now >= next_report_at:
                    elapsed = now - started_at
                    fps = inference_count / elapsed if elapsed > 0 else 0.0
                    average_ms = total_inference_ms / inference_count
                    print(
                        f"STATUS elapsed={elapsed:.1f}s frames={frame_count} "
                        f"fps={fps:.2f} avg_inference={average_ms:.1f}ms "
                        f"detections={len(detections)}",
                        flush=True,
                    )
                    next_report_at = now + 1.0
        except KeyboardInterrupt:
            print("Stopping...", flush=True)
        finally:
            if camera_started:
                picam2.stop()
            picam2.close()
            buffer.close()
            http_server.shutdown()
            http_server.server_close()
            if server_thread is not None:
                server_thread.join(timeout=2.0)

        elapsed = time.monotonic() - started_at if started_at else 0.0
        fps = inference_count / elapsed if elapsed > 0 else 0.0
        average_ms = total_inference_ms / inference_count if inference_count else 0.0
        print(
            f"SUMMARY frames={frame_count} inferences={inference_count} "
            f"fps={fps:.2f} avg_inference={average_ms:.1f}ms",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
