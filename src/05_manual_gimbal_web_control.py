#!/usr/bin/env python3
"""Phase 5M: MJPEG映像を見ながらブラウザーのキーで2軸ジンバルを操作する。

Pi側はPicamera2のハードウェアMJPEG配信とSG90制御を担当する。
PCには追加アプリを入れず、ブラウザーで映像、キー操作、現在角度を確認する。

配線:
  PAN  GPIO12 / 物理ピン32
  TILT GPIO13 / 物理ピン33
  外部5V電源を使用し、外部GNDとPi物理ピン34を共通化する。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import socketserver
import threading
import time
from http import server
from typing import Any
from urllib.parse import urlparse

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput


PAN_GPIO = 12
TILT_GPIO = 13
PAN_PHYSICAL_PIN = 32
TILT_PHYSICAL_PIN = 33
SERVO_MIN_PULSE_S = 0.0005
SERVO_MAX_PULSE_S = 0.0025
DEFAULT_CENTER = 90
DEFAULT_MIN_ANGLE = 80
DEFAULT_MAX_ANGLE = 100
DEFAULT_STEP = 2
DEFAULT_CONTROL_INTERVAL_MS = 40
DEFAULT_SIZE = (640, 480)
DEFAULT_FPS = 10.0
DEFAULT_PORT = 8000
MAX_REQUEST_BYTES = 1024
SHUTDOWN_CENTER_HOLD_S = 1.0


PAGE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pi Camera Gimbal Manual Control</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111827; color: #f9fafb; }
main { width: min(960px, 100%); margin: auto; padding: 16px; box-sizing: border-box; }
h1 { font-size: 1.35rem; margin: 0 0 8px; }
.notice { margin: 6px 0 12px; color: #cbd5e1; }
.stream-frame { position: relative; width: 100%; max-width: __WIDTH__px; aspect-ratio: __WIDTH__ / __HEIGHT__; box-sizing: border-box; background: #000; border: 2px solid #334155; border-radius: 8px; overflow: hidden; }
.stream-frame::before, .stream-frame::after { content: ""; position: absolute; z-index: 2; pointer-events: none; background: rgba(250, 204, 21, .78); }
.stream-frame::before { top: 0; bottom: 0; left: 50%; width: 1px; transform: translateX(-50%); }
.stream-frame::after { left: 0; right: 0; top: 50%; height: 1px; transform: translateY(-50%); }
.stream { width: 100%; height: 100%; max-width: none; object-fit: contain; background: #000; display: block; }
.panel { display: grid; grid-template-columns: minmax(230px, 1fr) minmax(220px, 1fr); gap: 16px; margin-top: 14px; }
.card { background: #1f2937; border-radius: 8px; padding: 14px; }
.status { font-size: 1.1rem; line-height: 1.8; }
.ok { color: #86efac; }
.warn { color: #fca5a5; }
.keys { display: grid; grid-template-columns: repeat(3, 64px); gap: 7px; justify-content: center; }
button { min-height: 48px; border: 1px solid #64748b; border-radius: 7px; background: #334155; color: white; font-size: 1rem; cursor: pointer; touch-action: none; }
button:active, button.active { background: #0284c7; }
.wide { grid-column: span 3; }
.help { color: #cbd5e1; line-height: 1.7; font-size: .95rem; }
kbd { background: #0f172a; border: 1px solid #64748b; border-radius: 4px; padding: 2px 6px; }
@media (max-width: 650px) { .panel { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main tabindex="0" id="app">
  <h1>Pi Camera Gimbal 手動操作</h1>
  <p class="notice">このページを一度クリックしてからキーを操作してください。</p>
  <div class="stream-frame" aria-label="Piカメラ映像と中央ガイド線">
    <img class="stream" src="/stream.mjpg" alt="Piカメラ映像">
  </div>
  <section class="panel">
    <div class="card">
      <div class="status">
        PAN: <strong id="pan">--</strong>°<br>
        TILT: <strong id="tilt">--</strong>°<br>
        PWM: <strong id="pwm">--</strong><br>
        状態: <strong id="message">接続中</strong>
      </div>
      <div class="keys">
        <span></span><button data-pan="0" data-tilt="1">↑ / E</button><span></span>
        <button data-pan="-1" data-tilt="0">← / S</button>
        <button id="center">中央 / C</button>
        <button data-pan="1" data-tilt="0">→ / F</button>
        <span></span><button data-pan="0" data-tilt="-1">↓ / D</button><span></span>
        <button class="wide" id="release">PWM解放 / X</button>
      </div>
    </div>
    <div class="card help">
      <strong>キー操作</strong><br>
      左右: <kbd>←</kbd> <kbd>→</kbd> または <kbd>S</kbd> <kbd>F</kbd><br>
      上下: <kbd>↑</kbd> <kbd>↓</kbd> または <kbd>E</kbd> <kbd>D</kbd><br>
      中央: <kbd>C</kbd>　PWM解放: <kbd>X</kbd><br><br>
      キーを押している間、Pi側が__INTERVAL_MS__msごとに__STEP__°ずつ動かします。
      斜め方向は横キーと縦キーを同時に押します。ブラウザーが非表示になった場合は移動を止めます。
    </div>
  </section>
</main>
<script>
const keyMap = {
  ArrowLeft:  { pan: -1, tilt:  0 },
  ArrowRight: { pan:  1, tilt:  0 },
  ArrowUp:    { pan:  0, tilt:  1 },
  ArrowDown:  { pan:  0, tilt: -1 },
  s:          { pan: -1, tilt:  0 },
  f:          { pan:  1, tilt:  0 },
  e:          { pan:  0, tilt:  1 },
  d:          { pan:  0, tilt: -1 }
};
const pressed = new Set();
let motionBusy = false;
let desiredMotion = { pan: 0, tilt: 0 };
let sentMotion = { pan: 0, tilt: 0 };

function normalizedKey(event) {
  return event.key.length === 1 ? event.key.toLowerCase() : event.key;
}

function updateStatus(data) {
  document.getElementById('pan').textContent = data.pan_angle;
  document.getElementById('tilt').textContent = data.tilt_angle;
  const pwm = document.getElementById('pwm');
  pwm.textContent = data.attached ? '出力中' : '解放';
  pwm.className = data.attached ? 'ok' : 'warn';
  document.getElementById('message').textContent = '接続正常';
}

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store'
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  updateStatus(data);
}

function currentMotion() {
  let pan = 0;
  let tilt = 0;
  for (const key of pressed) {
    const command = keyMap[key];
    if (command) { pan += command.pan; tilt += command.tilt; }
  }
  pan = Math.max(-1, Math.min(1, pan));
  tilt = Math.max(-1, Math.min(1, tilt));
  return { pan, tilt };
}

function sameMotion(left, right) {
  return left.pan === right.pan && left.tilt === right.tilt;
}

async function synchronizeMotion() {
  desiredMotion = currentMotion();
  if (motionBusy || sameMotion(desiredMotion, sentMotion)) return;
  const target = { ...desiredMotion };
  motionBusy = true;
  try {
    await post('/api/motion', target);
    sentMotion = target;
  } catch (error) {
    sentMotion = target;
    document.getElementById('message').textContent = error.message;
  } finally {
    motionBusy = false;
    if (!sameMotion(desiredMotion, sentMotion)) synchronizeMotion();
  }
}

document.addEventListener('keydown', event => {
  const key = normalizedKey(event);
  if (key in keyMap) {
    event.preventDefault();
    pressed.add(key);
    synchronizeMotion();
  } else if (key === 'c') {
    event.preventDefault();
    pressed.clear();
    synchronizeMotion();
    post('/api/center').catch(console.error);
  } else if (key === 'x') {
    event.preventDefault();
    pressed.clear();
    synchronizeMotion();
    post('/api/release').catch(console.error);
  }
});

document.addEventListener('keyup', event => {
  const key = normalizedKey(event);
  if (pressed.delete(key)) synchronizeMotion();
});
window.addEventListener('blur', () => {
  if (pressed.size) {
    pressed.clear();
    synchronizeMotion();
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && pressed.size) {
    pressed.clear();
    synchronizeMotion();
  }
});

for (const button of document.querySelectorAll('[data-pan]')) {
  const start = event => {
    event.preventDefault();
    button.classList.add('active');
    const key = `button-${button.dataset.pan}-${button.dataset.tilt}`;
    keyMap[key] = { pan: Number(button.dataset.pan), tilt: Number(button.dataset.tilt) };
    pressed.add(key);
    synchronizeMotion();
  };
  const stop = () => {
    button.classList.remove('active');
    let changed = false;
    for (const key of [...pressed]) {
      if (key.startsWith('button-')) changed = pressed.delete(key) || changed;
    }
    if (changed) synchronizeMotion();
  };
  button.addEventListener('pointerdown', start);
  button.addEventListener('pointerup', stop);
  button.addEventListener('pointercancel', stop);
  button.addEventListener('pointerleave', stop);
}
document.getElementById('center').addEventListener('click', () => post('/api/center').catch(console.error));
document.getElementById('release').addEventListener('click', () => {
  pressed.clear();
  synchronizeMotion();
  post('/api/release').catch(console.error);
});
fetch('/api/status', { cache: 'no-store' }).then(r => r.json()).then(updateStatus).catch(error => {
  document.getElementById('message').textContent = error.message;
});
document.getElementById('app').focus();
</script>
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    """MJPEGエンコーダーから届いた最新JPEGだけを保持する。"""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class ServoController:
    """安全角度制限付きのPAN/TILT制御。"""

    def __init__(
        self,
        pan_min: int,
        pan_max: int,
        tilt_min: int,
        tilt_max: int,
        center: int,
        step: int,
        control_interval_ms: int,
        invert_pan: bool,
        invert_tilt: bool,
    ) -> None:
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.center_angle = center
        self.step = step
        self.control_interval_s = control_interval_ms / 1000.0
        self.control_interval_ms = control_interval_ms
        self.pan_sign = -1 if invert_pan else 1
        self.tilt_sign = -1 if invert_tilt else 1
        self.pan_angle = center
        self.tilt_angle = center
        self.pan_direction = 0
        self.tilt_direction = 0
        self.attached = False
        self.closing = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.factory = PiGPIOFactory()
        self.pan = AngularServo(
            PAN_GPIO,
            min_angle=0,
            max_angle=180,
            min_pulse_width=SERVO_MIN_PULSE_S,
            max_pulse_width=SERVO_MAX_PULSE_S,
            initial_angle=None,
            pin_factory=self.factory,
        )
        self.tilt = AngularServo(
            TILT_GPIO,
            min_angle=0,
            max_angle=180,
            min_pulse_width=SERVO_MIN_PULSE_S,
            max_pulse_width=SERVO_MAX_PULSE_S,
            initial_angle=None,
            pin_factory=self.factory,
        )
        self.center()
        self.motion_thread = threading.Thread(
            target=self._motion_loop,
            name="servo-motion",
            daemon=True,
        )
        self.motion_thread.start()

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "attached": self.attached,
            "step": self.step,
            "control_interval_ms": self.control_interval_ms,
            "motion": {"pan": self.pan_direction, "tilt": self.tilt_direction},
            "pan_range": [self.pan_min, self.pan_max],
            "tilt_range": [self.tilt_min, self.tilt_max],
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self._status_unlocked()

    def _ensure_active_unlocked(self) -> None:
        if self.closing:
            raise RuntimeError("Servo controller is shutting down")

    @staticmethod
    def _validate_direction(pan_direction: int, tilt_direction: int) -> None:
        if pan_direction not in (-1, 0, 1) or tilt_direction not in (-1, 0, 1):
            raise ValueError("panとtiltは-1、0、1のいずれかにしてください")

    def _move_once_unlocked(self, pan_direction: int, tilt_direction: int) -> None:
        if self.closing:
            return
        next_pan = self._clamp(
            self.pan_angle + pan_direction * self.step * self.pan_sign,
            self.pan_min,
            self.pan_max,
        )
        next_tilt = self._clamp(
            self.tilt_angle + tilt_direction * self.step * self.tilt_sign,
            self.tilt_min,
            self.tilt_max,
        )
        if next_pan != self.pan_angle:
            self.pan_angle = next_pan
            self.pan.angle = self.pan_angle
            self.attached = True
        if next_tilt != self.tilt_angle:
            self.tilt_angle = next_tilt
            self.tilt.angle = self.tilt_angle
            self.attached = True

    def _motion_loop(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                if self.closing:
                    break
                if self.pan_direction or self.tilt_direction:
                    self._move_once_unlocked(self.pan_direction, self.tilt_direction)
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.0, self.control_interval_s - elapsed))

    def set_motion(self, pan_direction: int, tilt_direction: int) -> dict[str, Any]:
        self._validate_direction(pan_direction, tilt_direction)
        with self.lock:
            self._ensure_active_unlocked()
            self.pan_direction = pan_direction
            self.tilt_direction = tilt_direction
            if pan_direction or tilt_direction:
                self.attached = True
            logging.debug("Servo motion PAN=%d TILT=%d", pan_direction, tilt_direction)
            return self._status_unlocked()

    def move(self, pan_direction: int, tilt_direction: int) -> dict[str, Any]:
        """互換用の1ステップ移動。通常のブラウザー操作はset_motionを使う。"""
        self._validate_direction(pan_direction, tilt_direction)
        with self.lock:
            self.pan_direction = 0
            self.tilt_direction = 0
            self._move_once_unlocked(pan_direction, tilt_direction)
            return self._status_unlocked()

    def _center_unlocked(self) -> dict[str, Any]:
        self.pan_direction = 0
        self.tilt_direction = 0
        self.pan_angle = self._clamp(self.center_angle, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.center_angle, self.tilt_min, self.tilt_max)
        self.pan.angle = self.pan_angle
        self.tilt.angle = self.tilt_angle
        self.attached = True
        logging.info("Servo centered PAN=%d TILT=%d", self.pan_angle, self.tilt_angle)
        return self._status_unlocked()

    def center(self) -> dict[str, Any]:
        with self.lock:
            self._ensure_active_unlocked()
            return self._center_unlocked()

    def _release_unlocked(self) -> dict[str, Any]:
        self.pan_direction = 0
        self.tilt_direction = 0
        self.pan.detach()
        self.tilt.detach()
        self.attached = False
        logging.info("Servo PWM released")
        return self._status_unlocked()

    def release(self) -> dict[str, Any]:
        with self.lock:
            self._ensure_active_unlocked()
            return self._release_unlocked()

    def close(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            self.pan_direction = 0
            self.tilt_direction = 0
        self.stop_event.set()
        self.motion_thread.join(timeout=1.0)
        try:
            with self.lock:
                self._center_unlocked()
            # Keep center PWM active until a full-range SG90 move can finish.
            time.sleep(SHUTDOWN_CENTER_HOLD_S)
        except Exception as error:
            logging.warning("Centering on shutdown failed: %s", error)
        try:
            with self.lock:
                self._release_unlocked()
        finally:
            self.pan.close()
            self.tilt.close()
            self.factory.close()


class ControlHandler(server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "PiCameraGimbalManual/0.1"

    def _send_bytes(self, status: int, content_type: str, content: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(content))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        content = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", content)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            content = (
                PAGE.replace("__WIDTH__", str(self.server.stream_width))
                .replace("__HEIGHT__", str(self.server.stream_height))
                .replace("__STEP__", str(self.server.controller.step))
                .replace("__INTERVAL_MS__", str(self.server.control_interval_ms))
                .encode("utf-8")
            )
            self._send_bytes(200, "text/html; charset=utf-8", content)
            return
        if path == "/health":
            self._send_bytes(200, "text/plain; charset=utf-8", b"ok\n")
            return
        if path == "/api/status":
            self._send_json(200, self.server.controller.status())
            return
        if path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with self.server.output.condition:
                        self.server.output.condition.wait_for(lambda: self.server.output.frame is not None)
                        frame = self.server.output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                logging.info("Streaming client disconnected")
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 0 or content_length > MAX_REQUEST_BYTES:
                raise ValueError("リクエストが大きすぎます")
            body = self.rfile.read(content_length) if content_length else b"{}"
            data = json.loads(body.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSONオブジェクトを指定してください")

            if path == "/api/move":
                result = self.server.controller.move(int(data.get("pan", 0)), int(data.get("tilt", 0)))
            elif path == "/api/motion":
                result = self.server.controller.set_motion(int(data.get("pan", 0)), int(data.get("tilt", 0)))
            elif path == "/api/center":
                result = self.server.controller.center()
            elif path == "/api/release":
                result = self.server.controller.release()
            else:
                self.send_error(404)
                return
            self._send_json(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            logging.exception("Control request failed")
            self._send_json(500, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        if urlparse(self.path).path in ("/api/move", "/api/motion"):
            return
        logging.info("HTTP %s - %s", self.address_string(), format % args)


class ControlServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address,
        output,
        controller,
        width: int,
        height: int,
        control_interval_ms: int,
    ):
        super().__init__(address, ControlHandler)
        self.output = output
        self.controller = controller
        self.stream_width = width
        self.stream_height = height
        self.control_interval_ms = control_interval_ms


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("サイズは640x480形式で指定してください") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=parse_size, default=DEFAULT_SIZE, metavar="WIDTHxHEIGHT")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--pan-min", type=int, default=DEFAULT_MIN_ANGLE)
    parser.add_argument("--pan-max", type=int, default=DEFAULT_MAX_ANGLE)
    parser.add_argument("--tilt-min", type=int, default=DEFAULT_MIN_ANGLE)
    parser.add_argument("--tilt-max", type=int, default=DEFAULT_MAX_ANGLE)
    parser.add_argument("--center", type=int, default=DEFAULT_CENTER)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument(
        "--control-interval-ms",
        type=int,
        default=DEFAULT_CONTROL_INTERVAL_MS,
        help="キー押下中の指令間隔。既定値: 40ms",
    )
    parser.add_argument("--invert-pan", action="store_true")
    parser.add_argument("--invert-tilt", action="store_true")
    parser.add_argument("--vflip", action="store_true", help="配信映像を上下反転")
    parser.add_argument("--duration", type=float, default=0.0, help="0はCtrl+Cまで継続")
    args = parser.parse_args()

    for name, value in (
        ("pan-min", args.pan_min), ("pan-max", args.pan_max),
        ("tilt-min", args.tilt_min), ("tilt-max", args.tilt_max),
        ("center", args.center),
    ):
        if not 0 <= value <= 180:
            parser.error(f"--{name}は0から180で指定してください")
    if not args.pan_min <= args.center <= args.pan_max:
        parser.error("pan-min <= center <= pan-maxにしてください")
    if not args.tilt_min <= args.center <= args.tilt_max:
        parser.error("tilt-min <= center <= tilt-maxにしてください")
    if not 1 <= args.step <= 20:
        parser.error("--stepは1から20で指定してください")
    if not 20 <= args.control_interval_ms <= 500:
        parser.error("--control-interval-msは20から500で指定してください")
    if args.fps <= 0:
        parser.error("--fpsは0より大きくしてください")
    if not 1 <= args.port <= 65535:
        parser.error("--portは1から65535で指定してください")
    if args.duration < 0:
        parser.error("--durationは0以上で指定してください")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    width, height = args.size

    controller: ServoController | None = None
    picam2: Picamera2 | None = None
    http_server: ControlServer | None = None
    recording_started = False
    try:
        controller = ServoController(
            args.pan_min, args.pan_max, args.tilt_min, args.tilt_max,
            args.center, args.step, args.control_interval_ms,
            args.invert_pan, args.invert_tilt,
        )
        picam2 = Picamera2()
        camera_model = picam2.camera_properties.get("Model", "unknown")
        config = picam2.create_video_configuration(
            main={"size": args.size},
            controls={"FrameRate": args.fps},
            transform=Transform(vflip=args.vflip),
            buffer_count=4,
        )
        picam2.configure(config)
        output = StreamingOutput()
        encoder = MJPEGEncoder()
        http_server = ControlServer(
            (args.host, args.port),
            output,
            controller,
            width,
            height,
            args.control_interval_ms,
        )
        picam2.start_recording(encoder, FileOutput(output))
        recording_started = True

        print(f"Camera model : {camera_model}", flush=True)
        print(f"Stream       : {width}x{height} @ {args.fps:.1f} fps", flush=True)
        print(f"Image vflip  : {'on' if args.vflip else 'off'}", flush=True)
        print(f"PAN          : GPIO{PAN_GPIO} / pin {PAN_PHYSICAL_PIN} / {args.pan_min}-{args.pan_max} deg", flush=True)
        print(f"TILT         : GPIO{TILT_GPIO} / pin {TILT_PHYSICAL_PIN} / {args.tilt_min}-{args.tilt_max} deg", flush=True)
        print(f"Step         : {args.step} deg", flush=True)
        print(f"Control rate : {args.control_interval_ms} ms", flush=True)
        print(f"Open in PC   : http://raspberrypi.local:{args.port}/", flush=True)
        print("Keys         : arrows or ESDF, C=center, X=release", flush=True)
        print("Stop         : Ctrl+C", flush=True)

        if args.duration > 0:
            def stop_later() -> None:
                time.sleep(args.duration)
                if http_server is not None:
                    http_server.shutdown()

            threading.Thread(target=stop_later, name="stop-timer", daemon=True).start()

        http_server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        if http_server is not None:
            http_server.server_close()
        if picam2 is not None:
            try:
                if recording_started:
                    picam2.stop_recording()
            finally:
                picam2.close()
        if controller is not None:
            controller.close()
        print("Camera closed and servo PWM released", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
