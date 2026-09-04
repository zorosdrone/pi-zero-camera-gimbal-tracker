#!/usr/bin/env python3
"""PCで対象を選択し、Pi Zero 2上のKCFで追跡してSG90を制御する。

Pi側:
  - Picamera2でOV5647映像を取得
  - KCF/MOSSE/CSRTで単一ROIを追跡
  - 追跡対象の中心ずれからPAN/TILT方向を計算
  - SG90を安全角度範囲内で制御
  - MJPEG映像と操作ページをHTTP配信

PC側:
  ブラウザーで http://raspberrypi.local:8002/ を開き、映像上で対象をドラッグする。
  ROI選択はPCで行うが、追跡処理とサーボ制御はPi側で行う。

配線:
  PAN  GPIO12 / 物理ピン32
  TILT GPIO13 / 物理ピン33
  外部5V電源を使用し、外部GNDとPi物理ピン34を共通化する。
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import threading
import time
from typing import Any

import cv2
from gpiozero import AngularServo
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

from pi_kcf_web import TrackingServer


PAN_GPIO = 12
TILT_GPIO = 13
PAN_PHYSICAL_PIN = 32
TILT_PHYSICAL_PIN = 33
SERVO_MIN_PULSE_S = 0.0005
SERVO_MAX_PULSE_S = 0.0025

DEFAULT_SIZE = (640, 480)
DEFAULT_TRACK_SIZE = (320, 240)
DEFAULT_FPS = 15.0
DEFAULT_PORT = 8002
DEFAULT_TRACKER = "kcf"
DEFAULT_CENTER = 90
DEFAULT_MIN_ANGLE = 30
DEFAULT_MAX_ANGLE = 150
DEFAULT_STEP = 2
DEFAULT_CONTROL_INTERVAL_MS = 120
DEFAULT_DEADBAND = 0.05
DEFAULT_RECOVERY_THRESHOLD = 0.55
DEFAULT_RECOVERY_CONFIRM_FRAMES = 3
DEFAULT_TRACKING_THRESHOLD = 0.35
RECOVERY_SCALES = (0.50, 0.65, 0.85, 1.0, 1.25, 1.55, 2.0)
RECOVERY_CANDIDATES_PER_SCALE = 2
RECOVERY_SEARCH_WIDTH = 160
DEFAULT_DURATION = 0.0
SHUTDOWN_CENTER_HOLD_S = 1.0


class StreamingOutput(io.BufferedIOBase):
    """ハードウェアMJPEGエンコーダーの最新フレームを保持する。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.sequence = 0
        self._closed = False

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = bytes(buf)
            self.sequence += 1
            self.condition.notify_all()
        return len(buf)

    def close(self) -> None:
        with self.condition:
            self._closed = True
            self.condition.notify_all()

    def wait_next(self, previous: int) -> tuple[int, bytes | None, bool]:
        with self.condition:
            self.condition.wait_for(lambda: self._closed or self.sequence > previous)
            return self.sequence, self.frame, self._closed


class TargetPreviewOutput:
    """ROI選択時の切り出しJPEGを保持する。追跡中は更新しない。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: bytes | None = None
        self.image: Any | None = None
        self.sequence = 0

    @staticmethod
    def _crop(frame: Any, box: tuple[float, float, float, float]) -> Any | None:
        height, width = frame.shape[:2]
        x, y, w, h = box
        left = max(0, min(width - 1, int((x - w * 0.18) * width)))
        top = max(0, min(height - 1, int((y - h * 0.18) * height)))
        right = max(left + 1, min(width, int((x + w * 1.18) * width)))
        bottom = max(top + 1, min(height, int((y + h * 1.18) * height)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return crop.copy()

    def _encode_unlocked(self) -> bytes | None:
        if self.image is None:
            return None
        result = self.image.copy()
        color = (134, 239, 172)
        cv2.rectangle(
            result,
            (0, 0),
            (result.shape[1] - 1, result.shape[0] - 1),
            color,
            max(1, min(result.shape[:2]) // 80),
        )
        encoded_ok, encoded = cv2.imencode(
            ".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        return bytes(encoded) if encoded_ok else None

    def set_selection(self, frame: Any, box: tuple[float, float, float, float]) -> None:
        crop = self._crop(frame, box)
        if crop is None:
            return
        with self.lock:
            self.image = crop
            encoded = self._encode_unlocked()
            if encoded is not None:
                self.frame = encoded
                self.sequence += 1

    def clear(self) -> None:
        with self.lock:
            self.frame = None
            self.image = None
            self.sequence += 1

    def snapshot(self) -> tuple[int, bytes | None]:
        with self.lock:
            return self.sequence, self.frame


class ServoController:
    """安全角度制限付きPAN/TILT制御。"""

    def __init__(
        self,
        pan_min: int,
        pan_max: int,
        tilt_min: int,
        tilt_max: int,
        center: int,
        step: int,
        control_interval_ms: int,
        deadband: float,
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
        self.deadband = deadband
        self.pan_sign = -1 if invert_pan else 1
        self.tilt_sign = -1 if invert_tilt else 1
        self.pan_angle = center
        self.tilt_angle = center
        self.pan_direction = 0
        self.tilt_direction = 0
        self.pan_error = 0.0
        self.tilt_error = 0.0
        self.attached = False
        self.closing = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.factory, self.gpio_backend = create_gpio_factory()
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
        self.motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self.motion_thread.start()

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, value))

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "pan_angle": self.pan_angle,
            "tilt_angle": self.tilt_angle,
            "attached": self.attached,
            "gpio_backend": self.gpio_backend,
            "motion": {"pan": self.pan_direction, "tilt": self.tilt_direction},
            "tracking_error": {
                "x": round(self.pan_error, 3),
                "y": round(self.tilt_error, 3),
            },
            "pan_range": [self.pan_min, self.pan_max],
            "tilt_range": [self.tilt_min, self.tilt_max],
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self._status_unlocked()

    def _move_once_unlocked(self) -> None:
        if self.closing:
            return
        pan_step = self._step_for_error(self.pan_error)
        tilt_step = self._step_for_error(self.tilt_error)
        next_pan = self._clamp(
            self.pan_angle + self.pan_direction * pan_step * self.pan_sign,
            self.pan_min,
            self.pan_max,
        )
        next_tilt = self._clamp(
            self.tilt_angle + self.tilt_direction * tilt_step * self.tilt_sign,
            self.tilt_min,
            self.tilt_max,
        )
        if next_pan != self.pan_angle:
            self.pan_angle = next_pan
            self.pan.angle = next_pan
            self.attached = True
        if next_tilt != self.tilt_angle:
            self.tilt_angle = next_tilt
            self.tilt.angle = next_tilt
            self.attached = True

    def _step_for_error(self, error: float) -> float:
        """誤差が小さいほど1回の角度更新を小さくする。"""
        magnitude = abs(error)
        if magnitude <= self.deadband:
            return 0.0
        span = max(0.001, 0.5 - self.deadband)
        ratio = min(1.0, (magnitude - self.deadband) / span)
        minimum_step = min(0.25, float(self.step))
        return max(minimum_step, float(self.step) * ratio)

    def _motion_loop(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                if self.closing:
                    break
                if self.pan_direction or self.tilt_direction:
                    self._move_once_unlocked()
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.0, self.control_interval_s - elapsed))

    def set_tracking_error(self, error_x: float, error_y: float) -> None:
        """画面中心からの正規化ずれをPAN/TILT方向へ変換する。"""
        pan = 0 if abs(error_x) <= self.deadband else (1 if error_x > 0 else -1)
        tilt = 0 if abs(error_y) <= self.deadband else (1 if error_y > 0 else -1)
        with self.lock:
            if self.closing:
                return
            self.pan_error = error_x
            self.tilt_error = error_y
            self.pan_direction = pan
            self.tilt_direction = tilt
            if pan or tilt:
                self.attached = True

    def stop_motion(self) -> dict[str, Any]:
        with self.lock:
            self.pan_direction = 0
            self.tilt_direction = 0
            self.pan_error = 0.0
            self.tilt_error = 0.0
            return self._status_unlocked()

    def _center_unlocked(self) -> dict[str, Any]:
        self.pan_direction = 0
        self.tilt_direction = 0
        self.pan_error = 0.0
        self.tilt_error = 0.0
        self.pan_angle = self._clamp(self.center_angle, self.pan_min, self.pan_max)
        self.tilt_angle = self._clamp(self.center_angle, self.tilt_min, self.tilt_max)
        self.pan.angle = self.pan_angle
        self.tilt.angle = self.tilt_angle
        self.attached = True
        return self._status_unlocked()

    def center(self) -> dict[str, Any]:
        with self.lock:
            return self._center_unlocked()

    def _release_unlocked(self) -> dict[str, Any]:
        self.pan_direction = 0
        self.tilt_direction = 0
        self.pan_error = 0.0
        self.tilt_error = 0.0
        self.pan.detach()
        self.tilt.detach()
        self.attached = False
        return self._status_unlocked()

    def release(self) -> dict[str, Any]:
        with self.lock:
            return self._release_unlocked()

    def close(self) -> None:
        with self.lock:
            if self.closing:
                return
            self.closing = True
            self.pan_direction = 0
            self.tilt_direction = 0
            self.pan_error = 0.0
            self.tilt_error = 0.0
        self.stop_event.set()
        self.motion_thread.join(timeout=1.0)
        try:
            with self.lock:
                self._center_unlocked()
            time.sleep(SHUTDOWN_CENTER_HOLD_S)
        except Exception as error:
            logging.warning("終了時の中央移動に失敗: %s", error)
        try:
            with self.lock:
                self._release_unlocked()
        finally:
            self.pan.close()
            self.tilt.close()
            self.factory.close()


def clamp_norm(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def create_gpio_factory() -> tuple[Any, str]:
    """pigpioのハードウェアタイミングPWMを優先する。"""
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory

        return PiGPIOFactory(), "pigpio"
    except Exception as pigpio_error:
        try:
            from gpiozero.pins.lgpio import LGPIOFactory

            return LGPIOFactory(), "lgpio-fallback"
        except Exception as lgpio_error:
            raise RuntimeError(
                "gpiozeroのGPIOバックエンドを初期化できません。"
                f" pigpio={pigpio_error}; lgpio={lgpio_error}"
            ) from lgpio_error


class TrackingState:
    """ブラウザーのROIとKCF追跡結果を共有する。"""

    def __init__(self, tracker_name: str, width: int, height: int) -> None:
        self.tracker_name = tracker_name
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.pending_roi: tuple[float, float, float, float] | None = None
        self.roi: tuple[float, float, float, float] | None = None
        self.bbox: tuple[float, float, float, float] | None = None
        self.tracking = False
        self.recovering = False
        self.lost_count = 0
        self.recovery_count = 0
        self.recovery_success_count = 0
        self.recovery_score: float | None = None
        self.match_score: float | None = None
        self.frame_count = 0
        self.tracked_frame_count = 0
        self.tracking_started_at: float | None = None
        self.started_at = time.monotonic()
        self.last_error = "ROI未選択"

    @staticmethod
    def _validate_box(box: dict[str, Any]) -> tuple[float, float, float, float]:
        x = clamp_norm(float(box.get("x", 0.0)))
        y = clamp_norm(float(box.get("y", 0.0)))
        w = max(0.0, min(1.0 - x, float(box.get("w", 0.0))))
        h = max(0.0, min(1.0 - y, float(box.get("h", 0.0))))
        if w < 0.03 or h < 0.03:
            raise ValueError("ROIが小さすぎます")
        return x, y, w, h

    def request_roi(self, box: dict[str, Any]) -> dict[str, Any]:
        normalized = self._validate_box(box)
        with self.lock:
            self.pending_roi = normalized
            self.roi = normalized
            self.bbox = normalized
            self.tracking = False
            self.recovering = False
            self.lost_count = 0
            self.recovery_count = 0
            self.recovery_success_count = 0
            self.recovery_score = None
            self.match_score = None
            self.last_error = "ROIを受信、初期化待ち"
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.pending_roi = None
            self.roi = None
            self.bbox = None
            self.tracking = False
            self.recovering = False
            self.tracking_started_at = None
            self.recovery_score = None
            self.match_score = None
            self.last_error = "追跡停止"
        return self.status()

    def take_pending_roi(self) -> tuple[float, float, float, float] | None:
        with self.lock:
            pending = self.pending_roi
            self.pending_roi = None
            return pending

    def mark_initialized(self, box: tuple[float, float, float, float]) -> None:
        with self.lock:
            self.roi = box
            self.bbox = box
            self.tracking = True
            self.recovering = False
            self.recovery_score = None
            self.match_score = None
            self.tracked_frame_count = 0
            self.tracking_started_at = time.monotonic()
            self.last_error = "追跡中"

    def mark_frame(self) -> None:
        with self.lock:
            self.frame_count += 1

    def mark_update(
        self,
        box: tuple[float, float, float, float] | None,
        score: float | None = None,
    ) -> None:
        with self.lock:
            self.match_score = score
            if box is None:
                self.tracking = False
                self.recovering = True
                self.lost_count += 1
                self.recovery_count += 1
                self.recovery_score = None
                self.last_error = "対象を見失いました。画面内で再捕捉中"
            else:
                self.tracking = True
                self.recovering = False
                self.recovery_score = None
                self.bbox = box
                self.tracked_frame_count += 1
                self.last_error = "追跡中"

    def mark_recovery_score(self, score: float | None) -> None:
        with self.lock:
            if self.recovering:
                self.recovery_score = score

    def mark_recovered(
        self,
        box: tuple[float, float, float, float],
        score: float,
    ) -> None:
        with self.lock:
            self.tracking = True
            self.recovering = False
            self.bbox = box
            self.recovery_success_count += 1
            self.recovery_score = score
            self.match_score = score
            self.tracked_frame_count += 1
            self.last_error = f"画面内で再捕捉しました score={score:.2f}"

    def status(self) -> dict[str, Any]:
        with self.lock:
            elapsed = (
                max(0.001, time.monotonic() - self.tracking_started_at)
                if self.tracking_started_at is not None
                else 0.0
            )
            return {
                "tracker": self.tracker_name.upper(),
                "roi": self.roi,
                "bbox": self.bbox,
                "tracking": self.tracking,
                "recovering": self.recovering,
                "lost_count": self.lost_count,
                "recovery_count": self.recovery_count,
                "recovery_success_count": self.recovery_success_count,
                "recovery_score": self.recovery_score,
                "match_score": self.match_score,
                "tracking_fps": self.tracked_frame_count / elapsed if elapsed else 0.0,
                "message": self.last_error,
            }


def create_tracker(name: str) -> Any:
    """OpenCV 4.xの通常API/legacy API両方を探す。"""
    normalized = name.lower()
    class_name = {"kcf": "TrackerKCF", "mosse": "TrackerMOSSE", "csrt": "TrackerCSRT"}.get(normalized)
    if class_name is None:
        raise ValueError(f"未対応のトラッカーです: {name}")

    modules = [getattr(cv2, "legacy", None), cv2]
    candidates = []
    for module in modules:
        if module is None:
            continue
        candidates.extend(
            [
                getattr(module, f"{class_name}_create", None),
                getattr(getattr(module, class_name, None), "create", None),
            ]
        )
    for factory in candidates:
        if callable(factory):
            return factory()
    raise RuntimeError(
        f"OpenCVの{class_name}が見つかりません。"
        "python3-opencvのtracking/legacy APIを確認してください。"
    )


def norm_to_pixels(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    left = max(0, min(width - 2, round(x * width)))
    top = max(0, min(height - 2, round(y * height)))
    right = max(left + 1, min(width - 1, round((x + w) * width)))
    bottom = max(top + 1, min(height - 1, round((y + h) * height)))
    return left, top, right - left, bottom - top


def pixels_to_norm(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = box
    left = max(0.0, min(float(width - 1), x))
    top = max(0.0, min(float(height - 1), y))
    right = max(left + 1.0, min(float(width), x + w))
    bottom = max(top + 1.0, min(float(height), y + h))
    if right <= left or bottom <= top:
        return None
    return left / width, top / height, (right - left) / width, (bottom - top) / height


def recovery_boxes_match(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """連続フレームの再捕捉候補が同じ場所・大きさか確認する。"""
    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    first_center = (first_x + first_w / 2.0, first_y + first_h / 2.0)
    second_center = (second_x + second_w / 2.0, second_y + second_h / 2.0)
    center_distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    width_ratio = second_w / max(0.001, first_w)
    height_ratio = second_h / max(0.001, first_h)
    return (
        center_distance <= 0.06
        and 0.50 <= width_ratio <= 2.0
        and 0.50 <= height_ratio <= 2.0
    )


class TemplateRecovery:
    """KCFが失敗したとき、現在フレーム内のテンプレート位置を探す。"""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.template: Any | None = None
        self.template_color: Any | None = None
        self.last_score: float | None = None

    @staticmethod
    def _gray(frame: Any) -> Any:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _bgr(frame: Any) -> Any:
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def set_template(
        self,
        frame: Any,
        box: tuple[float, float, float, float],
    ) -> bool:
        color = self._bgr(frame)
        gray = self._gray(color)
        left, top, width, height = norm_to_pixels(box, gray.shape[1], gray.shape[0])
        crop = gray[top:top + height, left:left + width]
        color_crop = color[top:top + height, left:left + width]
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            self.template = None
            self.template_color = None
            self.last_score = None
            return False
        self.template = crop.copy()
        self.template_color = color_crop.copy()
        self.last_score = None
        return True

    def score_box(
        self,
        frame: Any,
        box: tuple[float, float, float, float],
    ) -> float | None:
        """KCFが返した枠が、保持中のテンプレートと似ているか評価する。"""
        if self.template is None or self.template_color is None:
            return None
        color = self._bgr(frame)
        gray = self._gray(color)
        left, top, width, height = norm_to_pixels(box, gray.shape[1], gray.shape[0])
        crop = gray[top:top + height, left:left + width]
        color_crop = color[top:top + height, left:left + width]
        if crop.size == 0:
            return None
        template_height, template_width = self.template.shape[:2]
        resized = cv2.resize(crop, (template_width, template_height), interpolation=cv2.INTER_AREA)
        resized_color = cv2.resize(
            color_crop,
            (template_width, template_height),
            interpolation=cv2.INTER_AREA,
        )
        result = cv2.matchTemplate(resized, self.template, cv2.TM_CCOEFF_NORMED)
        correlation_score = float(result[0, 0])
        if not math.isfinite(correlation_score):
            return None

        _, template_std = cv2.meanStdDev(self.template)
        _, crop_std = cv2.meanStdDev(resized)
        template_std_value = float(template_std[0, 0])
        crop_std_value = float(crop_std[0, 0])
        if template_std_value < 1.0 or crop_std_value < 1.0:
            return None
        variance_ratio = crop_std_value / template_std_value
        variance_score = min(1.0, variance_ratio, 1.0 / variance_ratio)
        if variance_score < 0.35:
            return None

        template_normalized = cv2.normalize(self.template, None, 0, 255, cv2.NORM_MINMAX)
        crop_normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        difference = cv2.absdiff(template_normalized, crop_normalized)
        structure_score = max(0.0, 1.0 - float(cv2.mean(difference)[0]) / 255.0)
        color_scores = []
        for channel in range(3):
            color_result = cv2.matchTemplate(
                resized_color[:, :, channel],
                self.template_color[:, :, channel],
                cv2.TM_CCOEFF_NORMED,
            )
            color_score = float(color_result[0, 0])
            if not math.isfinite(color_score):
                return None
            color_scores.append(color_score)
        color_score = min(color_scores)
        score = min(correlation_score, variance_score, structure_score, color_score)
        return score if math.isfinite(score) else None

    def find(self, frame: Any) -> tuple[tuple[float, float, float, float], float] | None:
        self.last_score = None
        if self.template is None or self.template_color is None:
            return None
        color_frame = self._bgr(frame)
        gray = self._gray(color_frame)
        frame_height, frame_width = gray.shape[:2]
        base_height, base_width = self.template.shape[:2]
        search_width = min(frame_width, RECOVERY_SEARCH_WIDTH)
        search_scale = search_width / frame_width
        search_height = max(1, round(frame_height * search_scale))
        search_gray = cv2.resize(gray, (search_width, search_height), interpolation=cv2.INTER_AREA)
        best: tuple[float, int, int, int, int] | None = None

        # 白黒で候補を絞り、上位候補だけをカラー・構造検証する。
        # RGB全画面検索を毎回行うとPi Zero 2では再捕捉が遅くなる。
        for scale in RECOVERY_SCALES:
            template_width = max(8, round(base_width * scale * search_scale))
            template_height = max(8, round(base_height * scale * search_scale))
            if template_width >= search_width or template_height >= search_height:
                continue
            template = cv2.resize(
                self.template,
                (template_width, template_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
            gray_result = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
            candidates = gray_result.copy()
            for _ in range(RECOVERY_CANDIDATES_PER_SCALE):
                _, raw_score, _, location = cv2.minMaxLoc(candidates)
                raw_score = float(raw_score)
                if not math.isfinite(raw_score) or raw_score < max(0.2, self.threshold * 0.5):
                    break
                full_left = round(location[0] / search_scale)
                full_top = round(location[1] / search_scale)
                full_width = max(12, round(template_width / search_scale))
                full_height = max(12, round(template_height / search_scale))
                box = (
                    full_left / frame_width,
                    full_top / frame_height,
                    full_width / frame_width,
                    full_height / frame_height,
                )
                score = self.score_box(frame, box)
                if score is not None and (best is None or score > best[0]):
                    best = (score, full_left, full_top, full_width, full_height)
                cv2.rectangle(
                    candidates,
                    location,
                    (location[0] + template_width, location[1] + template_height),
                    -1,
                    -1,
                )

        if best is None:
            return None
        score, left, top, template_width, template_height = best
        self.last_score = score
        if score < self.threshold:
            return None
        box = (
            left / frame_width,
            top / frame_height,
            template_width / frame_width,
            template_height / frame_height,
        )
        return box, score


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("サイズは640x480形式で指定してください") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")
    return width, height


class JapaneseDefaultsHelpFormatter(argparse.HelpFormatter):
    """引数の意味と既定値を日本語で表示する。"""

    @staticmethod
    def _format_default(value: Any) -> str:
        if isinstance(value, tuple) and len(value) == 2:
            return f"{value[0]}x{value[1]}"
        if isinstance(value, bool):
            return "有効" if value else "無効"
        return str(value)

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.help is None or action.default is argparse.SUPPRESS:
            return action.help
        return f"{action.help}（既定値: {self._format_default(action.default)}）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=JapaneseDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size", type=parse_size, default=DEFAULT_SIZE, metavar="WIDTHxHEIGHT",
        help="PCへ配信するカメラ映像の解像度",
    )
    parser.add_argument(
        "--track-size", type=parse_size, default=DEFAULT_TRACK_SIZE, metavar="WIDTHxHEIGHT",
        help="KCF追跡・画面内再捕捉に使う縮小映像の解像度",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="カメラ撮影フレームレート")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP配信ポート")
    parser.add_argument(
        "--tracker", choices=("kcf", "mosse", "csrt"), default=DEFAULT_TRACKER,
        help="使用するOpenCVトラッカー",
    )
    parser.add_argument("--check-tracker", action="store_true", help="トラッカーAPIだけ確認して終了")
    parser.add_argument("--pan-min", type=int, default=DEFAULT_MIN_ANGLE, help="PAN最小角度")
    parser.add_argument("--pan-max", type=int, default=DEFAULT_MAX_ANGLE, help="PAN最大角度")
    parser.add_argument("--tilt-min", type=int, default=DEFAULT_MIN_ANGLE, help="TILT最小角度")
    parser.add_argument("--tilt-max", type=int, default=DEFAULT_MAX_ANGLE, help="TILT最大角度")
    parser.add_argument("--center", type=int, default=DEFAULT_CENTER, help="起動時・終了時の中央角度")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP, help="1回のサーボ更新で使う最大角度")
    parser.add_argument(
        "--control-interval-ms", type=int, default=DEFAULT_CONTROL_INTERVAL_MS,
        help="サーボ制御の更新間隔（ミリ秒）",
    )
    parser.add_argument("--deadband", type=float, default=DEFAULT_DEADBAND, help="画面中心とみなして停止する誤差")
    parser.add_argument(
        "--tracking-threshold",
        type=float,
        default=DEFAULT_TRACKING_THRESHOLD,
        help="KCF結果を追跡中と認めるテンプレート一致度",
    )
    parser.add_argument(
        "--recovery-threshold",
        type=float,
        default=DEFAULT_RECOVERY_THRESHOLD,
        help="画面内再捕捉に必要なテンプレート一致度",
    )
    parser.add_argument(
        "--recovery-confirm-frames",
        type=int,
        default=DEFAULT_RECOVERY_CONFIRM_FRAMES,
        help="再捕捉候補を連続確認するフレーム数",
    )
    parser.add_argument(
        "--invert-pan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PAN方向を反転する（解除は--no-invert-pan）",
    )
    parser.add_argument(
        "--invert-tilt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="TILT方向を反転する（有効化は--invert-tilt）",
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="自動終了までの秒数。0は無期限")
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fpsは0より大きくしてください")
    if args.track_size[0] > args.size[0] or args.track_size[1] > args.size[1]:
        parser.error("--track-sizeは--size以下にしてください")
    if not 1 <= args.port <= 65535:
        parser.error("--portは1から65535で指定してください")
    if not 0.0 <= args.deadband < 0.5:
        parser.error("--deadbandは0以上0.5未満で指定してください")
    if not 0.0 <= args.recovery_threshold <= 1.0:
        parser.error("--recovery-thresholdは0以上1以下で指定してください")
    if not 1 <= args.recovery_confirm_frames <= 10:
        parser.error("--recovery-confirm-framesは1から10で指定してください")
    if not 0.0 <= args.tracking_threshold <= 1.0:
        parser.error("--tracking-thresholdは0以上1以下で指定してください")
    if not 1 <= args.step <= 10:
        parser.error("--stepは1から10で指定してください")
    if not 20 <= args.control_interval_ms <= 500:
        parser.error("--control-interval-msは20から500で指定してください")
    if args.duration < 0:
        parser.error("--durationは0以上で指定してください")
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
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.check_tracker:
        tracker = create_tracker(args.tracker)
        print(f"Tracker API OK: {args.tracker.upper()} ({type(tracker).__name__})", flush=True)
        return 0
    width, height = args.size
    track_width, track_height = args.track_size

    controller: ServoController | None = None
    picam2: Picamera2 | None = None
    http_server: TrackingServer | None = None
    output = StreamingOutput()
    target_output = TargetPreviewOutput()
    recovery = TemplateRecovery(args.recovery_threshold)
    recovery_candidate: tuple[float, float, float, float] | None = None
    recovery_candidate_frames = 0
    recording_started = False

    try:
        controller = ServoController(
            args.pan_min, args.pan_max, args.tilt_min, args.tilt_max,
            args.center, args.step, args.control_interval_ms, args.deadband,
            args.invert_pan, args.invert_tilt,
        )
        tracking_state = TrackingState(args.tracker, track_width, track_height)
        picam2 = Picamera2()
        model = picam2.camera_properties.get("Model", "unknown")
        config = picam2.create_video_configuration(
            main={"size": args.size, "format": "RGB888"},
            controls={"FrameRate": args.fps},
            buffer_count=4,
        )
        picam2.configure(config)
        http_server = TrackingServer(
            ("0.0.0.0", args.port), output, target_output,
            tracking_state, controller, width, height
        )
        encoder = MJPEGEncoder()
        picam2.start_recording(encoder, FileOutput(output))
        recording_started = True

        print(f"Camera model : {model}", flush=True)
        print(f"Stream       : {width}x{height} @ {args.fps:.1f} fps", flush=True)
        print(f"Track input  : {track_width}x{track_height} RGB888 resized from main", flush=True)
        print(f"Tracker      : {args.tracker.upper()}", flush=True)
        print(f"Recovery     : template match / threshold {args.recovery_threshold:.2f}", flush=True)
        print(f"Recovery confirm: {args.recovery_confirm_frames} frames", flush=True)
        print(f"Validation   : KCF match / threshold {args.tracking_threshold:.2f}", flush=True)
        print(f"PAN          : GPIO{PAN_GPIO} / pin {PAN_PHYSICAL_PIN} / {args.pan_min}-{args.pan_max} deg", flush=True)
        print(f"TILT         : GPIO{TILT_GPIO} / pin {TILT_PHYSICAL_PIN} / {args.tilt_min}-{args.tilt_max} deg", flush=True)
        print(f"GPIO backend : {controller.gpio_backend}", flush=True)
        print(f"Open in PC   : http://raspberrypi.local:{args.port}/", flush=True)
        print("Select       : PCブラウザー上で対象をドラッグ", flush=True)
        print("Stop         : Ctrl+C", flush=True)

        server_thread = threading.Thread(
            target=http_server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        server_thread.start()

        time.sleep(2.0)
        started_at = time.monotonic()
        next_report_at = started_at + 1.0
        deadline = started_at + args.duration if args.duration > 0 else None
        tracker: Any | None = None

        while deadline is None or time.monotonic() < deadline:
            main_frame = picam2.capture_array("main")
            frame = cv2.resize(
                main_frame,
                (track_width, track_height),
                interpolation=cv2.INTER_AREA,
            )
            tracking_state.mark_frame()

            pending = tracking_state.take_pending_roi()
            if pending is not None:
                recovery_candidate = None
                recovery_candidate_frames = 0
                tracker = create_tracker(args.tracker)
                pixel_box = norm_to_pixels(pending, track_width, track_height)
                initialized = tracker.init(frame, pixel_box)
                if initialized is False:
                    tracker = None
                    tracking_state.mark_update(None)
                    raise RuntimeError("トラッカーの初期化に失敗しました")
                tracking_state.mark_initialized(pending)
                target_output.set_selection(frame, pending)
                if not recovery.set_template(frame, pending):
                    logging.warning("ROI is too small for template recovery")
                controller.stop_motion()
                logging.info("ROI initialized: %s", pending)

            if tracker is not None:
                ok, updated_box = tracker.update(frame)
                candidate_box = pixels_to_norm(updated_box, track_width, track_height) if ok else None
                match_score = recovery.score_box(frame, candidate_box) if candidate_box is not None else None
                normalized_box = (
                    candidate_box
                    if match_score is not None and match_score >= args.tracking_threshold
                    else None
                )
                tracking_state.mark_update(normalized_box, match_score)
                if normalized_box is None:
                    controller.stop_motion()
                    recovered = recovery.find(frame)
                    if recovered is not None:
                        recovered_box, score = recovered
                        if recovery_candidate is not None and recovery_boxes_match(
                            recovery_candidate,
                            recovered_box,
                        ):
                            recovery_candidate_frames += 1
                        else:
                            recovery_candidate = recovered_box
                            recovery_candidate_frames = 1
                        tracking_state.mark_recovery_score(score)
                        if recovery_candidate_frames >= args.recovery_confirm_frames:
                            recovered_tracker = create_tracker(args.tracker)
                            recovered_initialized = recovered_tracker.init(
                                frame,
                                norm_to_pixels(recovered_box, track_width, track_height),
                            )
                            if recovered_initialized is not False:
                                tracker = recovered_tracker
                                tracking_state.mark_recovered(recovered_box, score)
                                controller.stop_motion()
                                recovery_candidate = None
                                recovery_candidate_frames = 0
                                logging.info(
                                    "ROI recovered after %d frames: box=%s score=%.2f",
                                    args.recovery_confirm_frames,
                                    recovered_box,
                                    score,
                                )
                    else:
                        recovery_candidate = None
                        recovery_candidate_frames = 0
                        tracking_state.mark_recovery_score(recovery.last_score)
                else:
                    recovery_candidate = None
                    recovery_candidate_frames = 0
                    x, y, w, h = normalized_box
                    center_x = x + w / 2.0
                    center_y = y + h / 2.0
                    controller.set_tracking_error(center_x - 0.5, center_y - 0.5)

            now = time.monotonic()
            if now >= next_report_at:
                elapsed = now - started_at
                status = tracking_state.status()
                fps = status["tracking_fps"]
                print(
                    f"STATUS elapsed={elapsed:.1f}s frames={tracking_state.frame_count} "
                    f"tracking_fps={fps:.2f} tracking={status['tracking']} "
                    f"lost={status['lost_count']}",
                    flush=True,
                )
                next_report_at = now + 1.0
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        if http_server is not None:
            http_server.shutdown()
            http_server.server_close()
        output.close()
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
