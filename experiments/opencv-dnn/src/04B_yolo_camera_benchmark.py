#!/usr/bin/env python3
"""Phase 4B: Picamera2の低解像度映像をYOLOへ連続入力するベンチマーク。

カメラ取得と推論を別処理に分け、推論側には常に最新フレームだけを渡す。
推論中に古いフレームをキューへため込まないため、後続の追跡処理で問題に
なりやすい遅延の増加を避けられる。検知枠の描画とサーボ制御はまだ行わない。

Pi側はOpenCV DNN + ONNXだけを使用する。Picamera2のloresはYUV420で取得し、
OpenCVでRGBへ変換してからYOLOへ渡す。カメラ配列はRGB変換済みのため
blob作成時はswapRB=Falseとする。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from picamera2 import Picamera2


DEFAULT_MODEL = Path("models/yolo.onnx")
DEFAULT_LABELS = Path("models/coco.names")
DEFAULT_MAIN_SIZE = (640, 480)
DEFAULT_LORES_SIZE = (320, 240)
DEFAULT_CAMERA_FPS = 15.0
DEFAULT_DURATION = 30.0
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS = 0.45
DEFAULT_WARMUP = 2.0


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("サイズはWIDTHxHEIGHTで指定してください") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO ONNXモデル")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="クラス名ファイル")
    parser.add_argument(
        "--main-size",
        type=parse_size,
        default=DEFAULT_MAIN_SIZE,
        metavar="WIDTHxHEIGHT",
        help="配信用候補のmainサイズ。既定値: 640x480",
    )
    parser.add_argument(
        "--lores-size",
        type=parse_size,
        default=DEFAULT_LORES_SIZE,
        metavar="WIDTHxHEIGHT",
        help="YOLOへ渡すloresサイズ。Picamera2はYUV420。既定値: 320x240",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help="カメラの設定fps。既定値: 15",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="ベンチマーク時間（秒）。既定値: 30",
    )
    parser.add_argument(
        "--input-size",
        type=parse_size,
        default=DEFAULT_INPUT_SIZE,
        metavar="WIDTHxHEIGHT",
        help="YOLO入力サイズ。既定値: 640x640",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS)
    parser.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP,
        help="自動露出を安定させる待ち時間。推論計測前。既定値: 2",
    )
    parser.add_argument("--json", type=Path, default=None, help="結果JSONの保存先")
    args = parser.parse_args()

    if args.camera_fps <= 0:
        parser.error("--camera-fpsは0より大きくしてください")
    if args.duration <= 0:
        parser.error("--durationは0より大きくしてください")
    if args.warmup < 0:
        parser.error("--warmupは0以上で指定してください")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidenceは0より大きく1以下で指定してください")
    if not 0.0 < args.nms <= 1.0:
        parser.error("--nmsは0より大きく1以下で指定してください")
    return args


def read_labels(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_output(output: np.ndarray) -> np.ndarray:
    """YOLO出力を[候補数, 特徴量数]へそろえる。"""
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise RuntimeError(f"未対応のONNX出力形状です: {predictions.shape}")

    # YOLOv8系は[特徴量数, 候補数]、YOLOv5系は[候補数, 特徴量数]が一般的。
    if predictions.shape[0] <= 256 and predictions.shape[1] > predictions.shape[0]:
        predictions = predictions.T
    if predictions.shape[1] < 6:
        raise RuntimeError(f"検知出力の特徴量数が少なすぎます: {predictions.shape}")
    return predictions


def decode_detections(
    predictions: np.ndarray,
    image_size: tuple[int, int],
    input_size: tuple[int, int],
    labels: list[str],
    confidence_threshold: float,
    nms_threshold: float,
) -> list[dict[str, Any]]:
    image_width, image_height = image_size
    input_width, input_height = input_size
    boxes: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    has_objectness = predictions.shape[1] >= 85
    for row in predictions:
        box_values = row[:4]
        class_scores = row[5:] if has_objectness else row[4:]
        if not class_scores.size:
            continue

        class_id = int(np.argmax(class_scores))
        class_confidence = float(class_scores[class_id])
        objectness = float(row[4]) if has_objectness else 1.0
        confidence = class_confidence * objectness if has_objectness else class_confidence
        if confidence < confidence_threshold:
            continue

        center_x, center_y, width, height = [float(value) for value in box_values]
        scale_x = image_width / input_width
        scale_y = image_height / input_height
        left = int((center_x - width / 2.0) * scale_x)
        top = int((center_y - height / 2.0) * scale_y)
        box_width = int(width * scale_x)
        box_height = int(height * scale_y)
        boxes.append([left, top, box_width, box_height])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    kept_indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)
    kept = np.asarray(kept_indices).reshape(-1).tolist() if len(kept_indices) else []
    detections: list[dict[str, Any]] = []
    for index in kept:
        left, top, width, height = boxes[index]
        class_id = class_ids[index]
        detections.append(
            {
                "class_id": class_id,
                "label": labels[class_id] if class_id < len(labels) else f"class_{class_id}",
                "confidence": round(float(scores[index]), 6),
                "box": {"x": left, "y": top, "width": width, "height": height},
                "center": {"x": left + width // 2, "y": top + height // 2},
            }
        )
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def read_cpu_times() -> tuple[int, int] | None:
    """Linux全体のCPU busy/total jiffiesを読む。"""
    try:
        first_line = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0]
        values = [int(value) for value in first_line.split()[1:]]
    except (FileNotFoundError, IndexError, ValueError):
        return None
    if len(values) < 5:
        return None
    idle = values[3]
    iowait = values[4]
    total = sum(values)
    return total - idle - iowait, total


def read_memory_snapshot() -> dict[str, int] | None:
    """自身のRSSとシステムのMemAvailableをbytesで読む。"""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                values["process_rss_bytes"] = int(line.split()[1]) * 1024
                break
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                values["mem_available_bytes"] = int(line.split()[1]) * 1024
                break
    except (FileNotFoundError, IndexError, ValueError):
        return None
    return values or None


def read_temperature_c() -> float | None:
    try:
        value = Path("/sys/class/thermal/thermal_zone0/temp").read_text(encoding="ascii").strip()
        return round(int(value) / 1000.0, 1)
    except (FileNotFoundError, ValueError):
        return None


def read_throttled() -> str | None:
    try:
        completed = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None


class LatestFrame:
    """最新フレームだけを保持する1要素キュー。"""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.captured_at: float | None = None
        self.sequence = 0

    def put(self, frame: np.ndarray, captured_at: float) -> None:
        with self.condition:
            self.frame = frame
            self.captured_at = captured_at
            self.sequence += 1
            self.condition.notify_all()

    def get_newer(self, last_sequence: int, timeout: float) -> tuple[np.ndarray, int, float] | None:
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > last_sequence, timeout=timeout)
            if self.frame is None or self.captured_at is None or self.sequence <= last_sequence:
                return None
            return self.frame, self.sequence, self.captured_at


def run_inference(
    net: cv2.dnn.Net,
    frame_rgb: np.ndarray,
    input_size: tuple[int, int],
    labels: list[str],
    confidence_threshold: float,
    nms_threshold: float,
) -> tuple[list[dict[str, Any]], tuple[int, ...], float]:
    image_height, image_width = frame_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_rgb,
        scalefactor=1.0 / 255.0,
        size=input_size,
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    start = time.perf_counter()
    raw_output = net.forward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    predictions = normalize_output(raw_output)
    detections = decode_detections(
        predictions,
        (image_width, image_height),
        input_size,
        labels,
        confidence_threshold,
        nms_threshold,
    )
    return detections, tuple(predictions.shape), elapsed_ms


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"ONNXモデルがありません: {model_path}")

    labels = read_labels(labels_path)
    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    picam2 = Picamera2()
    camera_model = picam2.camera_properties.get("Model", "unknown")
    config = picam2.create_video_configuration(
        main={"size": args.main_size, "format": "RGB888"},
        lores={"size": args.lores_size, "format": "YUV420"},
        controls={"FrameRate": args.camera_fps},
        buffer_count=4,
    )
    picam2.configure(config)

    latest = LatestFrame()
    stop_event = threading.Event()
    capture_errors: list[BaseException] = []

    def capture_loop() -> None:
        try:
            while not stop_event.is_set():
                yuv_frame = picam2.capture_array("lores")
                lores_height = args.lores_size[1]
                lores_width = args.lores_size[0]
                expected_height = lores_height * 3 // 2
                if yuv_frame.ndim != 2 or yuv_frame.shape[0] < expected_height or yuv_frame.shape[1] < lores_width:
                    raise RuntimeError(
                        f"YUV420フレーム形状ではありません: {tuple(yuv_frame.shape)} "
                        f"(expected at least {(expected_height, lores_width)})"
                    )
                yuv_frame = yuv_frame[:expected_height, :lores_width]
                frame_rgb = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV420p2RGB)
                latest.put(frame_rgb, time.monotonic())
        except BaseException as error:  # thread内の実機エラーをメインへ渡す
            capture_errors.append(error)
            stop_event.set()

    print(f"Camera model : {camera_model}", flush=True)
    print(f"Main stream  : {args.main_size[0]}x{args.main_size[1]} RGB888", flush=True)
    print(f"YOLO stream  : {args.lores_size[0]}x{args.lores_size[1]} YUV420 -> RGB", flush=True)
    print(f"Camera fps   : {args.camera_fps:.2f}", flush=True)
    print(f"Input size   : {args.input_size[0]}x{args.input_size[1]}", flush=True)
    print(f"Duration     : {args.duration:.1f} s", flush=True)

    capture_thread: threading.Thread | None = None
    records: list[dict[str, Any]] = []
    frame_shape: tuple[int, ...] | None = None
    benchmark_start = None
    benchmark_end = None
    cpu_start = None
    process_cpu_start = None
    memory_start = None
    temperature_start = None
    last_sequence = 0

    try:
        picam2.start()
        time.sleep(args.warmup)

        benchmark_start = time.monotonic()
        cpu_start = read_cpu_times()
        process_cpu_start = time.process_time()
        memory_start = read_memory_snapshot()
        temperature_start = read_temperature_c()

        capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
        capture_thread.start()
        deadline = benchmark_start + args.duration

        while time.monotonic() < deadline and not stop_event.is_set():
            item = latest.get_newer(last_sequence, timeout=0.5)
            if item is None:
                continue
            frame_rgb, sequence, captured_at = item
            last_sequence = sequence
            if frame_shape is None:
                frame_shape = tuple(frame_rgb.shape)
                if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
                    raise RuntimeError(f"RGB888フレーム形状ではありません: {frame_shape}")

            age_ms = (time.monotonic() - captured_at) * 1000.0
            detections, output_shape, inference_ms = run_inference(
                net,
                frame_rgb,
                args.input_size,
                labels,
                args.confidence,
                args.nms,
            )
            record = {
                "sequence": sequence,
                "frame_age_at_inference_ms": round(age_ms, 3),
                "inference_ms": round(inference_ms, 3),
                "output_shape": list(output_shape),
                "detections": detections,
            }
            records.append(record)
            labels_text = ", ".join(
                f"{item['label']} {item['confidence']:.3f}" for item in detections
            ) or "none"
            print(
                f"Inference {len(records)}: {inference_ms:.1f} ms, "
                f"frame age {age_ms:.1f} ms, detections={labels_text}",
                flush=True,
            )
    finally:
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=5.0)
        benchmark_end = time.monotonic()
        try:
            picam2.stop()
        finally:
            picam2.close()

    if capture_errors:
        raise RuntimeError(f"カメラ取得スレッドでエラー: {capture_errors[0]}")
    if benchmark_start is None or benchmark_end is None:
        raise RuntimeError("ベンチマーク時間を取得できません")

    elapsed = benchmark_end - benchmark_start
    cpu_end = read_cpu_times()
    process_cpu_end = time.process_time()
    memory_end = read_memory_snapshot()
    temperature_end = read_temperature_c()
    captured_frames = latest.sequence
    capture_fps = captured_frames / elapsed if elapsed > 0 else 0.0
    inference_fps = len(records) / elapsed if elapsed > 0 else 0.0

    system_cpu_percent = None
    if cpu_start is not None and cpu_end is not None:
        busy_delta = cpu_end[0] - cpu_start[0]
        total_delta = cpu_end[1] - cpu_start[1]
        if total_delta > 0:
            system_cpu_percent = round(100.0 * busy_delta / total_delta, 2)

    process_cpu_percent = None
    if process_cpu_start is not None:
        process_cpu_percent = round(100.0 * (process_cpu_end - process_cpu_start) / elapsed, 2)

    result = {
        "model": str(model_path),
        "camera_model": camera_model,
        "main": {"size": list(args.main_size), "format": "RGB888"},
        "lores": {
            "size": list(args.lores_size),
            "format": "YUV420 -> RGB",
            "array_shape": list(frame_shape) if frame_shape is not None else None,
        },
        "camera_fps_target": args.camera_fps,
        "input_size": {"width": args.input_size[0], "height": args.input_size[1]},
        "duration_seconds": round(elapsed, 3),
        "captured_frames": captured_frames,
        "capture_fps": round(capture_fps, 3),
        "inference_count": len(records),
        "inference_fps": round(inference_fps, 3),
        "system_cpu_percent": system_cpu_percent,
        "process_cpu_percent": process_cpu_percent,
        "memory_start": memory_start,
        "memory_end": memory_end,
        "temperature_start_c": temperature_start,
        "temperature_end_c": temperature_end,
        "throttled": read_throttled(),
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms,
        "inferences": records,
    }

    print(f"Elapsed      : {elapsed:.3f} s", flush=True)
    print(f"Camera frames: {captured_frames} ({capture_fps:.2f} fps)", flush=True)
    print(f"Inferences   : {len(records)} ({inference_fps:.3f} fps)", flush=True)
    if system_cpu_percent is not None:
        print(f"System CPU   : {system_cpu_percent:.2f} %", flush=True)
    if process_cpu_percent is not None:
        print(f"Process CPU  : {process_cpu_percent:.2f} %", flush=True)
    if temperature_start is not None or temperature_end is not None:
        print(f"Temperature  : {temperature_start} -> {temperature_end} C", flush=True)
    if result["throttled"] is not None:
        print(f"Throttled    : {result['throttled']}", flush=True)

    if args.json is not None:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON         : {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
