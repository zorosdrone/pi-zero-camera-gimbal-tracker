#!/usr/bin/env python3
"""Phase 4C: run MobileNet-SSD (Caffe/VOC) on one saved image.

This comparison test deliberately keeps the camera, servos, and streaming
stopped.  It uses the same OpenCV DNN CPU backend as Phase 4A, measures only
``net.forward()`` as inference time, saves detections to JSON, and writes an
annotated image for checking the bounding boxes on a PC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CONFIG = Path("models/mobilenet-ssd.prototxt")
DEFAULT_MODEL = Path("models/mobilenet-ssd.caffemodel")
DEFAULT_LABELS = Path("models/voc.names")
DEFAULT_IMAGE = Path("captures/phase1a_ov5647.jpg")
DEFAULT_OUTPUT_IMAGE = Path("captures/phase4c_mobilenet_ssd.jpg")
DEFAULT_INPUT_SIZE = (300, 300)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_THREADS = 4
SCALE_FACTOR = 1.0 / 127.5
MEAN_VALUE = (127.5, 127.5, 127.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Caffe prototxt")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Caffe学習済みモデル")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="VOCクラス名ファイル")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="入力画像")
    parser.add_argument(
        "--output-image",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE,
        help="検知枠付き画像の保存先",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--warmup", type=int, default=1, help="計測前の推論回数")
    parser.add_argument("--iterations", type=int, default=3, help="計測する推論回数")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="OpenCV CPUスレッド数")
    parser.add_argument("--json", type=Path, default=None, help="結果JSONの保存先")
    args = parser.parse_args()

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidenceは0より大きく1以下で指定してください")
    if args.warmup < 0:
        parser.error("--warmupは0以上で指定してください")
    if args.iterations < 1:
        parser.error("--iterationsは1以上で指定してください")
    if args.threads < 1:
        parser.error("--threadsは1以上で指定してください")
    return args


def read_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"クラス名ファイルがありません: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 21 or labels[0] != "background":
        raise SystemExit("VOCクラス名はbackgroundを含む21行で指定してください")
    return labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_detections(
    raw_output: np.ndarray,
    image_size: tuple[int, int],
    labels: list[str],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Decode Caffe DetectionOutput rows: image_id, class, score, x1, y1, x2, y2."""
    output = np.asarray(raw_output)
    if output.size % 7 != 0:
        raise RuntimeError(f"未対応のMobileNet-SSD出力形状です: {output.shape}")
    rows = output.reshape(-1, 7)
    image_width, image_height = image_size
    detections: list[dict[str, Any]] = []

    for row in rows:
        class_id = int(row[1])
        confidence = float(row[2])
        if confidence < confidence_threshold or not 0 < class_id < len(labels):
            continue

        x1 = int(round(float(row[3]) * image_width))
        y1 = int(round(float(row[4]) * image_height))
        x2 = int(round(float(row[5]) * image_width))
        y2 = int(round(float(row[6]) * image_height))
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        width = x2 - x1
        height = y2 - y1
        detections.append(
            {
                "class_id": class_id,
                "label": labels[class_id],
                "confidence": round(confidence, 6),
                "box": {"x": x1, "y": y1, "width": width, "height": height},
                "center": {"x": x1 + width // 2, "y": y1 + height // 2},
            }
        )
    return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        box = detection["box"]
        center = detection["center"]
        x1, y1 = box["x"], box["y"]
        x2, y2 = x1 + box["width"], y1 + box["height"]
        color = (0, 220, 0) if detection["label"] == "person" else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(
            annotated,
            (center["x"], center["y"]),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )
        text = f"{detection['label']} {detection['confidence']:.2f}"
        text_y = max(18, y1 - 6)
        cv2.putText(annotated, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_image_path = args.output_image.expanduser().resolve()

    if not config_path.is_file():
        raise SystemExit(f"Caffe設定ファイルがありません: {config_path}")
    if not model_path.is_file():
        raise SystemExit(f"Caffeモデルがありません: {model_path}")
    if not image_path.is_file():
        raise SystemExit(f"入力画像がありません: {image_path}")

    labels = read_labels(labels_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"画像を読み込めません: {image_path}")
    image_height, image_width = image.shape[:2]

    cv2.setNumThreads(args.threads)
    net = cv2.dnn.readNetFromCaffe(str(config_path), str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    preprocess_start = time.perf_counter()
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=SCALE_FACTOR,
        size=DEFAULT_INPUT_SIZE,
        mean=MEAN_VALUE,
        swapRB=False,
        crop=False,
    )
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    net.setInput(blob)

    for _ in range(args.warmup):
        net.forward()

    inference_times_ms: list[float] = []
    raw_output = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        raw_output = net.forward()
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)
    if raw_output is None:
        raise RuntimeError("推論出力がありません")

    postprocess_start = time.perf_counter()
    detections = decode_detections(
        raw_output,
        (image_width, image_height),
        labels,
        args.confidence,
    )
    annotated = draw_detections(image, detections)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_image_path), annotated):
        raise RuntimeError(f"検知枠付き画像を保存できません: {output_image_path}")

    average_ms = sum(inference_times_ms) / len(inference_times_ms)
    result = {
        "framework": "Caffe via OpenCV DNN",
        "model_family": "MobileNet-SSD",
        "dataset": "VOC0712",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "image": str(image_path),
        "output_image": str(output_image_path),
        "image_size": {"width": image_width, "height": image_height},
        "input_size": {"width": DEFAULT_INPUT_SIZE[0], "height": DEFAULT_INPUT_SIZE[1]},
        "output_shape": list(np.asarray(raw_output).shape),
        "opencv_threads": cv2.getNumThreads(),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "preprocess_ms": round(preprocess_ms, 3),
        "inference_times_ms": [round(value, 3) for value in inference_times_ms],
        "inference_average_ms": round(average_ms, 3),
        "inference_min_ms": round(min(inference_times_ms), 3),
        "inference_max_ms": round(max(inference_times_ms), 3),
        "estimated_inference_fps": round(1000.0 / average_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "confidence_threshold": args.confidence,
        "detections": detections,
    }

    print(f"Model       : {model_path}")
    print(f"Image       : {image_path}")
    print(f"Image size  : {image_width}x{image_height}")
    print(f"Input size  : {DEFAULT_INPUT_SIZE[0]}x{DEFAULT_INPUT_SIZE[1]}")
    print(f"Output shape: {tuple(np.asarray(raw_output).shape)}")
    print(f"Threads     : {cv2.getNumThreads()}")
    print(f"Inference   : {average_ms:.3f} ms average ({1000.0 / average_ms:.3f} fps)")
    print(f"Range       : {min(inference_times_ms):.3f} - {max(inference_times_ms):.3f} ms")
    print(f"Detections  : {len(detections)}")
    for detection in detections:
        print(
            f"  {detection['label']}: {detection['confidence']:.3f} "
            f"box={detection['box']} center={detection['center']}"
        )
    print(f"Output image: {output_image_path}")

    if args.json is not None:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON        : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
