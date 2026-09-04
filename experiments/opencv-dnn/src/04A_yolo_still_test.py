#!/usr/bin/env python3
"""Phase 4A: run a YOLO-style ONNX model on one saved image.

This baseline deliberately keeps camera, servo, and streaming out of the
test.  It uses OpenCV DNN on the Pi CPU and accepts the common YOLOv5-style
output with objectness as well as the YOLOv8-style output without it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL = Path("models/yolo.onnx")
DEFAULT_LABELS = Path("models/coco.names")
DEFAULT_IMAGE = Path("captures/phase1a_ov5647.jpg")
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS = 0.45


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
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="入力画像")
    parser.add_argument(
        "--input-size",
        type=parse_size,
        default=DEFAULT_INPUT_SIZE,
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS)
    parser.add_argument("--warmup", type=int, default=1, help="計測前の推論回数")
    parser.add_argument("--iterations", type=int, default=1, help="計測する推論回数")
    parser.add_argument("--json", type=Path, default=None, help="結果JSONの保存先")
    args = parser.parse_args()

    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidenceは0より大きく1以下で指定してください")
    if not 0.0 < args.nms <= 1.0:
        parser.error("--nmsは0より大きく1以下で指定してください")
    if args.warmup < 0:
        parser.error("--warmupは0以上で指定してください")
    if args.iterations < 1:
        parser.error("--iterationsは1以上で指定してください")
    return args


def read_labels(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_output(output: np.ndarray) -> np.ndarray:
    """Return predictions as [number_of_candidates, number_of_features]."""
    predictions = np.asarray(output)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise RuntimeError(f"未対応のONNX出力形状です: {predictions.shape}")

    # YOLOv8 exports are commonly [features, candidates], while YOLOv5
    # exports are commonly [candidates, features].
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

    # 85 features is the common YOLOv5 layout: xywh + objectness + classes.
    # 84 features is the common YOLOv8 layout: xywh + classes.
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


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    image_path = args.image.expanduser().resolve()

    if not model_path.is_file():
        raise SystemExit(f"ONNXモデルがありません: {model_path}")
    if not image_path.is_file():
        raise SystemExit(f"入力画像がありません: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"画像を読み込めません: {image_path}")
    image_height, image_width = image.shape[:2]

    net = cv2.dnn.readNetFromONNX(str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=args.input_size,
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)

    for _ in range(args.warmup):
        net.forward()

    start = time.perf_counter()
    raw_output = None
    for _ in range(args.iterations):
        raw_output = net.forward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iterations
    if raw_output is None:
        raise RuntimeError("推論出力がありません")

    predictions = normalize_output(raw_output)
    labels = read_labels(labels_path)
    detections = decode_detections(
        predictions,
        (image_width, image_height),
        args.input_size,
        labels,
        args.confidence,
        args.nms,
    )
    result = {
        "model": str(model_path),
        "image": str(image_path),
        "image_size": {"width": image_width, "height": image_height},
        "input_size": {"width": args.input_size[0], "height": args.input_size[1]},
        "output_shape": list(predictions.shape),
        "inference_ms": round(elapsed_ms, 3),
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms,
        "detections": detections,
    }

    print(f"Model       : {model_path}")
    print(f"Image       : {image_path}")
    print(f"Image size  : {image_width}x{image_height}")
    print(f"Output shape: {tuple(predictions.shape)}")
    print(f"Inference   : {elapsed_ms:.3f} ms")
    print(f"Detections  : {len(detections)}")
    for detection in detections:
        print(
            f"  {detection['label']}: {detection['confidence']:.3f} "
            f"box={detection['box']} center={detection['center']}"
        )

    if args.json is not None:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"JSON        : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
