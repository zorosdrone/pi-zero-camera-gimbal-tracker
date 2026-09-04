#!/usr/bin/env python3
"""Phase 6C: サンプル写真をEdge Impulse FOMOで推論する。

人物あり／人物なしの静止画を使い、ライブカメラやHTTP配信を介さずに
モデルの検出結果を確認する。元画像への枠付き画像と、FOMOが実際に見る
64x64モデル入力画像の両方を保存する。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
from edge_impulse_linux.image import ImageImpulseRunner


DEFAULT_MODEL = Path("~/src/fomo/models/person_fomo_pizero2_int8_v5.eim")
DEFAULT_INPUT_DIR = Path("~/src/fomo/sampleImg")
DEFAULT_OUTPUT_DIR = Path("~/src/fomo/results/fomo_still")
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--thresholdは0以上1以下で指定してください")
    return args


def get_model_features(runner: Any, image_rgb: Any) -> tuple[Any, Any]:
    """Edge Impulse Studioの画像リサイズ設定に合わせて入力を作る。"""
    studio_method = getattr(runner, "get_features_from_image_auto_studio_settings", None)
    if studio_method is not None:
        return studio_method(image_rgb)
    return runner.get_features_from_image(image_rgb)


def result_detections(result: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    detections = result.get("result", {}).get("bounding_boxes", [])
    return [
        item
        for item in detections
        if float(item.get("value", 0.0)) >= threshold
    ]


def map_box_to_image(
    detection: dict[str, Any],
    image_width: int,
    image_height: int,
    model_width: int,
    model_height: int,
    resize_mode: str,
) -> tuple[int, int, int, int] | None:
    """モデル入力座標を元画像座標へ戻す。"""
    x = float(detection.get("x", 0))
    y = float(detection.get("y", 0))
    width = float(detection.get("width", 0))
    height = float(detection.get("height", 0))

    if resize_mode == "fit-longest":
        scale = min(model_width / image_width, model_height / image_height)
        resized_width = image_width * scale
        resized_height = image_height * scale
        pad_x = (model_width - resized_width) / 2.0
        pad_y = (model_height - resized_height) / 2.0
        left = (x - pad_x) / scale
        top = (y - pad_y) / scale
        right = (x + width - pad_x) / scale
        bottom = (y + height - pad_y) / scale
    else:
        scale_x = image_width / model_width
        scale_y = image_height / model_height
        left = x * scale_x
        top = y * scale_y
        right = (x + width) * scale_x
        bottom = (y + height) * scale_y

    left = max(0, min(image_width - 1, round(left)))
    top = max(0, min(image_height - 1, round(top)))
    right = max(0, min(image_width - 1, round(right)))
    bottom = max(0, min(image_height - 1, round(bottom)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def draw_label(image: Any, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (x, max(18, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_original_overlay(
    image_rgb: Any,
    detections: list[dict[str, Any]],
    model_width: int,
    model_height: int,
    resize_mode: str,
) -> Any:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_height, image_width = image.shape[:2]

    for detection in detections:
        box = map_box_to_image(
            detection,
            image_width,
            image_height,
            model_width,
            model_height,
            resize_mode,
        )
        if box is None:
            continue
        left, top, right, bottom = box
        score = float(detection.get("value", 0.0))
        color = (0, 255, 0) if score >= 0.75 else (0, 165, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, 3)
        draw_label(image, f"{detection.get('label', 'unknown')} {score:.2f}", left, top, color)

    cv2.line(image, (image_width // 2, 0), (image_width // 2, image_height), (0, 215, 255), 2)
    cv2.line(image, (0, image_height // 2), (image_width, image_height // 2), (0, 215, 255), 2)
    return image


def draw_model_input_overlay(
    model_image_rgb: Any,
    detections: list[dict[str, Any]],
    display_size: int = 640,
) -> Any:
    """モデルが見た64x64画像へ、モデル座標のまま枠を描く。"""
    image = cv2.cvtColor(model_image_rgb, cv2.COLOR_RGB2BGR)
    model_height, model_width = image.shape[:2]
    for detection in detections:
        x = int(detection.get("x", 0))
        y = int(detection.get("y", 0))
        right = x + int(detection.get("width", 0))
        bottom = y + int(detection.get("height", 0))
        score = float(detection.get("value", 0.0))
        color = (0, 255, 0) if score >= 0.75 else (0, 165, 255)
        cv2.rectangle(image, (x, y), (right, bottom), color, 1)
        draw_label(image, f"{detection.get('label', 'unknown')} {score:.2f}", x, y, color)

    image = cv2.resize(image, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
    cv2.line(image, (display_size // 2, 0), (display_size // 2, display_size), (0, 215, 255), 2)
    cv2.line(image, (0, display_size // 2), (display_size, display_size // 2), (0, 215, 255), 2)
    return image


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise FileNotFoundError(f"画像がありません: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with ImageImpulseRunner(str(model_path)) as runner:
        model_info = runner.init()
        parameters = model_info.get("model_parameters", {})
        model_width = int(parameters.get("image_input_width", 64))
        model_height = int(parameters.get("image_input_height", 64))
        resize_mode = str(parameters.get("image_resize_mode", "squash"))
        print(f"Model       : {model_path}")
        print(f"Model input : {model_width}x{model_height} RGB")
        print(f"Resize mode : {resize_mode}")
        print(f"Threshold   : {args.threshold:.2f}")

        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"SKIP {image_path.name}: 読み込み失敗")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            started_at = time.monotonic()
            features, model_image = get_model_features(runner, image_rgb)
            result = runner.classify(features)
            wall_ms = (time.monotonic() - started_at) * 1000.0
            detections = result_detections(result, args.threshold)
            timing = result.get("timing", {})
            inference_ms = sum(
                float(timing.get(name, 0.0))
                for name in ("dsp", "classification", "anomaly")
            )

            original_output = output_dir / f"{image_path.stem}_fomo_overlay.jpg"
            model_output = output_dir / f"{image_path.stem}_fomo_model_input.jpg"
            cv2.imwrite(
                str(original_output),
                draw_original_overlay(
                    image_rgb,
                    detections,
                    model_width,
                    model_height,
                    resize_mode,
                ),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            cv2.imwrite(
                str(model_output),
                draw_model_input_overlay(model_image, detections),
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            print(
                f"{image_path.name}: detections={len(detections)} "
                f"inference={inference_ms:.1f}ms wall={wall_ms:.1f}ms"
            )
            for detection in detections:
                print(
                    f"  {detection.get('label', 'unknown')}="
                    f"{float(detection.get('value', 0.0)):.2f} "
                    f"({detection.get('x', 0)},{detection.get('y', 0)},"
                    f"{detection.get('width', 0)},{detection.get('height', 0)})"
                )
            print(f"  original: {original_output}")
            print(f"  model   : {model_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
