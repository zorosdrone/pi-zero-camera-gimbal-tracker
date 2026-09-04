#!/usr/bin/env python3
"""Phase 6A: Picamera2の映像をEdge Impulse FOMOへ入力する。

最初のライブ推論では、検出結果をコンソールへ表示する。
検出枠の映像合成、PC配信、サーボ制御は後続の段階で追加する。

想定環境:
  Raspberry Pi Zero 2 W / 64-bit aarch64
  Picamera2
  Edge Impulse Linux Python SDK
  Linux AARCH64用の.eimモデル
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from edge_impulse_linux.image import ImageImpulseRunner
from picamera2 import Picamera2


DEFAULT_MODEL = Path("~/src/fomo/models/person_fomo_pizero2_int8_v5.eim")
DEFAULT_SIZE = (640, 480)
DEFAULT_CAMERA_FPS = 15.0
DEFAULT_DURATION = 30.0
DEFAULT_WARMUP = 2.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_REPORT_INTERVAL = 1.0


def parse_size(value: str) -> tuple[int, int]:
    """WIDTHxHEIGHT形式を正の整数2個へ変換する。"""
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
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Linux AARCH64用.eimモデル。既定値: ~/src/fomo/models/person_fomo_pizero2_int8_v5.eim",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=DEFAULT_SIZE,
        metavar="WIDTHxHEIGHT",
        help="Picamera2の入力サイズ。既定値: 640x480",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help="カメラ設定fps。既定値: 15",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="実行時間（秒）。0はCtrl+Cまで継続。既定値: 30",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=DEFAULT_WARMUP,
        help="自動露出を安定させる待ち時間（秒）。既定値: 2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="表示する最小信頼度。既定値: 0.5",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=DEFAULT_REPORT_INTERVAL,
        help="進捗表示間隔（秒）。既定値: 1",
    )
    args = parser.parse_args()

    if args.camera_fps <= 0:
        parser.error("--camera-fpsは0より大きくしてください")
    if args.duration < 0:
        parser.error("--durationは0以上で指定してください")
    if args.warmup < 0:
        parser.error("--warmupは0以上で指定してください")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--thresholdは0以上1以下で指定してください")
    if args.report_interval <= 0:
        parser.error("--report-intervalは0より大きくしてください")
    return args


def detection_text(detection: dict[str, Any]) -> str:
    """Edge Impulseの検出結果を短い表示文字列へ変換する。"""
    label = detection.get("label", "unknown")
    score = float(detection.get("value", 0.0))
    x = int(detection.get("x", 0))
    y = int(detection.get("y", 0))
    width = int(detection.get("width", 0))
    height = int(detection.get("height", 0))
    return f"{label}={score:.2f} ({x},{y},{width},{height})"


def result_detections(result: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    """結果からしきい値以上のBounding Boxを取り出す。"""
    detections = result.get("result", {}).get("bounding_boxes", [])
    return [
        detection
        for detection in detections
        if float(detection.get("value", 0.0)) >= threshold
    ]


def get_model_features(runner: Any, frame: Any) -> tuple[Any, Any]:
    """Studioの画像リサイズ設定に合わせて特徴量を作成する。

    新しいSDKではfit-longestなどのStudio設定を反映するAPIを使う。
    古いSDKで未提供の場合だけ、互換用の従来APIへフォールバックする。
    """
    studio_method = getattr(runner, "get_features_from_image_auto_studio_settings", None)
    if studio_method is not None:
        return studio_method(frame)
    return runner.get_features_from_image(frame)


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")

    picam2: Picamera2 | None = None
    camera_started = False

    print(f"Model        : {model_path}", flush=True)

    with ImageImpulseRunner(str(model_path)) as runner:
        model_info = runner.init()
        project = model_info.get("project", {})
        parameters = model_info.get("model_parameters", {})
        labels = parameters.get("labels", [])

        print(
            f"Project      : {project.get('owner', '?')} / {project.get('name', '?')}",
            flush=True,
        )
        print(
            f"Model input  : {parameters.get('image_input_width', '?')}x"
            f"{parameters.get('image_input_height', '?')} RGB",
            flush=True,
        )
        print(f"Labels       : {', '.join(labels) if labels else '?'}", flush=True)
        print(f"Camera input : {args.size[0]}x{args.size[1]} RGB888", flush=True)
        print(f"Camera fps   : {args.camera_fps:.1f}", flush=True)
        print(f"Threshold    : {args.threshold:.2f}", flush=True)

        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": args.size, "format": "RGB888"},
            controls={"FrameRate": args.camera_fps},
            buffer_count=4,
        )
        picam2.configure(camera_config)

        frame_count = 0
        inference_count = 0
        detected_frame_count = 0
        total_inference_ms = 0.0
        started_at = 0.0
        next_report_at = 0.0

        try:
            picam2.start()
            camera_started = True
            time.sleep(args.warmup)

            started_at = time.monotonic()
            next_report_at = started_at + args.report_interval
            deadline = started_at + args.duration if args.duration > 0 else None
            print("Live inference started. Stop with Ctrl+C.", flush=True)

            while deadline is None or time.monotonic() < deadline:
                frame = picam2.capture_array("main")
                frame_count += 1

                features, _model_image = get_model_features(runner, frame)
                result = runner.classify(features)
                inference_count += 1

                timing = result.get("timing", {})
                inference_ms = sum(
                    float(timing.get(name, 0.0))
                    for name in ("dsp", "classification", "anomaly")
                )
                total_inference_ms += inference_ms

                detections = result_detections(result, args.threshold)
                if detections:
                    detected_frame_count += 1
                    text = ", ".join(detection_text(item) for item in detections)
                    print(f"DETECTED {text} inference={inference_ms:.1f}ms", flush=True)

                now = time.monotonic()
                if now >= next_report_at:
                    elapsed = now - started_at
                    loop_fps = inference_count / elapsed if elapsed > 0 else 0.0
                    average_ms = (
                        total_inference_ms / inference_count
                        if inference_count
                        else 0.0
                    )
                    print(
                        f"STATUS elapsed={elapsed:.1f}s frames={frame_count} "
                        f"inferences={inference_count} fps={loop_fps:.2f} "
                        f"avg_inference={average_ms:.1f}ms "
                        f"detected_frames={detected_frame_count}",
                        flush=True,
                    )
                    next_report_at = now + args.report_interval
        except KeyboardInterrupt:
            print("Stopping...", flush=True)
        finally:
            if camera_started:
                picam2.stop()
            picam2.close()

        elapsed = time.monotonic() - started_at if started_at else 0.0
        average_ms = total_inference_ms / inference_count if inference_count else 0.0
        measured_fps = inference_count / elapsed if elapsed > 0 else 0.0
        print(
            f"SUMMARY frames={frame_count} inferences={inference_count} "
            f"fps={measured_fps:.2f} avg_inference={average_ms:.1f}ms "
            f"detected_frames={detected_frame_count}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
