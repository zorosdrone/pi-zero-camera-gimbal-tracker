#!/usr/bin/env python3
"""Phase 1B: Picamera2のmainとloresを同時に連続取得する。

参考:
  https://github.com/raspberrypi/picamera2/blob/main/examples/capture_motion.py
  Picamera2 is licensed under the BSD 2-Clause License.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from picamera2 import Picamera2


DEFAULT_MAIN_SIZE = (640, 480)
DEFAULT_LORES_SIZE = (320, 240)


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
        description="配信用mainと解析用loresを同時に取得し、連続取得fpsを測定します。"
    )
    parser.add_argument(
        "--main-size",
        type=parse_size,
        default=DEFAULT_MAIN_SIZE,
        metavar="WIDTHxHEIGHT",
        help="mainサイズ。既定値: 640x480",
    )
    parser.add_argument(
        "--lores-size",
        type=parse_size,
        default=DEFAULT_LORES_SIZE,
        metavar="WIDTHxHEIGHT",
        help="loresサイズ。既定値: 320x240",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="カメラの設定fps。既定値: 15",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="連続取得時間（秒）。既定値: 10",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="自動露出を安定させる待ち時間（秒）。既定値: 2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("captures/phase1b"),
        help="確認画像とJSON結果の保存先。既定値: captures/phase1b",
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fpsには0より大きい値を指定してください")
    if args.duration <= 0:
        parser.error("--durationには0より大きい値を指定してください")
    if args.warmup < 0:
        parser.error("--warmupには0以上を指定してください")

    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_width, main_height = args.main_size
    lores_width, lores_height = args.lores_size

    picam2 = Picamera2()
    model = picam2.camera_properties.get("Model", "unknown")
    config = picam2.create_video_configuration(
        main={"size": args.main_size, "format": "RGB888"},
        lores={"size": args.lores_size, "format": "YUV420"},
        controls={"FrameRate": args.fps},
        buffer_count=4,
    )
    picam2.configure(config)

    print(f"Camera model : {model}")
    print(f"Main stream  : {main_width}x{main_height} RGB888")
    print(f"Lores stream : {lores_width}x{lores_height} YUV420")
    print(f"Target fps   : {args.fps:.2f}")
    print(f"Duration     : {args.duration:.1f} s")

    frame_count = 0
    first_sensor_timestamp = None
    last_sensor_timestamp = None
    main_shape = None
    lores_shape = None
    start_time = None

    try:
        picam2.start()
        time.sleep(args.warmup)

        start_time = time.monotonic()
        deadline = start_time + args.duration

        while time.monotonic() < deadline:
            arrays, metadata = picam2.capture_arrays(["main", "lores"])
            main_array, lores_array = arrays

            if frame_count == 0:
                main_shape = tuple(main_array.shape)
                lores_shape = tuple(lores_array.shape)

                if main_shape[:2] != (main_height, main_width):
                    raise RuntimeError(
                        f"main shape mismatch: expected {(main_height, main_width)}, got {main_shape}"
                    )
                if lores_shape[0] < lores_height or lores_shape[1] < lores_width:
                    raise RuntimeError(
                        f"lores shape mismatch: expected at least {(lores_height, lores_width)}, got {lores_shape}"
                    )

                lores_y = lores_array[:lores_height, :lores_width]
                Image.fromarray(main_array).save(output_dir / "phase1b_main.jpg", quality=90)
                Image.fromarray(lores_y).save(output_dir / "phase1b_lores_y.jpg", quality=90)

                print(f"Main array   : {main_shape}")
                print(f"Lores array  : {lores_shape} (Y plane: {lores_width}x{lores_height})")

            sensor_timestamp = metadata.get("SensorTimestamp")
            if sensor_timestamp is not None:
                if first_sensor_timestamp is None:
                    first_sensor_timestamp = sensor_timestamp
                last_sensor_timestamp = sensor_timestamp

            frame_count += 1

        elapsed = time.monotonic() - start_time
    finally:
        picam2.stop()
        picam2.close()

    measured_fps = frame_count / elapsed
    sensor_fps = None
    if (
        first_sensor_timestamp is not None
        and last_sensor_timestamp is not None
        and last_sensor_timestamp > first_sensor_timestamp
        and frame_count > 1
    ):
        sensor_elapsed = (last_sensor_timestamp - first_sensor_timestamp) / 1_000_000_000
        sensor_fps = (frame_count - 1) / sensor_elapsed

    result = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_model": model,
        "main": {"size": list(args.main_size), "format": "RGB888", "array_shape": main_shape},
        "lores": {
            "size": list(args.lores_size),
            "format": "YUV420",
            "array_shape": lores_shape,
            "analysis_plane": "Y",
        },
        "target_fps": args.fps,
        "duration_seconds": elapsed,
        "frame_count": frame_count,
        "measured_fps": measured_fps,
        "sensor_fps": sensor_fps,
    }
    result_path = output_dir / "phase1b_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Frames       : {frame_count}")
    print(f"Elapsed      : {elapsed:.3f} s")
    print(f"Measured fps : {measured_fps:.2f}")
    if sensor_fps is not None:
        print(f"Sensor fps   : {sensor_fps:.2f}")
    print(f"Result       : {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
