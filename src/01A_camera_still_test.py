#!/usr/bin/env python3
"""Phase 1A: Picamera2で静止画を1枚保存するヘッドレス動作確認。

参考:
  https://github.com/raspberrypi/picamera2
  Picamera2 is licensed under the BSD 2-Clause License.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

from picamera2 import Picamera2


DEFAULT_SIZE = (1296, 972)


def parse_size(value: str) -> tuple[int, int]:
    """WIDTHxHEIGHT形式を正の整数2個へ変換する。"""
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width = int(width_text)
        height = int(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("サイズは1296x972の形式で指定してください") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("幅と高さには正の整数を指定してください")

    return width, height


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("captures") / f"camera_still_{timestamp}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Picamera2で静止画を1枚保存し、OV5647の基本動作を確認します。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="保存先。省略時はcaptures/camera_still_日時.jpg",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=DEFAULT_SIZE,
        metavar="WIDTHxHEIGHT",
        help="撮影サイズ。既定値: 1296x972",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="自動露出を安定させる待ち時間（秒）。既定値: 2.0",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmupには0以上を指定してください")

    return args


def main() -> int:
    args = parse_args()
    output_path = args.output or default_output_path()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()
    model = picam2.camera_properties.get("Model", "unknown")
    config = picam2.create_still_configuration(
        main={"size": args.size, "format": "RGB888"}
    )
    picam2.configure(config)

    print(f"Camera model : {model}")
    print(f"Capture size : {args.size[0]}x{args.size[1]}")
    print(f"Warm-up      : {args.warmup:.1f} s")

    try:
        picam2.start()
        time.sleep(args.warmup)
        picam2.capture_file(str(output_path))
    finally:
        picam2.stop()
        picam2.close()

    file_size = output_path.stat().st_size
    print(f"Saved        : {output_path}")
    print(f"File size    : {file_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
