#!/usr/bin/env python3
"""Phase 2B: Pi Camera H.264/MPEG-TS stream test.

This is a small transport test for the later YOLO and gimbal stages.  The
Pi's hardware H.264 encoder produces the video, Picamera2/PyAV muxes it as
MPEG-TS, and one PC client receives it over TCP.  The PC does not need a
desktop session on the Pi.
"""

from __future__ import annotations

import argparse
import socket
import time
from contextlib import closing
from typing import Optional

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10001
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 10.0
DEFAULT_BITRATE = 1_000_000


def parse_size(value: str) -> tuple[int, int]:
    """Parse a WIDTHxHEIGHT command-line value."""
    try:
        width_text, height_text = value.lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizeはWIDTHxHEIGHTで指定してください") from exc

    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError("sizeは幅・高さとも16以上にしてください")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="待受アドレス")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP待受ポート")
    parser.add_argument(
        "--size",
        type=parse_size,
        default=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
        help="映像サイズ（例: 640x480）",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="目標フレームレート")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE, help="H.264ビットレート(bit/s)")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="サーバーの実行秒数。0はCtrl+Cまで実行",
    )
    return parser


def remaining_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def stream_one_client(
    picam2: Picamera2,
    client: socket.socket,
    size: tuple[int, int],
    fps: float,
    bitrate: int,
    deadline: Optional[float],
) -> None:
    """Encode and send video until the client or the test duration ends."""
    client_address = client.getpeername()
    print(f"Client connected: {client_address[0]}:{client_address[1]}", flush=True)

    stopped = False
    encoder = H264Encoder(bitrate=bitrate, repeat=True, framerate=fps)
    output = PyavOutput(f"pipe:{client.fileno()}", format="mpegts")
    output_error: list[BaseException] = []

    def on_output_error(error: BaseException) -> None:
        output_error.append(error)

    output.error_callback = on_output_error

    try:
        configuration = picam2.create_video_configuration(
            main={"size": size},
            controls={"FrameRate": fps},
            buffer_count=4,
        )
        picam2.configure(configuration)
        picam2.start_recording(encoder, output)

        while True:
            seconds_left = remaining_seconds(deadline)
            if seconds_left is not None and seconds_left <= 0:
                break
            if output_error:
                raise ConnectionError(f"MPEG-TS output failed: {output_error[0]}")

            # The encoder/output threads do the work.  A short sleep keeps the
            # control loop responsive without polling the camera aggressively.
            time.sleep(min(0.25, seconds_left) if seconds_left is not None else 0.25)
    finally:
        try:
            picam2.stop_recording()
            stopped = True
        except RuntimeError:
            # The output may already have stopped after a client disconnect.
            pass
        finally:
            if not stopped:
                try:
                    output.stop()
                except Exception:
                    pass

    print("Client disconnected; encoder stopped", flush=True)


def run(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    picam2 = Picamera2()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.settimeout(0.5)

    width, height = args.size
    print(f"Camera model: {picam2.camera_properties.get('Model', 'unknown')}", flush=True)
    print(f"Stream: H.264/MPEG-TS {width}x{height} at {args.fps:g} fps", flush=True)
    print(f"Bitrate: {args.bitrate} bit/s", flush=True)
    print(f"Listening: tcp://0.0.0.0:{args.port} (one client)", flush=True)
    print(
        f"PC test: ffplay -fflags nobuffer -flags low_delay tcp://raspberrypi.local:{args.port}",
        flush=True,
    )
    if args.duration > 0:
        print(f"Auto stop: {args.duration:g} seconds", flush=True)

    try:
        while True:
            seconds_left = remaining_seconds(deadline)
            if seconds_left is not None and seconds_left <= 0:
                break

            try:
                client, address = server.accept()
            except TimeoutError:
                continue

            with closing(client):
                try:
                    stream_one_client(
                        picam2,
                        client,
                        args.size,
                        args.fps,
                        args.bitrate,
                        deadline,
                    )
                except (BrokenPipeError, ConnectionError, OSError) as error:
                    print(f"Client connection ended: {error}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted", flush=True)
    finally:
        server.close()
        picam2.close()
        print("Camera closed; server stopped", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("portは1から65535の範囲で指定してください")
    if args.fps <= 0:
        raise SystemExit("fpsは0より大きくしてください")
    if args.bitrate <= 0:
        raise SystemExit("bitrateは0より大きくしてください")
    if args.duration < 0:
        raise SystemExit("durationは0以上で指定してください")
    run(args)


if __name__ == "__main__":
    main()
