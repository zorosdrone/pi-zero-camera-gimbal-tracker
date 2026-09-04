#!/usr/bin/env python3
"""Phase 3: safe SG90 pan/tilt motion test.

The test uses PiGPIO-backed gpiozero PWM, moves only a small angle range by
default, and releases PWM after returning the selected servo(s) to center.
Servo power must come from an external 5 V supply with a common ground.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import sleep

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory


SERVO_MIN_PULSE_S = 0.0005
SERVO_MAX_PULSE_S = 0.0025
CENTER_ANGLE = 90
DEFAULT_MIN_ANGLE = 80
DEFAULT_MAX_ANGLE = 100
DEFAULT_HOLD_S = 0.7
DEFAULT_STEP_DELAY_S = 0.01


@dataclass(frozen=True)
class AxisConfig:
    name: str
    gpio: int
    physical_pin: int


AXES = {
    "pan": AxisConfig("PAN", 12, 32),
    "tilt": AxisConfig("TILT", 13, 33),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        choices=("pan", "tilt", "both"),
        default="pan",
        help="試験軸。初期値はPANのみ",
    )
    parser.add_argument("--min-angle", type=int, default=DEFAULT_MIN_ANGLE)
    parser.add_argument("--max-angle", type=int, default=DEFAULT_MAX_ANGLE)
    parser.add_argument("--center-angle", type=int, default=CENTER_ANGLE)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD_S, help="各位置の保持秒数")
    parser.add_argument("--step-delay", type=float, default=DEFAULT_STEP_DELAY_S, help="1度移動する間隔")
    parser.add_argument("--cycles", type=int, default=1, help="往復回数")
    return parser


def selected_axes(axis_name: str) -> list[AxisConfig]:
    if axis_name == "both":
        return [AXES["pan"], AXES["tilt"]]
    return [AXES[axis_name]]


def validate_args(args: argparse.Namespace) -> None:
    for name, value in (
        ("min-angle", args.min_angle),
        ("max-angle", args.max_angle),
        ("center-angle", args.center_angle),
    ):
        if not 0 <= value <= 180:
            raise SystemExit(f"{name}は0から180の範囲で指定してください")
    if not args.min_angle < args.center_angle < args.max_angle:
        raise SystemExit("min-angle < center-angle < max-angleにしてください")
    if args.hold <= 0:
        raise SystemExit("holdは0より大きくしてください")
    if args.step_delay <= 0:
        raise SystemExit("step-delayは0より大きくしてください")
    if args.cycles < 1:
        raise SystemExit("cyclesは1以上で指定してください")


def set_group_angle(servos: list[AngularServo], angle: int) -> None:
    for servo in servos:
        servo.angle = angle


def move_group(
    servos: list[AngularServo],
    current_angle: int,
    target_angle: int,
    step_delay_s: float,
) -> int:
    """Move all selected servos together in one-degree steps."""
    if current_angle == target_angle:
        return current_angle

    direction = 1 if target_angle > current_angle else -1
    angle = current_angle
    while angle != target_angle:
        angle += direction
        set_group_angle(servos, angle)
        sleep(step_delay_s)
    return target_angle


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    axes = selected_axes(args.axis)

    print("Phase 3 SG90 motion test", flush=True)
    print("Power: external 5 V; external GND and Pi GND must be common", flush=True)
    for axis in axes:
        print(
            f"{axis.name}: GPIO{axis.gpio} / physical pin {axis.physical_pin}",
            flush=True,
        )
    print(
        f"Range: {args.min_angle} -> {args.center_angle} -> {args.max_angle} degrees",
        flush=True,
    )

    factory = PiGPIOFactory()
    servos: list[AngularServo] = []
    current_angle = args.center_angle

    try:
        for axis in axes:
            servos.append(
                AngularServo(
                    axis.gpio,
                    min_angle=0,
                    max_angle=180,
                    min_pulse_width=SERVO_MIN_PULSE_S,
                    max_pulse_width=SERVO_MAX_PULSE_S,
                    initial_angle=None,
                    pin_factory=factory,
                )
            )

        set_group_angle(servos, current_angle)
        sleep(args.hold)
        sequence = (args.min_angle, args.center_angle, args.max_angle, args.center_angle)

        for cycle in range(1, args.cycles + 1):
            print(f"Cycle {cycle}/{args.cycles}", flush=True)
            for target_angle in sequence:
                print(f"MOVE {target_angle} deg", flush=True)
                current_angle = move_group(
                    servos,
                    current_angle,
                    target_angle,
                    args.step_delay,
                )
                sleep(args.hold)

        print("PASS: command sequence completed", flush=True)
    finally:
        print(f"CENTER {args.center_angle} deg, then detach PWM", flush=True)
        if servos:
            try:
                set_group_angle(servos, args.center_angle)
                sleep(min(args.hold, 0.5))
            except Exception as error:
                print(f"Centering warning: {error}", flush=True)
            for servo in servos:
                try:
                    servo.detach()
                finally:
                    servo.close()
        factory.close()
        print("PWM released", flush=True)


if __name__ == "__main__":
    main()
