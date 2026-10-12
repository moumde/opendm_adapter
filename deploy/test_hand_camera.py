"""Read-only connection test for the right LinkerHand O6 and three D435 cameras.

RS485 example::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    python deploy/test_hand_camera.py \
        --transport rs485 \
        --modbus /dev/ttyUSB0

CAN example::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    python deploy/test_hand_camera.py \
        --transport can \
        --can can0

The script only initializes the right O6, reads its current position, starts
the three RGB camera streams, and reads frames.  It does not send hand
commands, power on the CR3, open ServoJ, or move any hardware.
"""

from __future__ import annotations

import argparse
from functools import wraps
import inspect
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .camera_interface import CAMERA_NAMES
    from .camera_interface import DEFAULT_CAMERA_SERIALS
    from .camera_interface import RealSenseCameraInterface
except ImportError:
    # Support ``python deploy/test_hand_camera.py`` from the repository root.
    from camera_interface import CAMERA_NAMES
    from camera_interface import DEFAULT_CAMERA_SERIALS
    from camera_interface import RealSenseCameraInterface


LOGGER = logging.getLogger("opendm_hand_camera_test")
DEFAULT_LINKER_HAND_ROOT = Path("/home/je/code/linkerhand-python-sdk")


def _load_hand_api():
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
    except ImportError as exc:
        raise RuntimeError(
            "LinkerHand SDK import failed. Set PYTHONPATH, for example: "
            "export PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH"
        ) from exc
    return LinkerHandApi


def _patch_pymodbus_slave_argument() -> None:
    """Bridge the SDK's ``slave=`` calls to newer pymodbus ``device_id=`` APIs."""

    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError as exc:
        raise RuntimeError("LinkerHand RS485 requires pymodbus") from exc

    for method_name in ("read_input_registers", "write_registers"):
        method = getattr(ModbusSerialClient, method_name, None)
        if method is None:
            continue
        parameters = inspect.signature(method).parameters
        if "slave" in parameters or "device_id" not in parameters:
            continue
        if getattr(method, "_opendm_slave_compat", False):
            continue

        original = method

        def compatible_method(self: Any, *args: Any, _original=original, **kwargs: Any):
            if "slave" in kwargs:
                if "device_id" in kwargs:
                    raise TypeError("Specify only one of slave or device_id")
                kwargs["device_id"] = kwargs.pop("slave")
            return _original(self, *args, **kwargs)

        compatible_method = wraps(original)(compatible_method)
        compatible_method._opendm_slave_compat = True
        setattr(ModbusSerialClient, method_name, compatible_method)


def _validate_hand_state(state: Sequence[float]) -> list[int]:
    values = [float(value) for value in state]
    if len(values) != 6:
        raise AssertionError(f"Expected 6 O6 joint values, got {len(values)}: {values!r}")
    if not all(math.isfinite(value) and 0 <= value <= 255 for value in values):
        raise AssertionError(f"O6 joint values must be finite and in [0, 255], got {values!r}")
    return [int(round(value)) for value in values]


def _close_hand(hand_api: Any) -> None:
    """Close whichever transport implementation is behind LinkerHandApi."""

    hand = getattr(hand_api, "hand", None)
    if hand is None:
        return
    for method_name in ("close", "close_can_interface"):
        method = getattr(hand, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                LOGGER.exception("Failed to close LinkerHand using %s()", method_name)
            return
    LOGGER.warning("No close method found on LinkerHand backend %s", type(hand).__name__)


def _validate_frames(
    frames: dict[str, np.ndarray], *, width: int, height: int
) -> None:
    if tuple(frames) != CAMERA_NAMES:
        raise AssertionError(
            f"Expected camera keys in order {CAMERA_NAMES}, got {tuple(frames)}"
        )
    expected_shape = (height, width, 3)
    for name in CAMERA_NAMES:
        image = np.asarray(frames[name])
        if image.shape != expected_shape:
            raise AssertionError(
                f"Camera {name} returned shape {image.shape}, "
                f"expected {expected_shape}"
            )
        if image.dtype != np.uint8:
            raise AssertionError(
                f"Camera {name} returned dtype {image.dtype}, expected uint8"
            )
        if not image.flags.c_contiguous:
            raise AssertionError(f"Camera {name} returned a non-contiguous image")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("rs485", "can"), default="rs485")
    parser.add_argument("--modbus", default="/dev/ttyUSB0", help="O6 RS485 device")
    parser.add_argument("--can", default="can0", help="O6 CAN interface")
    parser.add_argument(
        "--linker-hand-root",
        type=Path,
        default=DEFAULT_LINKER_HAND_ROOT,
        help="LinkerHand SDK root, used in the error message and sys.path fallback",
    )
    parser.add_argument("--frames", type=int, default=3, help="Camera frames to read")
    parser.add_argument("--camera-timeout-ms", type=int, default=500)
    parser.add_argument("--max-frame-age-s", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # Make the SDK import work even when the caller did not export PYTHONPATH.
    linker_hand_root = args.linker_hand_root.expanduser().resolve()
    if str(linker_hand_root) not in sys.path:
        sys.path.insert(0, str(linker_hand_root))

    hand_api = None
    cameras = None
    try:
        LinkerHandApi = _load_hand_api()
        if args.transport == "rs485":
            _patch_pymodbus_slave_argument()
            modbus = args.modbus
            can = "None"
            transport_description = f"RS485 at {modbus}"
        else:
            modbus = "None"
            can = args.can
            transport_description = f"CAN at {can}"

        LOGGER.info("Connecting right LinkerHand O6 over %s", transport_description)
        hand_api = LinkerHandApi(
            hand_type="right",
            hand_joint="O6",
            modbus=modbus,
            can=can,
        )
        if hand_api.hand_type != "right" or str(hand_api.hand_joint).upper() != "O6":
            raise AssertionError(
                "Unexpected hand configuration: "
                f"type={hand_api.hand_type!r}, joint={hand_api.hand_joint!r}"
            )
        hand_state = _validate_hand_state(hand_api.get_state())
        LOGGER.info("Right O6 connected; current state [0..255]: %s", hand_state)

        serials = dict(DEFAULT_CAMERA_SERIALS)
        cameras = RealSenseCameraInterface(
            camera_serials=serials,
            width=args.width,
            height=args.height,
            fps=args.fps,
            frame_timeout_ms=args.camera_timeout_ms,
            warmup_frames=args.warmup_frames,
        )
        LOGGER.info("Connecting cameras: %s", serials)
        cameras.connect()
        if not cameras.ready:
            raise AssertionError("Camera interface is connected but not ready")

        for index in range(args.frames):
            frames = cameras.read(max_age_s=args.max_frame_age_s)
            _validate_frames(frames, width=args.width, height=args.height)
            LOGGER.info(
                "Camera frame %d/%d received: %s",
                index + 1,
                args.frames,
                {name: frames[name].shape for name in CAMERA_NAMES},
            )

        LOGGER.info("Right O6 + three-camera connection test passed")
        return 0
    except (AssertionError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Connection test failed: %s", exc)
        return 1
    finally:
        if cameras is not None:
            cameras.disconnect()
        if hand_api is not None:
            _close_hand(hand_api)


if __name__ == "__main__":
    sys.exit(main())
