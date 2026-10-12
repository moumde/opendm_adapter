#!/usr/bin/env python3
"""Deploy the CR3/O6 robot with an OpenDM DM05 HTTP inference service.

Start the matching DM05 service first, for example::

    script/dm05_launcher.sh \
      --exp playground/dm05_cr3_o6_sft.py \
      --task inference \
      --model-config.model-name-or-path ./user_checkpoints/dm05_cr3_o6_500/checkpoint-50000 \
      --model-config.chunk-size 50 \
      --model-config.precision-policy bf16_mixed \
      --model-config.llm-attn-implementation sdpa \
      --model-config.vision-attn-implementation sdpa \
      --model-config.action-attn-implementation sdpa \
      --inference-config.output-action-dim 12 \
      --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
      --inference-config.port 7891

Preview the full hardware and HTTP path without sending motion::

    PYTHONPATH=/home/je/code/linkerhand-python-sdk:$PYTHONPATH \
    python deploy/deploy_dm05_cr3_o6.py \
      --preview-only \
      --duration-s 10

Physical control requires the explicit ``--confirm-motion`` flag.  Add
``--power-on`` if this client should power on the CR3 when it is not already
running.

The action contract is the one used by ``playground/dm05_cr3_o6_sft.py``:

* state/action dimensions are 12;
* CR3 values are six absolute joint targets in radians;
* O6 values are six absolute positions in the native 0..255 range;
* actions are consumed at 20 Hz by default;
* the server returns an action chunk of 50 rows.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from dataclasses import dataclass
from functools import wraps
from io import BytesIO
import inspect
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

try:
    from .camera_interface import RealSenseCameraInterface
except ImportError:
    from camera_interface import RealSenseCameraInterface


LOGGER = logging.getLogger("opendm_dm05_cr3_o6_deploy")

ACTION_DIM = 12
ACTION_HORIZON = 50
CONTROL_HZ = 20.0
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 7891
DEFAULT_ROBOT_IP = "192.168.2.14"
DEFAULT_LINKER_HAND_ROOT = Path("/home/je/code/linkerhand-python-sdk")
DEFAULT_PROMPT = (
    "['Pick up the motor and place it on the right side with the protruding side facing left.']"
)
DEFAULT_ROBOT_TYPE = "CR3_O6"
DEFAULT_JPEG_QUALITY = 95
DEFAULT_MAX_JOINT_SPEED_RAD_S = 0.5
DEFAULT_MAX_O6_STEP = 12.0
DEFAULT_O6_SPEED = 100
DEFAULT_INITIAL_MOVE_DURATION_S = 5.0
INITIAL_ROBOT_POSITION = np.asarray(
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32
)
INITIAL_HAND_POSITION = np.asarray((157, 95, 175, 0, 0, 0), dtype=np.float32)


class DeploymentError(RuntimeError):
    """Raised when the DM05 deployment cannot safely continue."""


def _finite_vector(values: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise DeploymentError(
            f"{name} must be a finite vector with shape ({length},), got {array.shape}"
        )
    return np.ascontiguousarray(array)


def _load_hand_api(linker_hand_root: Path):
    root = linker_hand_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
    except ImportError as exc:
        raise DeploymentError(
            "LinkerHand SDK import failed. Set PYTHONPATH or pass "
            f"--linker-hand-root; current root is {root}"
        ) from exc
    return LinkerHandApi


def _load_cr3_interface():
    """Load the NRC binding only after CLI validation and device setup."""

    try:
        from .interface import CR3Interface
    except ImportError:
        try:
            from interface import CR3Interface
        except ImportError as exc:
            raise DeploymentError(
                "CR3 NRC interface import failed. Check that the vendor "
                "_nrc_host.so matches the active Python version and that its "
                "libpython dependency is available."
            ) from exc
    return CR3Interface


def _patch_pymodbus_slave_argument() -> None:
    """Adapt the SDK's ``slave=`` calls to pymodbus versions using ``device_id``."""

    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError as exc:
        raise DeploymentError("LinkerHand RS485 requires pymodbus") from exc

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

        @wraps(original)
        def compatible_method(self: Any, *args: Any, _original=original, **kwargs: Any):
            if "slave" in kwargs:
                if "device_id" in kwargs:
                    raise TypeError("Specify only one of slave or device_id")
                kwargs["device_id"] = kwargs.pop("slave")
            return _original(self, *args, **kwargs)

        compatible_method._opendm_slave_compat = True
        setattr(ModbusSerialClient, method_name, compatible_method)


def _close_hand(hand_api: Any) -> None:
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


class DM05HTTPClient:
    """Small client for OpenDM's ``POST /v1/infer`` endpoint."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        timeout_s: float,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        action_horizon: int = ACTION_HORIZON,
    ) -> None:
        if not str(host).strip():
            raise ValueError("server host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"invalid server port: {port}")
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("HTTP timeout must be positive")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if int(action_horizon) <= 0:
            raise ValueError("action horizon must be positive")

        self.url = f"http://{host}:{int(port)}/v1/infer"
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.action_horizon = int(action_horizon)
        self.session = requests.Session()

    @staticmethod
    def _to_rgb_uint8(image: Any, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise DeploymentError(
                f"{name} must be an HxWx3 RGB image, got {array.shape}"
            )
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _encode_image(self, image: Any, name: str) -> str:
        array = self._to_rgb_uint8(image, name)
        buffer = BytesIO()
        Image.fromarray(array, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=self.jpeg_quality,
            subsampling=0,
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def infer(
        self,
        *,
        prompt: str,
        state: Any,
        images: dict[str, Any],
        robot_type: str = DEFAULT_ROBOT_TYPE,
    ) -> np.ndarray:
        state_array = _finite_vector(state, ACTION_DIM, "observation.state")
        expected_images = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        if tuple(images) != expected_images:
            raise DeploymentError(
                f"expected images in order {expected_images}, got {tuple(images)}"
            )

        payload = {
            "observation": {
                "prompt": str(prompt),
                "state": state_array.tolist(),
                "images": {
                    str(index): self._encode_image(images[name], name)
                    for index, name in enumerate(expected_images, start=1)
                },
                "robot_type": robot_type,
            }
        }

        started = time.monotonic()
        try:
            response = self.session.post(
                self.url,
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise DeploymentError(f"DM05 HTTP request failed: {exc}") from exc
        elapsed_ms = (time.monotonic() - started) * 1000.0

        if response.status_code != 200:
            raise DeploymentError(
                f"DM05 server returned HTTP {response.status_code}: {response.text[:2000]}"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise DeploymentError(
                f"DM05 server returned invalid JSON: {response.text[:1000]}"
            ) from exc

        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape != (self.action_horizon, ACTION_DIM)
            or not np.all(np.isfinite(actions))
        ):
            raise DeploymentError(
                "invalid DM05 action chunk: "
                f"got {actions.shape}, expected ({self.action_horizon}, {ACTION_DIM})"
            )

        metadata = result.get("metadata")
        server_latency = metadata.get("latency_ms") if isinstance(metadata, dict) else None
        LOGGER.info(
            "DM05 inference OK: chunk=%s round_trip_ms=%.1f server_latency_ms=%s",
            tuple(actions.shape),
            elapsed_ms,
            server_latency,
        )
        return np.ascontiguousarray(actions)

    def close(self) -> None:
        self.session.close()


class CR3O6Hardware:
    """Combine the existing CR3, O6, and RealSense interfaces."""

    def __init__(self, robot: CR3Interface, hand_api: Any, cameras: RealSenseCameraInterface):
        self.robot = robot
        self.hand_api = hand_api
        self.cameras = cameras
        self._hand_lock = threading.RLock()

    def read_observation(self, *, max_frame_age_s: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        arm_state = self.robot.get_joint_positions()
        with self._hand_lock:
            hand_state = self.hand_api.get_state()
        state = _finite_vector(list(arm_state) + list(hand_state), ACTION_DIM, "robot state")
        images = self.cameras.read(max_age_s=max_frame_age_s)
        return state, images

    def set_hand_speed(self, speed: int) -> None:
        with self._hand_lock:
            self.hand_api.set_speed([int(speed)] * 6)

    def move_to_initial_position(
        self,
        *,
        duration_s: float,
        control_hz: float,
        max_joint_speed_rad_s: float,
        max_o6_step: float,
    ) -> None:
        """Move the arm and hand smoothly to the configured initial position.

        The NRC interface exposes six arm joints publicly; the seventh value
        in ``INITIAL_ROBOT_POSITION`` is forwarded as the external-axis value
        and is normally zero for this CR3 setup.
        """

        duration_s = float(duration_s)
        control_hz = float(control_hz)
        max_joint_speed_rad_s = float(max_joint_speed_rad_s)
        max_o6_step = float(max_o6_step)
        if duration_s <= 0 or control_hz <= 0:
            raise ValueError("initial move duration and control_hz must be positive")
        if max_joint_speed_rad_s <= 0 or max_o6_step <= 0:
            raise ValueError("initial move speed limits must be positive")

        current_arm = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        if current_arm.shape != (6,):
            raise DeploymentError(f"CR3 state must have shape (6,), got {current_arm.shape}")
        # CR3Interface does not expose the current external-axis value.  The
        # configured CR3 setup uses zero, matching the requested target.
        current_arm = np.concatenate((current_arm, np.array([0.0], dtype=np.float32)))
        with self._hand_lock:
            current_hand = _finite_vector(self.hand_api.get_state(), 6, "O6 state")

        arm_delta = np.abs(INITIAL_ROBOT_POSITION - current_arm)
        hand_delta = np.abs(INITIAL_HAND_POSITION - current_hand)
        minimum_arm_duration = float(np.max(arm_delta[:6])) / max_joint_speed_rad_s
        minimum_hand_duration = float(np.max(hand_delta)) / (max_o6_step * control_hz)
        effective_duration = max(duration_s, minimum_arm_duration, minimum_hand_duration)
        steps = max(1, int(np.ceil(effective_duration * control_hz)))
        period = 1.0 / control_hz

        LOGGER.info(
            "Moving CR3/O6 to initial position over %.2fs (%d steps): arm=%s hand=%s",
            steps / control_hz,
            steps,
            INITIAL_ROBOT_POSITION.tolist(),
            INITIAL_HAND_POSITION.astype(int).tolist(),
        )
        started = time.monotonic()
        next_tick = started
        for step in range(1, steps + 1):
            ratio = step / steps
            arm_target = current_arm + ratio * (INITIAL_ROBOT_POSITION - current_arm)
            hand_target = current_hand + ratio * (INITIAL_HAND_POSITION - current_hand)
            with self._hand_lock:
                self.robot.send_joint_positions(
                    arm_target[:6].tolist(),
                    external_axis=float(arm_target[6]),
                )
                self.hand_api.finger_move(
                    np.rint(np.clip(hand_target, 0, 255)).astype(int).tolist()
                )
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        LOGGER.info("CR3/O6 initial position reached")

    def apply_action(self, action: np.ndarray) -> None:
        action = _finite_vector(action, ACTION_DIM, "action")
        with self._hand_lock:
            self.robot.send_joint_positions(action[:6].tolist())
            self.hand_api.finger_move(np.rint(np.clip(action[6:], 0, 255)).astype(int).tolist())


class ActionSlewLimiter:
    """Limit absolute target changes before they reach the hardware."""

    def __init__(
        self,
        initial_state: np.ndarray,
        *,
        control_hz: float,
        max_joint_speed_rad_s: float,
        max_o6_step: float,
    ) -> None:
        self.previous = _finite_vector(initial_state, ACTION_DIM, "initial state").copy()
        self.arm_step = float(max_joint_speed_rad_s) / float(control_hz)
        self.o6_step = float(max_o6_step)
        if self.arm_step <= 0 or self.o6_step <= 0:
            raise ValueError("action slew limits must be positive")

    def limit(self, target: np.ndarray) -> np.ndarray:
        target = _finite_vector(target, ACTION_DIM, "action")
        limited = self.previous.copy()
        limited[:6] += np.clip(target[:6] - limited[:6], -self.arm_step, self.arm_step)
        limited[6:] += np.clip(target[6:] - limited[6:], -self.o6_step, self.o6_step)
        limited[6:] = np.clip(limited[6:], 0, 255)
        self.previous = limited
        return limited


@dataclass
class _InferenceResult:
    actions: np.ndarray
    state: np.ndarray


class AsyncActionBuffer:
    """Request DM05 chunks in the background while the control loop consumes them."""

    def __init__(
        self,
        *,
        hardware: CR3O6Hardware,
        policy: DM05HTTPClient,
        prompt: str,
        robot_type: str,
        control_hz: float,
        request_lead_steps: int,
        max_frame_age_s: float,
    ) -> None:
        if request_lead_steps < 1:
            raise ValueError("request_lead_steps must be positive")
        self.hardware = hardware
        self.policy = policy
        self.prompt = prompt
        self.robot_type = robot_type
        self.control_hz = float(control_hz)
        self.request_lead_steps = int(request_lead_steps)
        self.max_frame_age_s = float(max_frame_age_s)
        self._condition = threading.Condition()
        self._chunks: deque[np.ndarray] = deque()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="DM05-inference",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                while (
                    not self._stop.is_set()
                    and sum(len(chunk) for chunk in self._chunks) > self.request_lead_steps
                ):
                    self._condition.wait(0.05)
            if self._stop.is_set():
                return

            try:
                state, images = self.hardware.read_observation(
                    max_frame_age_s=self.max_frame_age_s
                )
                actions = self.policy.infer(
                    prompt=self.prompt,
                    state=state,
                    images=images,
                    robot_type=self.robot_type,
                )
                with self._condition:
                    self._chunks.append(actions)
                    self._condition.notify_all()
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                self._stop.set()
                return

    def _raise_error(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise DeploymentError(
                f"DM05 inference worker failed: {type(error).__name__}: {error}"
            ) from error

    def wait_first_chunk(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._chunks:
                self._raise_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeploymentError("Timed out waiting for the first DM05 action chunk")
                self._condition.wait(min(0.1, remaining))
            chunk = self._chunks.popleft()
            self._condition.notify_all()
            return chunk

    def pop_action(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._chunks:
                self._raise_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeploymentError("DM05 action buffer underrun")
                self._condition.wait(min(0.05, remaining))

            chunk = self._chunks[0]
            action = chunk[0].copy()
            if len(chunk) == 1:
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[1:]
            self._condition.notify_all()
            return action

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=5.0)
        self._worker = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-host", default=DEFAULT_SERVER_HOST)
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--transport", choices=("rs485", "can"), default="rs485")
    parser.add_argument("--modbus", default="/dev/ttyUSB0")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--linker-hand-root", type=Path, default=DEFAULT_LINKER_HAND_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview-only", action="store_true")
    mode.add_argument("--confirm-motion", action="store_true")
    parser.add_argument("--power-on", action="store_true")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--request-lead-steps", type=int, default=10)
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-frame-age-s", type=float, default=0.5)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=DEFAULT_MAX_JOINT_SPEED_RAD_S)
    parser.add_argument("--max-o6-step", type=float, default=DEFAULT_MAX_O6_STEP)
    parser.add_argument("--o6-speed", type=int, default=DEFAULT_O6_SPEED)
    parser.add_argument(
        "--initial-move-duration-s",
        type=float,
        default=DEFAULT_INITIAL_MOVE_DURATION_S,
        help="Minimum seconds for the start/end move to the configured initial position",
    )
    parser.add_argument(
        "--skip-initial-position",
        action="store_true",
        help="Do not move to the configured initial position at start or end",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.duration_s < 0 or not np.isfinite(args.duration_s):
        raise ValueError("--duration-s must be >= 0")
    if not np.isfinite(args.control_hz) or args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if args.request_lead_steps < 1:
        raise ValueError("--request-lead-steps must be positive")
    if args.inference_timeout_s <= 0 or args.max_frame_age_s <= 0:
        raise ValueError("timeouts must be positive")
    if args.max_joint_speed_rad_s <= 0 or args.max_o6_step <= 0:
        raise ValueError("action speed limits must be positive")
    if not np.isfinite(args.initial_move_duration_s) or args.initial_move_duration_s <= 0:
        raise ValueError("--initial-move-duration-s must be positive")
    if not 10 <= args.o6_speed <= 255:
        raise ValueError("--o6-speed must be in [10, 255]")


def _connect_hand(args: argparse.Namespace) -> Any:
    LinkerHandApi = _load_hand_api(args.linker_hand_root)
    if args.transport == "rs485":
        _patch_pymodbus_slave_argument()
        modbus, can = args.modbus, "None"
        LOGGER.info("Connecting right O6 over RS485 at %s", modbus)
    else:
        modbus, can = "None", args.can
        LOGGER.info("Connecting right O6 over CAN at %s", can)

    hand_api = LinkerHandApi(
        hand_type="right",
        hand_joint="O6",
        modbus=modbus,
        can=can,
    )
    if hand_api.hand_type != "right" or str(hand_api.hand_joint).upper() != "O6":
        raise DeploymentError(
            f"Unexpected hand configuration: {hand_api.hand_type}/{hand_api.hand_joint}"
        )
    LOGGER.info("Right O6 state: %s", hand_api.get_state())
    return hand_api


def _run_preview(
    *,
    hardware: CR3O6Hardware,
    policy: DM05HTTPClient,
    prompt: str,
    robot_type: str,
    max_frame_age_s: float,
    duration_s: float,
) -> None:
    started = time.monotonic()
    request_count = 0
    while True:
        state, images = hardware.read_observation(max_frame_age_s=max_frame_age_s)
        actions = policy.infer(
            prompt=prompt,
            state=state,
            images=images,
            robot_type=robot_type,
        )
        request_count += 1
        LOGGER.info(
            "Preview request %d: state=%s action_chunk=%s first_arm_rad=%s first_o6=%s",
            request_count,
            tuple(state.shape),
            tuple(actions.shape),
            np.round(actions[0, :6], 5).tolist(),
            np.round(actions[0, 6:], 2).tolist(),
        )
        if duration_s <= 0 or time.monotonic() - started >= duration_s:
            return
        time.sleep(1.0)


def _run_motion(
    *,
    args: argparse.Namespace,
    hardware: CR3O6Hardware,
    policy: DM05HTTPClient,
) -> None:
    servo_opened = False
    buffer = None
    try:
        if hardware.robot.get_servo_state() != 3:
            if not args.power_on:
                raise DeploymentError(
                    "CR3 is not running. Use --power-on or power it on manually "
                    "before adding --confirm-motion."
                )
            LOGGER.info("CR3 is not running; powering it on")
            hardware.robot.power_on()
        if hardware.robot.get_servo_state() != 3:
            raise DeploymentError("CR3 is not in running state")

        hardware.set_hand_speed(args.o6_speed)
        hardware.robot.open_servoj()
        servo_opened = True

        if not args.skip_initial_position:
            hardware.move_to_initial_position(
                duration_s=args.initial_move_duration_s,
                control_hz=args.control_hz,
                max_joint_speed_rad_s=args.max_joint_speed_rad_s,
                max_o6_step=args.max_o6_step,
            )

        initial_state, _ = hardware.read_observation(max_frame_age_s=args.max_frame_age_s)
        limiter = ActionSlewLimiter(
            initial_state,
            control_hz=args.control_hz,
            max_joint_speed_rad_s=args.max_joint_speed_rad_s,
            max_o6_step=args.max_o6_step,
        )
        buffer = AsyncActionBuffer(
            hardware=hardware,
            policy=policy,
            prompt=args.prompt,
            robot_type=args.robot_type,
            control_hz=args.control_hz,
            request_lead_steps=args.request_lead_steps,
            max_frame_age_s=args.max_frame_age_s,
        )
        buffer.start()
        first_chunk = buffer.wait_first_chunk(args.inference_timeout_s)
        LOGGER.info("Received first DM05 action chunk: shape=%s", tuple(first_chunk.shape))

        # Put the first chunk back at the front by using a local first-action
        # variable.  The worker continues filling the buffer in parallel.
        first_action_index = 0
        started = time.monotonic()
        period = 1.0 / args.control_hz
        next_tick = started
        while True:
            if args.duration_s > 0 and time.monotonic() - started >= args.duration_s:
                return
            if first_action_index < len(first_chunk):
                raw_action = first_chunk[first_action_index]
                first_action_index += 1
            else:
                raw_action = buffer.pop_action(max(0.1, period * 2.0))
            hardware.apply_action(limiter.limit(raw_action))
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                LOGGER.warning("Control loop overrun by %.1f ms", -delay * 1000.0)
                next_tick = time.monotonic()
    finally:
        if buffer is not None:
            buffer.stop()
        if servo_opened:
            try:
                if not args.skip_initial_position:
                    try:
                        hardware.move_to_initial_position(
                            duration_s=args.initial_move_duration_s,
                            control_hz=args.control_hz,
                            max_joint_speed_rad_s=args.max_joint_speed_rad_s,
                            max_o6_step=args.max_o6_step,
                        )
                    except BaseException:
                        # Even a second Ctrl+C during the return move must not
                        # prevent ServoJ from being closed below.
                        LOGGER.exception("Failed to return CR3/O6 to initial position")
            finally:
                try:
                    hardware.robot.close_servoj()
                except BaseException:
                    LOGGER.exception("Failed to close CR3 ServoJ")


def main() -> int:
    args = _parser().parse_args()
    try:
        _validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    robot = None
    hand_api = None
    cameras = None
    policy = None
    try:
        CR3Interface = _load_cr3_interface()
        LOGGER.info("Connecting CR3 at %s", args.robot_ip)
        robot = CR3Interface(
            args.robot_ip,
            servo_vmax=[50.1] * 7,
            servo_amax=[100.0] * 7,
            servo_jmax=[200.0] * 7,
        )
        robot.connect()
        hand_api = _connect_hand(args)
        cameras = RealSenseCameraInterface()
        cameras.connect()
        hardware = CR3O6Hardware(robot, hand_api, cameras)
        policy = DM05HTTPClient(
            host=args.server_host,
            port=args.server_port,
            timeout_s=args.inference_timeout_s,
            jpeg_quality=args.jpeg_quality,
        )

        if args.confirm_motion:
            _run_motion(args=args, hardware=hardware, policy=policy)
        else:
            if not args.preview_only:
                LOGGER.info("No motion flag supplied; running one safe preflight request")
            _run_preview(
                hardware=hardware,
                policy=policy,
                prompt=args.prompt,
                robot_type=args.robot_type,
                max_frame_age_s=args.max_frame_age_s,
                duration_s=args.duration_s if args.preview_only else 0.0,
            )
        return 0
    except (DeploymentError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("DM05 CR3/O6 deployment failed: %s", exc)
        return 1
    finally:
        if policy is not None:
            policy.close()
        if cameras is not None:
            cameras.disconnect()
        if hand_api is not None:
            _close_hand(hand_api)
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    sys.exit(main())
