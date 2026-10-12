"""Direct RealSense interface for the OpenDM deployment stack.

This module deliberately depends on ``pyrealsense2`` rather than LeRobot.
OpenDM deployment only needs the latest RGB frame from each camera, while a
direct RealSense interface keeps the Python 3.10 runtime small and avoids
coupling camera acquisition to a dataset/robot abstraction.

The three logical camera names match the dataset fields and map to the D435
serials as follows::

    base_0_rgb       -> base camera
    left_wrist_0_rgb -> left wrist camera
    right_wrist_0_rgb -> right wrist camera

``read()`` returns HWC RGB ``numpy.uint8`` arrays.  A background thread is
used for every camera, so the caller always receives the newest available
frame without blocking the robot-control or inference loop on a camera
exposure.  The interface does not resize, normalize, encode, or display
images; those operations belong to the model/input layer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

CAMERA_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

DEFAULT_CAMERA_SERIALS = {
    "base_0_rgb": "254622075848",
    "left_wrist_0_rgb": "317222074617",
    "right_wrist_0_rgb": "262422074985",
}


class CameraInterfaceError(RuntimeError):
    """Base exception raised by the OpenDM RealSense interface."""


class CameraDependencyError(CameraInterfaceError):
    """Raised when the ``pyrealsense2`` package is unavailable."""


class CameraConnectionError(CameraInterfaceError):
    """Raised when one or more RealSense pipelines cannot be started."""


class CameraFrameTimeoutError(CameraInterfaceError):
    """Raised when a fresh frame is not available within the requested age."""


@dataclass(frozen=True)
class CameraFrame:
    """A copied RGB frame and its host-clock timestamp."""

    image: np.ndarray
    timestamp_ns: int


def _load_realsense() -> Any:
    """Import pyrealsense2 only when a real camera connection is requested."""

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraDependencyError(
            "pyrealsense2 is required for the RealSense camera interface. "
            "Install it in the OpenDM Python 3.10 environment with "
            "`python -m pip install pyrealsense2`."
        ) from exc
    return rs


class RealSenseCameraInterface:
    """Capture the three CR3/O6 D435 RGB streams.

    Args:
        camera_serials: Mapping from ``base_0_rgb``, ``left_wrist_0_rgb``, and
            ``right_wrist_0_rgb`` to RealSense serial numbers.
        width: Color stream width.
        height: Color stream height.
        fps: Color stream frame rate.
        frame_timeout_ms: Maximum time for one ``wait_for_frames`` call.
        warmup_frames: Number of frames discarded after each pipeline starts.
        connect_timeout_s: Maximum time to wait for the first frame from every
            camera after starting the pipelines.

    The class is safe to use from one reader/control thread.  Internally each
    camera has its own capture thread and lock-protected latest-frame buffer.
    """

    def __init__(
        self,
        *,
        camera_serials: Mapping[str, str | int] | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        frame_timeout_ms: int = 500,
        warmup_frames: int = 15,
        connect_timeout_s: float = 10.0,
    ) -> None:
        serials = dict(
            DEFAULT_CAMERA_SERIALS if camera_serials is None else camera_serials
        )
        if set(serials) != set(CAMERA_NAMES):
            raise ValueError(
                "camera_serials must contain exactly "
                f"{CAMERA_NAMES}, got {tuple(serials)}"
            )

        self.camera_serials = {
            name: str(serials[name]) for name in CAMERA_NAMES
        }
        self.width = self._positive_int(width, "width")
        self.height = self._positive_int(height, "height")
        self.fps = self._positive_int(fps, "fps")
        self.frame_timeout_ms = self._positive_int(
            frame_timeout_ms, "frame_timeout_ms"
        )
        self.warmup_frames = self._nonnegative_int(warmup_frames, "warmup_frames")
        self.connect_timeout_s = float(connect_timeout_s)
        if not np.isfinite(self.connect_timeout_s) or self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")

        self._lock = threading.RLock()
        self._pipelines: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._latest: dict[str, CameraFrame] = {}
        self._errors: dict[str, BaseException] = {}
        self._stop_event = threading.Event()
        self._connected = False
        self._rs: Any | None = None

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _nonnegative_int(value: int, name: str) -> int:
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @property
    def connected(self) -> bool:
        """Whether all three pipelines have been started."""

        with self._lock:
            return self._connected

    @property
    def ready(self) -> bool:
        """Whether every camera has delivered at least one frame."""

        with self._lock:
            return self._connected and set(self._latest) == set(CAMERA_NAMES)

    @property
    def errors(self) -> Mapping[str, BaseException]:
        """A snapshot of capture-thread errors, useful for diagnostics."""

        with self._lock:
            return dict(self._errors)

    def _make_pipeline(self, name: str) -> Any:
        assert self._rs is not None
        pipeline = self._rs.pipeline()
        config = self._rs.config()
        config.enable_device(self.camera_serials[name])
        config.enable_stream(
            self._rs.stream.color,
            self.width,
            self.height,
            self._rs.format.rgb8,
            self.fps,
        )
        return pipeline, config

    def connect(self) -> "RealSenseCameraInterface":
        """Start all pipelines and wait until every camera has a first frame."""

        with self._lock:
            if self._connected:
                return self

        self._rs = _load_realsense()
        try:
            for name in CAMERA_NAMES:
                pipeline, config = self._make_pipeline(name)
                LOGGER.info(
                    "Starting RealSense camera %s (serial=%s, %dx%d@%d RGB)",
                    name,
                    self.camera_serials[name],
                    self.width,
                    self.height,
                    self.fps,
                )
                pipeline.start(config)
                self._pipelines[name] = pipeline

            with self._lock:
                self._stop_event.clear()
                self._connected = True
                for name in CAMERA_NAMES:
                    thread = threading.Thread(
                        target=self._capture_loop,
                        args=(name,),
                        name=f"RealSense-{name}",
                        daemon=True,
                    )
                    self._threads[name] = thread
                    thread.start()

            deadline = time.monotonic() + self.connect_timeout_s
            while time.monotonic() < deadline:
                if self.ready:
                    LOGGER.info("All OpenDM RealSense cameras are ready")
                    return self
                time.sleep(0.01)

            missing = [name for name in CAMERA_NAMES if name not in self._latest]
            details = ", ".join(
                f"{name}: {error}" for name, error in self.errors.items()
            )
            raise CameraConnectionError(
                "Timed out waiting for first frames from: "
                + ", ".join(missing)
                + (f"; capture errors: {details}" if details else "")
            )
        except BaseException as exc:
            self.disconnect()
            if isinstance(exc, CameraInterfaceError):
                raise
            raise CameraConnectionError("Failed to start RealSense cameras") from exc

    def _capture_loop(self, name: str) -> None:
        pipeline = self._pipelines[name]
        discarded = 0
        while not self._stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                image = np.asanyarray(color_frame.get_data())
                if image.ndim != 3 or image.shape[2] != 3:
                    raise CameraInterfaceError(
                        f"camera {name} returned invalid color shape {image.shape}"
                    )
                image = np.ascontiguousarray(image, dtype=np.uint8)

                if discarded < self.warmup_frames:
                    discarded += 1
                    continue

                frame = CameraFrame(image=image.copy(), timestamp_ns=time.monotonic_ns())
                with self._lock:
                    self._latest[name] = frame
            except BaseException as exc:
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._errors[name] = exc
                # Avoid a tight error loop if a USB device disappears.
                self._stop_event.wait(0.05)

    def _raise_capture_errors(self) -> None:
        with self._lock:
            errors = dict(self._errors)
        if errors and not self.ready:
            details = "; ".join(f"{name}: {error}" for name, error in errors.items())
            raise CameraFrameTimeoutError(f"RealSense capture error: {details}")

    def read(self, *, max_age_s: float = 0.5) -> dict[str, np.ndarray]:
        """Return the newest frame from each camera.

        ``max_age_s`` is checked against the monotonic host timestamp.  This
        prevents a disconnected camera from silently supplying an old frame to
        the controller.
        """

        max_age_s = float(max_age_s)
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        with self._lock:
            if not self._connected:
                raise CameraInterfaceError("cameras are not connected; call connect() first")
            frames = dict(self._latest)

        self._raise_capture_errors()
        missing = [name for name in CAMERA_NAMES if name not in frames]
        if missing:
            raise CameraFrameTimeoutError(
                "No frame available from: " + ", ".join(missing)
            )

        now_ns = time.monotonic_ns()
        stale = [
            name
            for name in CAMERA_NAMES
            if (now_ns - frames[name].timestamp_ns) / 1e9 > max_age_s
        ]
        if stale:
            raise CameraFrameTimeoutError(
                f"Stale RealSense frame(s): {', '.join(stale)} "
                f"(max_age_s={max_age_s:.3f})"
            )

        return {name: frames[name].image.copy() for name in CAMERA_NAMES}

    get_observation = read

    def read_with_timestamps(
        self, *, max_age_s: float = 0.5
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Return frames together with host-clock timestamps in nanoseconds."""

        max_age_s = float(max_age_s)
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        with self._lock:
            if not self._connected:
                raise CameraInterfaceError("cameras are not connected; call connect() first")
            frames = dict(self._latest)

        self._raise_capture_errors()
        missing = [name for name in CAMERA_NAMES if name not in frames]
        if missing:
            raise CameraFrameTimeoutError(
                "No frame available from: " + ", ".join(missing)
            )
        now_ns = time.monotonic_ns()
        stale = [
            name
            for name in CAMERA_NAMES
            if (now_ns - frames[name].timestamp_ns) / 1e9 > max_age_s
        ]
        if stale:
            raise CameraFrameTimeoutError(
                f"Stale RealSense frame(s): {', '.join(stale)} "
                f"(max_age_s={max_age_s:.3f})"
            )
        return (
            {name: frames[name].image.copy() for name in CAMERA_NAMES},
            {name: frames[name].timestamp_ns for name in CAMERA_NAMES},
        )

    def read_open_dm_observation(
        self, *, max_age_s: float = 0.5
    ) -> dict[str, np.ndarray]:
        """Return images using the keys expected by OpenDM deployment code."""

        return {
            f"observation/{name}": image
            for name, image in self.read(max_age_s=max_age_s).items()
        }

    def disconnect(self) -> None:
        """Stop capture threads and all RealSense pipelines."""

        self._stop_event.set()
        threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=max(1.0, self.frame_timeout_ms / 1000.0 + 0.5))

        for name, pipeline in list(self._pipelines.items())[::-1]:
            try:
                pipeline.stop()
            except Exception:
                LOGGER.exception("Failed to stop RealSense camera %s", name)

        with self._lock:
            self._pipelines.clear()
            self._threads.clear()
            self._latest.clear()
            self._errors.clear()
            self._connected = False
            self._rs = None

    close = disconnect

    def __enter__(self) -> "RealSenseCameraInterface":
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()


__all__ = [
    "CAMERA_NAMES",
    "CameraConnectionError",
    "CameraDependencyError",
    "CameraFrame",
    "CameraFrameTimeoutError",
    "CameraInterfaceError",
    "DEFAULT_CAMERA_SERIALS",
    "RealSenseCameraInterface",
]
