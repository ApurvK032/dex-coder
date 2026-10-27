"""The controller's side of camera-only mode. Runs inside the sandbox, as its own process.

    python -I vworker.py SPEC.json

This process can import only the standard library, numpy and mujoco: the dexcoder and
humanoid_vla packages are not visible inside its sandbox, and no simulator runs here. Every
control step the trusted process (dexcoder.vision) sends what a real robot's computer would have,
namely joint readings and camera images, and this process answers with the controller's
action. Cube poses never reach it.

Protocol. From the trusted side, on stdin: one type byte, an 8-byte little-endian length, then
the payload. b"E" starts an episode (JSON task), b"O" is one observation (an .npz archive),
b"X" ends the episode, b"Q" quits. Replies on the original stdout: a 4-byte little-endian
length and a JSON object. File descriptors 0 and 1 are pointed at /dev/null once the protocol
streams are taken, so stray prints or writes cannot corrupt the protocol.
"""

import contextlib
import importlib.util
import io
import json
import os
import resource
import struct
import sys
import traceback
from pathlib import Path

import mujoco
import numpy as np

OUTPUT_LIMIT = 4000  # characters of printed output kept per episode


class LimitedWriter:
    def __init__(self, limit: int) -> None:
        self.limit, self.parts, self.size, self.dropped = limit, [], 0, 0

    def write(self, s: str) -> int:
        room = self.limit - self.size
        if room > 0:
            self.parts.append(s[:room])
            self.size += min(len(s), room)
        self.dropped += max(0, len(s) - max(room, 0))
        return len(s)

    def flush(self) -> None:
        pass

    def text(self) -> str:
        out = "".join(self.parts)
        if self.dropped:
            out += f"\n[... {self.dropped} more characters of output dropped]"
        return out


class Robot:
    """Facts about the robot, its cameras and the scene, fixed for the episode."""

    def __init__(self, spec: dict) -> None:
        r = spec["robot"]
        self.model = mujoco.MjModel.from_binary_path(spec["model_path"])
        self.arm_joints = tuple(r["arm_joints"])
        self.hand_joints = tuple(r["hand_joints"])
        for key in ("arm_qpos_adr", "arm_dof_adr", "hand_qpos_adr"):
            setattr(self, key, np.asarray(r[key], dtype=int))
        for key in ("arm_lower", "arm_upper", "home_arm"):
            setattr(self, key, np.asarray(r[key], dtype=float))
        for key in ("grasp_site", "control_dt", "max_joint_speed", "time_limit_s", "table_height",
                    "block_half_size", "bin_inner_half_size", "bin_wall_height", "bin_wall_thickness",
                    "released_closure", "settle_s"):
            setattr(self, key, r[key])
        self.bin_xy = {k: tuple(v) for k, v in r["bin_xy"].items()}
        self.cameras = {k: dict(v) for k, v in r["cameras"].items()}


class Observations:
    """Turns the trusted side's arrays into the obs dict a controller gets."""

    def __init__(self, robot: Robot) -> None:
        self.model = robot.model
        self.site = robot.model.site(robot.grasp_site).id
        self.cam_ids = {name: robot.model.camera(name).id for name in robot.cameras}

    def build(self, z) -> dict:
        kin = mujoco.MjData(self.model)
        kin.qpos[:] = z["kin_qpos"]
        kin.qvel[:] = z["kin_qvel"]
        mujoco.mj_forward(self.model, kin)
        cameras = {
            name: {
                "rgb": z[f"{name}_rgb"],
                "depth": z[f"{name}_depth"],
                "pos": kin.cam_xpos[cid].copy(),
                "mat": kin.cam_xmat[cid].reshape(3, 3).copy(),
            }
            for name, cid in self.cam_ids.items()
        }
        return {
            "t": float(z["t"]),
            "qpos": z["qpos"].copy(),
            "qvel": z["qvel"].copy(),
            "arm_target": z["arm_target"].copy(),
            "closure": float(z["closure"]),
            "grasp_site_pos": kin.site_xpos[self.site].copy(),
            "grasp_site_mat": kin.site_xmat[self.site].reshape(3, 3).copy(),
            "kin": kin,
            "cameras": cameras,
        }


def load_controller_class(path: str):
    spec = importlib.util.spec_from_file_location("controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "Controller"):
        raise AttributeError("controller.py must define a class named Controller")
    return module.Controller


def _short_traceback(max_lines: int = 30) -> str:
    lines = traceback.format_exc().rstrip().splitlines()
    return "\n".join(lines[-max_lines:])


def _read_exact(stream, n: int) -> bytes | None:
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    mem = int(spec.get("memory_limit_gb", 4) * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    proto_in = os.fdopen(os.dup(0), "rb", buffering=0)
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)

    def reply(obj: dict) -> None:
        data = json.dumps(obj).encode()
        proto_out.write(struct.pack("<I", len(data)) + data)

    robot = Robot(spec)
    observations = Observations(robot)
    out = LimitedWriter(OUTPUT_LIMIT)
    load_error = None
    controller_cls = None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            controller_cls = load_controller_class(spec["controller"])
    except BaseException:  # noqa: BLE001 - any failure of controller code is reported, not raised
        load_error = "controller.py failed to load:\n" + _short_traceback()
    controller = None

    while True:
        header = _read_exact(proto_in, 9)
        if header is None:
            return 0
        kind, length = header[:1], struct.unpack("<Q", header[1:])[0]
        payload = _read_exact(proto_in, length) if length else b""
        if kind == b"Q" or payload is None:
            return 0
        if kind == b"E":
            out = LimitedWriter(OUTPUT_LIMIT)
            controller = None
            if load_error:
                reply({"error": load_error})
                continue
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    controller = controller_cls(robot, json.loads(payload))
                reply({"ok": True})
            except BaseException:  # noqa: BLE001
                reply({"error": _short_traceback()})
        elif kind == b"O":
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as z:
                    obs = observations.build(z)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    action = controller.act(obs)
                action = np.asarray(action, dtype=float).reshape(-1)
                if action.shape != (8,):
                    raise ValueError(f"act() must return 8 numbers; it returned shape {action.shape}")
                if not np.all(np.isfinite(action)):
                    raise ValueError(f"act() returned a non-finite action: {action.tolist()}")
                reply({"action": action.tolist()})
            except BaseException:  # noqa: BLE001
                reply({"error": _short_traceback()})
        elif kind == b"X":
            reply({"output": out.text()})
        else:
            reply({"error": f"unknown message type {kind!r}"})


if __name__ == "__main__":
    sys.exit(main())
