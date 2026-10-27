"""Camera-only mode: controllers see camera images and joint angles, never cube poses.

The trusted side (this module) runs the physics and renders the cameras. The controller runs in
a separate process inside the sandbox (``dexcoder/vworker.py``) that can import only the standard
library, numpy and mujoco, and holds no simulator. Every control step this side sends it the
joint readings and the head and wrist camera images (colour and depth) and gets back an action,
so nothing else about the world can reach the controller. Episodes are judged here, on the
simulation that actually ran; no replay is needed.

Held-out conditions (VISION_VARIANTS): the physics variants of the first study (heavy and
slippery blocks, weak motors), "lighting" (dimmer, warmer light plus a strong side light that
casts shadows) and "noisy_camera" (pixel and depth noise, depth dropouts, a head camera knocked
slightly off its calibrated pose, and images one control step late).

    python -m dexcoder.vision SPEC.json   # one chunk of episodes (launched by run_vision_trials)
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import io  # noqa: E402
import json  # noqa: E402
import select  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from humanoid_vla.envs.task import COLORS  # noqa: E402
from humanoid_vla.sim.g1 import GRASP_SITE  # noqa: E402
from humanoid_vla.sim.scene import SceneConfig, build_model  # noqa: E402

from dexcoder import trials as T  # noqa: E402
from dexcoder.sandbox import sandboxed_command  # noqa: E402
from dexcoder.tasks import CAMERA_VARIANTS, TaskEnv  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
WORKER = REPO / "src/dexcoder/vworker.py"

IMAGE_HW = (240, 320)
CAMERAS = ("head", "wrist")
NO_READING_M = 5.0  # depth beyond this is reported as 0 (no reading)

# noisy_camera
RGB_NOISE_STD = 3.0  # uint8 levels
DEPTH_NOISE_STD_PER_M2 = 0.005  # m of noise per (m of depth)^2: about 1 mm at the cubes
DEPTH_STEP_M = 0.001
DEPTH_DROPOUT = 0.02  # fraction of depth pixels with no reading
HEAD_BUMP_POS_STD = 0.003  # m, per axis, not reported to the controller
HEAD_BUMP_ROT_STD = np.deg2rad(0.5)  # per axis

# lighting
HEADLIGHT_DIFFUSE = (0.32, 0.29, 0.24)  # nominal (0.6, 0.6, 0.6)
HEADLIGHT_AMBIENT = (0.12, 0.11, 0.09)  # nominal (0.1, 0.1, 0.1)
SIDE_LIGHT_DIFFUSE = (0.62, 0.55, 0.42)


# The robot as the controller knows it ----------------------------------------------------------


def robot_model() -> mujoco.MjModel:
    """The robot, table and bins with their cameras, without any cube."""
    return build_model(SceneConfig(block_xy={}))


def robot_facts(env: TaskEnv, model: mujoco.MjModel) -> dict:
    scene, cfg = env.config.scene, env.config
    arm_ids = [model.joint(j).id for j in env.arm_joints]
    hand_ids = [model.joint(j).id for j in env.hand_joints]
    return {
        "arm_joints": list(env.arm_joints),
        "hand_joints": list(env.hand_joints),
        "arm_qpos_adr": model.jnt_qposadr[arm_ids].tolist(),
        "arm_dof_adr": model.jnt_dofadr[arm_ids].tolist(),
        "hand_qpos_adr": model.jnt_qposadr[hand_ids].tolist(),
        "arm_lower": env.arm_lower.tolist(),
        "arm_upper": env.arm_upper.tolist(),
        "grasp_site": GRASP_SITE,
        "control_dt": float(env.control_dt),
        "max_joint_speed": float(cfg.max_joint_speed),
        "time_limit_s": float(cfg.time_limit_s),
        "table_height": float(scene.table_height),
        "block_half_size": float(scene.block_half_size),
        "bin_xy": {k: [float(x) for x in v] for k, v in scene.bin_xy.items()},
        "bin_inner_half_size": float(scene.bin_inner_half_size),
        "bin_wall_height": float(scene.bin_wall_height),
        "bin_wall_thickness": float(scene.bin_wall_thickness),
        "home_arm": [float(x) for x in cfg.home_arm],
        "released_closure": float(cfg.released_closure),
        "settle_s": float(cfg.settle_s),
        "cameras": {
            c: {"height": IMAGE_HW[0], "width": IMAGE_HW[1], "fovy_deg": float(model.cam_fovy[model.camera(c).id])}
            for c in CAMERAS
        },
    }


def kin_index(sim: mujoco.MjModel, kin: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """For each qpos / qvel entry of the cube-free robot model, its index in the simulator's."""
    nq = {0: 7, 1: 4, 2: 1, 3: 1}
    nv = {0: 6, 1: 3, 2: 1, 3: 1}
    q_idx, v_idx = np.full(kin.nq, -1), np.full(kin.nv, -1)
    for j in range(kin.njnt):
        s = sim.joint(kin.joint(j).name).id
        t = int(kin.jnt_type[j])
        q_idx[kin.jnt_qposadr[j]: kin.jnt_qposadr[j] + nq[t]] = np.arange(sim.jnt_qposadr[s], sim.jnt_qposadr[s] + nq[t])
        v_idx[kin.jnt_dofadr[j]: kin.jnt_dofadr[j] + nv[t]] = np.arange(sim.jnt_dofadr[s], sim.jnt_dofadr[s] + nv[t])
    assert (q_idx >= 0).all() and (v_idx >= 0).all()
    return q_idx, v_idx


# Cameras -------------------------------------------------------------------------------------


class Cameras:
    """Renders the head and wrist cameras (RGB and depth) under a variant's camera conditions."""

    def __init__(self, env: TaskEnv, variant: str) -> None:
        self.env, self.variant = env, variant
        m = env.model
        self.renderer = mujoco.Renderer(m, *IMAGE_HW)
        self.head = m.camera("head").id
        self.head_pos0 = m.cam_pos[self.head].copy()
        self.head_quat0 = m.cam_quat[self.head].copy()
        if variant == "lighting":
            m.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
            m.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
        self.rng = np.random.default_rng(0)
        self.previous: dict | None = None
        self.light_dir = np.array([0.0, 0.0, -1.0])

    def reset(self, seed: int) -> None:
        """Per-episode camera conditions; call after env.reset_task."""
        m, d = self.env.model, self.env.data
        self.rng = np.random.default_rng([seed, 11])
        self.previous = None
        m.cam_pos[self.head] = self.head_pos0
        m.cam_quat[self.head] = self.head_quat0
        if self.variant == "noisy_camera":
            m.cam_pos[self.head] = self.head_pos0 + self.rng.normal(0, HEAD_BUMP_POS_STD, 3)
            rot = self.rng.normal(0, HEAD_BUMP_ROT_STD, 3)
            angle = float(np.linalg.norm(rot))
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, rot / max(angle, 1e-12), angle)
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self.head_quat0, dq)
            m.cam_quat[self.head] = q
        if self.variant == "lighting":
            # From the robot's front left, high up, varying a little between episodes.
            d_ = np.array([-0.45, -0.55, -1.0]) + self.rng.normal(0, 0.12, 3) * [1, 1, 0]
            self.light_dir = d_ / np.linalg.norm(d_)
        mujoco.mj_camlight(m, d)

    def _add_side_light(self, scene) -> None:
        if scene.nlight >= len(scene.lights):
            return
        light = scene.lights[scene.nlight]
        scene.nlight += 1
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        light.dir[:] = self.light_dir
        light.pos[:] = np.array([0.3, -0.1, 0.8]) - 2.0 * self.light_dir
        light.diffuse[:] = SIDE_LIGHT_DIFFUSE
        light.ambient[:] = 0.0
        light.specular[:] = 0.2
        light.attenuation[:] = (1.0, 0.0, 0.0)
        light.cutoff, light.exponent, light.range = 45.0, 10.0, 10.0
        light.bulbradius, light.intensity = 0.02, 0.0
        light.castshadow, light.headlight = True, False
        light.texid, light.id = -1, -1

    def capture(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Images for this control step: {camera}_rgb (H, W, 3) uint8, {camera}_depth (H, W) float32."""
        r = self.renderer
        images = {}
        for cam in CAMERAS:
            r.update_scene(data, cam)
            if self.variant == "lighting":
                self._add_side_light(r.scene)
            images[f"{cam}_rgb"] = r.render().copy()
            r.enable_depth_rendering()
            depth = r.render().astype(np.float32)
            r.disable_depth_rendering()
            depth[depth > NO_READING_M] = 0.0
            images[f"{cam}_depth"] = depth
        if self.variant == "noisy_camera":
            images = self._add_noise(images)
            images, self.previous = (self.previous or images), images  # one control step late
        return images

    def _add_noise(self, images: dict) -> dict:
        rng, out = self.rng, {}
        for cam in CAMERAS:
            rgb = images[f"{cam}_rgb"].astype(np.float32) + rng.normal(0, RGB_NOISE_STD, images[f"{cam}_rgb"].shape)
            out[f"{cam}_rgb"] = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
            d = images[f"{cam}_depth"].astype(np.float64)
            valid = d > 0
            d = d + rng.normal(0, 1, d.shape) * DEPTH_NOISE_STD_PER_M2 * d * d
            d = np.round(d / DEPTH_STEP_M) * DEPTH_STEP_M
            d[~valid | (rng.random(d.shape) < DEPTH_DROPOUT)] = 0.0
            out[f"{cam}_depth"] = d.astype(np.float32)
        return out

    def close(self) -> None:
        self.renderer.close()


# The reference: humanoid-vla's scripted expert planning from cubes found in the head camera ----


HUE_RANGES = {"red": ((340, 360), (0, 20)), "green": ((80, 170),), "blue": ((190, 260),)}


def pixels_to_world(depth: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, fovy_deg: float) -> np.ndarray:
    """World point of every pixel (H, W, 3); depth is along the camera's viewing axis."""
    h, w = depth.shape
    f = (h / 2) / np.tan(np.deg2rad(fovy_deg) / 2)
    v, u = np.mgrid[0:h, 0:w]
    local = np.stack([(u + 0.5 - w / 2) * depth / f, -(v + 0.5 - h / 2) * depth / f, -depth], axis=-1)
    return cam_pos + local @ cam_mat.T


def hue_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rgb.astype(np.float64) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    delta = np.maximum(mx - mn, 1e-9)
    hue = np.where(mx == r, ((g - b) / delta) % 6, np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) * 60.0
    sat = (mx - mn) / np.maximum(mx, 1e-9)
    return hue, sat, mx


def detect_blocks(rgb, depth, cam_pos, cam_mat, fovy_deg, table_height, half) -> dict[str, tuple[np.ndarray, float]]:
    """Cube centre and yaw (mod 90 degrees) for each colour whose top face the camera sees."""
    pts = pixels_to_world(depth, cam_pos, cam_mat, fovy_deg)
    hue, sat, val = hue_saturation(rgb)
    top = (depth > 0) & (np.abs(pts[..., 2] - (table_height + 2 * half)) < 0.008) & (sat > 0.45) & (val > 0.12)
    found = {}
    for color, ranges in HUE_RANGES.items():
        mask = top & np.any([(hue >= lo) & (hue < hi) for lo, hi in ranges], axis=0)
        p = pts[mask][:, :2]
        if len(p) < 15:
            continue
        p = p[np.linalg.norm(p - np.median(p, axis=0), axis=1) < 0.045]
        best = None
        for th in np.deg2rad(np.arange(0.0, 90.0, 1.0)):
            rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            q = p @ rot.T
            lo, hi = q.min(axis=0), q.max(axis=0)
            area = float(np.prod(hi - lo))
            if best is None or area < best[0]:
                best = (area, th, rot.T @ ((lo + hi) / 2))
        _, yaw, center = best
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        found[color] = (np.array([center[0], center[1], table_height + half]), float(yaw))
    return found


class VisionExpert:
    """Reference for camera-only sort: blocks found by colour and depth in the first head image,
    then humanoid-vla's scripted expert plans from them (open loop, as it does with true poses)."""

    def __init__(self, env: TaskEnv, images: dict, head_pos: np.ndarray, head_mat: np.ndarray, fovy_deg: float) -> None:
        from humanoid_vla.envs.sort import PrivilegedState
        from humanoid_vla.policies.expert import ScriptedExpert

        scene = env.config.scene
        found = detect_blocks(images["head_rgb"], images["head_depth"], head_pos, head_mat, fovy_deg,
                              scene.table_height, scene.block_half_size)
        true_state = env.privileged_state()
        if true_state.target not in found:
            raise RuntimeError(f"reference: the {true_state.target} block was not found in the head image")
        self.found = found
        state = PrivilegedState(
            block_pos={c: found.get(c, (np.zeros(3), 0.0))[0] for c in COLORS},
            block_yaw={c: found.get(c, (np.zeros(3), 0.0))[1] for c in COLORS},
            bin_xy=true_state.bin_xy,
            target=true_state.target,
            destination=true_state.destination,
            qpos=true_state.qpos,
        )
        self.expert = ScriptedExpert(env)
        env.privileged_state = lambda: state  # instance attribute shadows the method
        try:
            self.expert.reset()
        finally:
            del env.privileged_state

    def act(self) -> np.ndarray:
        return self.expert.act()


# Talking to the sandboxed controller process ----------------------------------------------------


class WorkerDied(Exception):
    pass


class Worker:
    def __init__(self, cmd: list[str], log_path: Path) -> None:
        self.log = log_path.open("ab")
        env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "HOME": "/tmp", "MUJOCO_GL": "disable"}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, kind: bytes, payload: bytes = b"") -> None:
        try:
            self.proc.stdin.write(kind + struct.pack("<Q", len(payload)) + payload)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as err:
            raise WorkerDied(f"the controller process is gone ({err})") from err

    def recv(self, deadline: float) -> dict:
        n = struct.unpack("<I", self._read(4, deadline))[0]
        if n > 1_000_000:
            raise WorkerDied("the controller process sent an oversized message")
        try:
            msg = json.loads(self._read(n, deadline))
        except ValueError as err:
            raise WorkerDied("the controller process sent a malformed message") from err
        if not isinstance(msg, dict):
            raise WorkerDied("the controller process sent a malformed message")
        return msg

    def _read(self, n: int, deadline: float) -> bytes:
        fd, chunks, got = self.proc.stdout.fileno(), [], 0
        while got < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError
            b = os.read(fd, n - got)
            if not b:
                raise WorkerDied("the controller process ended (crash, memory limit or exit)")
            chunks.append(b)
            got += len(b)
        return b"".join(chunks)

    def close(self) -> None:
        if self.alive():
            try:
                self.send(b"Q")
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        if self.alive():
            self.proc.kill()
            self.proc.wait()
        self.log.close()


# One chunk of episodes (trusted process) ---------------------------------------------------------


def _parse_action(msg: dict) -> np.ndarray:
    a = msg.get("action")
    if not (isinstance(a, list) and len(a) == 8 and all(isinstance(x, (int, float)) for x in a)):
        raise WorkerDied("the controller process sent an invalid action")
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise WorkerDied("the controller process sent a non-finite action")
    return a


def run_chunk(spec: dict) -> list[dict]:
    out_dir = Path(spec["out"])
    inputs = Path(spec["inputs"])
    kind, variant = spec["kind"], spec["variant"]
    wall_limit = float(spec["episode_wall_s"])
    env = TaskEnv(kind, variant)
    cams = Cameras(env, variant)
    kin_model = mujoco.MjModel.from_binary_path(str(inputs / "robot.mjb"))
    q_idx, v_idx = kin_index(env.model, kin_model)
    m = env.model
    state_dof = m.jnt_dofadr[[m.joint(j).id for j in (*env.arm_joints, *env.hand_joints)]]
    site = m.site(GRASP_SITE).id
    builtin = spec["controller"] == "builtin:vision_expert"
    # The worker runs from its own copy in the inputs folder, so nothing under src/ is visible.
    cmd = [str(PYTHON), "-I", str(inputs / "vworker.py"), str(inputs / "worker_spec.json")]
    if spec["sandbox"]:
        cmd = sandboxed_command(cmd, writable=[], readable=[inputs], base_readonly=[REPO / ".venv"])
    worker: Worker | None = None
    episodes = []

    def observation(images: dict) -> bytes:
        d = env.data
        buf = io.BytesIO()
        np.savez(buf, t=np.float64(env.time), qpos=d.qpos[env._state_adr], qvel=d.qvel[state_dof],
                 arm_target=np.asarray(env._arm_target, dtype=float), closure=np.float64(env._closure),
                 kin_qpos=d.qpos[q_idx], kin_qvel=d.qvel[v_idx], **images)
        return buf.getvalue()

    try:
        for seed in spec["seeds"]:
            seed = int(seed)
            goal = env.reset_task(seed)
            cams.reset(seed)
            d = env.data
            start = {c: env.block_pos(c) for c in COLORS}
            qpos_traj, site_traj, closure_traj = [d.qpos.copy()], [d.site_xpos[site].copy()], [0.0]
            error, output, used = None, "", 0.0
            t_start = time.time()

            def step(action: np.ndarray) -> bool:
                _, _, terminated, truncated, _ = env.step(action)
                qpos_traj.append(d.qpos.copy())
                site_traj.append(d.site_xpos[site].copy())
                closure_traj.append(env._closure)
                return terminated or truncated

            if builtin:
                kin = mujoco.MjData(kin_model)
                kin.qpos[:] = d.qpos[q_idx]
                mujoco.mj_forward(kin_model, kin)
                head = kin_model.camera("head").id
                try:
                    ref = VisionExpert(env, cams.capture(d), kin.cam_xpos[head].copy(),
                                       kin.cam_xmat[head].reshape(3, 3).copy(), float(kin_model.cam_fovy[head]))
                    output = "found: " + ", ".join(f"{c} {np.round(p, 3).tolist()} yaw {np.rad2deg(y):.0f}" for c, (p, y) in ref.found.items())
                    while not step(ref.act()):
                        pass
                except RuntimeError as err:
                    error = str(err)
            else:
                try:
                    if worker is None or not worker.alive():
                        if worker is not None:
                            worker.close()
                        worker = Worker(cmd, out_dir / f"server_{spec['index']}_worker.log")
                    public = {"kind": goal.kind, "instruction": goal.instruction, "target": goal.target,
                              "destination": goal.destination, "second": goal.second}
                    t0 = time.monotonic()
                    worker.send(b"E", json.dumps(public).encode())
                    msg = worker.recv(t0 + wall_limit)
                    used += time.monotonic() - t0
                    if "error" in msg:
                        raise RuntimeError(str(msg["error"]))
                    while True:
                        payload = observation(cams.capture(d))
                        t0 = time.monotonic()
                        worker.send(b"O", payload)
                        msg = worker.recv(t0 + max(0.0, wall_limit - used))
                        used += time.monotonic() - t0
                        if "error" in msg:
                            raise RuntimeError(str(msg["error"]))
                        if step(_parse_action(msg)):
                            break
                except RuntimeError as err:
                    error = str(err)[:6000]
                except TimeoutError:
                    error = f"episode stopped: the controller used more than {wall_limit:.0f} s of wall-clock time"
                    worker.close()
                except WorkerDied as err:
                    error = str(err)
                    worker.close()
                if worker is not None and worker.alive():
                    try:
                        worker.send(b"X")
                        output = str(worker.recv(time.monotonic() + 10).get("output", ""))
                    except (TimeoutError, WorkerDied):
                        worker.close()
            qpos_arr = np.asarray(qpos_traj)
            np.savez_compressed(out_dir / f"traj_{seed}.npz", qpos=qpos_arr, site=np.asarray(site_traj),
                                closure=np.asarray(closure_traj), dt=env.control_dt)
            ep = T.Episode(
                seed=seed, instruction=goal.instruction,
                outcome="controller_error" if error else str(env.outcome),
                sim_time=round(env.time, 3), steps=len(qpos_arr) - 1, error=error, output=output,
                wall_s=round(time.time() - t_start, 3),
                **T._metrics(env, goal, start, qpos_arr, np.asarray(site_traj)),
            )
            episodes.append(asdict(ep))
    finally:
        if worker is not None:
            worker.close()
        cams.close()
    return episodes


def prepare_inputs(controller: str, kind: str, inputs: Path) -> None:
    """What the controller process may read: its code, the cube-free robot model and the robot facts."""
    inputs.mkdir(parents=True, exist_ok=True)
    model = robot_model()
    mujoco.mj_saveModel(model, str(inputs / "robot.mjb"), None)
    shutil.copy(WORKER, inputs / "vworker.py")
    spec = {"model_path": str(inputs / "robot.mjb"), "robot": robot_facts(TaskEnv(kind), model)}
    if controller != "builtin:vision_expert":
        shutil.copy(controller, inputs / "controller.py")
        spec["controller"] = str(inputs / "controller.py")
    (inputs / "worker_spec.json").write_text(json.dumps(spec))


def run_vision_trials(controller: str, kind: str, variant: str, seeds: list[int], out_dir: Path,
                      workers: int = 4, sandbox: bool = True, episode_wall_s: float = 60.0) -> list[T.Episode]:
    """Like trials.run_trials, in camera-only mode. ``controller`` is a path or ``builtin:vision_expert``."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if controller != "builtin:vision_expert":
        controller = str(Path(controller).resolve())
    inputs = out_dir / "worker_inputs"
    prepare_inputs(controller, kind, inputs)
    chunks = [list(c) for c in np.array_split(np.asarray(seeds, dtype=int), min(workers, len(seeds))) if len(c)]

    def launch(i: int, chunk: list[int]) -> list[dict]:
        spec = {"controller": controller, "kind": kind, "variant": variant, "seeds": [int(s) for s in chunk],
                "out": str(out_dir), "inputs": str(inputs), "index": i, "episode_wall_s": episode_wall_s,
                "sandbox": sandbox}
        spec_path = out_dir / f"server_{i}.spec.json"
        spec_path.write_text(json.dumps(spec))
        result = out_dir / f"server_{i}.episodes.json"
        env = {**os.environ, "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1"}
        timeout = (episode_wall_s + 30) * len(chunk) + 120
        try:
            proc = subprocess.run([str(PYTHON), "-m", "dexcoder.vision", str(spec_path)], capture_output=True,
                                  text=True, timeout=timeout, env=env, cwd=REPO)
            (out_dir / f"server_{i}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
            failure = None if result.exists() else "the episode runner failed:\n" + proc.stderr[-3000:]
        except subprocess.TimeoutExpired:
            failure = "the episode runner timed out"
        if failure:
            return [asdict(T.Episode(seed=s, instruction=None, outcome="sandbox_error", sim_time=0.0, steps=0,
                                     error=failure, output="", wall_s=0.0)) for s in chunk]
        return json.loads(result.read_text())

    with ThreadPoolExecutor(len(chunks)) as pool:
        records = [r for rs in pool.map(launch, range(len(chunks)), chunks) for r in rs]
    episodes = [T.Episode(**r) for r in records]
    episodes.sort(key=lambda e: list(seeds).index(e.seed))
    (out_dir / "episodes.json").write_text(json.dumps([asdict(e) for e in episodes], indent=1))
    return episodes


# Observations on demand, for the agent's get_observations tool (practice seeds only) -----------


def save_observations(kind: str, seed: int, times: list[float], traj_path: Path | None, out_dir: Path) -> list[str]:
    """Write what a controller would observe at each time (nominal conditions) plus the true cube
    poses, one .npz per time. Times after 0 need the episode's trajectory from run_trials."""
    env = TaskEnv(kind)
    env.reset_task(seed)
    cams = Cameras(env, "nominal")
    cams.reset(seed)
    kin_model = robot_model()
    q_idx, _ = kin_index(env.model, kin_model)
    kin = mujoco.MjData(kin_model)
    traj = dict(np.load(traj_path)) if traj_path is not None and traj_path.exists() else None
    d, saved = env.data, []
    qpos0 = d.qpos.copy()
    for t in times:
        i = int(round(t / env.control_dt))
        if i > 0:
            if traj is None:
                raise ValueError("times after 0 need this seed in the latest run_trials")
            i = min(i, len(traj["qpos"]) - 1)
            d.qpos[:] = traj["qpos"][i]
        else:
            d.qpos[:] = qpos0
        mujoco.mj_forward(env.model, d)
        images = cams.capture(d)
        kin.qpos[:] = d.qpos[q_idx]
        mujoco.mj_forward(kin_model, kin)
        arrays = dict(images)
        for cam in CAMERAS:
            cid = kin_model.camera(cam).id
            arrays[f"{cam}_pos"] = kin.cam_xpos[cid].copy()
            arrays[f"{cam}_mat"] = kin.cam_xmat[cid].reshape(3, 3).copy()
        arrays["qpos"] = d.qpos[env._state_adr].copy()
        arrays["closure"] = np.float64(traj["closure"][i] if traj is not None and i > 0 else 0.0)
        arrays["t"] = np.float64(i * env.control_dt)
        for c in COLORS:
            arrays[f"true_{c}_pos"] = env.block_pos(c)
            arrays[f"true_{c}_yaw"] = np.float64(_yaw(env.block_quat(c)))
        name = f"obs_{seed}_t{i * env.control_dt:.2f}.npz"
        np.savez_compressed(out_dir / name, **arrays)
        saved.append(name)
    cams.close()
    return saved


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> int:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    episodes = run_chunk(spec)
    (spec_path.parent / f"server_{spec['index']}.episodes.json").write_text(json.dumps(episodes))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CAMERA_VARIANTS", "Cameras", "detect_blocks", "run_vision_trials", "save_observations"]
