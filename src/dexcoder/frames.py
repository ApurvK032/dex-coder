"""Camera frames from replayed episodes (trusted side, GPU rendering through EGL)."""

import io
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from humanoid_vla.sim.render import FULL_QUALITY, CameraRenderer  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dexcoder.tasks import TaskEnv  # noqa: E402

CAMERAS = ("scene", "head", "wrist")
TILE_HW = (300, 400)


def contact_sheet(traj_path: Path, kind: str, times: list[float], cameras: list[str]) -> bytes:
    """PNG with one row per time and one column per camera."""
    env = TaskEnv(kind)
    m, d = env.model, mujoco.MjData(env.model)
    with np.load(traj_path) as z:
        qpos, dt = z["qpos"], float(z["dt"])
    h, w = TILE_HW
    times = times[:6]
    sheet = Image.new("RGB", (w * len(cameras), h * len(times)), "white")
    with CameraRenderer(m, h, w, FULL_QUALITY) as renderer:
        for row, t in enumerate(times):
            i = int(np.clip(round(t / dt), 0, len(qpos) - 1))
            d.qpos[:] = qpos[i]
            mujoco.mj_forward(m, d)
            for col, cam in enumerate(cameras):
                tile = Image.fromarray(renderer.render(d, cam))
                ImageDraw.Draw(tile).text((6, 4), f"t={i * dt:.2f}s {cam}", fill=(255, 255, 0))
                sheet.paste(tile, (col * w, row * h))
    env.close()
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def video(traj_path: Path, kind: str, out_path: Path, camera: str = "scene", fps: int = 25) -> None:
    import imageio.v3 as iio

    env = TaskEnv(kind)
    m, d = env.model, mujoco.MjData(env.model)
    with np.load(traj_path) as z:
        qpos = z["qpos"]
    frames = []
    with CameraRenderer(m, 480, 640, FULL_QUALITY) as renderer:
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            frames.append(renderer.render(d, camera))
    env.close()
    iio.imwrite(out_path, np.asarray(frames), fps=fps)
