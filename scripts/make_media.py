"""Small README GIFs of Claude-written controllers on held-out test layouts.

    .venv/bin/python scripts/make_media.py RUN_ID [RUN_ID ...] [--seed SEED]

For each run, picks its first successful nominal test episode (or --seed) from
outputs/eval/<run>/final/nominal/ and writes docs/media/<run>.gif (scene camera). With
--with-head the robot's head camera, what a camera-only controller sees, is placed beside it.
"""

import argparse
import json
import sys
from pathlib import Path

import dexcoder  # noqa: F401  (selects EGL before MuJoCo loads)
import mujoco
import numpy as np
from humanoid_vla.sim.render import FULL_QUALITY, CameraRenderer
from PIL import Image

from dexcoder.tasks import TaskEnv

REPO = Path(__file__).resolve().parents[1]


def frames_for(traj: Path, kind: str, camera: str = "scene") -> list[np.ndarray]:
    env = TaskEnv(kind)
    m, d = env.model, mujoco.MjData(env.model)
    with np.load(traj) as z:
        qpos = z["qpos"]
    out = []
    with CameraRenderer(m, 360, 480, FULL_QUALITY) as renderer:
        for q in qpos:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            out.append(renderer.render(d, camera))
    env.close()
    return out


def write_gif(frames: list[np.ndarray], path: Path, width: int = 480, every: int = 2, frame_ms: int = 40) -> None:
    """Like humanoid_vla.utils.media.write_gif, but the shared palette is built with the cubes'
    saturated pixels weighted up, so small red, green and blue cubes keep their colours."""
    height = round(frames[0].shape[0] * width / frames[0].shape[1])
    images = [Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS) for f in frames[::every]]
    samples = [np.asarray(im) for im in images[:: max(1, len(images) // 8)]]
    montage = np.concatenate(samples, axis=0)
    c = montage.astype(float) / 255
    sat = (c.max(-1) - c.min(-1)) / np.maximum(c.max(-1), 1e-6)
    vivid = montage[(sat > 0.5) & (c.max(-1) > 0.15)]
    if len(vivid):
        rows = max(1, montage.shape[0] // 3)
        strip = np.resize(vivid, (rows * montage.shape[1], 3)).reshape(rows, montage.shape[1], 3)
        montage = np.concatenate([montage, strip], axis=0)
    palette = Image.fromarray(montage.astype(np.uint8)).quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    images[0].save(path, save_all=True, append_images=images[1:], duration=frame_ms * every, loop=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seed", type=int)
    p.add_argument("--with-head", action="store_true", help="put the head camera view beside the scene view")
    a = p.parse_args()
    media = REPO / "docs" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for run_id in a.runs:
        kind = json.loads((REPO / "outputs/runs" / run_id / "run.json").read_text())["config"]["kind"]
        nominal = REPO / "outputs/eval" / run_id / "final" / "nominal"
        episodes = json.loads((nominal / "episodes.json").read_text())
        ep = next(e for e in episodes if (a.seed is None and e["outcome"] == "success") or e["seed"] == a.seed)
        frames = frames_for(nominal / f"traj_{ep['seed']}.npz", kind)
        width = 400
        if a.with_head:
            head = frames_for(nominal / f"traj_{ep['seed']}.npz", kind, camera="head")
            frames = [np.concatenate([f, h], axis=1) for f, h in zip(frames, head, strict=True)]
            width = 640
        path = media / f"{run_id}.gif"
        write_gif(frames, path, width=width, every=3)
        print(f"{path.relative_to(REPO)}: seed {ep['seed']}, {ep['instruction']} -> {ep['outcome']}, "
              f"{len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
