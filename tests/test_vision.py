"""Camera-only mode: the controller process sees images and joints only, the documented pixel
geometry is right, the reference solves the task from the camera, and the agent loop works."""

import json

import numpy as np
import pytest

from dexcoder import agent as A
from dexcoder import trials as T
from dexcoder.ledger import Ledger
from dexcoder.sandbox import SandboxUnavailable, check_sandbox
from dexcoder.tasks import TaskEnv
from dexcoder.vision import Cameras, pixels_to_world, hue_saturation, kin_index, robot_model
from tests.fake_client import FakeClient


def sandbox_available() -> bool:
    try:
        check_sandbox()
        return True
    except SandboxUnavailable:
        return False


needs_sandbox = pytest.mark.skipif(not sandbox_available(), reason="bubblewrap cannot start (AppArmor)")

PROBE = """
import importlib, os
import numpy as np
class Controller:
    def __init__(self, robot, task):
        self.home = np.asarray(robot.home_arm)
        print("TASK", sorted(task))
        for mod in ("dexcoder", "dexcoder.tasks", "humanoid_vla"):
            try:
                importlib.import_module(mod)
                print("IMPORTABLE", mod)
            except ImportError:
                pass
        print("HOME", os.path.exists(os.path.expanduser("~apurv/.config")) or os.path.exists("/home/apurv/.config"))
        self.n = 0
    def act(self, obs):
        if self.n == 0:
            print("OBS", sorted(obs), sorted(obs["cameras"]["head"]), obs["cameras"]["head"]["rgb"].shape)
            print("NQ", obs["kin"].qpos.shape[0])
        self.n += 1
        return np.append(self.home, 0.0)
"""


@needs_sandbox
def test_controller_process_gets_images_and_joints_only(tmp_path):
    ctl = tmp_path / "probe.py"
    ctl.write_text(PROBE)
    (ep,) = T.run_trials(str(ctl), "sort", "nominal", [100000], tmp_path / "out", workers=1, mode="vision")
    assert ep.outcome == "timeout" and ep.error is None
    assert "TASK ['destination', 'instruction', 'kind', 'second', 'target']" in ep.output
    assert "IMPORTABLE" not in ep.output
    assert "HOME False" in ep.output
    assert "OBS ['arm_target', 'cameras', 'closure', 'grasp_site_mat', 'grasp_site_pos', 'kin', 'qpos', 'qvel', 't'] ['depth', 'mat', 'pos', 'rgb'] (240, 320, 3)" in ep.output
    # The controller's kinematics model has no cube joints: only the robot's.
    assert f"NQ {robot_model().nq}" in ep.output and robot_model().nq == TaskEnv("sort").model.nq - 3 * 7


def test_documented_pixel_geometry():
    """Red top-face pixels, back-projected with the formula in the prompt, land on the red cube."""
    env = TaskEnv("sort")
    env.reset_task(100004)
    cams = Cameras(env, "nominal")
    cams.reset(100004)
    images = cams.capture(env.data)
    cams.close()
    kin_model = robot_model()
    import mujoco

    kin = mujoco.MjData(kin_model)
    q_idx, _ = kin_index(env.model, kin_model)
    kin.qpos[:] = env.data.qpos[q_idx]
    mujoco.mj_forward(kin_model, kin)
    cid = kin_model.camera("head").id
    pts = pixels_to_world(images["head_depth"], kin.cam_xpos[cid], kin.cam_xmat[cid].reshape(3, 3), kin_model.cam_fovy[cid])
    hue, sat, _ = hue_saturation(images["head_rgb"])
    top_z = env.block_pos("red")[2] + 0.025
    red_top = ((hue > 340) | (hue < 20)) & (sat > 0.45) & (np.abs(pts[..., 2] - top_z) < 0.004)
    assert red_top.sum() > 50
    w, x, y, z = env.block_quat("red")
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rel = pts[red_top][:, :2] - env.block_pos("red")[:2]
    local = rel @ np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    assert np.abs(local).max() < 0.025 + 0.003


def test_reference_sorts_from_the_camera(tmp_path):
    for variant in ("nominal", "noisy_camera", "lighting"):
        eps = T.run_trials("builtin:vision_expert", "sort", variant, list(range(100000, 100004)), tmp_path / variant,
                           workers=2, mode="vision")
        assert sum(e.outcome == "success" for e in eps) >= 3, (variant, T.outcome_counts(eps))


IDLE = """
import numpy as np
class Controller:
    def __init__(self, robot, task):
        self.home = np.asarray(robot.home_arm)
    def act(self, obs):
        return np.append(self.home, 0.0)
"""

LOAD = """
import numpy as np
z = np.load("obs_100000_t1.00.npz")
print(z["head_rgb"].shape, z["head_depth"].dtype, round(float(z["t"]), 2), z["true_red_pos"].shape)
"""


@pytest.mark.parametrize("sandbox", [False, pytest.param(True, marks=needs_sandbox)])
def test_scripted_vision_run(tmp_path, monkeypatch, sandbox):
    monkeypatch.setattr(A, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(A, "STOP_FILE", tmp_path / "STOP")
    script = [
        ("write_controller", {"code": IDLE}),
        ("run_trials", {"num_episodes": 2}),
        ("get_observations", {"seed": 100000, "times": [0, 1.0]}),
        ("get_observations", {"seed": 100500, "times": [2.0]}),
        ("run_python", {"code": LOAD}),
        ("view_frames", {"seed": 100001, "times": [0.5], "cameras": ["head"]}),
        ("submit", {"notes": "idle"}),
    ]
    client = FakeClient(script)
    cfg = A.RunConfig(run_id="vfake", kind="sort", model="claude-sonnet-5-5", max_turns=20, max_usd=5, mode="vision")
    state = A.AgentRun(cfg, Ledger(tmp_path / "ledger.jsonl", 50), client, sandbox=sandbox).run()
    assert state.end_reason == "submitted", state.end_reason
    run_dir = tmp_path / "runs" / "vfake"
    lines = [json.loads(x) for x in (run_dir / "transcript.jsonl").read_text().splitlines()]
    results = {i: line["tool_results"][0] for i, line in enumerate(lines)}
    assert state.trials[0]["outcomes"] == {"timeout": 2}
    assert "obs_100000_t0.00.npz, obs_100000_t1.00.npz" in results[2]["content"][0]["text"]
    assert results[3]["is_error"] and "latest run_trials" in results[3]["content"]
    assert "(240, 320, 3) float32 1.0 (3,)" in results[4]["content"][0]["text"]
    assert results[5]["content"][1]["type"] == "image"
    names = [t["name"] for t in client.requests[0]["tools"]]
    assert "get_observations" in names and "cameras" in client.requests[0]["system"]
    assert "block_xy" not in client.requests[0]["system"]
