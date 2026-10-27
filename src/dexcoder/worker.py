"""Run a controller's episodes and record its actions. Runs inside the sandbox.

    python -I -m dexcoder.worker SPEC.json

SPEC names the controller file (or ``builtin:expert``), task kind, physics variant, seeds and
limits. The worker writes ``<out>.json`` (one record per episode) and ``<out>.npz`` (the
executed actions). Outcomes here are only provisional: the trusted side replays the recorded
actions in a fresh simulator and takes the official outcome from that replay.
"""

import argparse
import contextlib
import importlib.util
import json
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUTPUT_LIMIT = 4000  # characters of printed output kept per episode


class EpisodeTimeout(Exception):
    pass


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


def _alarm(signum, frame):  # noqa: ARG001
    raise EpisodeTimeout


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


class ExpertAdapter:
    """humanoid-vla's scripted expert behind the controller interface (the human reference).

    Under ``noisy_sensing`` it plans from the same noisy block poses controllers observe.
    """

    def __init__(self, env, observer) -> None:
        from humanoid_vla.envs.sort import PrivilegedState
        from humanoid_vla.policies.expert import ScriptedExpert

        self.expert = ScriptedExpert(env)
        if env.variant == "noisy_sensing":
            obs = observer._snapshot()
            true_state = env.privileged_state()
            noisy = PrivilegedState(
                block_pos={c: b["pos"] for c, b in obs["blocks"].items()},
                block_yaw={c: b["yaw"] for c, b in obs["blocks"].items()},
                bin_xy=true_state.bin_xy,
                target=true_state.target,
                destination=true_state.destination,
                qpos=true_state.qpos,
            )
            env.privileged_state = lambda: noisy  # instance attribute shadows the method
            try:
                self.expert.reset()
            finally:
                del env.privileged_state
        else:
            self.expert.reset()

    def act(self, obs):  # noqa: ARG002
        return self.expert.act()


def run_episode(env, nominal, controller_cls, seed: int, wall_limit_s: float) -> tuple[dict, np.ndarray]:
    from dexcoder.robot_api import Observer, robot_info

    goal = env.reset_task(seed)
    observer = Observer(env, nominal, seed)
    out = LimitedWriter(OUTPUT_LIMIT)
    actions: list[np.ndarray] = []
    error = None
    start = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, wall_limit_s)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            if controller_cls == "builtin:expert":
                controller = ExpertAdapter(env, observer)
            else:
                controller = controller_cls(robot_info(env, nominal), goal.as_dict())
            while True:
                action = np.asarray(controller.act(observer.observe()), dtype=float).reshape(-1)
                _, _, terminated, truncated, info = env.step(action)
                actions.append(info["applied_action"])
                if terminated or truncated:
                    break
    except EpisodeTimeout:
        error = f"episode stopped: the controller used more than {wall_limit_s:.0f} s of wall-clock time"
    except BaseException:  # noqa: BLE001 - any failure of controller code ends the episode
        error = _short_traceback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    record = {
        "seed": seed,
        "instruction": goal.instruction,
        "provisional_outcome": "controller_error" if error else str(env.outcome),
        "error": error,
        "sim_time": env.time,
        "steps": len(actions),
        "wall_s": round(time.time() - start, 3),
        "output": out.text(),
        "final_qpos": env.data.qpos.tolist(),
    }
    arr = np.asarray(actions, dtype=np.float32).reshape(-1, 8)
    return record, arr


def _short_traceback(max_lines: int = 30) -> str:
    lines = traceback.format_exc().rstrip().splitlines()
    return "\n".join(lines[-max_lines:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    spec = json.loads(Path(parser.parse_args().spec).read_text())
    mem = int(spec.get("memory_limit_gb", 4) * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    from humanoid_vla.sim.scene import build_model

    from dexcoder.tasks import TaskEnv

    out = Path(spec["out"])
    records, arrays = [], {}
    nominal = build_model()
    env = TaskEnv(spec["kind"], spec["variant"])
    try:
        if spec["controller"] == "builtin:expert":
            controller_cls = "builtin:expert"
        else:
            controller_cls = load_controller_class(spec["controller"])
    except BaseException:  # noqa: BLE001
        error = "controller.py failed to load:\n" + _short_traceback()
        for seed in spec["seeds"]:
            records.append({"seed": seed, "provisional_outcome": "controller_error", "error": error,
                            "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "output": "", "final_qpos": None,
                            "instruction": None})
            arrays[f"seed_{seed}"] = np.zeros((0, 8), np.float32)
        controller_cls = None
    if controller_cls is not None:
        for seed in spec["seeds"]:
            record, arr = run_episode(env, nominal, controller_cls, int(seed), spec["episode_wall_s"])
            records.append(record)
            arrays[f"seed_{seed}"] = arr
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    out.with_suffix(".json").write_text(json.dumps(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
