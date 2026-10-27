"""Run a controller on seeds (sandboxed workers), then replay and judge every episode (trusted).

The workers execute controller code inside the sandbox and record the executed actions. This
process replays those actions in a fresh simulator. The replay's outcome is the official one;
if its final state differs from what the worker reported, the episode is marked
``integrity_mismatch``. Physics is deterministic, so an honest controller always matches.
"""

import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from humanoid_vla.envs.task import COLORS

from dexcoder.sandbox import sandboxed_command
from dexcoder.tasks import TaskEnv

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"


@dataclass
class Episode:
    seed: int
    instruction: str | None
    outcome: str
    sim_time: float
    steps: int
    error: str | None
    output: str
    wall_s: float
    blocks: dict = field(default_factory=dict)  # per color: start, final, max_lift, moved, where
    first_lift_s: float | None = None
    min_site_to_target: float | None = None


def run_trials(
    controller: str,
    kind: str,
    variant: str,
    seeds: list[int],
    out_dir: Path,
    workers: int = 4,
    sandbox: bool = True,
    episode_wall_s: float = 60.0,
    mode: str = "state",
) -> list[Episode]:
    """``controller`` is a path to controller.py or ``builtin:expert``. ``mode="vision"`` runs the
    camera-only mode (dexcoder.vision), where controllers get images instead of cube poses."""
    if mode == "vision":
        from dexcoder.vision import run_vision_trials

        return run_vision_trials(controller, kind, variant, seeds, out_dir, workers, sandbox, episode_wall_s)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [list(c) for c in np.array_split(np.asarray(seeds, dtype=int), min(workers, len(seeds))) if len(c)]
    if controller != "builtin:expert":
        controller = str(Path(controller).resolve())

    def launch(i: int, chunk: list[int]) -> Path:
        stem = out_dir / f"worker_{i}"
        spec = {
            "controller": controller,
            "kind": kind,
            "variant": variant,
            "seeds": [int(s) for s in chunk],
            "out": str(stem),
            "episode_wall_s": episode_wall_s,
        }
        spec_path = stem.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec))
        cmd = [str(PYTHON), "-I", "-m", "dexcoder.worker", str(spec_path)]
        if sandbox:
            cmd = sandboxed_command(cmd, writable=[out_dir], readable=_readable_paths(controller))
        timeout = episode_wall_s * len(chunk) + 60
        env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "HOME": "/tmp"}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=out_dir)
        (out_dir / f"worker_{i}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
        return stem

    with ThreadPoolExecutor(len(chunks)) as pool:
        futures = [pool.submit(launch, i, c) for i, c in enumerate(chunks)]
        stems = []
        for f, chunk in zip(futures, chunks, strict=True):
            try:
                stems.append((f.result(), chunk, None))
            except subprocess.TimeoutExpired:
                stems.append((None, chunk, "worker timed out"))

    episodes: list[Episode] = []
    env = TaskEnv(kind, variant)
    for stem, chunk, failure in stems:
        records, actions = _load_worker(stem, chunk, failure)
        for rec in records:
            episodes.append(_replay(env, rec, actions.get(f"seed_{rec['seed']}"), out_dir))
    episodes.sort(key=lambda e: seeds.index(e.seed))
    (out_dir / "episodes.json").write_text(json.dumps([asdict(e) for e in episodes], indent=1))
    return episodes


def _readable_paths(controller: str) -> list[Path]:
    return [] if controller == "builtin:expert" else [Path(controller).parent]


def _load_worker(stem: Path | None, chunk: list[int], failure: str | None) -> tuple[list[dict], dict]:
    def failed(msg: str) -> tuple[list[dict], dict]:
        return [
            {"seed": s, "instruction": None, "provisional_outcome": "sandbox_error", "error": msg,
             "sim_time": 0.0, "steps": 0, "wall_s": 0.0, "output": "", "final_qpos": None}
            for s in chunk
        ], {}

    if stem is None:
        return failed(failure or "worker failed")
    json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if not (json_path.exists() and npz_path.exists()):
        log = (stem.parent / f"{stem.name}.log")
        tail = log.read_text()[-3000:] if log.exists() else ""
        return failed("the worker process ended without results (crash, memory limit or os.exit?).\n" + tail)
    try:
        records = json.loads(json_path.read_text())
        with np.load(npz_path) as z:
            actions = {k: z[k] for k in z.files}
    except Exception as err:  # noqa: BLE001
        return failed(f"worker results unreadable: {err}")
    return records, actions


def _replay(env: TaskEnv, rec: dict, actions: np.ndarray | None, out_dir: Path) -> Episode:
    seed = int(rec["seed"])
    goal = env.reset_task(seed)
    m, d = env.model, env.data
    site = m.site("grasp_site").id
    start = {c: env.block_pos(c) for c in COLORS}
    qpos_traj = [d.qpos.copy()]
    site_traj = [d.site_xpos[site].copy()]
    closure_traj = [0.0]
    outcome = "running"
    actions = actions if actions is not None else np.zeros((0, 8), np.float32)
    ended_at = None
    for i, a in enumerate(actions):
        _, _, terminated, truncated, _ = env.step(a.astype(float))
        qpos_traj.append(d.qpos.copy())
        site_traj.append(d.site_xpos[site].copy())
        closure_traj.append(env._closure)
        if terminated or truncated:
            outcome = str(env.outcome)
            ended_at = i
            break
    provisional = rec["provisional_outcome"]
    if provisional in ("controller_error", "sandbox_error"):
        outcome = provisional
    elif ended_at is None or ended_at != len(actions) - 1:
        outcome = "integrity_mismatch"
    elif rec.get("final_qpos") is None or not np.array_equal(np.asarray(rec["final_qpos"]), d.qpos):
        outcome = "integrity_mismatch"
    qpos_traj = np.asarray(qpos_traj)
    np.savez_compressed(
        out_dir / f"traj_{seed}.npz",
        qpos=qpos_traj,
        site=np.asarray(site_traj),
        closure=np.asarray(closure_traj),
        dt=env.control_dt,
    )
    return Episode(
        seed=seed,
        instruction=goal.instruction,
        outcome=outcome,
        sim_time=round(env.time, 3),
        steps=len(qpos_traj) - 1,
        error=rec.get("error"),
        output=rec.get("output", ""),
        wall_s=rec.get("wall_s", 0.0),
        **_metrics(env, goal, start, qpos_traj, np.asarray(site_traj)),
    )


def _metrics(env: TaskEnv, goal, start: dict, qpos_traj: np.ndarray, site_traj: np.ndarray) -> dict:
    scene = env.config.scene
    rest_z = scene.table_height + scene.block_half_size
    blocks, first_lift = {}, None
    target = goal.target
    for c in COLORS:
        adr = env._block_qpos[c]
        traj = qpos_traj[:, adr : adr + 3]
        lift = traj[:, 2] - rest_z
        moved = float(np.linalg.norm(traj[-1] - traj[0]))
        blocks[c] = {
            "start": np.round(start[c], 3).tolist(),
            "final": np.round(traj[-1], 3).tolist(),
            "max_lift_cm": round(float(lift.max()) * 100, 1),
            "moved_cm": round(moved * 100, 1),
            "where": _where(env, traj[-1]),
        }
        if c == target:
            lifted = np.nonzero(lift > 0.02)[0]
            first_lift = round(float(lifted[0]) * env.control_dt, 2) if len(lifted) else None
    tgt = qpos_traj[:, env._block_qpos[target] : env._block_qpos[target] + 3]
    min_dist = float(np.min(np.linalg.norm(site_traj - tgt, axis=1)))
    return {"blocks": blocks, "first_lift_s": first_lift, "min_site_to_target": round(min_dist, 3)}


def _where(env: TaskEnv, pos: np.ndarray) -> str:
    from humanoid_vla.envs.outcome import in_bin, off_table

    scene = env.config.scene
    if off_table(pos, scene):
        return "off the table"
    for side, xy in scene.bin_xy.items():
        if in_bin(pos, xy, scene):
            return f"in the {side} bin"
    if pos[2] > scene.table_height + 3 * scene.block_half_size:
        return "above the table (held or on top of something)"
    if pos[2] > scene.table_height + 1.5 * scene.block_half_size:
        return "raised (on another block, a bin wall, or tilted)"
    return "on the table"


# Summaries for the agent and for reports ----------------------------------------------------


def outcome_counts(episodes: list[Episode]) -> dict[str, int]:
    return dict(Counter(e.outcome for e in episodes).most_common())


def success_rate(episodes: list[Episode]) -> float:
    return sum(e.outcome == "success" for e in episodes) / max(1, len(episodes))


def summarize_for_agent(episodes: list[Episode], max_detail: int = 12) -> str:
    n = len(episodes)
    k = sum(e.outcome == "success" for e in episodes)
    lines = [f"{k}/{n} succeeded. Outcomes: {outcome_counts(episodes)}", ""]
    shown = sorted(episodes, key=lambda e: e.outcome == "success")[:max_detail]
    for e in shown:
        lines.append(f"seed {e.seed}: {e.instruction} -> {e.outcome} at t={e.sim_time:.2f}s"
                     + (f", target first lifted at {e.first_lift_s}s" if e.first_lift_s is not None else ", target never lifted 2 cm")
                     + f", closest grasp_site-to-target distance {e.min_site_to_target} m")
        for c, b in e.blocks.items():
            if b["moved_cm"] < 0.5:
                lines.append(f"    {c}: untouched at {b['start']}")
            else:
                lines.append(f"    {c}: start {b['start']} -> final {b['final']} ({b['where']}), max lift {b['max_lift_cm']} cm")
        if e.error:
            lines.append("    error: " + e.error.replace("\n", "\n    "))
        if e.output:
            out = e.output if len(e.output) < 800 else e.output[:800] + " [...]"
            lines.append("    printed: " + out.strip().replace("\n", "\n      "))
    if n > max_detail:
        hidden = episodes and sorted(episodes, key=lambda e: e.outcome == "success")[max_detail:]
        lines.append(f"({n - max_detail} more episodes not shown: {outcome_counts(hidden)})")
    return "\n".join(lines)


def episode_table(out_dir: Path, seed: int, kind: str, start_s: float, end_s: float, every_s: float) -> str:
    """Time series of one replayed episode: grasp site, closure and block positions."""
    path = out_dir / f"traj_{seed}.npz"
    if not path.exists():
        return f"No episode with seed {seed} in the latest run."
    env = TaskEnv(kind)
    with np.load(path) as z:
        qpos, site, closure, dt = z["qpos"], z["site"], z["closure"], float(z["dt"])
    step = max(1, round(every_s / dt))
    lo, hi = max(0, round(start_s / dt)), min(len(qpos) - 1, round(end_s / dt))
    header = "t(s)   grasp_site xyz           closure  " + "  ".join(f"{c:>21s}" for c in COLORS)
    rows = [header]
    for i in range(lo, hi + 1, step):
        cols = []
        for c in COLORS:
            adr = env._block_qpos[c]
            cols.append("(" + ",".join(f"{v:6.3f}" for v in qpos[i, adr : adr + 3]) + ")")
        s = site[i]
        rows.append(f"{i * dt:5.2f}  ({s[0]:6.3f},{s[1]:6.3f},{s[2]:6.3f})  {closure[i]:5.2f}   " + "  ".join(cols))
    if len(rows) > 151:
        rows = rows[:151] + ["[table truncated: use a larger every_s or a shorter window]"]
    return "\n".join(rows)


def worker_env_note() -> str:
    return f"workers: {os.cpu_count()} CPUs"


def timed(fn, *args, **kwargs):
    t = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t
