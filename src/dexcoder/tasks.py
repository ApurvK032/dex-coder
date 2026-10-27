"""The three tasks, how their layouts are drawn, and how the simulator judges them.

All tasks use humanoid-vla's scene: a fixed-base Unitree G1 whose right arm and Dex3 hand are
active, three 5 cm blocks on a table and two bins. ``sort`` is humanoid-vla's own task and keeps
its layouts and judge exactly, so its scripted expert is a direct human reference. ``stack`` and
``pair`` are new and have no reference.

Seeds: the agent practises on ``PRACTICE_SEEDS``; final scores use ``TEST_SEEDS``, which the
agent never sees. Physics variants other than ``nominal`` are never shown to the agent either.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from humanoid_vla.envs.outcome import Outcome, in_bin, off_table
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.envs.task import BINS, COLORS, sample_task
from humanoid_vla.sim.scene import SceneConfig

TASK_KINDS = ("sort", "stack", "pair")
TIME_LIMIT_S = {"sort": 25.0, "stack": 25.0, "pair": 40.0}

PRACTICE_SEEDS = range(100_000, 101_000)
TEST_SEEDS = range(300_000, 300_100)

VARIANTS = ("nominal", "heavy_slippery", "weak_motors", "noisy_sensing")
HELD_OUT_VARIANTS = VARIANTS[1:]
# Camera-only mode (dexcoder.vision): the physics is nominal; the cameras change.
CAMERA_VARIANTS = ("lighting", "noisy_camera")
VISION_VARIANTS = ("nominal", "heavy_slippery", "weak_motors", *CAMERA_VARIANTS)

# Stacking tolerances: top block centred on the bottom one and resting on it.
STACK_XY_TOL = 0.02  # m
STACK_Z_TOL = 0.01  # m


@dataclass(frozen=True)
class Task:
    """One episode's goal and starting layout.

    ``sort``: put ``target`` in bin ``destination``.
    ``stack``: put ``target`` on top of ``destination`` (a block color).
    ``pair``: put ``target`` and ``second`` in bin ``destination``.
    """

    kind: str
    block_xy: dict[str, tuple[float, float]]
    block_yaw: dict[str, float]
    target: str
    destination: str
    second: str | None = None

    @property
    def instruction(self) -> str:
        if self.kind == "sort":
            return f"Put the {self.target} block in the {self.destination} bin."
        if self.kind == "stack":
            return f"Put the {self.target} block on top of the {self.destination} block."
        return f"Put the {self.target} block and the {self.second} block in the {self.destination} bin."

    def as_dict(self) -> dict:
        d = asdict(self)
        d["instruction"] = self.instruction
        return d


def make_task(kind: str, seed: int) -> Task:
    """Layout and goal for ``seed``. ``sort`` reproduces humanoid-vla's layouts seed for seed."""
    if kind == "sort":
        t = sample_task(np.random.default_rng(seed))
        return Task("sort", t.block_xy, t.block_yaw, t.target, t.destination)
    rng = np.random.default_rng([seed, TASK_KINDS.index(kind)])
    layout = sample_task(rng)
    if kind == "stack":
        top, bottom = (str(c) for c in rng.choice(COLORS, size=2, replace=False))
        return Task("stack", layout.block_xy, layout.block_yaw, top, bottom)
    if kind == "pair":
        a, b = (str(c) for c in rng.choice(COLORS, size=2, replace=False))
        side = str(rng.choice(BINS))
        return Task("pair", layout.block_xy, layout.block_yaw, a, side, second=b)
    raise ValueError(f"unknown task kind {kind!r}")


class TaskEnv(SortEnv):
    """humanoid-vla's sorting environment with the three task kinds and physics variants."""

    def __init__(self, kind: str, variant: str = "nominal") -> None:
        if kind not in TASK_KINDS:
            raise ValueError(f"unknown task kind {kind!r}")
        if variant not in (*VARIANTS, *CAMERA_VARIANTS):
            raise ValueError(f"unknown variant {variant!r}")
        scene = SceneConfig()
        if variant == "heavy_slippery":
            # Mass is set before compiling: editing a compiled model's masses leaves MuJoCo's
            # derived constants stale and made contacts unstable.
            scene = replace(scene, block_mass=scene.block_mass * HEAVY_MASS_SCALE)
        super().__init__(SortEnvConfig(scene=scene, cameras=(), time_limit_s=TIME_LIMIT_S[kind]))
        self.kind = kind
        self.variant = variant
        apply_physics_variant(self, variant)
        self._goal: Task | None = None

    def reset_task(self, seed: int) -> Task:
        goal = make_task(self.kind, seed)
        if self.kind == "sort":
            # Let SortEnv draw the layout itself so seeds match humanoid-vla exactly.
            self.reset(seed=seed)
            t = self.task
            assert (t.target, t.destination) == (goal.target, goal.destination)
        else:
            self.reset(seed=seed, options={"task": goal})
        self._goal = goal
        return goal

    @property
    def goal(self) -> Task:
        if self._goal is None:
            raise RuntimeError("call reset_task() first")
        return self._goal

    def block_pos(self, color: str) -> np.ndarray:
        return self.data.xpos[self._block_body[color]].copy()

    def block_quat(self, color: str) -> np.ndarray:
        return self.data.xquat[self._block_body[color]].copy()

    def block_linvel(self, color: str) -> np.ndarray:
        dof = self._block_dof[color]
        return self.data.qvel[dof : dof + 3].copy()

    # Judging ---------------------------------------------------------------------------------

    def _judge(self) -> Outcome:
        if self.kind == "sort":
            return super()._judge()
        d, scene, cfg, goal = self.data, self.config.scene, self.config, self.goal
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return Outcome.UNSTABLE
        if np.abs(d.qvel).max() > 1e3:
            return Outcome.UNSTABLE
        pos = {c: d.xpos[self._block_body[c]] for c in COLORS}
        if any(off_table(p, scene) for p in pos.values()):
            return Outcome.DROPPED

        def bin_of(color: str) -> str | None:
            bins = scene.bin_xy.items()
            return next((side for side, xy in bins if in_bin(pos[color], xy, scene)), None)

        h = scene.block_half_size
        on_table_z = scene.table_height + h
        if self.kind == "stack":
            top, bottom = goal.target, goal.destination
            held = [top]
            if any(bin_of(c) is not None for c in COLORS):
                candidate = Outcome.WRONG_OBJECT
            elif any(c != top and pos[c][2] > on_table_z + h for c in COLORS):
                candidate = Outcome.WRONG_OBJECT  # some other block ended up on top of a block
            else:
                rel = pos[top] - pos[bottom]
                stacked = (
                    abs(rel[0]) < STACK_XY_TOL
                    and abs(rel[1]) < STACK_XY_TOL
                    and abs(rel[2] - 2 * h) < STACK_Z_TOL
                    and abs(pos[bottom][2] - on_table_z) < STACK_Z_TOL
                )
                candidate = Outcome.SUCCESS if stacked else Outcome.RUNNING
            needs_release = candidate == Outcome.SUCCESS
        else:  # pair
            a, b, side = goal.target, goal.second, goal.destination
            held = [a, b]
            third = next(c for c in COLORS if c not in (a, b))
            bins = (bin_of(a), bin_of(b))
            if bin_of(third) is not None:
                candidate = Outcome.WRONG_OBJECT
            elif bins == (side, side):
                candidate = Outcome.SUCCESS
            elif any(x is not None and x != side for x in bins):
                candidate = Outcome.WRONG_BIN
            else:
                candidate = Outcome.RUNNING
            needs_release = candidate in (Outcome.SUCCESS, Outcome.WRONG_BIN)

        dofs = (self._block_dof[c] for c in COLORS)
        still = max(np.linalg.norm(d.qvel[dof : dof + 3]) for dof in dofs) < cfg.settle_speed
        released = self._closure < cfg.released_closure and not any(
            self._touches_robot(c) for c in held
        )
        holds = candidate != Outcome.RUNNING and still and (released or not needs_release)
        if holds and candidate == self._candidate:
            self._settle_time += self.control_dt
        elif holds:
            self._candidate, self._settle_time = candidate, self.control_dt
        else:
            self._candidate, self._settle_time = Outcome.RUNNING, 0.0
        return candidate if self._settle_time >= cfg.settle_s - 1e-9 else Outcome.RUNNING


# Physics variants (held out from the agent) -------------------------------------------------


HEAVY_MASS_SCALE = 3.0  # blocks weigh 150 g instead of 50 g
SLIPPERY_FRICTION_SCALE = 0.5  # sliding friction of every surface, fingers included
WEAK_STIFFNESS_SCALE = 0.5  # arm and hand position-servo stiffness


def apply_physics_variant(env: SortEnv, variant: str) -> None:
    """Change the compiled world in place (block mass is set before compiling, above).
    Sensing variants are applied by the observer, not here."""
    m = env.model
    if variant in ("nominal", "noisy_sensing", *CAMERA_VARIANTS):
        return
    if variant == "heavy_slippery":
        m.geom_friction[:, 0] *= SLIPPERY_FRICTION_SCALE
        return
    if variant == "weak_motors":
        for act in (*env._arm_act, *env._hand_act):
            m.actuator_gainprm[act, 0] *= WEAK_STIFFNESS_SCALE
            m.actuator_biasprm[act, 1] *= WEAK_STIFFNESS_SCALE
            m.actuator_biasprm[act, 2] *= np.sqrt(WEAK_STIFFNESS_SCALE)
        m.jnt_actgravcomp[:] = 0  # the arm now sags under its own weight
        return
    raise ValueError(f"unknown variant {variant!r}")
