"""Helpers for the agent's run_python tool (exploration only; scores never come from here).

    from dexcoder.explore import setup
    env, robot, task, observer = setup("sort", seed=100000)
    obs = observer.observe()
    env.step(action)  # 7 arm joint targets + closure

``env`` is a full simulator of the practice scene, so exploration may read anything in it
(``env.data``, ``env.model``). Controllers never get it: they see only ``robot``, ``task`` and
``obs``.
"""

from humanoid_vla.sim.scene import build_model

from dexcoder.robot_api import Observer, RobotInfo, robot_info
from dexcoder.tasks import PRACTICE_SEEDS, TaskEnv


def setup(kind: str = "sort", seed: int = PRACTICE_SEEDS[0]) -> tuple[TaskEnv, RobotInfo, dict, Observer]:
    if seed not in PRACTICE_SEEDS:
        raise ValueError(f"exploration uses practice seeds only ({PRACTICE_SEEDS.start}-{PRACTICE_SEEDS.stop - 1})")
    nominal = build_model()
    env = TaskEnv(kind)
    goal = env.reset_task(seed)
    return env, robot_info(env, nominal), goal.as_dict(), Observer(env, nominal, seed)
