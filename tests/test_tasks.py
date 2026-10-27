"""Tasks, judges and held-out variants."""

import numpy as np
import pytest
from humanoid_vla.envs.sort import SortEnv, SortEnvConfig
from humanoid_vla.envs.task import COLORS

from dexcoder.tasks import PRACTICE_SEEDS, TEST_SEEDS, VARIANTS, TaskEnv, make_task


def test_seed_ranges_are_disjoint():
    assert set(PRACTICE_SEEDS).isdisjoint(TEST_SEEDS)


@pytest.mark.parametrize("seed", [100000, 100007, 300003])
def test_sort_layouts_match_humanoid_vla(seed):
    ours = TaskEnv("sort")
    goal = ours.reset_task(seed)
    ref = SortEnv(SortEnvConfig(cameras=()))
    ref.reset(seed=seed)
    assert (goal.target, goal.destination) == (ref.task.target, ref.task.destination)
    for c in COLORS:
        np.testing.assert_array_equal(ours.block_pos(c), ref.privileged_state().block_pos[c])


def test_new_tasks_name_distinct_blocks():
    for seed in PRACTICE_SEEDS[:50]:
        s = make_task("stack", seed)
        p = make_task("pair", seed)
        assert s.target != s.destination
        assert p.target != p.second and p.destination in ("left", "right")


def _place(env, color, xyz):
    adr = env._block_qpos[color]
    env.data.qpos[adr : adr + 3] = xyz
    env.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
    env.data.qvel[env._block_dof[color] : env._block_dof[color] + 6] = 0


def _settle(env, steps=40):
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    outcome = "running"
    for _ in range(steps):
        _, _, term, trunc, info = env.step(np.append(env._arm_target, 0.0))
        outcome = str(info["outcome"])
        if term or trunc:
            break
    return outcome


def test_stack_judge_accepts_a_stack():
    env = TaskEnv("stack")
    goal = env.reset_task(100000)
    bottom = env.block_pos(goal.destination)
    h = env.config.scene.block_half_size
    _place(env, goal.target, bottom + [0, 0, 2 * h + 0.001])
    assert _settle(env) == "success"


def test_pair_judge_accepts_both_blocks_in_the_bin():
    env = TaskEnv("pair")
    goal = env.reset_task(100000)
    bx, by = env.config.scene.bin_xy[goal.destination]
    z = env.config.scene.table_height + 0.006 + env.config.scene.block_half_size + 0.001
    _place(env, goal.target, [bx - 0.027, by, z])
    _place(env, goal.second, [bx + 0.027, by, z])
    assert _settle(env) == "success"


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_build(variant):
    env = TaskEnv("sort", variant)
    env.reset_task(100000)
    assert _settle(env, steps=5) == "running"
