"""What a controller gets: robot facts at the start of an episode and an observation every step.

Controllers see simulator ground truth for block poses (like humanoid-vla's scripted expert, not
like its learned policy), measured joint angles, and a private kinematics copy of the robot for
Jacobians and forward kinematics. They act only through the 8-value action of humanoid-vla's
action contract. Nothing they do to the objects given here reaches the simulated robot.
"""

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from humanoid_vla.envs.task import COLORS
from humanoid_vla.sim.g1 import GRASP_SITE

from dexcoder.tasks import TaskEnv

# noisy_sensing: per-episode bias plus per-step jitter on block poses, and one control step
# (40 ms) of observation delay.
POS_BIAS_STD = 0.004  # m
POS_JITTER_STD = 0.0015  # m
YAW_BIAS_STD = np.deg2rad(4.0)


@dataclass
class RobotInfo:
    """Facts about the robot and scene, fixed for the episode."""

    model: mujoco.MjModel  # nominal model: a private copy, for kinematics only
    arm_joints: tuple[str, ...]
    hand_joints: tuple[str, ...]
    arm_qpos_adr: np.ndarray  # indices of the 7 arm joints in model.qpos
    arm_dof_adr: np.ndarray  # indices of the 7 arm joints in model.qvel / Jacobian columns
    hand_qpos_adr: np.ndarray
    arm_lower: np.ndarray
    arm_upper: np.ndarray
    grasp_site: str
    control_dt: float
    max_joint_speed: float
    time_limit_s: float
    table_height: float
    block_half_size: float
    bin_xy: dict[str, tuple[float, float]]
    bin_inner_half_size: float
    bin_wall_height: float
    bin_wall_thickness: float
    home_arm: np.ndarray
    released_closure: float
    settle_s: float


def robot_info(env: TaskEnv, nominal_model: mujoco.MjModel) -> RobotInfo:
    scene, cfg = env.config.scene, env.config
    m = nominal_model
    arm_ids = [m.joint(j).id for j in env.arm_joints]
    hand_ids = [m.joint(j).id for j in env.hand_joints]
    return RobotInfo(
        model=copy.copy(m),
        arm_joints=tuple(env.arm_joints),
        hand_joints=tuple(env.hand_joints),
        arm_qpos_adr=m.jnt_qposadr[arm_ids].copy(),
        arm_dof_adr=m.jnt_dofadr[arm_ids].copy(),
        hand_qpos_adr=m.jnt_qposadr[hand_ids].copy(),
        arm_lower=env.arm_lower.copy(),
        arm_upper=env.arm_upper.copy(),
        grasp_site=GRASP_SITE,
        control_dt=env.control_dt,
        max_joint_speed=cfg.max_joint_speed,
        time_limit_s=cfg.time_limit_s,
        table_height=scene.table_height,
        block_half_size=scene.block_half_size,
        bin_xy=dict(scene.bin_xy),
        bin_inner_half_size=scene.bin_inner_half_size,
        bin_wall_height=scene.bin_wall_height,
        bin_wall_thickness=scene.bin_wall_thickness,
        home_arm=np.asarray(cfg.home_arm, dtype=float),
        released_closure=cfg.released_closure,
        settle_s=cfg.settle_s,
    )


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Observer:
    """Builds each step's observation from the true simulator state."""

    def __init__(self, env: TaskEnv, nominal_model: mujoco.MjModel, seed: int) -> None:
        self.env = env
        self.kin_model = nominal_model
        self.site = nominal_model.site(GRASP_SITE).id
        m = env.model
        self.state_dof = m.jnt_dofadr[[m.joint(j).id for j in (*env.arm_joints, *env.hand_joints)]]
        self.noisy = env.variant == "noisy_sensing"
        rng = np.random.default_rng([seed, 7])
        self.rng = rng
        self.pos_bias = {c: rng.normal(0, POS_BIAS_STD, 3) * [1, 1, 0.3] for c in COLORS}
        self.yaw_bias = {c: float(rng.normal(0, YAW_BIAS_STD)) for c in COLORS}
        self._previous: dict | None = None

    def observe(self) -> dict:
        """The observation for this control step, with a fresh kinematics copy under "kin"."""
        current = self._snapshot()
        if self.noisy:  # 40 ms delay: return the previous step's snapshot
            obs, self._previous = (self._previous or current), current
        else:
            obs = current
        obs = dict(obs)  # the stored snapshot must keep its private keys
        kin = mujoco.MjData(self.kin_model)
        kin.qpos[:] = obs.pop("_qpos_full")
        kin.qvel[:] = obs.pop("_qvel_full")
        mujoco.mj_forward(self.kin_model, kin)
        obs["grasp_site_pos"] = kin.site_xpos[self.site].copy()
        obs["grasp_site_mat"] = kin.site_xmat[self.site].reshape(3, 3).copy()
        obs["kin"] = kin
        return obs

    def _snapshot(self) -> dict:
        env = self.env
        d = env.data
        qpos_full, blocks = d.qpos.copy(), {}
        for c in COLORS:
            pos, quat, vel = env.block_pos(c), env.block_quat(c), env.block_linvel(c)
            yaw = yaw_of(quat)
            if self.noisy:
                pos = pos + self.pos_bias[c] + self.rng.normal(0, POS_JITTER_STD, 3)
                yaw = yaw + self.yaw_bias[c]
                quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            blocks[c] = {"pos": pos, "quat": quat, "yaw": yaw, "linvel": vel}
            adr = env._block_qpos[c]
            qpos_full[adr : adr + 3] = pos
            qpos_full[adr + 3 : adr + 7] = quat
        return {
            "t": env.time,
            "qpos": d.qpos[env._state_adr].copy(),
            "qvel": d.qvel[self.state_dof].copy(),
            "arm_target": env._arm_target.copy(),
            "closure": float(env._closure),
            "blocks": blocks,
            "_qpos_full": qpos_full,
            "_qvel_full": d.qvel.copy(),
        }
