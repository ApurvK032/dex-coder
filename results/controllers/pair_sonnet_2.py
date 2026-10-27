import numpy as np
import mujoco

HOVER = 0.15      # site height above cube center for hover
CARRY_Z = 0.97    # site z during transit
TABLE = 0.78
FLOOR = 0.786     # bin floor top
GRASP_DZ = 0.006


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap90(a):
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.site = self.m.site(robot.grasp_site).id
        self.dofs = list(robot.arm_dof_adr)
        self.names = [task['target'], task['second']]
        self.dest = task['destination']
        self.bin = robot.bin_xy[self.dest]
        self.idx = 0
        self.phase = 'hover'
        self.phase_t = 0.0
        self.grasp_yaw = None
        self.cp_frozen = None
        self.lastp = None
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.first_pos = None
        self.lower_t = None
        self.rel_yaw = 0.0
        self.q_goal = None
        self.trans_t = None
        self.hover_t = None
        self.reached_high = False
        self.hand_yaw_cmd = 0.0

    # ---------- kinematic feasibility
    def solve(self, d, pos_t, yaw, q0, iters=120):
        m = self.m
        lo = np.array(self.robot.arm_lower); hi = np.array(self.robot.arm_upper)
        qa = list(self.robot.arm_qpos_adr)
        q = np.array(q0, dtype=float)
        c0 = np.array([np.cos(yaw), np.sin(yaw), 0.0]); c1 = np.array([0, 0, -1.0])
        Rd = np.stack([c0, c1, np.cross(c0, c1)], axis=1)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        ep = np.ones(3); er = np.ones(3)
        for i in range(iters):
            d.qpos[qa] = q
            mujoco.mj_forward(m, d)
            p = d.site_xpos[self.site].copy(); R = d.site_xmat[self.site].reshape(3, 3)
            er = 0.5 * sum(np.cross(R[:, k], Rd[:, k]) for k in range(3))
            ep = pos_t - p
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dofs], jr[:, self.dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([ep, er]))
            q = np.clip(q + 0.5 * dq, lo, hi)
        return np.linalg.norm(ep), np.linalg.norm(er), q

    def yaw_for(self, cube_yaw, cp, obs):
        d = mujoco.MjData(self.m)
        d.qpos[:] = obs['kin'].qpos
        q0 = np.array(self.robot.home_arm)
        cands = []
        for k in range(-3, 4):
            a = cube_yaw + k * np.pi / 2
            if -1.1 < a < 1.3:
                cands.append(a)
        cands.sort(key=lambda a: abs(a - 0.1))
        best = None; bs = 1e9
        for a in cands:
            worst = 0.0
            for dz in (0.0, HOVER):
                pe, oe, _ = self.solve(d, np.array([cp[0], cp[1], TABLE + 0.025 + GRASP_DZ + dz]), a, q0, iters=250)
                worst = max(worst, pe / 0.005, oe / 0.02)
            if worst < 1.0:
                return a
            if worst < bs:
                bs = worst; best = a
        return best if best is not None else 0.0

    # ---------- IK step
    def ik(self, obs, pos_t, yaw_t, vmax=0.05):
        pos = np.array(obs['grasp_site_pos'])
        R = np.array(obs['grasp_site_mat'])
        c0 = np.array([np.cos(yaw_t), np.sin(yaw_t), 0.0])
        c1 = np.array([0, 0, -1.0])
        Rd = np.stack([c0, c1, np.cross(c0, c1)], axis=1)
        ep = pos_t - pos
        n = np.linalg.norm(ep)
        if n > vmax:
            ep = ep * vmax / n
        er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
        er = np.clip(er, -0.08, 0.08)
        mujoco.mj_jacSite(self.m, obs['kin'], self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.dofs], self.jr[:, self.dofs]])
        dq = J.T @ np.linalg.solve(J @ J.T + 0.02 ** 2 * np.eye(6), np.concatenate([ep, er]))
        mx = np.max(np.abs(dq))
        if mx > 0.08:
            dq *= 0.08 / mx
        qm = np.array(obs['qpos'])[:7]
        q = np.clip(np.array(obs['arm_target']) + dq, qm - 0.1, qm + 0.1)
        return q, np.linalg.norm(pos_t - pos), np.linalg.norm(er)

    def hand_yaw(self, obs):
        R = np.array(obs['grasp_site_mat'])
        return np.arctan2(R[1, 0], R[0, 0])

    def act(self, obs):
        t = obs['t']
        if self.idx >= 2:
            q = np.array(self.robot.home_arm)
            qm = np.array(obs['arm_target'])
            q = qm + np.clip(q - qm, -0.06, 0.06)
            return list(q) + [0.0]
        name = self.names[self.idx]
        b = obs['blocks'][name]
        cube = np.array(b['pos'])
        site = np.array(obs['grasp_site_pos'])
        if self.phase in ('hover', 'descend'):
            self.cp_frozen = cube.copy()
            if self.phase == 'hover' and self.grasp_yaw is None:
                self.grasp_yaw = self.yaw_for(b['yaw'], cube, obs)
                d = mujoco.MjData(self.m)
                d.qpos[:] = obs['kin'].qpos
                self.reached_high = False
                self.hover_t = None
                _, _, self.q_goal = self.solve(d, np.array([cube[0], cube[1], 1.03]),
                                               self.grasp_yaw, np.array(self.robot.home_arm), iters=300)
        cp = self.cp_frozen
        gy = self.grasp_yaw
        bx, by = self.bin
        zc = TABLE + 0.025 + GRASP_DZ
        if self.idx == 0:
            slot = np.array([bx + 0.031, by - 0.034])
        else:
            slot = np.array([bx - 0.031, by + 0.034])
            if self.first_pos is not None:
                sx = -1.0 if self.first_pos[0] > bx else 1.0
                sy = -1.0 if self.first_pos[1] > by else 1.0
                slot = np.array([bx + sx * 0.031, by + sy * 0.034])
        # in-hand offset (site - cube) for carrying phases
        off = site - cube
        clo = 0.0
        err = 0.0; yerr = 0.0

        if self.phase == 'hover':
            tgt = np.array([cp[0], cp[1], zc + HOVER])
            q, err, yerr = self.ik(obs, tgt, gy)
            if self.idx >= 1 and self.q_goal is not None and not self.reached_high:
                qa = np.array(obs['arm_target'])
                q = qa + np.clip(self.q_goal - qa, -0.07, 0.07)
                if self.hover_t is None:
                    self.hover_t = t
                if np.max(np.abs(np.array(obs['qpos'])[:7] - self.q_goal)) < 0.12 or t - self.hover_t > 3.5:
                    self.reached_high = True
                err = 1.0
            if err < 0.012 and yerr < 0.1:
                self.phase = 'descend'
        elif self.phase == 'descend':
            tgt = np.array([cp[0], cp[1], zc])
            q, err, yerr = self.ik(obs, tgt, gy, vmax=0.03)
            if err < 0.008:
                self.phase = 'grasp'; self.phase_t = t
        elif self.phase == 'grasp':
            tgt = np.array([cp[0], cp[1], zc])
            q, err, yerr = self.ik(obs, tgt, gy, vmax=0.01)
            clo = min(1.0, (t - self.phase_t) / 1.0)
            if t - self.phase_t > 1.6:
                self.phase = 'lift'
                rel = wrap90(b['yaw'] - self.hand_yaw(obs))
                self.rel_yaw = float(np.clip(-rel, -0.6, 0.6))
        elif self.phase == 'lift':
            tgt = np.array([site[0], site[1], CARRY_Z])
            tgt[:2] = cp[:2]
            q, err, yerr = self.ik(obs, tgt, gy, vmax=0.03)
            clo = 1.0
            self.hand_yaw_cmd = gy
            if err < 0.015:
                self.phase = 'transit'
        elif self.phase in ('transit', 'lower', 'release'):
            yaw = self.rel_yaw
            dyaw = 0.0
            if self.phase == 'transit':
                tgt = np.array([slot[0] + off[0], slot[1] + off[1], CARRY_Z])
                q, err, yerr = self.ik(obs, tgt, yaw, vmax=0.04)
                clo = 1.0
                if self.trans_t is None:
                    self.trans_t = t
                if np.linalg.norm(cube[:2] - slot) < 0.012 or t - self.trans_t > 4.0:
                    self.phase = 'lower'
                    self.trans_t = None
            elif self.phase == 'lower':
                zt = 0.836 + 0.027 + 0.025
                tgt = np.array([slot[0] + off[0], slot[1] + off[1], zt + off[2]])
                q, err, yerr = self.ik(obs, tgt, yaw, vmax=0.03)
                clo = 1.0
                if self.lower_t is None:
                    self.lower_t = t
                if (cube[2] < zt + 0.006 and np.linalg.norm(cube[:2] - slot) < 0.005) or t - self.lower_t > 3.0:
                    self.phase = 'release'; self.phase_t = t
                    self.hold_site = site.copy()
            else:
                tgt = self.hold_site.copy()
                q, err, yerr = self.ik(obs, tgt, yaw, vmax=0.01)
                clo = max(0.0, 1.0 - (t - self.phase_t) / 0.5)
                if t - self.phase_t > 0.9:
                    self.phase = 'retreat'
                    self.lower_t = None
                    self.first_pos = np.array(cube)
        elif self.phase == 'retreat':
            tgt = np.array([self.hold_site[0], self.hold_site[1], CARRY_Z + 0.03])
            q, err, yerr = self.ik(obs, tgt, 0.0, vmax=0.04)
            clo = 0.0
            if err < 0.02:
                self.idx += 1
                self.phase = 'hover'
                self.grasp_yaw = None
        if self.lastp != self.phase:
            if self.phase in ('hover', 'release'):
                print('t=%.1f %s idx%d cube %s gy %s' % (t, self.phase, self.idx, np.round(cube, 3), None if self.grasp_yaw is None else round(self.grasp_yaw, 2)))
            self.lastp = self.phase
        return list(q) + [float(clo)]
