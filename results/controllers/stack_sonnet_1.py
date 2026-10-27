import numpy as np
import mujoco

HOVER = 0.91


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3)
    return q[1:] / s * 2 * np.arctan2(s, q[0])


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.adr = np.array(robot.arm_qpos_adr)
        self.dof = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.phase = 0
        self.R = None
        self.hover = None
        self.retries = 0
        self.t_phase = 0.0
        self.wp = None
        self.closure = 0.0
        self.q_cmd = None
        self.tgt = task['target']
        self.dst = task['destination']

    def rot(self, psi):
        c, s = np.cos(psi), np.sin(psi)
        xs = np.array([c, s, 0.0])
        ys = np.array([0, 0, -1.0])
        return np.stack([xs, ys, np.cross(xs, ys)], axis=1)

    def ik(self, q0, pos, R, iters=3):
        d = self.d
        q = q0.copy()
        for _ in range(iters):
            d.qpos[self.adr] = q
            mujoco.mj_forward(self.m, d)
            p = d.site_xpos[self.sid].copy()
            Rc = d.site_xmat[self.sid].reshape(3, 3).copy()
            ep = pos - p
            eo = rotvec(R @ Rc.T)
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dof], 0.5 * jr[:, self.dof]])
            e = np.concatenate([ep, 0.5 * eo])
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def choose_R(self, blk):
        tp = blk[self.tgt]['pos']; dp = blk[self.dst]['pos']
        psi0 = (blk[self.tgt]['yaw'] + np.pi / 4) % (np.pi / 2) - np.pi / 4
        best = None
        for k in (0, 1, -1, 2):
            psi = psi0 + k * np.pi / 2
            R = self.rot(psi)
            q = np.array(self.robot.home_arm, dtype=float)
            cost = 0.0
            pick_ep = 0.0
            for pos in ([tp[0], tp[1], 0.91], [tp[0], tp[1], 0.82],
                        [dp[0], dp[1], 0.91], [dp[0], dp[1], 0.87]):
                pos = np.array(pos)
                for _ in range(60):
                    q = self.ik(q, pos, R, iters=1)
                d = self.d
                d.qpos[self.adr] = q
                mujoco.mj_forward(self.m, d)
                ep = np.linalg.norm(pos - d.site_xpos[self.sid])
                eo = np.linalg.norm(rotvec(R @ d.site_xmat[self.sid].reshape(3, 3).T))
                mg = float(np.min(np.minimum(q - self.lo, self.hi - q)))
                cost += 100 * ep + 3 * eo + 2 * max(0.0, 0.3 - mg)
                if pos[2] < 0.85 and pos[0] == tp[0]:
                    pick_ep = max(pick_ep, ep)
            cost += 0.05 * abs(k)
            if best is None or cost < best[0]:
                best = (cost, R, pick_ep)
        self.pick_ep = best[2]
        return best[1]

    def retry(self):
        self.phase = 0
        self.t_phase = 0.0
        self.R = None
        self.hover = None
        self.closure = 0.0
        self.retries += 1

    def pick_hover(self, blk):
        tp = blk[self.tgt]['pos']; dp = blk[self.dst]['pos']
        for h in (0.98, 0.95, 0.93):
            ok = True
            q = np.array(self.robot.home_arm, dtype=float)
            for pos in ([tp[0], tp[1], h], [tp[0], tp[1], 0.82],
                        [dp[0], dp[1], h], [dp[0], dp[1], 0.87]):
                pos = np.array(pos)
                for _ in range(60):
                    q = self.ik(q, pos, self.R, iters=1)
                d = self.d
                d.qpos[self.adr] = q
                mujoco.mj_forward(self.m, d)
                ep = np.linalg.norm(pos - d.site_xpos[self.sid])
                eo = np.linalg.norm(rotvec(self.R @ d.site_xmat[self.sid].reshape(3, 3).T))
                mg = float(np.min(np.minimum(q - self.lo, self.hi - q)))
                if ep > 0.004 or eo > 0.03 or mg < 0.08:
                    ok = False
                    break
            if ok:
                return h
        return 0.91

    def move_wp(self, goal, speed=0.012):
        v = goal - self.wp
        n = np.linalg.norm(v)
        if n > speed:
            v = v / n * speed
        self.wp = self.wp + v

    def act(self, obs):
        blk = obs['blocks']
        tp = blk[self.tgt]['pos']; dp = blk[self.dst]['pos']
        site = obs['grasp_site_pos']
        if self.q_cmd is None:
            self.q_cmd = np.array(obs['arm_target'], dtype=float)
            self.wp = site.copy()
        if self.R is None:
            self.R = self.choose_R(blk)
            if self.retries > 0 and (self.pick_ep > 0.008 or tp[2] < 0.78):
                self.phase = 11
        R = self.R
        global HOVER
        if self.hover is None:
            self.hover = self.pick_hover(blk)
        HOVER = self.hover
        z0 = 0.78 + 0.025
        ph = self.phase
        closure = self.closure
        self.t_phase += 0.04

        def goto(goal, speed=0.012, tol=0.01, nxt=None):
            self.move_wp(goal, speed)
            if nxt is not None and (np.linalg.norm(site - goal) < tol or self.t_phase > 5.0):
                self.phase = nxt; self.t_phase = 0.0

        zg = 0.015
        dist = np.linalg.norm(tp - site)
        if (ph == 4 and self.t_phase > 1.5 and (tp[2] < 0.84 or dist > 0.07)) or \
                (ph in (5, 6) and (tp[2] < 0.84 or dist > 0.08)):
            if obs['t'] < 18.0:
                self.retry()
                ph = 0
        if ph == 10:
            ok = (np.linalg.norm(tp[:2] - dp[:2]) < 0.012 and abs(tp[2] - dp[2] - 0.05) < 0.006)
            if ok or obs['t'] > 18.0:
                self.phase = 11
            else:
                self.retry()
                ph = 0
        if ph == 0:
            closure = 0.0
            if site[2] < HOVER - 0.03 and np.linalg.norm(site[:2] - tp[:2]) > 0.03:
                goto(np.array([site[0], site[1], HOVER]), 0.015, 0.0, None)
            else:
                goto(np.array([tp[0], tp[1], HOVER]), 0.015, 0.015, 1)
        elif ph == 11:
            closure = 0.0
            self.wp = self.wp
        elif ph == 1:
            closure = 0.0
            goto(np.array([tp[0], tp[1], HOVER]), 0.015, 0.008, 2)
        elif ph == 2:
            closure = 0.0
            goto(np.array([tp[0], tp[1], z0 + zg]), 0.008, 0.006, 3)
        elif ph == 3:
            closure = min(1.0, self.t_phase / 1.5)
            if self.t_phase > 2.2:
                self.phase = 4; self.t_phase = 0.0
        elif ph == 4:
            closure = 1.0
            goto(np.array([site[0], site[1], HOVER + 0.01]), 0.01, 0.01, 5)
            if self.phase == 5:
                self.off = tp - site
        elif ph == 5:
            closure = 1.0
            off = self.off
            g = np.array([dp[0] - off[0], dp[1] - off[1], HOVER + 0.01])
            goto(g, 0.012, 0.008, 6)
            if self.phase == 6:
                self.off = tp - site
                self.dgoal = np.array([dp[0], dp[1]])
        elif ph == 6:
            closure = 1.0
            off = self.off
            offz = tp[2] - site[2]
            g = np.array([self.dgoal[0] - off[0], self.dgoal[1] - off[1], 0.78 + 0.05 + 0.025 + 0.007 - offz])
            goto(g, 0.006, 0.004, 7)
            if tp[2] < 0.78 + 0.05 + 0.025 + 0.006 and self.phase == 6:
                self.phase = 7; self.t_phase = 0.0
        elif ph == 7:
            closure = 1.0
            if self.t_phase > 0.3:
                self.phase = 8; self.t_phase = 0.0
        elif ph == 8:
            closure = max(0.0, 1.0 - self.t_phase / 2.4)
            if self.t_phase > 2.6:
                self.phase = 9; self.t_phase = 0.0
        else:
            closure = 0.0
            goto(np.array([site[0], site[1], HOVER]), 0.012, 0.01, 10)
        self.closure = closure
        self.q_cmd = self.ik(self.q_cmd, self.wp, R, iters=3)
        prev = np.array(obs['arm_target'])
        dq = np.clip(self.q_cmd - prev, -0.08, 0.08)
        out = prev + dq
        self.q_cmd = out
        return list(out) + [float(closure)]
