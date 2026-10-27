
import numpy as np
import mujoco


def _rdes(psi):
    x = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def _seg_dist(p, a, b):
    ab = b - a
    s = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + s * ab))


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower, float)
        self.hi = np.array(robot.arm_upper, float)
        self.home = np.array(robot.home_arm, float)
        self.target = task["target"]
        self.bin = np.array(robot.bin_xy[task["destination"]], dtype=float)
        self.table = robot.table_height
        self.h = robot.block_half_size
        self.z_pre = self.table + 0.14
        self.z_grasp = self.table + self.h + 0.005
        self.z_carry = self.table + 0.175
        self.z_place = self.table + 0.045
        self.phase = "init"
        self.t_phase = 0.0
        self.q_cmd = self.home.copy()
        self.closure = 0.0
        self.attempt = 0
        self.mode = None
        self.started = False

    # ---------------- kinematics ----------------
    def ik(self, q0, p, R, iters=60, free_yaw=False):
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        wrot = 0.5
        mid = 0.5 * (self.lo + self.hi)
        span = (self.hi - self.lo)
        e1 = e2 = 1.0
        for _ in range(iters):
            self.d.qpos[self.qa] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            cp = self.d.site_xpos[self.sid]
            cR = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = p - cp
            if free_yaw:
                er = np.cross(cR[:, 1], R[:, 1])
            else:
                er = 0.5 * (np.cross(cR[:, 0], R[:, 0]) + np.cross(cR[:, 1], R[:, 1]) + np.cross(cR[:, 2], R[:, 2]))
            e1, e2 = np.linalg.norm(ep), np.linalg.norm(er)
            if e1 < 2e-4 and e2 < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], wrot * jr[:, self.da]])
            e = np.r_[ep, wrot * er]
            JJ = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            g = -(q - mid) / (span ** 2) * 0.5
            N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
            dq = dq + N @ g
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, e1, e2

    def limit_margin(self, q):
        return float(np.min(np.minimum(q - self.lo, self.hi - q)))

    def solve_multi(self, p, R, seeds, free_yaw=False):
        best = None
        for s in seeds:
            q, e1, e2 = self.ik(s, p, R, iters=200, free_yaw=free_yaw)
            ok = e1 < 0.003 and e2 < 0.03
            score = (0 if ok else 1, -min(self.limit_margin(q), 0.3) + 0.05 * np.linalg.norm(q - self.home))
            if best is None or score < best[0]:
                best = (score, q, ok)
        return best[1], best[2]

    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def yaw_of(self, q):
        _, R = self.fk(q)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ---------------- planning ----------------
    def plan_grasp(self, obs, bp, yaw):
        others = [np.array(b["pos"][:2]) for c, b in obs["blocks"].items() if c != self.target]
        seeds = [self.home, self.q_cmd]
        best = None
        for k in range(-4, 5):
            psi = yaw + k * np.pi / 2
            if psi < -2.4 or psi > 1.6:
                continue
            R = _rdes(psi)
            qpre, ok1 = self.solve_multi(np.array([bp[0], bp[1], self.z_pre]), R, seeds)
            if not ok1:
                continue
            qg, e1, e2 = self.ik(qpre, np.array([bp[0], bp[1], self.z_grasp]), R, iters=200)
            if e1 > 0.003 or e2 > 0.03:
                continue
            marg = min(self.limit_margin(qpre), self.limit_margin(qg))
            xd = np.array([np.cos(psi), np.sin(psi)])
            a = bp[:2] - 0.06 * xd
            b = bp[:2] + 0.10 * xd
            clear = min([_seg_dist(o, a, b) for o in others] + [1.0])
            cost = -2.0 * min(clear, 0.08) - 0.5 * min(marg, 0.2) + 0.03 * abs(psi + 0.35)
            if best is None or cost < best[0]:
                best = (cost, psi, qpre, qg)
        if best is None:
            psi = yaw + np.round((-0.35 - yaw) / (np.pi / 2)) * np.pi / 2
            R = _rdes(psi)
            qpre, _ = self.solve_multi(np.array([bp[0], bp[1], self.z_pre]), R, seeds)
            return psi, qpre
        return best[1], best[2]

    def plan_bin(self):
        # pose above the bin, free yaw, warm-started from current command
        p = np.array([self.bin[0], self.bin[1], self.z_carry])
        pl = np.array([self.bin[0], self.bin[1], self.z_place])
        y0 = self.yaw_of(self.q_cmd)
        best = None
        for dy in [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2, 1.5, -1.5]:
            R = _rdes(y0 + dy)
            q, e1, e2 = self.ik(self.q_cmd, p, R, iters=150)
            if e1 > 0.003 or e2 > 0.03:
                q, ok = self.solve_multi(p, R, [self.home, self.q_cmd])
                if not ok:
                    continue
            ql, f1, f2 = self.ik(q, pl, R, iters=150)
            if f1 > 0.003 or f2 > 0.03:
                continue
            worst = 1.0
            for dx, dyy in [(0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]:
                qq, g1, g2 = self.ik(ql, pl + np.array([dx, dyy, 0]), R, iters=60)
                if g1 > 0.003 or g2 > 0.03:
                    worst = 0.0
            marg = min(self.limit_margin(q), self.limit_margin(ql))
            cost = 0.2 * abs(dy) - 0.5 * min(marg, 0.2) + (1.0 if worst == 0.0 else 0.0) + 0.05 * np.max(np.abs(q - self.q_cmd))
            if best is None or cost < best[0]:
                best = (cost, q)
        if best is None:
            q, _ = self.solve_multi(p, _rdes(0.0), [self.home, self.q_cmd], free_yaw=True)
            return q
        return best[1]

    # ---------------- motion primitives ----------------
    def start_joint(self, t, q_goal, speed=1.2):
        self.jq0 = self.q_cmd.copy()
        self.jq1 = q_goal.copy()
        self.seg_T = max(0.5, np.max(np.abs(q_goal - self.jq0)) / speed)
        self.seg_t0 = t
        self.mode = "joint"

    def start_cart(self, t, p_goal, speed=0.2, min_T=0.4):
        self.p_start, self.R = self.fk(self.q_cmd)
        # keep the current yaw, enforce palm-down
        self.R = _rdes(float(np.arctan2(self.R[1, 0], self.R[0, 0])))
        self.p_goal = np.array(p_goal, float)
        self.seg_T = max(min_T, np.linalg.norm(self.p_goal - self.p_start) / speed)
        self.seg_t0 = t
        self.mode = "cart"

    def step_motion(self, t):
        s = _smooth((t - self.seg_t0) / self.seg_T)
        if self.mode == "joint":
            self.q_cmd = self.jq0 + s * (self.jq1 - self.jq0)
        elif self.mode == "cart":
            p = self.p_start + s * (self.p_goal - self.p_start)
            q, e1, e2 = self.ik(self.q_cmd, p, self.R, iters=40)
            if e1 < 0.02:
                self.q_cmd = q
        return (t - self.seg_t0) >= self.seg_T

    def start_center(self, t, bp, site, down=False):
        self.n_center = getattr(self, "n_center", 0) + 1
        p, _ = self.fk(self.q_cmd)
        corr = bp[:2] - site[:2]
        n = np.linalg.norm(corr)
        if n > 0.03:
            corr *= 0.03 / n
        z = self.z_place if down else p[2]
        self.start_cart(t, [p[0] + corr[0], p[1] + corr[1], z], speed=0.05)
        self.set_phase("center", t)

    def set_phase(self, name, t):
        self.phase = name
        self.t_phase = t

    # ---------------- main ----------------
    def act(self, obs):
        t = obs["t"]
        blk = obs["blocks"][self.target]
        bp = np.array(blk["pos"], float)
        site = np.array(obs["grasp_site_pos"], float)
        if not self.started:
            self.q_cmd = np.array(obs["arm_target"], float)
            self.started = True

        if self.phase == "init":
            self.psi, qpre = self.plan_grasp(obs, bp, blk["yaw"])
            print("plan t=%.2f psi=%.2f yaw=%.2f bp=%s qpre=%s" % (t, self.psi, blk["yaw"], np.round(bp, 3), np.round(qpre, 2)))
            self.closure = 0.0
            self.start_joint(t, qpre)
            self.set_phase("approach", t)
        el = t - self.t_phase

        ph = self.phase
        if ph == "approach":
            done = self.step_motion(t)
            if done and el > self.seg_T + 0.2:
                self.start_cart(t, [bp[0], bp[1], self.z_pre], speed=0.1, min_T=0.3)
                self.set_phase("align", t)
        elif ph == "align":
            done = self.step_motion(t)
            if done:
                self.start_cart(t, [bp[0], bp[1], self.z_grasp], speed=0.15)
                self.set_phase("descend", t)
        elif ph == "descend":
            done = self.step_motion(t)
            if done and el > self.seg_T + 0.3:
                self.set_phase("close", t)
        elif ph == "close":
            if el < 0.05:
                self.p_hold, _ = self.fk(self.q_cmd)
                self.corr = np.zeros(3)
            # closed-loop: keep the measured grasp site at the grasp point while fingers push
            err = self.p_hold - site
            self.corr = np.clip(self.corr + 0.6 * err, [-0.03, -0.03, -0.04], [0.03, 0.03, 0.005])
            q, e1, e2 = self.ik(self.q_cmd, self.p_hold + self.corr, self.R, iters=30)
            if e1 < 0.02:
                self.q_cmd = q
            if el < 0.5:
                self.closure = 0.2 + 0.3 * el / 0.5
            else:
                self.closure = min(1.0, 0.5 + 0.5 * (el - 0.5) / 0.4)
            if el > 1.3:
                p, _ = self.fk(self.q_cmd)
                print("closed: site-bp=%s corr=%s" % (np.round(site - bp, 3), np.round(self.corr, 3)))
                self.start_cart(t, [p[0], p[1], self.z_carry], speed=0.2)
                self.set_phase("lift", t)
        elif ph == "lift":
            done = self.step_motion(t)
            if done and el > self.seg_T + 0.2:
                if bp[2] < self.table + self.h + 0.04 or np.linalg.norm(bp - site) > 0.05:
                    self.set_phase("regrasp", t)
                else:
                    self.start_joint(t, self.plan_bin(), speed=1.0)
                    self.set_phase("transport", t)
        elif ph == "transport":
            done = self.step_motion(t)
            if np.linalg.norm(bp - site) > 0.06:
                self.set_phase("regrasp", t)
            elif done and el > self.seg_T + 0.3:
                off = bp[:2] - site[:2]
                n = np.linalg.norm(off)
                if n > 0.04:
                    off = off * 0.04 / n
                yw = self.yaw_of(self.q_cmd)
                xd = np.array([np.cos(yw), np.sin(yw)])
                goal = self.bin + 0.01 * xd - off
                self.start_cart(t, [goal[0], goal[1], self.z_place], speed=0.12)
                self.set_phase("lower", t)
        elif ph == "lower":
            done = self.step_motion(t)
            if done and el > self.seg_T + 0.3:
                self.set_phase("release", t)
        elif ph == "release":
            if el < 0.6:
                self.closure = 1.0 - 0.5 * el / 0.6
            else:
                self.closure = max(0.0, 0.5 - 0.5 * (el - 0.6) / 1.5)
            if el > 2.4:
                self.start_center(t, bp, site)
        elif ph == "center":
            self.closure = 0.0
            done = self.step_motion(t)
            if done and el > self.seg_T + 0.2:
                p, _ = self.fk(self.q_cmd)
                self.rest_z = bp[2]
                self.start_cart(t, [p[0], p[1], self.table + 0.20], speed=0.06)
                self.set_phase("retreat", t)
        elif ph == "retreat":
            self.closure = 0.0
            self.step_motion(t)
            p, _ = self.fk(self.q_cmd)
            if bp[2] > self.rest_z + 0.008 and p[2] < self.table + 0.16 and self.n_center < 4:
                # cube is being dragged up: go back down and re-center
                self.start_center(t, bp, site, down=True)
        elif ph == "regrasp":
            # open, move up, then replan
            self.closure = 0.0
            if el > 0.5 and self.mode != "up":
                p, _ = self.fk(self.q_cmd)
                self.start_cart(t, [p[0], p[1], max(p[2], self.z_pre)], speed=0.15)
                self.mode_up_T = self.seg_T
                self.mode = "cart"
                self.t_up = t
                self.phase = "regrasp_up"
        elif ph == "regrasp_up":
            self.closure = 0.0
            done = self.step_motion(t)
            if done:
                self.attempt += 1
                self.phase = "init"

        return np.r_[self.q_cmd, self.closure]
