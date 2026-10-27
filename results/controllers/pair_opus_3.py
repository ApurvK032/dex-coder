
import numpy as np
import mujoco

TABLE = 0.78
HALF = 0.025


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_down(psi):
    c, s = np.cos(psi), np.sin(psi)
    x = np.array([c, s, 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


class Controller:
    ROLL_A = 0.08
    ROLL_B = 0.2
    ROLL_CAP = 0.15
    HAND_OFF = 0.013
    DROP = 0.007
    THUMB_X = 0.048
    NUDGE_Z = 0.835
    REL_T1 = 0.4
    REL_C1 = 0.5
    REL_RATE = 0.04

    def __init__(self, robot, task, verbose=False):
        self.verbose = verbose
        self.r = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.Q0 = np.array([0.44, 0.01, -0.31, -0.35, -1.43, 0.09, -0.05])
        self.seeds = [self.Q0,
                      np.array([-0.91, -0.77, -0.14, 1.24, -1.21, -0.55, -0.12]),
                      np.array([-0.69, 0.12, -0.05, 0.31, -1.7, 0.17, 0.35]),
                      np.array([-0.5, 0.0, 0.6, 0.3, -1.0, 0.6, 0.3]),
                      np.array([-0.61, 0.1, 0.33, 0.43, -1.48, 0.4, 0.15]),
                      self.home.copy()]
        self.q_cmd = self.home.copy()
        self.names = [task["target"], task["second"]]
        bx, by = robot.bin_xy[task["destination"]]
        self.bin = np.array([bx, by])
        self.plan_slots(bx, by)
        self.idx = 0
        self.phase = "start"
        self.t_phase = 0.0
        self.cur_p = None
        self.cur_psi = None
        self.closure = 0.0
        self.attempts = 0
        self.grasp_xy = None
        self.grasp_psi = 0.0
        self.psi_for = None
        self.off_f = None
        self.dyaw_f = 0.0
        self.place_psi_goal = 0.0
        self.done = False

    # ---------------- kinematics ----------------
    def roll_max(self, pitch):
        return float(np.clip(self.ROLL_A - self.ROLL_B * (pitch + 0.5), 0.0, self.ROLL_CAP))

    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, p, R, q0, iters=15, wrot=0.3):
        q = q0.copy()
        q[1] = min(q[1], self.roll_max(q[0]))
        m, d = self.m, self.d
        for _ in range(iters):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cp = d.site_xpos[self.sid]
            cR = d.site_xmat[self.sid].reshape(3, 3)
            ep = p - cp
            er = 0.5 * (np.cross(cR[:, 0], R[:, 0]) + np.cross(cR[:, 1], R[:, 1]) + np.cross(cR[:, 2], R[:, 2]))
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], wrot * jr[:, self.da]])
            e = np.r_[ep, wrot * er]
            A = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(A, e)
            N = np.eye(7) - J.T @ np.linalg.solve(A, J)
            dq += N @ (0.05 * (self.Q0 - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
            q[1] = min(q[1], self.roll_max(q[0]))
        return q

    def pose_err(self, q, p, R):
        pp, RR = self.fk(q)
        return np.linalg.norm(pp - p) + 0.03 * np.linalg.norm(RR - R)

    def solve(self, p, R, q0, iters=15):
        q = self.ik(p, R, q0, iters)
        err = self.pose_err(q, p, R)
        if err > 0.002:
            for s in self.seeds:
                q2 = self.ik(p, R, s, 80)
                e2 = self.pose_err(q2, p, R)
                if e2 < err - 0.001:
                    q, err = q2, e2
        return q, err

    def reachable(self, p, psi):
        R = rot_down(psi)
        best = 9.0
        for s in self.seeds:
            q = self.ik(p, R, s, 80)
            best = min(best, self.pose_err(q, p, R))
            if best < 0.002:
                break
        return best

    def pick_psi(self, bp, yaw, pref=-0.15):
        cands = sorted([wrap(yaw + k * np.pi / 2) for k in range(4)], key=lambda c: abs(wrap(c - pref)))
        best, beste = cands[0], 9.0
        for c in cands[:3]:
            e = max(self.reachable(np.array([bp[0], bp[1], TABLE + HALF + 0.003]), c),
                    self.reachable(np.array([bp[0], bp[1], 0.93]), c))
            if e < 0.003:
                return c
            if e < beste:
                best, beste = c, e
        return best

    def plan_slots(self, bx, by):
        zl = TABLE + 0.006 + self.r.bin_wall_height + HALF + self.DROP
        best = None
        a = 0.026
        patterns = []
        # diagonal layouts first (most tolerant), then side-by-side along y
        for sx in [1, -1]:
            patterns.append(("diag", [np.array([bx + sx * a, by - a]), np.array([bx - sx * a, by + a])]))
            patterns.append(("diag", [np.array([bx - sx * a, by + a]), np.array([bx + sx * a, by - a])]))
        for dx in [-0.02, -0.015, -0.025]:
            patterns.append(("side", [np.array([bx + dx, by - 0.026]), np.array([bx + dx, by + 0.026])]))
        for kind, sl in patterns:
            for psi in [0.0, 0.05, -0.05, 0.1, -0.1]:
                hx = np.array([np.cos(psi), np.sin(psi)])
                e = 0.0
                for s in sl:
                    sp = s + self.HAND_OFF * hx
                    e = max(e, self.reachable(np.array([sp[0], sp[1], zl]), psi))
                    if e > 0.003:
                        break
                    e = max(e, self.reachable(np.array([sp[0], sp[1], 0.93]), psi))
                score = e + 0.01 * abs(psi) + (0.002 if kind == "side" else 0.0)
                if best is None or score < best[0]:
                    best = (score, psi, sl, e)
                if e < 0.002:
                    break
            if best[3] < 0.002:
                break
        self.place_psi = best[1]
        self.slots = best[2]
        # optional thumb nudge of the first cube toward its wall (side-by-side layouts only)
        self.nudged = False
        self.nudge_ok = False
        dvec = self.slots[0] - self.slots[1]
        self.nudge_dir = dvec / max(np.linalg.norm(dvec), 1e-6)
        if abs(self.slots[0][0] - self.slots[1][0]) < 1e-6:
            hx = np.array([np.cos(self.place_psi), np.sin(self.place_psi)])
            e = 0.0
            for s in [self.slots[0] - self.nudge_dir * 0.037 + self.nudge_dir * 0.03,
                      self.slots[0] - self.nudge_dir * 0.033]:
                sp = s + self.THUMB_X * hx
                e = max(e, self.reachable(np.array([sp[0], sp[1], self.NUDGE_Z]), self.place_psi))
            self.nudge_ok = e < 0.004
            if self.verbose:
                print("nudge reach err", round(e, 4))
        if self.verbose:
            print("slots", np.round(self.slots, 3), "psi", self.place_psi, "err", round(best[3], 4))

    # ---------------- helpers ----------------
    def choose_psi(self, yaw, pref):
        best = None
        for k in range(-3, 4):
            c = wrap(yaw + k * np.pi / 2)
            if best is None or abs(wrap(c - pref)) < abs(wrap(best - pref)):
                best = c
        return best

    def slot_goal(self, obs, bp, site):
        off = bp[:2] - site[:2]
        Rm = np.array(obs["grasp_site_mat"])
        hpsi = np.arctan2(Rm[1, 0], Rm[0, 0])
        dyaw = wrap(obs["blocks"][self.names[self.idx]]["yaw"] - hpsi)
        dyaw = (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        dyaw = float(np.clip(dyaw, -0.25, 0.25))
        if self.off_f is None:
            self.off_f = off
            self.dyaw_f = dyaw
        else:
            self.off_f = 0.8 * self.off_f + 0.2 * off
            self.dyaw_f = 0.8 * self.dyaw_f + 0.2 * dyaw
        # rotate the hand so the cube itself ends up axis-aligned with the bin
        self.place_psi_goal = float(np.clip(-self.dyaw_f, self.place_psi - 0.15, self.place_psi + 0.15))
        return self.slots[self.idx] - self.off_f

    def set_phase(self, ph, t):
        if self.verbose and self.cur_p is not None:
            print(f"t={t:.2f} {self.phase}->{ph} idx={self.idx} cur_p={np.round(self.cur_p,3)} psi={self.cur_psi:.2f}")
        self.phase = ph
        self.t_phase = t

    def move_toward(self, goal_p, goal_psi, speed=0.25, wspeed=1.5):
        dt = self.r.control_dt
        dp = goal_p - self.cur_p
        n = np.linalg.norm(dp)
        step = speed * dt
        if n > step:
            self.cur_p = self.cur_p + dp / n * step
        else:
            self.cur_p = goal_p.copy()
        dpsi = wrap(goal_psi - self.cur_psi)
        ws = wspeed * dt
        self.cur_psi = self.cur_psi + np.clip(dpsi, -ws, ws)
        return n <= step and abs(dpsi) <= ws

    # ---------------- main ----------------
    def act(self, obs):
        t = obs["t"]
        site = np.array(obs["grasp_site_pos"])
        if self.cur_p is None:
            self.cur_p = site.copy()
            Rm = np.array(obs["grasp_site_mat"])
            self.cur_psi = np.arctan2(Rm[1, 0], Rm[0, 0])
            self.set_phase("approach", t)
        blocks = obs["blocks"]
        el = t - self.t_phase
        z_travel = 0.93
        z_grasp = TABLE + HALF + 0.003
        if self.idx < 2:
            name = self.names[self.idx]
            bp = np.array(blocks[name]["pos"])
        ph = self.phase

        if self.done:
            goal = np.array([0.2, -0.2, 0.98])
            self.move_toward(goal, -0.15)
            self.closure = 0.0
        elif ph == "approach":
            self.closure = 0.0
            if self.psi_for != (self.idx, self.attempts):
                self.grasp_psi = self.pick_psi(bp, blocks[name]["yaw"])
                self.psi_for = (self.idx, self.attempts)
            psi = self.choose_psi(blocks[name]["yaw"], self.grasp_psi)
            self.grasp_psi = psi
            goal = np.array([bp[0], bp[1], z_travel])
            if self.cur_p[2] < z_travel - 0.02 and np.linalg.norm(self.cur_p[:2] - bp[:2]) > 0.02:
                goal = np.array([self.cur_p[0], self.cur_p[1], z_travel])
            arrived = self.move_toward(goal, psi)
            if arrived and np.linalg.norm(site - goal) < 0.015 and goal[2] == z_travel and \
                    np.linalg.norm(goal[:2] - bp[:2]) < 1e-6:
                self.set_phase("descend", t)
            elif arrived and el > 6.0:
                self.set_phase("descend", t)
        elif ph == "descend":
            psi = self.choose_psi(blocks[name]["yaw"], self.grasp_psi)
            self.grasp_psi = psi
            if self.cur_p[2] > z_grasp + 0.04:
                self.grasp_xy = bp[:2].copy()
            goal = np.array([self.grasp_xy[0], self.grasp_xy[1], z_grasp])
            arrived = self.move_toward(goal, psi, speed=0.15)
            if arrived and (np.linalg.norm(site - goal) < 0.008 or el > 3.0):
                self.set_phase("close", t)
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.04)
            if self.closure >= 1.0 and el > 1.2:
                self.set_phase("lift", t)
        elif ph == "lift":
            self.closure = 1.0
            goal = np.array([self.cur_p[0], self.cur_p[1], z_travel])
            arrived = self.move_toward(goal, self.cur_psi, speed=0.2)
            if arrived and el > 0.5:
                if np.linalg.norm(bp - site) < 0.04 and bp[2] > TABLE + 0.06:
                    self.off_f = None
                    self.set_phase("carry", t)
                else:
                    self.attempts += 1
                    self.set_phase("approach", t)
                    self.closure = 0.0
        elif ph == "carry":
            self.closure = 1.0
            slot = self.slot_goal(obs, bp, site)
            goal = np.array([slot[0], slot[1], z_travel])
            arrived = self.move_toward(goal, self.place_psi_goal, speed=0.25)
            if np.linalg.norm(bp - site) > 0.06:
                self.attempts += 1
                self.set_phase("approach", t)
                self.closure = 0.0
            elif arrived and (np.linalg.norm(site - goal) < 0.01 or el > 6.0):
                self.set_phase("lower", t)
        elif ph == "lower":
            self.closure = 1.0
            slot = self.slot_goal(obs, bp, site)
            zr = TABLE + 0.006 + self.r.bin_wall_height
            goal = np.array([slot[0], slot[1], zr + HALF + self.DROP])
            arrived = self.move_toward(goal, self.place_psi_goal, speed=0.12)
            if arrived and (np.linalg.norm(site - goal) < 0.006 or el > 3.0):
                self.set_phase("release", t)
        elif ph == "release":
            if el < self.REL_T1:
                self.closure = self.REL_C1
            else:
                self.closure = max(0.0, self.closure - self.REL_RATE)
            if el > self.REL_T1 + 0.6 and self.closure <= 0.0:
                self.set_phase("retreat", t)
        elif ph == "retreat":
            self.closure = 0.0
            goal = np.array([self.cur_p[0], self.cur_p[1], z_travel + 0.02])
            arrived = self.move_toward(goal, self.cur_psi, speed=0.2)
            if arrived:
                if self.idx == 0 and self.nudge_ok and not self.nudged:
                    self.nudged = True
                    c1 = np.array(blocks[self.names[0]]["pos"])
                    d = self.nudge_dir
                    inward = float(np.dot(self.slots[0] - c1[:2], d))
                    if 0.003 < inward < 0.03 and c1[2] < TABLE + 0.04:
                        self.nudge_start = c1[:2] - d * 0.037
                        self.nudge_end = self.slots[0] - d * 0.037 + d * 0.004
                        self.set_phase("nudge_go", t)
                        return self.finish(obs, t, site)
                self.idx += 1
                if self.idx >= 2:
                    self.done = True
                else:
                    self.set_phase("approach", t)
        elif ph in ("nudge_go", "nudge_down", "nudge_push", "nudge_up"):
            self.closure = 0.0
            hx = np.array([np.cos(self.place_psi), np.sin(self.place_psi)])
            zp = self.NUDGE_Z
            if ph == "nudge_go":
                s = self.nudge_start + self.THUMB_X * hx
                goal = np.array([s[0], s[1], z_travel])
                if self.move_toward(goal, self.place_psi, speed=0.2) and np.linalg.norm(site - goal) < 0.012:
                    self.set_phase("nudge_down", t)
            elif ph == "nudge_down":
                s = self.nudge_start + self.THUMB_X * hx
                goal = np.array([s[0], s[1], zp])
                if self.move_toward(goal, self.place_psi, speed=0.1) and (np.linalg.norm(site - goal) < 0.008 or el > 2.0):
                    self.set_phase("nudge_push", t)
            elif ph == "nudge_push":
                s = self.nudge_end + self.THUMB_X * hx
                goal = np.array([s[0], s[1], zp])
                if self.move_toward(goal, self.place_psi, speed=0.04) and el > 1.0:
                    self.set_phase("nudge_up", t)
            else:
                goal = np.array([self.cur_p[0], self.cur_p[1], z_travel + 0.02])
                if self.move_toward(goal, self.place_psi, speed=0.15):
                    self.idx += 1
                    self.set_phase("approach", t)

        return self.finish(obs, t, site)

    def finish(self, obs, t, site):
        R = rot_down(self.cur_psi)
        q, ikerr = self.solve(self.cur_p, R, self.q_cmd)
        self.q_cmd = q
        if self.verbose and int(round(t / 0.04)) % 50 == 0:
            print(f"  t={t:.2f} ph={self.phase} site={np.round(site,3)} cur={np.round(self.cur_p,3)} ikerr={ikerr:.3f}")
        prev = np.array(obs["arm_target"])
        q_out = prev + np.clip(q - prev, -0.08, 0.08)
        return np.r_[q_out, self.closure]
