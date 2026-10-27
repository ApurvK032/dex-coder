import numpy as np
import mujoco

HALF = 0.025
GZ = 0.004
GX = 0.0
REL_FWD = 0.0
PLACE_BACK = 0.0
REL_BACK = 0.0


def rdes(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [s, 0, c], [0, -1, 0]], dtype=float)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.ha = np.array(robot.hand_qpos_adr)
        self.lo = np.array(robot.arm_lower) + 0.03
        self.hi = np.array(robot.arm_upper) - 0.03
        self.home = np.array(robot.home_arm, dtype=float)
        self.target = task["target"]
        self.dest = task["destination"]
        self.table = robot.table_height
        self.q = self.home.copy()
        self.phase = "init"
        self.t_phase = 0.0
        self.closure = 0.0
        self.sp = None
        self.yaw = 0.0
        self.carry_z = self.table + 0.17
        self.rng = np.random.default_rng(0)
        m = self.m
        self.arm_bodies = set()
        for b in range(m.nbody):
            n = m.body(b).name
            if n.startswith("right_") and ("shoulder" in n or "elbow" in n or "wrist" in n or "hand" in n):
                self.arm_bodies.add(b)
        self.block_body = {}
        for c in ("red", "green", "blue"):
            self.block_body[c] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block_" + c)
        self.traj = None
        self.off_local = np.zeros(3)
        self.yaw_rel = 0.0
        self.attempts = 0
        self.kin_qpos = None
        self.rel_sp = None

    # ---------------- kinematics -----------------
    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, p, R, q0, iters=20, posture=0.03):
        q = q0.copy()
        for _ in range(iters):
            self.d.qpos[self.qa] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            cp = self.d.site_xpos[self.sid]
            cR = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = p - cp
            eR = 0.5 * (np.cross(cR[:, 0], R[:, 0]) + np.cross(cR[:, 1], R[:, 1]) + np.cross(cR[:, 2], R[:, 2]))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(eR) < 2e-3:
                break
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], 0.3 * jr[:, self.da]])
            e = np.concatenate([ep, 0.3 * eR])
            JJ = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if posture > 0:
                Jp = J.T @ np.linalg.inv(JJ)
                n = np.eye(7) - Jp @ J
                dq += n @ (posture * (self.home - q))
            mx = np.abs(dq).max()
            if mx > 0.3:
                dq *= 0.3 / mx
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def pose_err(self, q, p, R):
        pp, RR = self.fk(q)
        return np.linalg.norm(pp - p) + 0.05 * np.linalg.norm(RR - R)

    def collisions(self, q, ignore, hand_q=None):
        d = self.d
        d.qpos[:] = self.kin_qpos
        d.qpos[self.qa] = q
        if hand_q is not None:
            d.qpos[self.ha] = hand_q
        mujoco.mj_forward(self.m, d)
        pen = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            a1 = b1 in self.arm_bodies
            a2 = b2 in self.arm_bodies
            if a1 == a2:
                continue
            other = b2 if a1 else b1
            if other in ignore:
                continue
            if c.dist < -0.001:
                pen += -c.dist + 0.01
        return pen

    def solve(self, p, R, seed, ignore, hand_q=None, tries=6):
        best = None
        for s in range(tries):
            if s == 0:
                qi = seed
            elif s == 1:
                qi = self.home
            else:
                qi = np.clip(seed + self.rng.normal(0, 0.5, 7), self.lo, self.hi)
            q = self.ik(p, R, qi, iters=120)
            e = self.pose_err(q, p, R)
            if e > 0.004:
                continue
            col = self.collisions(q, ignore, hand_q)
            cost = col * 100 + 0.1 * np.linalg.norm(q - seed)
            if best is None or cost < best[1]:
                best = (q, cost, col)
            if col == 0 and s >= 1:
                break
        return best

    def plan_pair(self, low, high, yaws, seed, ignore, hand_q=None):
        """Find yaw such that both low and high poses are feasible."""
        best = None
        for yaw in yaws:
            R = rdes(yaw)
            sl = self.solve(low, R, seed, ignore, hand_q)
            if sl is None:
                continue
            sh = self.solve(high, R, sl[0], ignore, hand_q, tries=3)
            if sh is None:
                continue
            cost = sl[1] + sh[1] + 0.2 * np.linalg.norm(sh[0] - seed) + 0.1 * np.linalg.norm(sl[0] - self.home)
            if best is None or cost < best[0]:
                best = (cost, yaw, sl[0], sh[0])
        return best

    # ---------------- helpers -----------------
    def set_phase(self, name, t):
        self.phase = name
        self.t_phase = t

    def start_traj(self, qgoal, t, speed=1.0):
        dq = np.abs(qgoal - self.q).max()
        T = max(0.4, dq / speed * 1.6)
        self.traj = (self.q.copy(), qgoal.copy(), t, T)

    def traj_step(self, t):
        q0, q1, t0, T = self.traj
        s = min(1.0, (t - t0) / T)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        return q0 + (q1 - q0) * s, (t - t0) >= T

    def move_toward(self, goal, speed):
        diff = goal - self.sp
        n = np.linalg.norm(diff)
        step = speed * self.r.control_dt
        if n <= step:
            self.sp = goal.copy()
            return True
        self.sp = self.sp + diff * step / n
        return False

    def cart_q(self):
        return self.ik(self.sp, rdes(self.yaw), self.q, iters=15, posture=0.0)

    def plan_pick(self, obs, t):
        b = obs["blocks"][self.target]
        tp = np.array(b["pos"])
        yaws = [wrap(b["yaw"] + k * np.pi / 2) for k in range(4)]
        low = tp + np.array([0, 0, GZ])
        high = np.array([tp[0], tp[1], max(self.carry_z, tp[2] + 0.12)])
        ignore = {self.block_body[self.target]}
        plan = self.plan_pair(low, high, yaws, self.q, ignore, hand_q=np.zeros(7))
        if plan is None:
            return False
        _, self.yaw, ql, qh = plan
        self.start_traj(qh, t)
        self.sp = high.copy()
        self.sp[0:2] += GX * rdes(self.yaw)[0:2, 0]
        self.set_phase("to_pick", t)
        return True

    def plan_place(self, obs, t):
        b = obs["blocks"][self.dest]
        dp = np.array(b["pos"])
        # desired cube yaw aligned with dest; hand yaw = cube yaw - yaw_rel
        yaws = [wrap(b["yaw"] + k * np.pi / 2 - self.yaw_rel) for k in range(4)]
        ignore = {self.block_body[self.target], self.block_body[self.dest]}
        best = None
        for yaw in yaws:
            R = rdes(yaw)
            off = R @ self.off_local
            low = dp + np.array([0, 0, 2 * HALF + 0.006]) - off
            high = np.array([low[0], low[1], self.carry_z])
            plan = self.plan_pair(low, high, [yaw], self.q, ignore, hand_q=None)
            if plan is not None and (best is None or plan[0] < best[0]):
                best = (plan[0], yaw, plan[2], plan[3], low, high)
        if best is None:
            return False
        _, self.yaw, ql, qh, low, high = best
        self.place_low = low
        self.start_traj(qh, t, speed=0.9)
        self.sp = high.copy()
        self.set_phase("to_place", t)
        return True

    # ---------------- main -----------------
    def act(self, obs):
        t = obs["t"]
        blocks = obs["blocks"]
        tp = np.array(blocks[self.target]["pos"])
        dp = np.array(blocks[self.dest]["pos"])
        site = np.array(obs["grasp_site_pos"])
        smat = np.array(obs["grasp_site_mat"])
        self.kin_qpos = obs["kin"].qpos.copy()
        el = t - self.t_phase

        if self.phase == "init":
            self.q = np.array(obs["arm_target"], dtype=float)
            self.set_phase("plan_pick", t)

        if self.phase == "plan_pick":
            self.closure = 0.0
            if not self.plan_pick(obs, t):
                print("pick plan failed", t)
                self.set_phase("plan_pick", t)
        elif self.phase == "to_pick":
            self.closure = 0.0
            self.q, done = self.traj_step(t)
            if done and el > 0.2:
                self.set_phase("descend", t)
        elif self.phase == "descend":
            self.closure = 0.0
            R = rdes(self.yaw)
            g = tp + np.array([0, 0, GZ]) + GX * R[:, 0]
            self.sp[0:2] = g[0:2]
            ok = self.move_toward(g, 0.12)
            self.q = self.cart_q()
            if ok and (np.linalg.norm(site - g) < 0.006 or el > 2.5):
                self.set_phase("close", t)
        elif self.phase == "close":
            self.closure = min(1.0, el / 0.7)
            if el > 1.0:
                self.set_phase("lift", t)
        elif self.phase == "lift":
            self.closure = 1.0
            g = np.array([self.sp[0], self.sp[1], self.carry_z])
            ok = self.move_toward(g, 0.2)
            self.q = self.cart_q()
            if ok and el > 0.8:
                if np.linalg.norm(tp - site) < 0.04 and tp[2] > self.table + 0.08:
                    self.off_local = smat.T @ (tp - site)
                    hand_yaw = np.arctan2(smat[1, 0], smat[0, 0])
                    self.yaw_rel = wrap(blocks[self.target]["yaw"] - hand_yaw)
                    self.yaw_rel = wrap(self.yaw_rel - np.round(self.yaw_rel / (np.pi / 2)) * np.pi / 2)
                    if not self.plan_place(obs, t):
                        print("place plan failed", t)
                        self.set_phase("lift", t)
                else:
                    self.attempts += 1
                    self.closure = 0.0
                    self.set_phase("plan_pick", t)
        elif self.phase == "to_place":
            self.closure = 1.0
            self.q, done = self.traj_step(t)
            if np.linalg.norm(tp - site) > 0.07:
                self.set_phase("recover", t)
            elif done and el > 0.2:
                self.set_phase("lower", t)
        elif self.phase == "lower":
            self.closure = 1.0
            # refine target using current measurements
            off_now = tp - site
            g = dp + np.array([0, 0, 2 * HALF - 0.006]) - off_now - PLACE_BACK * rdes(self.yaw)[:, 0]
            self.sp[0:2] += np.clip(g[0:2] - self.sp[0:2], -0.002, 0.002)
            gap = tp[2] - dp[2] - 2 * HALF
            spd = 0.08 if gap > 0.025 else 0.03
            gg = np.array([self.sp[0], self.sp[1], min(self.sp[2], g[2])])
            ok = self.move_toward(gg, spd)
            self.q = self.cart_q()
            if np.linalg.norm(tp - site) > 0.07:
                self.set_phase("recover", t)
            elif el > 0.5 and ((gap < 0.003 and abs(blocks[self.target]["linvel"][2]) < 0.02) or el > 4.0):
                self.set_phase("release", t)
        elif self.phase == "release":
            if el < 0.05:
                # remove downward pressure: hold at measured height
                self.sp[2] = max(self.sp[2], site[2])
                self.q = self.cart_q()
            if el < 0.5:
                self.closure = max(0.5, 1.0 - el / 0.4)
            elif el < 1.1:
                self.closure = 0.5
                if el >= 0.6:
                    if self.rel_sp is None:
                        self.rel_sp = self.sp.copy()
                    s = min(1.0, (el - 0.6) / 0.5)
                    self.sp = self.rel_sp + REL_FWD * s * rdes(self.yaw)[:, 0]
                    self.q = self.cart_q()
            elif el < 1.9:
                self.closure = max(0.0, 0.5 - (el - 1.1) / 0.7)
            else:
                self.closure = 0.0
                s = min(1.0, (el - 1.9) / 0.5)
                self.sp = self.rel_sp + (REL_FWD - REL_BACK * s) * rdes(self.yaw)[:, 0]
                self.q = self.cart_q()
            if el > 2.5:
                self.rel_sp = None
                self.set_phase("retreat", t)
        elif self.phase == "retreat":
            self.closure = 0.0
            g = np.array([self.sp[0], self.sp[1], self.carry_z + 0.03])
            lifted = tp[2] - dp[2] - 2 * HALF > 0.004 and np.linalg.norm(tp[:2] - dp[:2]) < 0.03
            if not hasattr(self, "unhook"):
                self.unhook = 0
            if lifted and site[2] - tp[2] < 0.06 and self.unhook < 3:
                # cube is being dragged up: go back down and shift away from the thumb contact
                self.unhook += 1
                Rh = rdes(self.yaw)
                self.sp = self.sp - np.array([0, 0, tp[2] - dp[2] - 2 * HALF + 0.002])
                self.sp = self.sp + 0.008 * (Rh[:, 0] if self.unhook % 2 else -Rh[:, 0])
                self.q = self.cart_q()
                self.set_phase("retreat", t)
            else:
                self.move_toward(g, 0.05 if el < 1.0 else 0.12)
                self.q = self.cart_q()
            on = (abs(tp[2] - dp[2] - 2 * HALF) < 0.01 and np.linalg.norm(tp[:2] - dp[:2]) < 0.02)
            if el > 2.0 and not on:
                self.attempts += 1
                self.set_phase("plan_pick", t)
        elif self.phase == "recover":
            self.closure = 0.0
            g = np.array([self.sp[0], self.sp[1], self.carry_z])
            ok = self.move_toward(g, 0.2)
            self.q = self.cart_q()
            if ok and el > 0.8:
                self.set_phase("plan_pick", t)

        return np.concatenate([self.q, [self.closure]])
