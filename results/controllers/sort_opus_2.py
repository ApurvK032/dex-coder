
import numpy as np
import mujoco


def _rot_err(R_cur, R_des):
    Re = R_des @ R_cur.T
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, Re.flatten())
    if q[0] < 0:
        q = -q
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    GRASP_DZ = 0.01      # site z above cube center at grasp
    PRE_DZ = 0.10        # pre-grasp height above grasp
    CARRY_Z = 0.97       # site z while carrying
    PLACE_Z = 0.786 + 0.025 + 0.035  # site z at release
    SPEED = 0.25
    PRECURL = 0.0
    YAW_RANGE = {"left": (-0.35, 0.7), "right": (-0.75, -0.05)}

    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower) + 0.03
        self.hi = np.array(robot.arm_upper) - 0.03
        self.rest = np.array(robot.home_arm, dtype=float)
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest], dtype=float)
        self.dt = robot.control_dt
        self.q_cmd = None
        self.phase = "init"
        self.t_phase = 0.0
        self.closure = 0.0
        self.cur_pos = None
        self.cur_R = None
        self.attempts = 0
        self.log = True
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros((3, self.model.nv))
        self.rng = np.random.default_rng(0)

    # ---------------- kinematics ----------------
    def fk(self, q):
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pos, R, iters=40, wrot=0.5, rest=None):
        rest = self.rest if rest is None else rest
        q = q0.copy()
        mid = 0.5 * (self.lo + self.hi)
        half = 0.5 * (self.hi - self.lo)
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = _rot_err(Rc, R)
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], wrot * self.jr[:, self.dadr]])
            e = np.r_[ep, wrot * er]
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            # nullspace: pull to rest and away from joint limits
            lim = -((q - mid) / half) ** 3
            dq += N @ (0.03 * (rest - q) + 0.05 * lim)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, Rc = self.fk(q)
        return q, (np.linalg.norm(pos - p), np.linalg.norm(_rot_err(Rc, R)))

    def ik_multi(self, seeds, pos, R, iters=80):
        best = None
        for s in seeds:
            q, err = self.ik(s, pos, R, iters=iters)
            c = 20 * err[0] + 2 * err[1]
            if best is None or c < best[0] - 1e-3:
                best = (c, q, err)
            if c < 0.01:
                break
        return best[1], best[2]

    # ---------------- helpers ----------------
    @staticmethod
    def grasp_R(yaw):
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y = np.array([0.0, 0.0, -1.0])
        z = np.cross(x, y)
        return np.column_stack([x, y, z])

    def set_phase(self, name, t):
        if self.log:
            print(f"t={t:.2f} phase {name}")
        self.phase = name
        self.t_phase = t

    def neighbor_cost(self, cpos, a, blocks):
        """penetration of the finger/thumb sweep footprint into other cubes and bins"""
        d = np.array([np.cos(a), np.sin(a)])
        p = np.array([-d[1], d[0]])
        pts = []
        for al in np.arange(0.0, 0.105, 0.01):
            for lat in (-0.04, -0.02, 0.02, 0.04):
                pts.append(cpos[:2] + al * d + lat * p)
        for al in (-0.065, -0.05, -0.035):
            for lat in (-0.015, 0.015):
                pts.append(cpos[:2] + al * d + lat * p)
        pts = np.array(pts)
        cost = 0.0
        for name, b in blocks.items():
            if name == self.target:
                continue
            v = pts - np.array(b["pos"][:2])
            cy, sy = np.cos(b["yaw"]), np.sin(b["yaw"])
            loc = np.abs(np.column_stack([v[:, 0] * cy + v[:, 1] * sy, -v[:, 0] * sy + v[:, 1] * cy]))
            pen = np.clip(np.minimum(0.034 - loc[:, 0], 0.034 - loc[:, 1]), 0, None)
            cost += 1.5 * pen.sum()
        h = self.robot.bin_inner_half_size + self.robot.bin_wall_thickness + 0.006
        for bxy in self.robot.bin_xy.values():
            v = np.abs(pts - np.array(bxy))
            pen = np.clip(np.minimum(h - v[:, 0], h - v[:, 1]), 0, None)
            cost += pen.sum()
        return 40.0 * cost

    def plan_grasp(self, cpos, cube_yaw, blocks, qnow):
        best = None
        seeds = [qnow, self.rest]
        for k in range(4):
            a = _wrap(cube_yaw + k * np.pi / 2)
            R = self.grasp_R(a)
            pref = abs(_wrap(a + 0.2))
            cost = 0.3 * pref + self.neighbor_cost(cpos, a, blocks)
            pre = np.array([cpos[0], cpos[1], max(cpos[2], 0.80) + self.GRASP_DZ + self.PRE_DZ])
            qp, e1 = self.ik_multi(seeds, pre, R)
            qg, e2 = self.ik(qp, pre - [0, 0, self.PRE_DZ], R, iters=60)
            ql, e3 = self.ik(qg, np.array([cpos[0], cpos[1], self.CARRY_Z]), R, iters=60)
            for e in (e1, e2, e3):
                cost += 20 * e[0] + 2 * e[1]
                if e[0] > 0.006 or e[1] > 0.06:
                    cost += 15.0
            cost += 0.02 * np.linalg.norm(qp - qnow)
            if best is None or cost < best[0]:
                best = (cost, a, qp, pre)
        _, self.yaw, self.q_pre, self.pre_pos = best
        self.gR = self.grasp_R(self.yaw)
        self.plan_xy = cpos[:2].copy()
        if self.log:
            print(f"plan yaw={self.yaw:.2f} cost={best[0]:.3f}")

    def plan_place(self, qnow):
        lo, hi = self.YAW_RANGE[self.dest]
        best = None
        for off in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            # hand yaw does not matter for the cube; stay close to current
            pass
        a = float(np.clip(self.yaw, lo, hi))
        self.pR = self.grasp_R(a)
        pos = np.array([self.bin[0], self.bin[1], self.CARRY_Z])
        self.q_place, err = self.ik_multi([qnow, self.rest], pos, self.pR)
        if self.log:
            print(f"place yaw={a:.2f} err={err[0]:.3f},{err[1]:.3f}")

    def start_joint_move(self, q_from, q_to, t, vmax=1.0):
        self.jm_from = q_from.copy()
        self.jm_to = q_to.copy()
        self.jm_T = max(0.4, np.max(np.abs(q_to - q_from)) / vmax * 1.5)
        self.jm_t0 = t

    def joint_move(self, t):
        s = min(1.0, (t - self.jm_t0) / self.jm_T)
        s2 = 0.5 - 0.5 * np.cos(np.pi * s)
        q = self.jm_from + s2 * (self.jm_to - self.jm_from)
        return q, s >= 1.0

    def move_toward(self, goal_pos, goal_R, speed=None):
        speed = speed or self.SPEED
        step = speed * self.dt
        d = goal_pos - self.cur_pos
        n = np.linalg.norm(d)
        self.cur_pos = self.cur_pos + (d * step / n if n > step else d)
        er = _rot_err(self.cur_R, goal_R)
        a = np.linalg.norm(er)
        amax = 0.8 * self.dt
        if a > 1e-6:
            frac = min(1.0, amax / a)
            q = np.empty(4)
            mujoco.mju_axisAngle2Quat(q, er / a, a * frac)
            Rq = np.empty(9)
            mujoco.mju_quat2Mat(Rq, q)
            self.cur_R = Rq.reshape(3, 3) @ self.cur_R
        return max(n - step, 0.0) + 0.05 * max(a - amax, 0.0)

    def cart_cmd(self):
        q, _ = self.ik(self.q_cmd, self.cur_pos, self.cur_R, iters=30, rest=self.q_cmd)
        return q

    def sync_cart(self):
        self.cur_pos, self.cur_R = self.fk(self.q_cmd)

    # ---------------- main ----------------
    def act(self, obs):
        t = obs["t"]
        if self.q_cmd is None:
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            self.sync_cart()
        blocks = obs["blocks"]
        cube = blocks[self.target]
        cpos = np.array(cube["pos"])
        site = np.array(obs["grasp_site_pos"])
        ph = self.phase
        el = t - self.t_phase
        q = self.q_cmd

        if ph == "init":
            self.plan_grasp(cpos, cube["yaw"], blocks, self.q_cmd)
            self.start_joint_move(self.q_cmd, self.q_pre, t)
            self.set_phase("goto_pre", t)
            ph, el = "goto_pre", 0.0

        if ph == "goto_pre":
            q, done = self.joint_move(t)
            self.q_cmd = q
            self.closure = min(self.PRECURL, self.closure + 0.04)
            if done:
                self.sync_cart()
                self.set_phase("align", t)
        elif ph == "align":
            if np.linalg.norm(cpos[:2] - self.plan_xy) > 0.02 and self.attempts < 6:
                self.attempts += 1
                self.set_phase("init", t)
                return np.r_[self.q_cmd, self.closure]
            goal = np.array([cpos[0], cpos[1], self.pre_pos[2]])
            rem = self.move_toward(goal, self.gR, speed=0.1)
            self.q_cmd = q = self.cart_cmd()
            if (rem < 1e-4 and np.linalg.norm(site - goal) < 0.01) or el > 2.0:
                self.set_phase("descend", t)
        elif ph == "descend":
            goal = np.array([cpos[0], cpos[1], max(cpos[2], 0.80) + self.GRASP_DZ])
            rem = self.move_toward(goal, self.gR, speed=0.12)
            self.q_cmd = q = self.cart_cmd()
            if (rem < 1e-4 and np.linalg.norm(site - goal) < 0.008) or el > 2.5:
                self.set_phase("close", t)
        elif ph == "close":
            self.closure = min(1.0, self.PRECURL + el / 0.5)
            if el > 0.8:
                self.set_phase("lift", t)
        elif ph == "lift":
            goal = np.array([self.cur_pos[0], self.cur_pos[1], self.CARRY_Z])
            rem = self.move_toward(goal, self.gR, speed=0.2)
            self.q_cmd = q = self.cart_cmd()
            if rem < 1e-4 and el > 0.3:
                held = np.linalg.norm(cpos - site) < 0.045 and cpos[2] > 0.86
                if held:
                    self.plan_place(self.q_cmd)
                    self.start_joint_move(self.q_cmd, self.q_place, t, vmax=0.8)
                    self.set_phase("carry", t)
                else:
                    self.retry(t)
        elif ph == "carry":
            q, done = self.joint_move(t)
            self.q_cmd = q
            if np.linalg.norm(cpos - site) > 0.06:
                self.retry(t)
            elif done and el > self.jm_T + 0.2:
                self.sync_cart()
                # lower until the held cube is just above the bin floor
                off = site[2] - cpos[2]
                self.place_z = float(np.clip(0.786 + 0.025 + 0.004 + off, 0.815, 0.86))
                self.set_phase("lower", t)
        elif ph == "lower":
            goal = np.array([self.bin[0], self.bin[1], self.place_z])
            rem = self.move_toward(goal, self.pR, speed=0.12)
            self.q_cmd = q = self.cart_cmd()
            if np.linalg.norm(cpos - site) > 0.06:
                self.retry(t)
            elif (rem < 1e-4 and abs(site[2] - goal[2]) < 0.008) or el > 2.5:
                self.set_phase("release", t)
        elif ph == "release":
            self.closure = max(self.PRECURL, 1.0 - el / 0.7)
            if el > 1.0:
                self.set_phase("retreat", t)
        elif ph == "retreat":
            goal = np.array([self.bin[0], self.bin[1], self.CARRY_Z + 0.03])
            self.move_toward(goal, self.pR, speed=0.15)
            self.q_cmd = q = self.cart_cmd()
            if el > 0.6:
                self.closure = max(0.0, self.closure - 0.05)
            if el > 1.5:
                inbin = np.all(np.abs(cpos[:2] - self.bin) < self.robot.bin_inner_half_size) and cpos[2] < 0.84
                if not inbin and np.linalg.norm(cube["linvel"]) < 0.05:
                    self.retry(t)
                else:
                    self.set_phase("done", t)
        elif ph == "reopen":
            goal = np.array([self.cur_pos[0], self.cur_pos[1], max(self.cur_pos[2], 0.93)])
            self.move_toward(goal, self.cur_R, speed=0.15)
            self.q_cmd = q = self.cart_cmd()
            if el > 0.8 and np.linalg.norm(cube["linvel"]) < 0.05:
                self.set_phase("init", t)
        elif ph == "done":
            pass

        return np.r_[self.q_cmd, self.closure]

    def retry(self, t):
        self.attempts += 1
        self.closure = 0.0
        if self.log:
            print(f"t={t:.2f} retry {self.attempts}")
        self.sync_cart()
        self.set_phase("reopen", t)
