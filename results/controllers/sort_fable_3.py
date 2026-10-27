
import numpy as np
import mujoco


class IK:
    def __init__(self, robot, qpos_full):
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = qpos_full
        mujoco.mj_forward(self.m, self.d)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower, dtype=float)
        self.hi = np.array(robot.arm_upper, dtype=float)
        # tighter limits to avoid wild / self-colliding postures
        self.lo[0] = max(self.lo[0], -1.6); self.hi[0] = min(self.hi[0], 1.3)
        self.lo[1] = max(self.lo[1], -1.6); self.hi[1] = min(self.hi[1], 0.0)
        self.lo[2] = max(self.lo[2], -1.6); self.hi[2] = min(self.hi[2], 1.6)
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.home = np.clip(np.array(robot.home_arm, dtype=float), self.lo, self.hi)
        # body classification for collision checks
        self.body_is_arm = np.zeros(self.m.nbody, dtype=bool)
        self.body_is_hand = np.zeros(self.m.nbody, dtype=bool)
        self.body_ignore = np.zeros(self.m.nbody, dtype=bool)
        for b in range(self.m.nbody):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if n.startswith("right_hand"):
                self.body_is_hand[b] = True
            elif n.startswith("right_") and ("shoulder" in n or "elbow" in n or "wrist" in n):
                self.body_is_arm[b] = True
            if n.startswith("block") or n == "table" or n.startswith("bin"):
                self.body_ignore[b] = True
        # sample database for seeding
        rng = np.random.default_rng(12345)
        N = 2500
        self.P = np.zeros((N, 3)); self.Q = np.zeros((N, 7))
        self.X = np.zeros((N, 3)); self.Y = np.zeros((N, 3))
        for i in range(N):
            q = rng.uniform(self.lo, self.hi)
            p, R = self.fk(q)
            self.Q[i] = q; self.P[i] = p; self.X[i] = R[:, 0]; self.Y[i] = R[:, 1]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def collision(self, q):
        """Penetration sum between the arm/hand and the robot's other bodies (self-collision)."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        pen = 0.0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.dist >= 0:
                continue
            b1 = self.m.geom_bodyid[c.geom1]; b2 = self.m.geom_bodyid[c.geom2]
            if self.body_ignore[b1] or self.body_ignore[b2]:
                continue
            r1 = self.body_is_arm[b1] or self.body_is_hand[b1]
            r2 = self.body_is_arm[b2] or self.body_is_hand[b2]
            if self.body_is_hand[b1] and self.body_is_hand[b2]:
                continue
            if r1 or r2:
                pen += -c.dist
        return pen

    def refine(self, q0, pt, xdir, ydir, wx=0.3, wy=0.3, iters=40, step=0.3):
        q = np.clip(q0, self.lo, self.hi)
        lo, hi = self.lo, self.hi
        for it in range(iters):
            p, R = self.fk(q)
            ep = pt - p
            ex = np.cross(R[:, 0], xdir); ey = np.cross(R[:, 1], ydir)
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(ex) < 2e-3 and np.linalg.norm(ey) < 2e-3:
                break
            err = np.concatenate([ep, wx * ex, wy * ey])
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            Jp = self.jacp[:, self.dadr]; Jr = self.jacr[:, self.dadr]
            J = np.vstack([Jp, wx * Jr, wy * Jr])
            atlim = (q <= lo + 1e-6) | (q >= hi - 1e-6)
            J[:, atlim] *= 0.3
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(9), err)
            q = np.clip(q + np.clip(dq, -step, step), lo, hi)
        p, R = self.fk(q)
        perr = np.linalg.norm(pt - p)
        xang = np.arccos(np.clip(R[:, 0] @ xdir, -1, 1))
        yang = np.arccos(np.clip(R[:, 1] @ ydir, -1, 1))
        return q, perr, xang, yang

    def solve(self, pt, xdir, ydir, q_seed=None, nseeds=6, tol_p=2e-3, tol_a=0.05, check_col=True):
        seeds = []
        if q_seed is not None:
            seeds.append(q_seed)
        seeds.append(self.home)
        score = np.linalg.norm(self.P - pt, axis=1) + 0.05 * np.linalg.norm(self.Y - ydir, axis=1) \
            + 0.03 * np.linalg.norm(self.X - xdir, axis=1)
        idx = np.argsort(score)[:nseeds]
        seeds += [self.Q[i] for i in idx]
        best = None
        for s in seeds:
            q, perr, xang, yang = self.refine(s, pt, xdir, ydir)
            sc = perr + 0.02 * (xang + yang)
            col = 0.0
            if check_col:
                col = self.collision(q)
                sc += 3.0 * col
            if q_seed is not None:
                sc += 0.002 * np.linalg.norm(q - q_seed)
            if best is None or sc < best[0]:
                best = (sc, q, perr, xang, yang, col)
            if perr < tol_p and xang < tol_a and yang < tol_a and col == 0.0:
                break
        return best[1:]


def yaw_dir(yaw):
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    DOWN = np.array([0.0, 0.0, -1.0])
    GRASP_DX = -0.008
    GRASP_DZ = 0.0
    YAW_OFF = np.pi / 4
    RELEASE_Z = 0.095
    RELEASE_T = 0.5
    PRE_Z = 0.11
    LIFT_Z = 0.17

    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest], dtype=float)
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.ik = None
        self.phase = "init"
        self.phase_t0 = 0.0
        self.q_cmd = np.array(robot.home_arm, dtype=float)
        self.closure = 0.0
        self.cart = None
        self.yaw = 0.0
        self.grasp_yaw = 0.0
        self.bin_yaw = self.pref_yaw(self.bin[1])
        self.attempts = 0
        self.v_max = 0.6
        self.yaw_rate = 2.0
        self.last_t = 0.0
        self.lift_ref = None
        self.dbg = []

    # ---------- helpers ----------
    @staticmethod
    def pref_yaw(y):
        return float(np.clip(0.5 + (y + 0.1) * 3.0, -0.3, 0.6))

    def choose_yaw(self, bpos, block_yaw):
        pref = self.pref_yaw(bpos[1])
        cands = []
        for k in range(-2, 3):
            y = wrap(block_yaw + self.YAW_OFF + k * np.pi / 2)
            if abs(y - pref) <= np.pi / 4 + 0.05:
                cands.append(y)
        cands = sorted(set(np.round(cands, 4)), key=lambda y: abs(y - pref))
        if not cands:
            cands = [pref]
        best = None
        for y in cands:
            xd = yaw_dir(y)
            p1 = np.array([bpos[0], bpos[1], self.table + self.half + self.GRASP_DZ])
            p2 = np.array([bpos[0], bpos[1], self.table + self.PRE_Z])
            cost = 0.0
            for p in (p1, p2):
                q, perr, xang, yang, col = self.ik.solve(p, xd, self.DOWN, q_seed=self.q_cmd, nseeds=4)
                cost += perr + 0.02 * (xang + yang) + 3.0 * col
            cost += 0.002 * abs(y - pref)
            # penalize other cubes sitting behind the hand (wrist/forearm side) or beside it
            for name, ob in self.other_blocks.items():
                rel = np.array(ob[:2]) - bpos[:2]
                along = rel @ xd[:2]
                lat = abs(rel[0] * (-xd[1]) + rel[1] * xd[0])
                if -0.16 < along < 0.0 and lat < 0.07:
                    cost += 0.05
                if np.hypot(along, lat) < 0.09:
                    cost += 0.02
            if best is None or cost < best[0]:
                best = (cost, y)
        return float(best[1])

    def set_phase(self, name, t):
        self.phase = name
        self.phase_t0 = t

    def move_toward(self, wp, yaw_goal, dt):
        if self.cart is None:
            self.cart = wp.copy()
        delta = wp - self.cart
        dist = np.linalg.norm(delta)
        maxd = self.v_max * dt
        if dist > maxd:
            self.cart = self.cart + delta / dist * maxd
        else:
            self.cart = wp.copy()
        dy = wrap(yaw_goal - self.yaw)
        maxy = self.yaw_rate * dt
        self.yaw += np.clip(dy, -maxy, maxy)
        xd = yaw_dir(self.yaw)
        q, perr, xang, yang = self.ik.refine(self.q_cmd, self.cart, xd, self.DOWN, iters=15)
        if perr > 0.01 or xang > 0.15 or yang > 0.15:
            q2, perr2, xang2, yang2, col2 = self.ik.solve(self.cart, xd, self.DOWN, q_seed=self.q_cmd, nseeds=4)
            if perr2 + 0.02 * (xang2 + yang2) + 3.0 * col2 < perr + 0.02 * (xang + yang) - 0.003:
                q = q2
        self.q_cmd = q
        return dist

    def arrived(self, obs, wp, tol=0.015):
        p = obs["grasp_site_pos"]
        return np.linalg.norm(p - wp) < tol and np.linalg.norm(obs["qvel"][:7]) < 0.6

    def in_bin(self, bpos):
        return (abs(bpos[0] - self.bin[0]) < 0.05 and abs(bpos[1] - self.bin[1]) < 0.05
                and bpos[2] < self.table + 0.06)

    # ---------- main ----------
    def act(self, obs):
        t = obs["t"]
        dt = max(t - self.last_t, 1e-3) if self.ik is not None else 0.04
        self.last_t = t
        if self.ik is None:
            self.ik = IK(self.robot, obs["kin"].qpos.copy())
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            self.cart = obs["grasp_site_pos"].copy()
            R = obs["grasp_site_mat"]
            self.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            self.set_phase("plan", t)

        blk = obs["blocks"][self.target]
        bpos = np.array(blk["pos"], dtype=float)
        ph = self.phase
        elapsed = t - self.phase_t0

        if ph == "plan":
            self.other_blocks = {k: np.array(v["pos"], dtype=float)
                                 for k, v in obs["blocks"].items() if k != self.target}
            self.grasp_yaw = self.choose_yaw(bpos, float(blk["yaw"]))
            self.closure = 0.0
            self.v_max = 0.6
            self.set_phase("pregrasp", t)
            ph = "pregrasp"

        if ph == "pregrasp":
            wp = np.array([bpos[0], bpos[1], self.table + self.PRE_Z])
            self.closure = 0.0
            self.move_toward(wp, self.grasp_yaw, dt)
            if self.arrived(obs, wp, 0.012) or elapsed > 4.0:
                self.set_phase("descend", t)
                self.v_max = 0.3
        elif ph == "descend":
            xd = yaw_dir(self.grasp_yaw)
            wp = np.array([bpos[0] + self.GRASP_DX * xd[0], bpos[1] + self.GRASP_DX * xd[1],
                           self.table + self.half + self.GRASP_DZ])
            self.move_toward(wp, self.grasp_yaw, dt)
            if self.arrived(obs, wp, 0.008) or elapsed > 2.5:
                self.set_phase("close", t)
                self.v_max = 0.6
        elif ph == "close":
            self.closure = min(1.0, elapsed / 0.5)
            if elapsed > 0.9:
                self.lift_ref = bpos.copy()
                self.set_phase("lift", t)
                self.v_max = 0.4
        elif ph == "lift":
            self.closure = 1.0
            wp = np.array([self.lift_ref[0], self.lift_ref[1], self.table + self.LIFT_Z])
            self.move_toward(wp, self.grasp_yaw, dt)
            if self.arrived(obs, wp, 0.02) or elapsed > 2.0:
                self.v_max = 0.6
                if bpos[2] > self.table + self.half + 0.06:
                    self.set_phase("transport", t)
                else:
                    self.attempts += 1
                    self.set_phase("reopen", t)
        elif ph == "reopen":
            self.closure = 0.0
            if elapsed > 0.5:
                self.set_phase("plan", t)
        elif ph == "transport":
            self.closure = 1.0
            wp = np.array([self.bin[0], self.bin[1], self.table + self.LIFT_Z])
            self.move_toward(wp, self.bin_yaw, dt)
            if self.arrived(obs, wp, 0.02) or elapsed > 3.0:
                self.set_phase("lower", t)
                self.v_max = 0.3
        elif ph == "lower":
            self.closure = 1.0
            wp = np.array([self.bin[0], self.bin[1], self.table + self.RELEASE_Z])
            self.move_toward(wp, self.bin_yaw, dt)
            if self.arrived(obs, wp, 0.012) or elapsed > 2.0:
                self.set_phase("release", t)
                self.v_max = 0.6
        elif ph == "release":
            self.closure = max(0.0, 1.0 - elapsed / self.RELEASE_T)
            if elapsed > self.RELEASE_T + 0.3:
                self.set_phase("retreat", t)
        elif ph == "retreat":
            self.closure = 0.0
            wp = np.array([self.bin[0], self.bin[1], self.table + 0.22])
            self.move_toward(wp, self.bin_yaw, dt)
            if self.arrived(obs, wp, 0.02) or elapsed > 2.0:
                self.set_phase("done", t)
        elif ph == "done":
            self.closure = 0.0
            if not self.in_bin(bpos) and elapsed > 1.0 and t < 18.0:
                self.set_phase("plan", t)

        return np.concatenate([self.q_cmd, [self.closure]])
