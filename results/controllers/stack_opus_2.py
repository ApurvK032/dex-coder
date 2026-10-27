
import numpy as np
import mujoco

TABLE_Z = 0.78
H = 0.025

SEEDS = [
    [0.875, -0.751, -1.223, -0.331, -0.24, -0.28, 0.391],
    [-0.26, -0.79, 0.18, -0.12, 0.02, -1.29, 0.77],
    [0.18, -0.71, -0.45, -0.11, -0.77, -0.31, 0.21],
    [0.4, -0.44, -0.14, -0.72, -0.81, -0.65, 0.41],
    [-0.03, -0.42, -0.07, -0.35, -0.82, -0.64, 0.48],
    [0.59, -0.61, -0.67, -0.33, -0.66, -0.44, 0.14],
]


def rotvec_err(Rd, Rc):
    E = Rd @ Rc.T
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, E.flatten())
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return np.zeros(3)
    return v / s * 2 * np.arctan2(s, q[0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tilt_deg(quat):
    w, x, y, z = quat
    return 2 * np.degrees(np.arccos(min(1.0, np.hypot(w, z))))


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.qnom = self.home.copy()
        self.q = self.home.copy()
        self.target = task["target"]
        self.dest = task["destination"]
        self.phase = None
        self.phase_t0 = 0.0
        self.closure = 0.0
        self.cur_pos = None
        self.cur_yaw = None
        self.corr = np.zeros(3)
        self.log = True
        self.press = 0.0
        self.open_time = 0.8
        self.rel_up = 0.01
        self.gdz = 0.001
        self.gdx = 0.008
        self.attempts = 0
        self.HOVER = TABLE_Z + H + 0.08
        self.CARRY = TABLE_Z + H + 0.13
        jb = set()
        for j in list(robot.arm_joints) + list(robot.hand_joints):
            jb.add(self.m.jnt_bodyid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)])
        self.armbodies = jb
        self.blockbodies = set(
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "block_" + c) for c in ("red", "green", "blue"))
        self.tablebody = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "table")

    # ---------- kinematics ----------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pdes, Rdes, iters=15, qnom=None):
        if qnom is None:
            qnom = self.qnom
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mid = 0.5 * (self.lo + self.hi)
        half = 0.5 * (self.hi - self.lo)
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pdes - p
            er = rotvec_err(Rdes, R)
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dadr], 0.3 * jr[:, self.dadr]])
            e = np.r_[ep, 0.3 * er]
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            lim = -((q - mid) / half) ** 5
            dq += N @ (0.05 * (qnom - q) + 0.05 * lim)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def ik_err(self, q, pdes, Rdes):
        p, R = self.fk(q)
        return np.linalg.norm(pdes - p) + 0.05 * np.linalg.norm(rotvec_err(Rdes, R))

    def coll_depth(self, q, kin):
        d = self.d
        d.qpos[:] = kin.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, d)
        worst = 0.0
        for k in range(d.ncon):
            con = d.contact[k]
            b1 = self.m.geom_bodyid[con.geom1]
            b2 = self.m.geom_bodyid[con.geom2]
            a1 = b1 in self.armbodies
            a2 = b2 in self.armbodies
            if a1 == a2:
                continue
            o = b2 if a1 else b1
            if o in self.blockbodies:
                continue
            dist = con.dist
            if o == self.tablebody:
                dist += 0.004
            worst = min(worst, dist)
        return worst

    @staticmethod
    def Rgrip(yaw):
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y = np.array([0.0, 0.0, -1.0])
        return np.column_stack([x, y, np.cross(x, y)])

    def clutter_penalty(self, p, yaw):
        """Penalty if another cube lies where fingers sweep (+x) or the thumb hangs (-x)."""
        pen = 0.0
        c, s = np.cos(yaw), np.sin(yaw)
        for name, bp in self.others.items():
            dx = bp[0] - p[0]
            dy = bp[1] - p[1]
            if abs(bp[2] - p[2]) > 0.035:
                continue
            ax = c * dx + s * dy
            lat = -s * dx + c * dy
            if abs(lat) < 0.065:
                if 0.0 < ax < 0.145:
                    pen += 3.0 * (1.0 - ax / 0.145) + 1.0
                elif -0.10 < ax <= 0.0:
                    pen += 1.0
        return pen

    def plan(self, base_yaw, p_low, p_high, kin):
        """Choose yaw (base + k*90deg) and joint configs for hover (p_high) and low pose."""
        best = None
        seeds = [self.q.copy()] + [np.array(s) for s in SEEDS]
        for k in range(4):
            yaw = wrap(base_yaw + k * np.pi / 2)
            R = self.Rgrip(yaw)
            for s in seeds:
                qh = self.ik(s, p_high, R, iters=150, qnom=s)
                if self.ik_err(qh, p_high, R) > 0.003:
                    continue
                ql = self.ik(qh, p_low, R, iters=150, qnom=qh)
                if self.ik_err(ql, p_low, R) > 0.003:
                    continue
                c = min(self.coll_depth(qh, kin), self.coll_depth(ql, kin))
                marg = min(np.min(np.minimum(qh - self.lo, self.hi - qh)),
                           np.min(np.minimum(ql - self.lo, self.hi - ql)))
                score = min(marg, 0.5) + (0 if c > -0.001 else -5 + 20 * c)
                score -= 0.03 * np.max(np.abs(qh - self.q))
                score -= self.clutter_penalty(p_low, yaw)
                if best is None or score > best[0]:
                    best = (score, yaw, qh, ql)
        if best is None:
            yaw = base_yaw
            qh = self.ik(self.q, p_high, self.Rgrip(yaw), iters=150)
            best = (-99, yaw, qh, qh)
        if self.log:
            print(f"  plan yaw {best[1]:.2f} score {best[0]:.2f}")
        return best[1], best[2], best[3]

    # ---------- motion helpers ----------
    def set_phase(self, name, t):
        if self.log:
            print(f"t={t:.2f} phase {name} site={np.round(self.site, 3)}")
        self.phase = name
        self.phase_t0 = t

    def start_joint_move(self, qgoal, t):
        self.jm_q0 = self.q.copy()
        self.jm_q1 = qgoal.copy()
        self.jm_T = max(0.6, np.max(np.abs(qgoal - self.q)) / 1.0)
        self.jm_t0 = t

    def joint_move(self, t):
        s = min(1.0, (t - self.jm_t0) / self.jm_T)
        s = s * s * (3 - 2 * s)
        return self.jm_q0 + s * (self.jm_q1 - self.jm_q0), s >= 1.0

    def move_toward(self, goal, yaw_goal, vmax, wmax=1.5):
        dt = self.r.control_dt
        d = goal - self.cur_pos
        n = np.linalg.norm(d)
        step = vmax * dt
        if n > step:
            self.cur_pos = self.cur_pos + d / n * step
        else:
            self.cur_pos = goal.copy()
        dy = wrap(yaw_goal - self.cur_yaw)
        sy = wmax * dt
        self.cur_yaw = self.cur_yaw + np.clip(dy, -sy, sy)
        return n <= step and abs(dy) <= sy

    def begin_cartesian(self, yaw):
        self.cur_pos = self.site.copy()
        self.cur_yaw = yaw
        self.corr[:] = 0.0

    def act(self, obs):
        t = obs["t"]
        kin = obs["kin"]
        self.site = np.array(obs["grasp_site_pos"])
        blocks = obs["blocks"]
        tb = np.array(blocks[self.target]["pos"])
        db = np.array(blocks[self.dest]["pos"])
        if self.phase is None:
            self.q = np.array(obs["arm_target"], dtype=float)
            self.phase = "plan_grasp"
        self.others = {n: np.array(b["pos"]) for n, b in blocks.items() if n != self.target}
        pt = t - self.phase_t0
        cart = False
        qcmd = None

        if self.phase == "plan_grasp":
            self.attempts += 1
            by = blocks[self.target]["yaw"]
            self.grip_yaw, qh, self.qgrasp = self.plan(
                by, np.r_[tb[:2], tb[2] + self.gdz], np.r_[tb[:2], self.HOVER], kin)
            gx = np.array([np.cos(self.grip_yaw), np.sin(self.grip_yaw)])
            qh = self.ik(qh, np.r_[tb[:2] + self.gdx * gx, self.HOVER], self.Rgrip(self.grip_yaw), iters=100, qnom=qh)
            self.qnom = self.qgrasp
            self.start_joint_move(qh, t)
            self.set_phase("approach", t)
        if self.phase == "approach":
            qcmd, done = self.joint_move(t)
            if done and pt > self.jm_T + 0.2:
                self.begin_cartesian(self.grip_yaw)
                self.set_phase("descend", t)
        elif self.phase == "descend":
            cart = True
            gx = np.array([np.cos(self.grip_yaw), np.sin(self.grip_yaw)])
            goal = np.r_[tb[:2] + self.gdx * gx, tb[2] + self.gdz]
            done = self.move_toward(goal, self.grip_yaw, 0.12)
            err = np.linalg.norm(self.site - self.cur_pos)
            if done and (err < 0.004 or pt > 2.5):
                if self.site[2] > goal[2] + 0.012 and self.attempts < 4:
                    self.set_phase("regrasp_open", t)
                else:
                    self.set_phase("close", t)
        elif self.phase == "close":
            cart = True
            self.closure = min(1.0, pt / 0.6)
            if pt > 1.0:
                self.set_phase("lift", t)
        elif self.phase == "lift":
            cart = True
            goal = np.r_[self.cur_pos[:2], self.CARRY]
            done = self.move_toward(goal, self.cur_yaw, 0.15)
            if done:
                held = tb[2] > TABLE_Z + H + 0.05 and np.linalg.norm(tb - self.site) < 0.04
                if not held and self.attempts < 4:
                    self.closure = 0.0
                    self.set_phase("regrasp_open", t)
                else:
                    self.off = self.site - tb
                    rel = wrap(blocks[self.dest]["yaw"] - blocks[self.target]["yaw"])
                    pd = np.r_[db[:2] + self.off[:2], db[2] + 2 * H + self.off[2]]
                    self.place_yaw, qh, ql = self.plan(
                        self.grip_yaw + rel, pd + self.corr, np.r_[pd[:2], self.CARRY] + self.corr, kin)
                    self.qnom = ql
                    self.start_joint_move(qh, t)
                    self.set_phase("carry", t)
        elif self.phase == "regrasp_open":
            cart = True
            self.closure = 0.0
            done = self.move_toward(np.r_[self.cur_pos[:2], self.HOVER], self.cur_yaw, 0.1)
            if done and pt > 0.6:
                self.phase = "plan_grasp"
                self.phase_t0 = t
                qcmd = self.q
        elif self.phase == "carry":
            qcmd, done = self.joint_move(t)
            if done and pt > self.jm_T + 0.3:
                self.begin_cartesian(self.place_yaw)
                self.frozen = None
                self.set_phase("lower", t)
        elif self.phase == "lower":
            cart = True
            if self.frozen is None:
                off = self.site - tb
                if tb[2] - (db[2] + 2 * H) < 0.02:
                    self.frozen = off.copy()
            else:
                off = self.frozen
            goal = np.r_[db[:2] + off[:2], db[2] + 2 * H - self.press + off[2]]
            v = 0.1 if self.cur_pos[2] > goal[2] + 0.03 else 0.03
            done = self.move_toward(goal, self.place_yaw, v)
            err = np.linalg.norm(self.site - self.cur_pos)
            if done and (err < 0.004 or pt > 4.0):
                self.rel_z0 = self.cur_pos[2]
                self.set_phase("release", t)
        elif self.phase == "release":
            cart = True
            self.closure = max(0.0, 1.0 - pt / self.open_time)
            if pt > 0.15:
                goal = np.r_[self.cur_pos[:2], self.rel_z0 + self.rel_up]
                self.move_toward(goal, self.cur_yaw, self.rel_up / self.open_time)
            if pt > self.open_time + 0.3:
                self.set_phase("retreat", t)
        elif self.phase == "retreat":
            cart = True
            goal = np.r_[self.cur_pos[:2], self.CARRY + 0.05]
            self.move_toward(goal, self.cur_yaw, 0.1)

        if cart:
            self.corr += 0.15 * (self.cur_pos - self.site)
            self.corr = np.clip(self.corr, -0.03, 0.03)
            qcmd = self.ik(self.q, self.cur_pos + self.corr, self.Rgrip(self.cur_yaw))
        if qcmd is None:
            qcmd = self.q
        prev = np.array(obs["arm_target"], dtype=float)
        qcmd = np.clip(qcmd, prev - 0.08, prev + 0.08)
        self.q = qcmd
        return np.r_[qcmd, self.closure]
