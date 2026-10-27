
import numpy as np
import mujoco


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_90(a):
    """Wrap angle to [-pi/4, pi/4) (cube symmetry)."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def R_topdown(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def rot_err(R_des, R):
    Rerr = R_des @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rerr.flatten())
    rv = np.zeros(3)
    mujoco.mju_quat2Vel(rv, q, 1.0)
    return rv


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.kd = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.adr = np.array(robot.arm_qpos_adr)
        self.dof = np.array(robot.arm_dof_adr)
        self.lower = np.array(robot.arm_lower)
        self.upper = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.dt = robot.control_dt
        self.target = task["target"]
        self.dest = task["destination"]
        self.other = [c for c in ("red", "green", "blue") if c not in (self.target, self.dest)][0]

        self.q_cmd = self.home.copy()
        self.bias = np.zeros(7)
        self.closure = 0.0
        self.phase = "init"
        self.phase_t0 = 0.0
        self.t = 0.0
        # cartesian waypoint state
        self.wp_p = None      # current interpolated site position
        self.wp_yaw = 0.0
        self.goal_p = None
        self.goal_yaw = 0.0
        self.seg_p0 = None
        self.seg_yaw0 = 0.0
        self.seg_t0 = 0.0
        self.seg_T = 0.1
        self.filt = {}
        self.grasp_yaw = 0.0
        self.off_site = np.array([0.0, 0.0, -0.012])
        self.null_w = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.grasp_xoff = 0.015
        self.grasp_zoff = 0.004
        self.lift_speed = 0.12
        self.squeeze_hold = 0.3
        self.off_samples = []
        self.attempts = 0
        self.place_attempts = 0
        self.done = False
        self.reached_t = None
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.log = []

    # ------------------------------------------------------------------ kinematics
    def fk(self, q):
        self.kd.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.kd)
        mujoco.mj_comPos(self.m, self.kd)
        return self.kd.site_xpos[self.sid].copy(), self.kd.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, p_des, R_des, iters=12, k_null=0.2):
        q = q0.copy()
        I6 = np.eye(6)
        for _ in range(iters):
            p, R = self.fk(q)
            ep = p_des - p
            rv = rot_err(R_des, R)
            mujoco.mj_jacSite(self.m, self.kd, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dof], 0.5 * self.jr[:, self.dof]])
            e = np.concatenate([ep, 0.5 * rv])
            Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * I6, I6)
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (k_null * self.null_w * (self.home - q))
            dq = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + dq, self.lower, self.upper)
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(rv) < 2e-3 and np.linalg.norm(dq) < 1e-3:
                break
        return q

    def collides(self, q):
        self.kd.qpos[self.adr] = q
        mujoco.mj_forward(self.m, self.kd)
        d = self.kd
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = self.m.body(self.m.geom_bodyid[c.geom1]).name
            b2 = self.m.body(self.m.geom_bodyid[c.geom2]).name
            if "block" in b1 or "block" in b2 or "table" in (b1, b2) or "bin" in b1 or "bin" in b2:
                continue
            return True
        return False

    def ik_feasible(self, p, yaw, q0=None):
        q0 = self.home if q0 is None else q0
        q = self.ik(q0, p, R_topdown(yaw), iters=60)
        pp, RR = self.fk(q)
        ok = np.linalg.norm(pp - p) < 0.004 and np.linalg.norm(rot_err(R_topdown(yaw), RR)) < 0.05
        if not ok:
            return False
        return not self.collides(q)

    def pick_yaw(self, points, cube_yaw, prefer=None):
        """Hand yaw aligned with a cube face, feasible (reachable, collision free) at all points."""
        base = wrap_90(cube_yaw)
        cands = []
        for k in (0, 1, -1):
            y = base + k * np.pi / 2
            if -np.deg2rad(50) <= y <= np.deg2rad(95):
                cands.append(y)
        if prefer is not None:
            cands.sort(key=lambda y: abs(wrap_pi(y - prefer)))
        cands += [np.deg2rad(15), np.deg2rad(30), 0.0, np.deg2rad(45)]
        for y in cands:
            if all(self.ik_feasible(p, y) for p in points):
                return y
        return np.deg2rad(15)

    # ------------------------------------------------------------------ motion
    def set_goal(self, p, yaw, speed=0.25, yaw_speed=1.5):
        p = np.array(p, dtype=float)
        if self.wp_p is None:
            self.wp_p = p.copy()
            self.wp_yaw = yaw
        self.seg_p0 = self.wp_p.copy()
        self.seg_yaw0 = self.wp_yaw
        self.goal_p = p
        self.goal_yaw = yaw
        dist = np.linalg.norm(p - self.seg_p0)
        dyaw = abs(wrap_pi(yaw - self.seg_yaw0))
        self.seg_T = max(dist / speed, dyaw / yaw_speed, 0.12)
        self.seg_t0 = self.t
        self.reached_t = None

    def update_wp(self):
        s = min(1.0, (self.t - self.seg_t0) / self.seg_T)
        s = 0.5 - 0.5 * np.cos(np.pi * s)
        self.wp_p = self.seg_p0 + s * (self.goal_p - self.seg_p0)
        self.wp_yaw = self.seg_yaw0 + s * wrap_pi(self.goal_yaw - self.seg_yaw0)
        return s >= 1.0

    def goto(self, phase, p, yaw, **kw):
        self.phase = phase
        self.phase_t0 = self.t
        self.set_goal(p, yaw, **kw)

    def at_goal(self, obs, tol=0.006, patience=1.0):
        done = (self.t - self.seg_t0) >= self.seg_T
        if not done:
            return False
        err = np.linalg.norm(obs["grasp_site_pos"] - self.goal_p)
        if err < tol:
            return True
        return (self.t - self.seg_t0 - self.seg_T) > patience

    # ------------------------------------------------------------------ helpers
    def bpos(self, obs, c):
        return self.filt.get(c, obs["blocks"][c]["pos"])

    def update_filters(self, obs):
        for c, b in obs["blocks"].items():
            p = np.array(b["pos"], dtype=float)
            if c not in self.filt:
                self.filt[c] = p.copy()
            else:
                v = np.linalg.norm(b["linvel"])
                a = 0.5 if v > 0.03 else 0.25
                self.filt[c] = self.filt[c] + a * (p - self.filt[c])

    def stacked(self, obs):
        pt = np.array(obs["blocks"][self.target]["pos"])
        pd = np.array(obs["blocks"][self.dest]["pos"])
        dxy = np.linalg.norm(pt[:2] - pd[:2])
        dz = pt[2] - pd[2]
        return dxy < 0.017 and abs(dz - 2 * self.half) < 0.009

    # ------------------------------------------------------------------ main
    def act(self, obs):
        self.t = obs["t"]
        self.update_filters(obs)
        site = obs["grasp_site_pos"]
        try:
            self.step_fsm(obs)
        except Exception as e:  # never crash
            print("fsm error", repr(e))
        # waypoint interpolation
        if self.wp_p is None:
            self.wp_p = site.copy()
            self.wp_yaw = 0.0
            self.goal_p = site.copy()
            self.goal_yaw = 0.0
            self.seg_p0 = site.copy()
            self.seg_yaw0 = 0.0
            self.seg_t0 = self.t
            self.seg_T = 0.1
        finished = self.update_wp()
        R_des = R_topdown(self.wp_yaw)
        q_ik = self.ik(self.q_cmd - self.bias, self.wp_p, R_des)
        # sag compensation: integrate position error once the waypoint has arrived
        if finished:
            self.kd.qpos[self.adr] = obs["qpos"][:7]
            mujoco.mj_kinematics(self.m, self.kd)
            mujoco.mj_comPos(self.m, self.kd)
            p_meas = self.kd.site_xpos[self.sid].copy()
            mujoco.mj_jacSite(self.m, self.kd, self.jp, self.jr, self.sid)
            J = self.jp[:, self.dof]
            err = self.wp_p - p_meas
            if np.linalg.norm(err) < 0.05:
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
                self.bias = np.clip(self.bias + 0.15 * dq, -0.25, 0.25)
        q_cmd = np.clip(q_ik + self.bias, self.lower, self.upper)
        self.q_cmd = q_cmd
        return np.concatenate([q_cmd, [self.closure]])

    def step_fsm(self, obs):
        t = self.t
        site = obs["grasp_site_pos"]
        ph = self.phase
        tgt = self.bpos(obs, self.target)
        dst = self.bpos(obs, self.dest)

        if ph == "init":
            if t > 0.1:
                self.start_pick(obs)
        elif ph == "approach":
            if self.at_goal(obs, tol=0.01):
                p = self.bpos(obs, self.target) + np.array([0, 0, self.grasp_zoff]) - R_topdown(self.grasp_yaw) @ np.array([self.grasp_xoff, 0, 0])
                self.goto("descend", p, self.grasp_yaw, speed=0.15)
        elif ph == "descend":
            if self.at_goal(obs, tol=0.004, patience=0.8):
                print(f"t={t:.2f} descend done, site err={np.round(site - self.goal_p, 3)}")
                self.phase = "close"
                self.phase_t0 = t
        elif ph == "close":
            dt = t - self.phase_t0
            if dt < 0.5:
                self.closure = 0.5 * dt / 0.5
            elif dt < 0.7:
                self.closure = 0.5
            elif dt < 1.2:
                self.closure = 0.5 + 0.5 * (dt - 0.7) / 0.5
            elif dt < 1.2 + self.squeeze_hold:
                self.closure = 1.0
            else:
                self.closure = 1.0
                self.off_samples = []
                self.goto("lift", self.wp_p + np.array([0, 0, 0.10]), self.grasp_yaw, speed=self.lift_speed)
        elif ph == "lift":
            R = obs["grasp_site_mat"]
            self.off_samples.append(R.T @ (np.array(obs["blocks"][self.target]["pos"]) - site))
            if self.at_goal(obs, tol=0.01, patience=0.5):
                pt = np.array(obs["blocks"][self.target]["pos"])
                if pt[2] > self.table + self.half + 0.04:
                    self.off_site = np.mean(self.off_samples[-5:], axis=0)
                    print(f"t={t:.2f} lifted, off_site={np.round(self.off_site,3)}")
                    self.start_transport(obs)
                else:
                    print(f"t={t:.2f} grasp failed, retry")
                    self.closure = 0.0
                    self.attempts += 1
                    self.goto("reopen", self.wp_p, self.grasp_yaw)
        elif ph == "reopen":
            if t - self.phase_t0 > 0.5:
                self.start_pick(obs)
        elif ph == "transport":
            if self.at_goal(obs, tol=0.008, patience=0.8):
                R = R_topdown(self.place_yaw)
                dz = self.bpos(obs, self.dest)[2] + 2 * self.half + 0.006
                p_cube = np.array([dst[0], dst[1], dz])
                self.goto("lower", p_cube - R @ self.off_site, self.place_yaw, speed=0.08)
        elif ph == "lower":
            if self.at_goal(obs, tol=0.004, patience=0.8):
                self.phase = "release"
                self.phase_t0 = t
                self.closure = 0.0
        elif ph == "release":
            if t - self.phase_t0 > 0.4:
                self.goto("retreat", self.wp_p + np.array([0, 0, 0.09]), self.place_yaw, speed=0.12)
        elif ph == "retreat":
            if self.at_goal(obs, tol=0.01, patience=0.5):
                self.phase = "check"
                self.phase_t0 = t
        elif ph == "check":
            if t - self.phase_t0 > 0.4:
                if self.stacked(obs):
                    print(f"t={t:.2f} stacked ok")
                    self.phase = "done"
                else:
                    print(f"t={t:.2f} stack failed, retry")
                    self.place_attempts += 1
                    self.start_pick(obs)
        elif ph == "done":
            pass

    def start_pick(self, obs):
        tgt = np.array(obs["blocks"][self.target]["pos"], dtype=float)
        self.filt[self.target] = tgt.copy()
        yaw_c = obs["blocks"][self.target]["yaw"]
        above = tgt + np.array([0, 0, 0.10])
        low = tgt + np.array([0, 0, self.grasp_zoff])
        self.grasp_yaw = self.pick_yaw([above, low], yaw_c)
        above = above - R_topdown(self.grasp_yaw) @ np.array([self.grasp_xoff, 0, 0])
        self.closure = 0.0
        self.goto("approach", above, self.grasp_yaw, speed=0.3)
        print(f"t={self.t:.2f} pick {self.target} at {np.round(tgt,3)} yaw={np.degrees(self.grasp_yaw):.0f}")

    def start_transport(self, obs):
        dst = self.bpos(obs, self.dest)
        yaw_d = obs["blocks"][self.dest]["yaw"]
        above = np.array([dst[0], dst[1], dst[2] + 2 * self.half + 0.04])
        low = np.array([dst[0], dst[1], dst[2] + 2 * self.half + 0.006])
        yaw = self.pick_yaw([above, low], yaw_d, prefer=self.grasp_yaw)
        self.place_yaw = yaw
        p = above - R_topdown(yaw) @ self.off_site
        self.goto("transport", p, yaw, speed=0.25)
        print(f"t={self.t:.2f} transport to {np.round(dst,3)} yaw={np.degrees(yaw):.0f}")
