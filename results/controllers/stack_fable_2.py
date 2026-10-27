
import numpy as np
import mujoco


def _rotvec_from_mat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def target_rot(psi):
    """Site rotation: x along fingers (horizontal, yaw psi), y out of palm (down)."""
    ex = np.array([np.cos(psi), np.sin(psi), 0.0])
    ey = np.array([0.0, 0.0, -1.0])
    ez = np.cross(ex, ey)
    return np.column_stack([ex, ey, ez])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.site = self.m.site(robot.grasp_site).id
        self.arm_qadr = np.array(robot.arm_qpos_adr)
        self.arm_dadr = np.array(robot.arm_dof_adr)
        self.lower = np.array(robot.arm_lower, dtype=float)
        self.upper = np.array(robot.arm_upper, dtype=float)
        # tightened limits for IK (avoid arm crossing into the torso)
        self.ik_lo = self.lower + 0.05
        self.ik_up = self.upper - 0.05
        self.ik_up[1] = min(self.ik_up[1], -0.05)
        self.q_rest = np.array([0.3, -0.3, -0.3, 0.8, -0.5, 0.0, 0.0])
        self.home = np.array(robot.home_arm, dtype=float)
        self.h = robot.block_half_size
        self.table = robot.table_height
        self.target = task["target"]
        self.dest = task["destination"]
        self.phase = "pre"
        self.phase_t0 = 0.0
        self.q_cmd = self.home.copy()
        self.closure = 0.0
        self.psi = None
        self.bad_psi = []
        self.attempts = 0
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.z_transit = self.table + 0.12
        self.carrot = None
        self.cmd_err = 1.0
        self.cmd_err_ik = 0.0
        self.lower_tries = 0
        self.hold_offset = np.zeros(3)
        self.stall_t = None
        # body classification for collision checks
        self.arm_bodies = set()
        self.block_bodies = {}
        for i in range(self.m.nbody):
            n = self.m.body(i).name
            if n.startswith("right_") and ("shoulder" in n or "elbow" in n or "wrist" in n or "hand" in n):
                self.arm_bodies.add(i)
            if n.startswith("block_"):
                self.block_bodies[i] = n[6:]
        self.step_max = 0.014  # m per control step for the cartesian carrot (0.35 m/s)

    # ---------------- kinematics ----------------
    def fk(self, kin, q):
        kin.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.m, kin)
        mujoco.mj_comPos(self.m, kin)
        p = kin.site_xpos[self.site].copy()
        R = kin.site_xmat[self.site].reshape(3, 3).copy()
        return p, R

    def ik(self, kin, q0, p_t, R_t, iters=30, w_rot=0.3, k_null=0.3):
        q = np.clip(q0.copy(), self.ik_lo, self.ik_up)
        I7 = np.eye(7)
        for it in range(iters):
            p, R = self.fk(kin, q)
            e_p = p_t - p
            e_r = _rotvec_from_mat(R_t @ R.T)
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, kin, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.arm_dadr], w_rot * self.jacr[:, self.arm_dadr]])
            e = np.concatenate([e_p, w_rot * e_r])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e
            if it < iters // 2:
                dq += (I7 - Jp @ J) @ (k_null * (self.q_rest - q))
            n = np.linalg.norm(dq)
            if n > 0.5:
                dq *= 0.5 / n
            q = np.clip(q + dq, self.ik_lo, self.ik_up)
        p, R = self.fk(kin, q)
        return q, np.linalg.norm(p_t - p), np.linalg.norm(_rotvec_from_mat(R_t @ R.T))

    def ik_multi(self, kin, p_t, R_t, iters=30):
        best = None
        for q0 in (self.q_cmd, self.home, self.q_rest):
            q, ep, er = self.ik(kin, q0, p_t, R_t, iters=iters)
            cost = ep + 0.05 * er
            if best is None or cost < best[0]:
                best = (cost, q, ep, er)
            if ep < 0.003 and er < 0.03:
                break
        return best[1], best[2], best[3]

    def ik_track(self, kin, p_t, R_t, max_dq=0.07):
        """Continuous IK from the current command; joint step limited to stay on the same branch."""
        q, ep, er = self.ik(kin, self.q_cmd, p_t, R_t, iters=20, k_null=0.1)
        dq = q - self.q_cmd
        n = np.max(np.abs(dq))
        if n > max_dq:
            q = self.q_cmd + dq * (max_dq / n)
            p, R = self.fk(kin, q)
            ep = np.linalg.norm(p_t - p)
            er = np.linalg.norm(_rotvec_from_mat(R_t @ R.T))
        return q, ep, er

    def collisions(self, kin, q, ignore_block=None):
        """Number of penetrating contacts between the arm/hand and the rest (kin model)."""
        kin.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.m, kin)
        mujoco.mj_collision(self.m, kin)
        n = 0
        for i in range(kin.ncon):
            con = kin.contact[i]
            if con.dist > -0.001:
                continue
            b1 = self.m.geom_bodyid[con.geom1]
            b2 = self.m.geom_bodyid[con.geom2]
            a1, a2 = b1 in self.arm_bodies, b2 in self.arm_bodies
            if a1 == a2:
                continue
            other = b2 if a1 else b1
            if other in self.block_bodies and self.block_bodies[other] == ignore_block:
                continue
            n += 1
        return n

    # ---------------- helpers ----------------
    def pref_psi(self, y):
        # near the body (y close to 0) point the fingers more to the left
        return float(np.clip(0.3 + 3.0 * (y + 0.15), -0.5, 0.3))

    def pose_cost(self, kin, psi, p, block):
        R_t = target_rot(psi)
        q, ep, er = self.ik_multi(kin, p, R_t, iters=60)
        ncol = self.collisions(kin, q, ignore_block=block)
        return 100.0 * ep + 5.0 * er + 1.0 * ncol

    def select_psi(self, obs, yaw, p_grasp, block, p_place, dest):
        """Hand yaw for the grasp: feasible at the grasp pose and (preferably) at the place pose."""
        kin = obs["kin"]
        pref = self.pref_psi(p_grasp[1])
        best = None
        for k in range(4):
            psi = wrap(yaw + k * np.pi / 2)
            cost = self.pose_cost(kin, psi, p_grasp, block)
            cost += 0.5 * self.pose_cost(kin, psi, p_place, dest)
            cost += abs(wrap(psi - pref)) / np.pi
            for b in self.bad_psi:
                if abs(wrap(psi - b)) < 0.3:
                    cost += 2.0
            if best is None or cost < best[0]:
                best = (cost, psi)
        return best[1]

    def select_place_psi(self, obs, psi0, p_place, dest):
        """Hand yaw for placing: near the grasp yaw, but feasible at the place pose."""
        kin = obs["kin"]
        best = None
        for k in range(-2, 3):
            psi = wrap(psi0 + k * np.pi / 4)
            cost = self.pose_cost(kin, psi, p_place, dest) + 0.15 * abs(k)
            if best is None or cost < best[0]:
                best = (cost, psi)
        return best[1]

    def choose_psi(self, yaw, pref):
        best = None
        for k in range(-3, 4):
            c = yaw + k * np.pi / 2
            if best is None or abs(c - pref) < abs(best - pref):
                best = c
        return best

    def set_phase(self, name, t):
        self.phase = name
        self.phase_t0 = t
        self.stall_t = None

    def go(self, obs, p_goal, R_t, step=None):
        kin = obs["kin"]
        if self.carrot is None:
            self.carrot = np.array(obs["grasp_site_pos"], dtype=float)
        d = p_goal - self.carrot
        n = np.linalg.norm(d)
        smax = self.step_max if step is None else step
        sp = np.array(obs["grasp_site_pos"], dtype=float)
        lag = np.linalg.norm(sp - self.carrot)
        if self.cmd_err_ik > 0.015 or lag > 0.03:
            smax *= 0.2  # IK or the arm lagging: slow the carrot down so the arm catches up
        elif lag > 0.015:
            smax *= 0.5
        if n > smax:
            d *= smax / n
        self.carrot = self.carrot + d
        q, ep, er = self.ik_track(kin, self.carrot, R_t)
        self.q_cmd = q
        self.cmd_err_ik = ep
        self.cmd_err = np.linalg.norm(self.carrot - p_goal) + ep
        self.goal = p_goal
        return ep, er

    def reached(self, obs, p_t, tol):
        sp = np.array(obs["grasp_site_pos"])
        return self.cmd_err < 0.004 and np.linalg.norm(sp - p_t) < tol

    def stalled(self, obs, t):
        """Commanded pose reached but measured site far away for a while."""
        sp = np.array(obs["grasp_site_pos"])
        if self.cmd_err < 0.004 and np.linalg.norm(sp - self.goal) > 0.02:
            if self.stall_t is None:
                self.stall_t = t
            return t - self.stall_t > 1.0
        self.stall_t = None
        return False

    def restart(self, t):
        self.attempts += 1
        self.psi = None
        self.set_phase("pre", t)

    # ---------------- main ----------------
    def act(self, obs):
        t = obs["t"]
        blocks = obs["blocks"]
        tb = blocks[self.target]
        db = blocks[self.dest]
        tp = np.array(tb["pos"], dtype=float)
        dp = np.array(db["pos"], dtype=float)
        sp = np.array(obs["grasp_site_pos"])
        h = self.h
        dt = t - self.phase_t0

        if self.phase == "pre":
            if self.psi is None:
                p_place = np.array([dp[0], dp[1], dp[2] + 2 * h + 0.012])
                self.psi = self.select_psi(obs, tb["yaw"], tp.copy(), self.target, p_place, self.dest)
            else:
                self.psi = self.choose_psi(tb["yaw"], self.psi)
            p_t = np.array([tp[0], tp[1], self.z_transit])
            R_t = target_rot(self.psi)
            self.go(obs, p_t, R_t)
            self.closure = 0.0
            if (self.reached(obs, p_t, 0.012) and dt > 0.3) or dt > 5.0:
                self.set_phase("descend", t)
            elif self.stalled(obs, t):
                self.bad_psi.append(self.psi)
                self.restart(t)
        elif self.phase == "descend":
            self.psi = self.choose_psi(tb["yaw"], self.psi)
            p_t = np.array([tp[0], tp[1], tp[2]])
            R_t = target_rot(self.psi)
            self.go(obs, p_t, R_t)
            self.closure = 0.0
            if self.reached(obs, p_t, 0.006) or dt > 3.0:
                self.grasp_pos = p_t.copy()
                self.grasp_R = R_t.copy()
                self.set_phase("grasp", t)
            elif self.stalled(obs, t):
                self.bad_psi.append(self.psi)
                self.restart(t)
        elif self.phase == "grasp":
            self.go(obs, self.grasp_pos, self.grasp_R)
            self.closure = min(1.0, dt / 0.6)
            if dt > 1.0:
                self.set_phase("lift", t)
        elif self.phase == "lift":
            p_t = np.array([self.grasp_pos[0], self.grasp_pos[1], self.z_transit])
            self.go(obs, p_t, self.grasp_R)
            self.closure = 1.0
            if (dt > 0.8 and self.reached(obs, p_t, 0.02)) or dt > 3.0:
                if tp[2] > self.table + h + 0.05:
                    p_place = np.array([dp[0], dp[1], dp[2] + 2 * h + 0.012])
                    psi_p = self.select_place_psi(obs, self.psi, p_place, self.dest)
                    self.place_R = target_rot(psi_p)
                    self.hold_offset = tp - sp
                    self.set_phase("move", t)
                else:
                    self.restart(t)
        elif self.phase == "move":
            p_t = np.array([dp[0], dp[1], self.z_transit]) - self.hold_offset * [1, 1, 0]
            self.go(obs, p_t, self.place_R)
            self.closure = 1.0
            if tp[2] < self.table + h + 0.03 and dt > 0.5:
                self.restart(t)
            elif (self.reached(obs, p_t, 0.012) and dt > 0.3) or dt > 6.0:
                self.hold_offset = tp - sp
                self.set_phase("lower", t)
        elif self.phase == "lower":
            p_t = np.array([dp[0], dp[1], dp[2] + 2 * h + 0.012]) - self.hold_offset * [1, 1, 0]
            self.go(obs, p_t, self.place_R, step=0.012)
            self.closure = 1.0
            if self.reached(obs, p_t, 0.006) or dt > 3.0 and self.lower_tries >= 2 or dt > 6.0:
                self.place_pos = p_t.copy()
                self.set_phase("release", t)
            elif dt > 3.0:
                # stuck (held cube pressing on something): back up and try the descent again
                self.lower_tries += 1
                self.set_phase("move", t)
        elif self.phase == "release":
            self.go(obs, self.place_pos, self.place_R)
            self.closure = max(0.0, 1.0 - dt / 0.5)
            if dt > 0.8:
                self.set_phase("retreat", t)
        elif self.phase == "retreat":
            p_t = np.array([self.place_pos[0], self.place_pos[1], self.z_transit + 0.02])
            self.go(obs, p_t, self.place_R)
            self.closure = 0.0
            if dt > 1.5 and (self.reached(obs, p_t, 0.02) or dt > 4.0):
                dxy = np.linalg.norm(tp[:2] - dp[:2])
                dz = tp[2] - dp[2]
                ok = dxy < 0.015 and abs(dz - 2 * h) < 0.008
                if not ok and self.attempts < 4 and t < 19.0:
                    self.restart(t)
                else:
                    self.set_phase("done", t)
        else:
            self.closure = 0.0

        return np.concatenate([self.q_cmd, [self.closure]])
