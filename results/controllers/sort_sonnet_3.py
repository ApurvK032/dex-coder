import numpy as np, mujoco


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.d = mujoco.MjData(self.m)
        self.base = self.d.qpos.copy()
        self.table_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.tgt = task["target"]
        self.dest = robot.bin_xy[task["destination"]]
        self.q_home = np.array(robot.home_arm)
        self.phase = -1
        self.t_phase = 0.0
        self.closure = 0.0
        self.path = None
        self.hover = 0.15
        self.grasp_dz = 0.012
        self.carry_z = 1.0
        self.release_z = 0.90
        self.vmax = 0.06
        self.vcarry = 0.04
        self.R_home = None

    # ---------- kinematics ----------
    def fk(self, q):
        d = self.d
        d.qpos[:] = self.base
        d.qpos[self.robot.arm_qpos_adr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.site_xpos[self.sid].copy(), d.site_xmat[self.sid].reshape(3, 3).copy()

    def _geoms(self, names):
        out = []
        for n in names:
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            out += [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b
                    and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        return out

    def clearance(self, q):
        """min distance between torso/pelvis geoms and the arm links (after fk)."""
        if not hasattr(self, "_tg"):
            self._tg = self._geoms(["torso_link", "waist_yaw_link", "waist_roll_link", "pelvis"])
            self._ag = self._geoms(["right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
                                    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"])
        self.fk(q)
        fromto = np.zeros(6)
        best = 1.0
        for a in self._ag:
            for t in self._tg:
                dist = mujoco.mj_geomDistance(self.m, self.d, a, t, 0.1, fromto)
                if dist < best:
                    best = dist
        return best

    def solve(self, q, p_t, R_t, it=60):
        m, d = self.m, self.d
        dof = self.robot.arm_dof_adr
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        for _ in range(it):
            p, R = self.fk(q)
            ep = p_t - p
            Re = R_t @ R.T
            v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * 0.5
            s = np.linalg.norm(v)
            er = v / s * np.arctan2(s, (np.trace(Re) - 1) / 2) if s > 1e-9 else np.zeros(3)
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, dof], jr[:, dof]])
            e = np.concatenate([ep, er])
            JJ = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
            dq += N @ (0.05 * (self.q_home - q))
            mx = np.abs(dq).max()
            if mx > 0.15:
                dq *= 0.15 / mx
            q = np.clip(q + dq, self.robot.arm_lower, self.robot.arm_upper)
        return q

    def plan(self, q0, wps, step=0.01, y0=0.0, check=True):
        """wps: list of (pos, yaw_offset). Cartesian polyline -> joint path."""
        err = 0.0
        ncol = 1.0
        p0, _ = self.fk(q0)
        Q = [q0.copy()]
        q = q0.copy()
        cur, cy = p0, y0
        k = 0
        for w, wy in wps:
            w = np.asarray(w, float)
            n = max(1, int(np.ceil(np.linalg.norm(w - cur) / step)))
            for i in range(1, n + 1):
                f = i / n
                pt = cur + (w - cur) * f
                R_t = rotz(cy + (wy - cy) * f) @ self.R_home
                q = self.solve(q, pt, R_t)
                Q.append(q.copy())
                k += 1
                if check and k % 3 == 0:
                    ncol = min(ncol, self.clearance(q))
            p, R = self.fk(q)
            err = max(err, np.linalg.norm(p - w))
            cur, cy = w, wy
        self.last_err = err
        self.last_ncol = ncol
        return np.array(Q)

    def set_path(self, Q, v):
        self.path = Q
        diffs = np.abs(np.diff(Q, axis=0)).max(axis=1)
        self.cum = np.concatenate([[0], np.cumsum(diffs)])
        self.s = 0.0
        self.v = v

    def follow(self):
        self.s = min(self.s + self.v, self.cum[-1])
        i = np.searchsorted(self.cum, self.s)
        if i == 0:
            return self.path[0].copy()
        a = (self.s - self.cum[i - 1]) / max(self.cum[i] - self.cum[i - 1], 1e-9)
        return self.path[i - 1] * (1 - a) + self.path[i] * a

    def path_done(self):
        return self.s >= self.cum[-1] - 1e-9

    def go(self, ph, t):
        self.phase = ph
        self.t_phase = t

    def full_wps(self, bp, a, c):
        return [(bp + [0, 0, self.hover], a), (bp + [0, 0, self.grasp_dz], a),
                ([bp[0], bp[1], self.carry_z], a),
                ([self.dest[0], self.dest[1], self.carry_z], c),
                ([self.dest[0], self.dest[1], self.release_z], c)]

    def choose(self, q_now, bp, yaw, h):
        a0 = (yaw - h) % (np.pi / 2)
        cands = []
        for k in (-2, -1, 0, 1, 2):
            a = a0 + k * np.pi / 2
            if -1.0 <= a <= 2.0:
                cands.append(a)
        cands = [(a, 0.0) for a in cands]
        for k in (-2, -1, 0, 1):
            for dl in (-0.25, 0.25):
                a = a0 + k * np.pi / 2 + dl
                if -1.0 <= a <= 2.0:
                    cands.append((a, abs(dl)))
        cands.sort(key=lambda x: abs(x[0] - 0.8) + 3.0 * x[1] + (1.0 if x[1] > 0 else 0.0))
        best = None
        for a, mis in cands:
            bp_ = np.asarray(bp, float)
            wps = self.full_wps(bp_, a, a)[:3]
            self.plan(q_now, wps, step=0.02)
            s1 = 10 * self.last_err + 30 * max(0.0, 0.02 - self.last_ncol)
            if s1 > 0.5:
                continue
            for c in (a, 0.8, 0.4, 0.0):
                self.plan(q_now, self.full_wps(bp_, a, c), step=0.02)
                score = (10 * self.last_err + 30 * max(0.0, 0.02 - self.last_ncol) + 0.02 * abs(a - 0.8)
                         + 0.01 * abs(c - a) + 0.3 * mis)
                if best is None or score < best[0]:
                    best = (score, a, c, self.last_err, self.last_ncol)
                if self.last_err < 0.003 and self.last_ncol >= 0.015:
                    return (score, a, c, self.last_err, self.last_ncol)
        if best is None:
            best = (0, cands[0][0], cands[0][0], 0, 0)
        return best

    def act(self, obs):
        r = self.robot
        t = obs["t"]
        q_now = np.array(obs["arm_target"])
        if self.phase == -1:
            self.R_home = obs["grasp_site_mat"].copy()
            h = np.arctan2(self.R_home[1, 0], self.R_home[0, 0])
            b = obs["blocks"][self.tgt]
            self.bp = np.array(b["pos"])
            best = self.choose(q_now, self.bp, b["yaw"], h)
            self.a, self.c = best[1], best[2]
            Q = self.plan(q_now, [(self.bp + [0, 0, self.hover], self.a)], step=0.02, check=False)
            self.set_path(Q, self.vmax)
            self.go(0, t)
        ph = self.phase
        dt = t - self.t_phase
        q = q_now
        if ph == 0:
            q = self.follow()
            if self.path_done() and dt > 0.3:
                b = obs["blocks"][self.tgt]
                self.bp = np.array(b["pos"]) + [getattr(self, "dx_off", 0.0), 0, 0]
                Qd = self.plan(q_now, [(self.bp + [0, 0, self.hover], self.a), (self.bp + [0, 0, self.grasp_dz], self.a)],
                               step=0.01, y0=self.a, check=False)
                self.set_path(Qd, 0.04)
                self.go(1, t)
        elif ph == 1:
            q = self.follow()
            if self.path_done() and dt > 0.3:
                self.go(2, t)
        elif ph == 2:
            self.closure = min(1.0, self.closure + 0.04)
            if self.closure >= 1.0 and dt > 1.5:
                p = obs["grasp_site_pos"]
                wps = [([p[0], p[1], self.carry_z], self.a),
                       ([self.dest[0], self.dest[1], self.carry_z], self.c),
                       ([self.dest[0], self.dest[1], self.release_z], self.c)]
                Q = self.plan(q_now, wps, step=0.01, y0=self.a, check=False)
                self.set_path(Q, self.vcarry)
                self.go(3, t)
        elif ph == 3:
            q = self.follow()
            bpos = np.array(obs["blocks"][self.tgt]["pos"])
            ps = np.array(obs["grasp_site_pos"])
            if (getattr(self, "retries", 0) < 2 and ps[2] > 0.92 and bpos[2] < 0.85
                    and np.linalg.norm(bpos - ps) > 0.05):
                self.retries = getattr(self, "retries", 0) + 1
                self.go(7, t)
            elif self.path_done() and dt > 0.3:
                self.go(4, t)
        elif ph == 7:
            self.closure = max(0.0, self.closure - 0.08)
            if self.closure <= 0 and dt > 0.4:
                if self.retries <= 1:
                    self.dx_off, self.grasp_dz = 0.0, 0.0
                else:
                    self.dx_off, self.grasp_dz = -0.03, 0.012
                self.bp = np.array(obs["blocks"][self.tgt]["pos"]) + [self.dx_off, 0, 0]
                Q = self.plan(q_now, [(self.bp + [0, 0, self.hover], self.a)], step=0.02, y0=self.a, check=False)
                self.set_path(Q, 0.05)
                self.go(0, t)
        elif ph == 4:
            self.closure = max(0.0, self.closure - 0.08)
            if self.closure <= 0 and dt > 0.6:
                p = obs["grasp_site_pos"]
                Q = self.plan(q_now, [([p[0], p[1], self.carry_z + 0.05], self.c)], step=0.01, y0=self.c, check=False)
                self.set_path(Q, 0.05)
                self.go(5, t)
        else:
            q = self.follow()
        q = np.clip(q, q_now - 0.08, q_now + 0.08)
        q = np.clip(q, r.arm_lower, r.arm_upper)
        return list(q) + [self.closure]
