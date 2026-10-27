import numpy as np, mujoco


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
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest])
        self.phase = "approach"
        self.t_phase = 0.0
        self.cpos = None
        self.psi = None
        self.gpsi = None
        self.th = 0.0
        self.gth = 0.0
        self.closure = 0.0
        self.q = np.array(robot.home_arm, float)
        self.hover_z = 0.95
        self.grab_z = 0.808
        self.off = 0.012
        self.off_desc = -0.012
        self.tilt = 0.2
        self.qref = None
        self.qgoal = None
        self.qref_grasp = None
        self.qref_carry = None

    # ---------- kinematics
    def Rdes(self, psi, th):
        x0 = np.array([np.cos(psi), np.sin(psi), 0]); y0 = np.array([0, 0, -1.])
        x = x0 * np.cos(th) - y0 * np.sin(th)
        y = y0 * np.cos(th) + x0 * np.sin(th)
        z = np.cross(x, y)
        return np.stack([x, y, z], 1)

    def ik(self, pos, psi, th, q0, iters=6, qref=None):
        q = q0.copy(); R = self.Rdes(psi, th); m = self.m; d = self.d
        for _ in range(iters):
            d.qpos[self.adr] = q; mujoco.mj_forward(m, d)
            e = np.zeros(6); e[:3] = pos - d.site_xpos[self.sid]
            Rc = d.site_xmat[self.sid].reshape(3, 3)
            Re = R @ Rc.T
            e[3:] = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dof], jr[:, self.dof]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e
            if qref is not None:
                dq = dq + (np.eye(7) - Jp @ J) @ (0.3 * (qref - q))
            q = np.clip(q + 0.8 * dq, self.lo, self.hi)
        return q

    def ik_ok(self, pos, psi, th, q0, iters=60):
        q = self.ik(pos, psi, th, q0, iters)
        self.d.qpos[self.adr] = q; mujoco.mj_forward(self.m, self.d)
        e = np.linalg.norm(pos - self.d.site_xpos[self.sid])
        Rc = self.d.site_xmat[self.sid].reshape(3, 3)
        eo = np.linalg.norm(self.Rdes(psi, th) - Rc)
        ok = e < 2e-3 and eo < 2e-2
        return ok, q

    def goal_xy(self, bpos, psi, off=None):
        if off is None:
            off = self.off
        return np.array([bpos[0], bpos[1]]) + off * np.array([np.cos(psi), np.sin(psi)])

    def feasible(self, xy, psi, th):
        q0 = np.array(self.robot.home_arm, float)
        ok1, q1 = self.ik_ok(np.r_[xy, self.grab_z], psi, th, q0)
        if not ok1:
            return False, None
        ok2, q2 = self.ik_ok(np.r_[xy, self.hover_z], psi, th, q1)
        return ok2, q1

    def hazard(self, xy, psi):
        tip = xy + 0.11 * np.array([np.cos(psi), np.sin(psi)])
        for name in ("left", "right"):
            bx, by = self.robot.bin_xy[name]
            if tip[0] > bx - 0.061 - 0.012 and abs(tip[1] - by) < 0.061 + 0.04:
                return True
        return False

    def pick_psi(self, yaw, bpos):
        base = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        cands = [base + k * np.pi / 2 for k in (-1, 0, 1)]
        cands = [base + k * np.pi / 2 for k in (-2, -1, 0, 1, 2)]
        cands = [p for p in cands if abs(p) < np.deg2rad(115)]
        cands.sort(key=lambda p: abs(p - 0.3))
        # fallback: slightly misaligned angles
        extra = []
        for mis in (0.1, 0.2, 0.3):
            for p in cands:
                extra += [p + mis, p - mis]
        cands = cands + extra
        nal = len([1 for k in (-2, -1, 0, 1, 2) if abs(base + k * np.pi / 2) < np.deg2rad(115)])
        for p in cands[:nal]:
            xy = self.goal_xy(bpos, p)
            if not self.hazard(xy, p):
                ok, qg = self.feasible(xy, p, 0.0)
                if ok:
                    return float(p), 0.0, qg
        for p in cands:
            xy = self.goal_xy(bpos, p)
            ok, qg = self.feasible(xy, p, 0.0)
            if ok:
                gth = 0.0
                if self.hazard(xy, p):
                    for th in (self.tilt, 0.15):
                        ok2, qg2 = self.feasible(xy, p, th)
                        if ok2:
                            gth = th
                            qg = qg2
                            break
                return float(p), gth, qg
        return float(cands[0]), 0.0, None

    def move(self, goal, psi_goal, th_goal, speed, psi_speed=1.0, dt=0.04):
        diff = goal - self.cpos
        dist = np.linalg.norm(diff)
        step = speed * dt
        if dist <= step:
            self.cpos = goal.copy()
        else:
            self.cpos = self.cpos + diff / dist * step
        dp = psi_goal - self.psi
        ps = psi_speed * dt
        self.psi = psi_goal if abs(dp) <= ps else self.psi + np.sign(dp) * ps
        dt_ = th_goal - self.th
        ts = 0.6 * dt
        self.th = th_goal if abs(dt_) <= ts else self.th + np.sign(dt_) * ts
        return dist <= step and abs(dp) <= ps and abs(dt_) <= ts

    def act(self, obs):
        dt = 0.04
        if self.cpos is None:
            self.cpos = np.array(obs["grasp_site_pos"], float)
            Rc = np.array(obs["grasp_site_mat"])
            self.psi = float(np.arctan2(Rc[1, 0], Rc[0, 0]))
            self.q = np.array(obs["arm_target"], float)
            b = obs["blocks"][self.target]
            self.gpsi, self.gth, self.qref_grasp = self.pick_psi(b["yaw"], np.array(b["pos"]))
            self.qref = self.qref_grasp
            # carry reference posture
            q0 = self.qref_grasp if self.qref_grasp is not None else self.q
            ok, qc = self.ik_ok(np.r_[self.bin, 0.93], 0.0, 0.0, q0)
            self.qref_carry = None
            self.cpsi = 0.0
            best = None
            q0h = np.array(self.robot.home_arm, float)
            for deg in range(-100, 101, 10):
                p = np.deg2rad(deg)
                tgt = 0.0 if abs(self.gpsi) < np.deg2rad(50) else 0.5 * self.gpsi
                if best is not None and abs(p - tgt) >= abs(best - tgt):
                    continue
                ok1, qa = self.ik_ok(np.r_[self.bin, 0.93], p, 0.0, q0h)
                if not ok1:
                    continue
                ok2, qb = self.ik_ok(np.r_[self.bin, 0.90], p, 0.0, qa)
                if ok2:
                    best = p
                    self.qgoal = qa
            if best is not None:
                self.cpsi = float(best)
        blk = obs["blocks"][self.target]
        bpos = np.array(blk["pos"])
        site = np.array(obs["grasp_site_pos"])
        self.t_phase += dt
        ph = self.phase
        gxy = self.goal_xy(bpos, self.gpsi)
        gz = self.grab_z + (0.012 if self.gth > 0 else 0.0)

        if ph == "approach":
            goal = np.array([gxy[0], gxy[1], self.hover_z])
            self.closure = 0.0
            done = self.move(goal, self.gpsi, self.gth, 0.35, 1.5)
            if done and np.linalg.norm(site[:2] - gxy) < 0.012 and self.t_phase > 0.5:
                self.set_phase("descend")
        elif ph == "descend":
            gxy0 = self.goal_xy(bpos, self.gpsi, self.off_desc)
            goal = np.array([gxy0[0], gxy0[1], gz])
            self.closure = 0.0
            done = self.move(goal, self.gpsi, self.gth, 0.12)
            if (done and abs(site[2] - gz) < 0.006) or self.t_phase > 5.0:
                self.set_phase("slide")
        elif ph == "slide":
            goal = np.array([gxy[0], gxy[1], gz])
            self.closure = 0.0
            done = self.move(goal, self.gpsi, self.gth, 0.05)
            if (done and self.t_phase > 0.3) or self.t_phase > 3.0:
                self.set_phase("close")
        elif ph == "close":
            self.closure = min(1.0, self.closure + dt / 0.8)
            if self.closure >= 1.0 and self.t_phase > 1.6:
                self.set_phase("lift")
        elif ph == "lift":
            goal = np.array([self.cpos[0], self.cpos[1], self.hover_z])
            self.move(goal, self.gpsi, self.gth, 0.2)
            if abs(site[2] - self.hover_z) < 0.01:
                self.set_phase("carry")
        elif ph == "carry":
            self.qref = self.qref_carry
            if self.qgoal is not None:
                dq = self.qgoal - self.q
                mx = float(np.max(np.abs(dq)))
                if mx < 0.01:
                    self.q = self.qgoal.copy()
                    self.d.qpos[self.adr] = self.q; mujoco.mj_forward(self.m, self.d)
                    self.cpos = self.d.site_xpos[self.sid].copy()
                    self.psi = self.cpsi; self.th = 0.0
                    self.qref = self.qgoal
                    self.set_phase("lower")
                else:
                    self.q = self.q + dq * min(1.0, 0.06 / mx)
                return np.r_[self.q, self.closure]
            goal = np.array([self.bin[0], self.bin[1], self.hover_z])
            done = self.move(goal, self.cpsi, 0.0, 0.3)
            if done and np.linalg.norm(site[:2] - self.bin) < 0.015:
                self.set_phase("lower")
        elif ph == "lower":
            goal = np.array([self.bin[0], self.bin[1], 0.90])
            done = self.move(goal, self.cpsi, 0.0, 0.15)
            if done:
                self.set_phase("release")
        elif ph == "release":
            self.closure = max(0.0, self.closure - dt / 0.4)
            if self.closure <= 0.0 and self.t_phase > 0.8:
                self.set_phase("retreat")
        elif ph == "retreat":
            self.closure = 0.0
            goal = np.array([self.bin[0], self.bin[1], 1.0])
            self.move(goal, self.cpsi, 0.0, 0.2)

        qn = self.ik(self.cpos, self.psi, self.th, self.q, qref=self.qref)
        dq = np.clip(qn - self.q, -0.08, 0.08)
        self.q = np.clip(self.q + dq, self.lo, self.hi)
        return np.r_[self.q, self.closure]

    def set_phase(self, p):
        self.phase = p
        self.t_phase = 0.0
