import numpy as np, mujoco

TABLE = 0.78
HALF = 0.025


def yaw_rot(psi):
    c, s = np.cos(psi), np.sin(psi)
    x = np.array([c, s, 0.0]); y = np.array([0, 0, -1.0]); z = np.cross(x, y)
    return np.column_stack([x, y, z])


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.d = mujoco.MjData(self.m)
        self.phase = 0
        self.tphase = 0.0
        self.cmd = None   # commanded site pos
        self.closure = 0.0
        self.q = np.array(robot.home_arm, float)
        self.psi = None
        self.hover = 0.07
        self.lift = 0.09
        self.zgrasp = 0.0
        self.grip_dz = 0.0
        self.off = np.array([-0.01, -0.01, -0.01])
        self.close_rate = 0.04
        self.kn = 0.3
        self.carry_speed = 0.2
        self.qref = np.array(robot.home_arm, float)
        self.wt = 0.5
        self.wy = 0.5
        self.deltas = [0.0, -0.17, 0.17, -0.35, 0.35]

    # ---------------- IK
    def ik(self, q0, pos, R, iters=15):
        m, d = self.m, self.d
        robot = self.robot
        q = q0.copy()
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        dofs = list(robot.arm_dof_adr)
        for _ in range(iters):
            d.qpos[:] = self.base_qpos
            d.qpos[list(robot.arm_qpos_adr)] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site].copy()
            Rc = d.site_xmat[self.site].reshape(3, 3).copy()
            ep = pos - p
            eo = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            W = np.diag([self.wt, self.wy, self.wt])
            J = np.vstack([jp[:, dofs], W @ (R.T @ jr[:, dofs])])
            e = np.concatenate([ep, W @ (R.T @ eo)])
            lam = 0.02
            Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
            dq = Jp @ e
            N = np.eye(7) - Jp @ J
            dq = dq + N @ (self.kn * (self.qref - q))
            q = np.clip(q + dq, robot.arm_lower, robot.arm_upper)
            if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(eo) < 2e-2:
                break
        return q

    def dry_run(self, psi, tgt, dst, zc):
        Rp = yaw_rot(psi)
        sp0 = self.start_site
        zg = zc + self.off[2]
        o = Rp @ self.off
        g = np.array([tgt[0], tgt[1], zc]) + o
        d_xy = np.array([dst[0] - tgt[0], dst[1] - tgt[1]])
        wps = [sp0,
               g + np.array([0, 0, self.hover]),
               g,
               g + np.array([0, 0, self.lift]),
               g + np.array([d_xy[0], d_xy[1], self.lift]),
               g + np.array([d_xy[0], d_xy[1], 2 * HALF + 0.004])]
        q = self.q.copy()
        worst = 0.0
        jl = 0.0
        p = wps[0]
        for w in wps[1:]:
            n = max(1, int(np.linalg.norm(w - p) / 0.015))
            for i in range(1, n + 1):
                pos = p + (w - p) * i / n
                qn = self.ik(q, pos, Rp, iters=4)
                worst = max(worst, self.fk_err(qn, pos, Rp), 0.5 * max(0.0, qn[1] + 0.35))
                jl += np.abs(qn - q).sum()
                q = qn
            p = w
        return worst * 5 + 0.02 * jl + 0.01 * abs(psi - 0.3)

    def fk_err(self, q, pos, R):
        d = self.d
        d.qpos[:] = self.base_qpos
        d.qpos[list(self.robot.arm_qpos_adr)] = q
        mujoco.mj_kinematics(self.m, d)
        p = d.site_xpos[self.site]
        Rc = d.site_xmat[self.site].reshape(3, 3)
        return np.linalg.norm(p - pos) + 0.5 * np.linalg.norm(Rc - R)

    def move_cmd(self, goal, speed, dt=0.04):
        diff = goal - self.cmd
        n = np.linalg.norm(diff)
        step = speed * dt
        if n <= step:
            self.cmd = goal.copy()
            return True
        self.cmd = self.cmd + diff / n * step
        return False

    def act(self, obs):
        robot = self.robot
        task = self.task
        dt = 0.04
        self.base_qpos = np.array(obs["kin"].qpos)
        sp = np.array(obs["grasp_site_pos"])
        if self.cmd is None:
            self.cmd = sp.copy()
            self.q = np.array(obs["arm_target"], float)
        tgt = obs["blocks"][task["target"]]["pos"]
        dst = obs["blocks"][task["destination"]]["pos"]
        tgt = np.array(tgt); dst = np.array(dst)
        yaw_t = obs["blocks"][task["target"]]["yaw"]
        zc = TABLE + HALF
        self.start_site = sp.copy()
        if self.psi is None:
            # hand yaw: cube yaw + k*pi/2 nearest to preferred
            best = None
            poses = [np.array([tgt[0], tgt[1], zc + 0.0]), np.array([tgt[0], tgt[1], zc + self.hover]),
                     np.array([dst[0], dst[1], zc + 2 * HALF + 0.01]), np.array([dst[0], dst[1], zc + 0.12])]
            for k in range(-3, 4):
              for dl in self.deltas:
                psi = yaw_t + k * np.pi / 2 + dl
                if psi < -1.0 or psi > 1.6:
                    continue
                score = self.dry_run(psi, tgt, dst, zc) + 0.1 * abs(dl)
                if best is None or score < best[0]:
                    best = (score, psi)
            self.psi = best[1]
            print("psi", self.psi, "score", best[0])
        R = yaw_rot(self.psi)
        self.tphase += dt
        done = False
        zc = TABLE + HALF
        ph = self.phase
        if ph == 0:   # move above target
            goal = np.array([tgt[0], tgt[1], zc]) + R @ self.off + np.array([0, 0, self.hover])
            # first rise up if low
            done = self.move_cmd(goal, 0.25) and np.linalg.norm(sp - goal) < 0.02
            if done: self.next()
        elif ph == 1:  # descend
            goal = np.array([tgt[0], tgt[1], zc]) + R @ self.off
            # track cube xy
            self.move_cmd(goal, 0.10)
            if np.linalg.norm(sp - goal) < 0.012 and self.tphase > 0.5:
                self.next()
        elif ph == 2:  # close
            self.closure = min(1.0, self.closure + self.close_rate)
            if self.closure >= 1.0 and self.tphase > 1.6:
                self.next()
        elif ph == 3:  # lift
            goal = np.array([self.cmd[0], self.cmd[1], zc + self.lift])
            if self.move_cmd(goal, 0.12) and self.tphase > 0.3:
                self.next()
        elif ph == 4:  # move over dest
            off = tgt[:2] - sp[:2]
            goal = np.array([self.cmd[0] + (dst[0] - tgt[0]), self.cmd[1] + (dst[1] - tgt[1]), zc + self.lift])
            self.move_cmd(goal, self.carry_speed)
            if np.linalg.norm(dst[:2] - tgt[:2]) < 0.006 and self.tphase > 0.3:
                self.next()
        elif ph == 5:  # lower
            zt = dst[2] + 2 * HALF + 0.004
            dz = tgt[2] - zt
            goal = np.array([self.cmd[0] + (dst[0] - tgt[0]) * 0.5, self.cmd[1] + (dst[1] - tgt[1]) * 0.5, self.cmd[2] - dz])
            goal[2] = max(goal[2], self.cmd[2] - 0.003) if dz > 0 else goal[2]
            self.cmd = goal
            if dz < 0.002:
                self.next()
        elif ph == 6:  # release
            self.closure = max(0.0, self.closure - 0.05)
            if self.closure <= 0.0 and self.tphase > 0.6:
                self.next()
        elif ph == 7:  # retreat up
            goal = np.array([self.cmd[0], self.cmd[1], self.cmd[2] + 0.15])
            self.move_cmd(goal, 0.15)
        qn = self.ik(self.q, self.cmd, R)
        dq = np.clip(qn - self.q, -0.07, 0.07)
        self.q = self.q + dq
        return np.concatenate([self.q, [self.closure]])

    def next(self):
        self.phase += 1
        self.tphase = 0.0
