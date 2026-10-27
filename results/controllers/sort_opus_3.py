
import numpy as np
import mujoco


def yaw_R(a):
    xs = np.array([np.cos(a), np.sin(a), 0.0])
    ys = np.array([0.0, 0.0, -1.0])
    zs = np.cross(xs, ys)
    return np.column_stack([xs, ys, zs])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_err(R, Rd):
    E = Rd @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


class IK:
    def __init__(self, robot):
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower) + 0.03
        self.hi = np.array(robot.arm_upper) - 0.03
        self.hi[1] = min(self.hi[1], 0.0)  # shoulder roll: keep arm off the torso
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        m = self.m
        # body classification for collision checks
        names = [m.body(b).name for b in range(m.nbody)]
        self.arm_bodies = set(b for b in range(m.nbody) if names[b].startswith("right_") and
                              ("shoulder" in names[b] or "elbow" in names[b] or "wrist" in names[b] or "hand" in names[b]))
        self.hand_bodies = set(b for b in self.arm_bodies if "hand" in names[b] or "wrist_yaw" in names[b])
        self.robot_bodies = set(b for b in range(1, m.nbody) if not (names[b] in ("table",) or names[b].startswith("bin") or names[b].startswith("block")))
        self.bin_bodies = set(b for b in range(m.nbody) if names[b].startswith("bin"))
        self.table_body = names.index("table")
        self.block_bodies = {n[6:]: b for b, n in enumerate(names) if n.startswith("block_")}
        self.names = names

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def set_blocks(self, blocks):
        # place free-joint blocks in private model at observed poses
        m = self.m
        for n, b in self.block_bodies.items():
            if n not in blocks:
                continue
            jid = m.body_jntadr[b]
            if jid < 0:
                continue
            adr = m.jnt_qposadr[jid]
            self.d.qpos[adr:adr + 3] = blocks[n]["pos"]
            self.d.qpos[adr + 3:adr + 7] = blocks[n]["quat"]

    def collisions(self, q, ignore_block=None, table_ok=False):
        self.fk(q)
        mujoco.mj_collision(self.m, self.d)
        pen = 0.0
        m = self.m
        ign = self.block_bodies.get(ignore_block, -1)
        for k in range(self.d.ncon):
            c = self.d.contact[k]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            if c.dist > -0.001:
                continue
            for x, y in ((b1, b2), (b2, b1)):
                if x in self.arm_bodies:
                    if y in self.arm_bodies:
                        continue
                    if y == ign:
                        continue
                    if y == self.table_body and table_ok and x in self.hand_bodies:
                        continue
                    pen += -c.dist + 0.01
                    break
        return pen

    def solve(self, q0, pd, Rd, iters=100, wrot=0.5, step_lim=0.2, tol=1e-4, q_ref=None):
        q = np.clip(np.array(q0, float).copy(), self.lo, self.hi)
        for i in range(iters):
            p, R = self.fk(q)
            ep = pd - p
            er = rot_err(R, Rd)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
            err = np.r_[ep, wrot * er]
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], wrot * self.jr[:, self.dadr]])
            # mask joints pinned at limits pushing outward
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            if q_ref is not None:
                N = np.eye(7) - np.linalg.pinv(J) @ J
                dq += N @ (0.05 * (q_ref - q))
            n = np.max(np.abs(dq))
            if n > step_lim:
                dq *= step_lim / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        return q, np.linalg.norm(pd - p), np.linalg.norm(rot_err(R, Rd))


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.ik = IK(robot)
        self.home = np.array(robot.home_arm, float)
        self.target = task["target"]
        self.bin = np.array(robot.bin_xy[task["destination"]], float)
        self.th = robot.table_height
        self.phase = "init"
        self.q_cmd = None
        self.closure = 0.0
        self.gz = 0.006
        self.pre_h = 0.10
        self.transit_z = self.th + 0.18
        self.bias_p = {}
        self.bias_y = {}
        self.bias_acc = {}
        self.t_bias = 0.3
        self.integ = np.zeros(7)
        self.preclose = 0.0
        self.retries = 0
        self.seeds = [self.home,
                      np.array([0.5, -0.3, 0.0, -0.3, -1.4, -0.3, -0.1]),
                      np.array([0.3, -0.4, 0.3, 0.0, -1.0, 0.5, 0.0]),
                      np.array([0.0, -0.3, 0.3, 0.5, -1.2, 1.0, 0.2])]

    def plan_grasp(self, bpos, byaw):
        best = None
        self.other_blocks = {n: np.array(p["pos"], float) for n, p in self.cur_blocks.items() if n != self.target}
        for k in range(4):
            a = wrap(byaw + k * np.pi / 2)
            if abs(a) > 2.2:
                continue
            R = yaw_R(a)
            for s in self.seeds:
                pd = bpos + np.array([0, 0, self.pre_h])
                q, ep, er = self.ik.solve(s, pd, R, iters=150, q_ref=self.home)
                cost = 10 * ep + er
                if cost > 0.05:
                    continue
                q2, ep2, er2 = self.ik.solve(q, bpos + np.array([0, 0, self.gz]), R, iters=100)
                qm, _, _ = self.ik.solve(q, bpos + np.array([0, 0, self.gz + 0.5 * self.pre_h]), R, iters=60)
                col = (self.ik.collisions(q, self.target) + self.ik.collisions(q2, self.target, True)
                       + self.ik.collisions(qm, self.target, True))
                cost += 10 * ep2 + er2 + 5 * col + 0.01 * abs(a) + 0.005 * np.abs(q - self.home).sum()
                cost += 15 * self.sweep_penalty(bpos, a)
                if best is None or cost < best[0]:
                    best = (cost, a, q, R)
        if best is None:
            a = wrap(byaw)
            a = wrap(a - np.round(a / (np.pi / 2)) * np.pi / 2)
            R = yaw_R(a)
            q, _, _ = self.ik.solve(self.home, bpos + np.array([0, 0, self.pre_h]), R, iters=200, q_ref=self.home)
            best = (9.0, a, q, R)
        return best

    def sweep_penalty(self, bpos, a):
        # other cubes inside the zone swept by the curling fingers / thumb
        pen = 0.0
        xs = np.array([np.cos(a), np.sin(a)])
        zs = np.array([-np.sin(a), np.cos(a)])
        for n, ob in self.other_blocks.items():
            d = np.asarray(ob)[:2] - bpos[:2]
            u = d @ xs
            v = d @ zs
            du = min(u + 0.08, 0.12 - u)
            dv = 0.085 - abs(v)
            if du > 0 and dv > 0:
                pen += min(du, dv) + 0.02
        return pen

    def retry(self, t, blocks):
        self.retries += 1
        self.closure = 0.0
        b = blocks[self.target]
        self.ik.set_blocks(blocks)
        # first rise straight up to clear things, then go to pre-grasp
        cost, a, q, R = self.plan_grasp(b["pos"], b["yaw"])
        self.q_pre = q
        self.next_yaw = a
        self.next_R = R
        self.phase = "rise"
        self.t0 = t
        self.p_start = self.ik.fk(self.q_cmd)[0]
        self.grasp_xy = b["pos"][:2].copy()
        self.grasp_z = b["pos"][2]
        return np.r_[self.q_cmd, self.closure]

    def plan_place(self, a_grasp):
        best = None
        for k in range(-2, 3):
            a = a_grasp + k * np.pi / 2
            if abs(a) > 2.2:
                continue
            R = yaw_R(a)
            off = np.clip(R @ self.off_local, -0.006, 0.006)
            pd = np.array([self.bin[0] - off[0], self.bin[1] - off[1], self.transit_z])
            q, ep, er = self.ik.solve(self.q_cmd, pd, R, iters=200, q_ref=self.home)
            pd2 = pd.copy()
            pd2[2] = self.place_z
            q2, ep2, er2 = self.ik.solve(q, pd2, R, iters=100)
            col = self.ik.collisions(q, self.target) + self.ik.collisions(q2, self.target)
            cost = 10 * (ep + ep2) + (er + er2) + 5 * col + 0.01 * np.abs(q - self.q_cmd).sum()
            if best is None or cost < best[0]:
                best = (cost, a)
        return best[1]

    def corrected_blocks(self, obs):
        out = {}
        for n, b in obs["blocks"].items():
            p = np.array(b["pos"], float) - self.bias_p.get(n, 0.0)
            y = wrap(b["yaw"] - self.bias_y.get(n, 0.0))
            out[n] = {"pos": p, "yaw": y, "quat": np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)]),
                      "linvel": np.array(b.get("linvel", np.zeros(3)), float)}
        return out

    def update_bias(self, obs):
        bxy = self.task.get("block_xy", {}) or {}
        byaw = self.task.get("block_yaw", {}) or {}
        for n, b in obs["blocks"].items():
            if n not in bxy:
                continue
            e = np.array(b["pos"], float) - np.array([bxy[n][0], bxy[n][1], self.th + self.robot.block_half_size])
            if np.linalg.norm(e[:2]) > 0.03 or np.linalg.norm(b["linvel"]) > 0.02:
                continue
            ey = b["yaw"] - byaw.get(n, b["yaw"])
            ey = wrap(4 * ey) / 4  # cube symmetry
            self.bias_acc.setdefault(n, []).append((e, ey))
            L = self.bias_acc[n]
            self.bias_p[n] = np.mean([x[0] for x in L], axis=0)
            self.bias_p[n][2] = np.clip(self.bias_p[n][2], -0.01, 0.01)
            yb = np.mean([x[1] for x in L])
            self.bias_y[n] = yb if abs(yb) < 0.3 else 0.0

    def act(self, obs):
        if self.q_cmd is None or obs["t"] < self.t_bias:
            self.update_bias(obs)
        prev_cmd = None if self.q_cmd is None else self.q_cmd.copy()
        out = self.act_inner(obs)
        # integral correction of steady-state joint tracking error (gravity sag, weak motors)
        qm = np.array(obs["qpos"][:7], float)
        if prev_cmd is not None and np.max(np.abs(self.q_cmd - prev_cmd)) < 0.012:
            e = self.q_cmd - qm
            self.integ += 0.15 * np.clip(e, -0.05, 0.05)
            self.integ = np.clip(self.integ, -0.15, 0.15)
        q_out = np.clip(self.q_cmd + self.integ, self.robot.arm_lower, self.robot.arm_upper)
        return np.r_[q_out, out[7]]

    def act_inner(self, obs):
        t = obs["t"]
        blocks = self.corrected_blocks(obs)
        self.cur_blocks = blocks
        b = blocks[self.target]
        bpos = b["pos"]
        if self.q_cmd is None:
            self.q_cmd = np.array(obs["arm_target"], float)
            self.ik.set_blocks(blocks)
            cost, a, q, R = self.plan_grasp(bpos, b["yaw"])
            self.yaw = a
            self.R = R
            self.q_pre = q
            self.phase = "to_pre"
            self.t0 = t
            self.q_start = self.q_cmd.copy()
            self.grasp_xy = bpos[:2].copy()
            self.grasp_z = bpos[2]
        el = t - self.t0

        def go(phase):
            self.phase = phase
            self.t0 = t
            self.p_start = self.ik.fk(self.q_cmd)[0]
            self.yaw_start = self.yaw

        def cart(pd, dur, yaw_end=None):
            s = min(1.0, el / dur)
            s = 3 * s * s - 2 * s ** 3
            p = self.p_start + s * (pd - self.p_start)
            if yaw_end is not None:
                self.yaw = self.yaw_start + s * (yaw_end - self.yaw_start)
                self.R = yaw_R(self.yaw)
            q, _, _ = self.ik.solve(self.q_cmd, p, self.R, iters=10)
            self.q_cmd = q
            return el >= dur

        if self.phase == "rise":
            z = max(self.p_start[2], self.transit_z)
            if cart(np.array([self.p_start[0], self.p_start[1], z]), 0.8) and el > 1.0:
                self.yaw = self.next_yaw
                self.R = self.next_R
                self.q_start = self.q_cmd.copy()
                self.phase = "to_pre"
                self.t0 = t
        elif self.phase == "to_pre":
            dur = 2.0
            s = min(1.0, max(0.0, el / dur))
            s = 3 * s * s - 2 * s ** 3
            self.q_cmd = self.q_start + s * (self.q_pre - self.q_start)
            self.closure = self.preclose * s
            if el >= dur + 0.3:
                go("descend")
        elif self.phase == "descend":
            # track block xy (observed) while descending, until close to it
            if el < 1.0:
                self.grasp_xy = bpos[:2].copy()
                self.grasp_z = bpos[2]
            pd = np.array([self.grasp_xy[0], self.grasp_xy[1], self.grasp_z + self.gz])
            if cart(pd, 1.5) and el > 2.0:
                go("close")
        elif self.phase == "close":
            self.closure = min(1.0, self.preclose + el / 0.6)
            if el > 0.9:
                go("lift")
        elif self.phase == "lift":
            if cart(np.array([self.p_start[0], self.p_start[1], self.transit_z]), 1.0):
                sp = np.array(obs["grasp_site_pos"])
                sR = np.array(obs["grasp_site_mat"])
                self.off_local = sR.T @ (bpos - sp)
                cz = (bpos - sp)[2]
                if bpos[2] < self.th + 0.06:
                    return self.retry(t, blocks)
                self.place_z = self.th + 0.052 - cz
                self.place_yaw = self.plan_place(self.yaw)
                go("transit")
        elif self.phase == "transit":
            if np.linalg.norm(bpos - np.array(obs["grasp_site_pos"])) > 0.06:
                return self.retry(t, blocks)
            off = np.clip(yaw_R(self.place_yaw) @ self.off_local, -0.006, 0.006)
            self.place_xy = self.bin - off[:2]
            if cart(np.array([self.place_xy[0], self.place_xy[1], self.transit_z]), 2.0,
                    self.place_yaw) and el > 2.3:
                go("lower")
        elif self.phase == "lower":
            if cart(np.array([self.place_xy[0], self.place_xy[1], self.place_z]), 1.0) and el > 1.3:
                go("release")
        elif self.phase == "release":
            self.closure = max(0.0, 1.0 - el / 0.6)
            if el > 1.0:
                go("retreat")
        elif self.phase == "retreat":
            cart(np.array([self.p_start[0], self.p_start[1], self.transit_z + 0.05]), 1.2)
            if el > 2.0:
                # verify: if target not in bin, try again
                inside = (np.all(np.abs(bpos[:2] - self.bin) < self.robot.bin_inner_half_size)
                          and bpos[2] < self.th + 0.06)
                if not inside and np.linalg.norm(b.get("linvel", np.zeros(3))) < 0.05 and self.retries < 3:
                    return self.retry(t, blocks)
        return np.r_[self.q_cmd, self.closure]
