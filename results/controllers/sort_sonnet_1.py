import numpy as np, mujoco

P = dict(ox=-0.015, oz=0.0, zoff=0.0, ramp=0.05, widen=False, wmis=0.0, soft=False, wa=0.0, rel=0.1)


def _rot_des(a):
    x = np.array([np.cos(a), np.sin(a), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.tgt = task["target"]
        self.dest = task["destination"]
        self.phase = 0
        self.q = np.array(robot.home_arm, float)
        self.qgoal = self.q.copy()
        self.closure = 0.0
        self.a = None
        self.hold = 0
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.cur = None
        self.R = None
        self.newphase = True
        self.tries = 0
        self.z_hover = 0.97
        self.z_drop = 0.93
        self.qh = None
        self.qg = None
        self.jlim = 0.07

    # ---- kinematics
    def fk(self, q):
        self.d.qpos[self.r.arm_qpos_adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, p, R, iters=6, ow=0.7):
        q = q0.copy()
        r = self.r
        for _ in range(iters):
            self.d.qpos[r.arm_qpos_adr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            cp = self.d.site_xpos[self.sid].copy()
            cR = self.d.site_xmat[self.sid].reshape(3, 3).copy()
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp, jr])[:, r.arm_dof_adr]
            ep = p - cp
            eo = 0.5 * sum(np.cross(cR[:, i], R[:, i]) for i in range(3))
            e = np.concatenate([ep, ow * eo])
            lam = 0.02
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def solve_one(self, p, a, q_seed):
        R = _rot_des(a)
        p0, _ = self.fk(q_seed)
        for mode in (0, 1):
            q = q_seed.copy()
            if mode == 0:
                for _ in range(25):
                    q = self.ik(q, p, R, 6)
            else:
                for w in np.linspace(0, 1, 7)[1:]:
                    for _ in range(6):
                        q = self.ik(q, p0 + (p - p0) * w, R, 6)
            cp, cR = self.fk(q)
            pe = np.linalg.norm(cp - p)
            oe = np.arccos(np.clip((np.trace(cR.T @ R) - 1) / 2, -1, 1))
            if pe > 0.004 or oe > 0.05:
                continue
            margin = float(np.min(np.minimum(q - self.lo, self.hi - q)))
            return q, margin
        return None

    def solve_pose(self, p, a_list, q_seed, soft=True, a_ref=None, wa=0.0):
        best = None
        for a in a_list:
            res = self.solve_one(p, a, q_seed)
            if res is None:
                continue
            q, margin = res
            if soft:
                cost = np.linalg.norm(q - q_seed) + 3.0 * max(0.0, 0.3 - margin)
            else:
                cost = np.linalg.norm(q - q_seed) + (1.0 if margin < 0.1 else 0.0)
            if a_ref is not None:
                cost += wa * abs(a - a_ref)
            if best is None or cost < best[0]:
                best = (cost, q, a)
        return best

    def step_joint(self):
        dq = self.qgoal - self.q
        m = np.max(np.abs(dq))
        lim = self.jlim
        if m > lim:
            dq = dq * lim / m
        self.q = self.q + dq

    def joint_done(self, obs, tol=0.03):
        return (np.max(np.abs(self.qgoal - self.q)) < 1e-3 and
                np.max(np.abs(np.array(obs["qpos"][:7]) - self.qgoal)) < tol)

    def step_cart(self, goal, speed=0.01):
        dv = goal - self.cur
        n = np.linalg.norm(dv)
        if n > speed:
            dv = dv / n * speed
        self.cur = self.cur + dv
        q_new = self.ik(self.q, self.cur, self.R, 4)
        dq = np.clip(q_new - self.q, -0.08, 0.08)
        self.q = self.q + dq
        return n

    def plan_grasp(self, bp, yaw):
        """Pick finger yaw a and joint configs for hover (qh) and grasp (qg)."""
        best = None
        cands = [yaw + k * np.pi / 2 for k in range(-3, 4)]
        aligned = [x for x in cands if -1.6 < x < 1.6]
        cands = aligned + list(np.linspace(-1.5, 1.5, 13))
        for a in cands:
            if (not P.get('widen', True)) and best is not None and a not in aligned:
                break
            d = (a - yaw) % (np.pi / 2)
            mis = min(d, np.pi / 2 - d)
            ca, sa = np.cos(a), np.sin(a)
            ph = np.array([bp[0], bp[1], self.z_hover])
            pg = np.array([bp[0] + P['ox'] * ca - P['oz'] * sa,
                           bp[1] + P['ox'] * sa + P['oz'] * ca,
                           bp[2] + P['zoff']])
            h = self.solve_one(ph, a, self.q)
            if h is None:
                continue
            g = self.solve_one(pg, a, h[0])
            if g is None:
                continue
            margin = min(h[1], g[1])
            cost = (np.linalg.norm(h[0] - self.q) + 2.0 * np.linalg.norm(g[0] - h[0])
                    + 4.0 * max(0.0, 0.3 - margin) + P.get('wmis', 1.5) * mis)
            if best is None or cost < best[0]:
                best = (cost, a, h[0], g[0])
        return best

    def act(self, obs):
        r = self.r
        t = obs["t"]
        site = np.array(obs["grasp_site_pos"])
        blk = obs["blocks"][self.tgt]
        bp = np.array(blk["pos"])
        binx, biny = r.bin_xy[self.dest]
        closure = self.closure
        new = self.newphase
        self.newphase = False

        def goto(ph):
            self.phase = ph
            self.newphase = True
            self.hold = 0

        ph = self.phase
        if ph == 0:   # joint move to hover above cube
            closure = 0.0
            if new:
                self.jlim = 0.07
                res = self.plan_grasp(bp, blk["yaw"])
                if res is None:
                    print("no grasp plan")
                    res2 = self.solve_pose(np.array([bp[0], bp[1], self.z_hover]),
                                           np.linspace(-1.5, 1.5, 13), self.q)
                    if res2 is None:
                        return np.concatenate([self.q, [0.0]])
                    self.a = res2[2]; self.qh = res2[1]; self.qg = None
                else:
                    self.a = res[1]; self.qh = res[2]
                    self.qg = res[3] if np.max(np.abs(res[3] - res[2])) < P.get('jdesc', 0.0) else None
                self.R = _rot_des(self.a)
                self.qgoal = self.qh
            self.step_joint()
            if self.joint_done(obs):
                goto(1)
        elif ph == 1:  # descend
            closure = 0.0
            if new:
                self.jlim = 0.035
                if self.qg is None:
                    self.cur, _ = self.fk(self.q)
                else:
                    self.qgoal = self.qg
            if self.qg is not None:
                self.step_joint()
                if self.joint_done(obs, 0.02):
                    goto(2)
            else:
                ca, sa = np.cos(self.a), np.sin(self.a)
                goal = np.array([bp[0] + P['ox'] * ca - P['oz'] * sa,
                                 bp[1] + P['ox'] * sa + P['oz'] * ca, bp[2] + P['zoff']])
                n = self.step_cart(goal, 0.008)
                if n < 0.003 and np.linalg.norm(site - goal) < 0.01:
                    goto(2)
        elif ph == 2:  # close
            self.hold += 1
            closure = min(1.0, self.hold * P['ramp'])
            if self.hold > 28 + int(0.8 / P['ramp']) - 16:
                goto(3)
        elif ph == 3:  # lift
            closure = 1.0
            if new:
                self.jlim = 0.035
                if self.qh is not None and self.qg is not None:
                    self.qgoal = self.qh
                else:
                    self.cur, _ = self.fk(self.q)
            if self.qg is not None:
                self.step_joint()
                done = self.joint_done(obs, 0.03)
            else:
                goal = np.array([self.cur[0], self.cur[1], self.z_hover + 0.02])
                n = self.step_cart(goal, 0.008)
                done = n < 0.003
            if done and site[2] > self.z_hover - 0.03:
                if np.linalg.norm(bp - site) > 0.05:
                    print("lost cube; retry")
                    self.tries += 1
                    goto(0)
                else:
                    goto(4)
        elif ph == 4:  # to bin
            closure = 1.0
            if new:
                self.jlim = 0.07
                p = np.array([binx, biny, self.z_drop])
                res = self.solve_pose(p, np.linspace(-1.5, 1.5, 25), self.q,
                                      soft=P.get('soft', True), a_ref=self.a, wa=P.get('wa', 0.0))
                if res is None:
                    print("no bin IK")
                    return np.concatenate([self.q, [closure]])
                self.qgoal = res[1]
                self.R = _rot_des(res[2])
            self.step_joint()
            if self.joint_done(obs):
                goto(5)
        elif ph == 5:  # open
            self.hold += 1
            closure = max(0.0, 1.0 - self.hold * P.get('rel', 0.1))
            if self.hold > 20:
                goto(6)
        else:
            closure = 0.0
            if new:
                self.cur, _ = self.fk(self.q)
            goal = np.array([self.cur[0], self.cur[1], self.z_hover + 0.05])
            self.step_cart(goal, 0.008)

        self.closure = closure
        if int(round(t / 0.04)) % 25 == 0:
            print(f"t={t:.2f} ph={self.phase} site={np.round(site,3)} blk={np.round(bp,3)}")
        return np.concatenate([self.q, [closure]])
