import numpy as np, mujoco

H_ABOVE = 0.13


class Controller:
    UB1 = 0.0
    avoid = False
    DLP = 3.0
    GX = -0.02
    GZ = 0.0
    GH = 0.008

    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.task = task
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.q = None
        self.phase = -1
        self.t_phase = 0.0
        self.closure = 0.0
        self.cur = None
        self.tgt = task['target']
        self.dst = task['destination']
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.qhome = np.array(robot.home_arm, float)
        self.plan_q = None
        self.psi = 0.0
        self.psi2 = 0.0
        self.psi_cur = None
        m = self.m
        self.torso_geoms = set()
        self.block_geoms = set()
        self.hand_geoms = set()
        self.table_geoms = set()
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if bn.startswith('block'):
                self.block_geoms.add(g)
            elif 'hand' in bn:
                self.hand_geoms.add(g)
            if bn == 'table':
                self.table_geoms.add(g)
            if bn == 'torso_link':
                self.torso_geoms.add(g)
        self.ub = np.array(robot.arm_upper, float)
        self.ub[1] = self.UB1
        self.planned = False
        self.lag = 0.0
        self.plan_cost = -1
        try:
            tb = task['block_xy'][self.tgt]
            db = task['block_xy'][self.dst]
            hs = robot.block_half_size
            zc = robot.table_height + hs
            self.plan_all(np.array([tb[0], tb[1], zc]), task['block_yaw'][self.tgt],
                          np.array([db[0], db[1], zc]), task['block_yaw'][self.dst])
            if self.plan_cost >= 1e8:
                first = (self.psi, self.psi2, self.plan_q)
                self.ub[1] = 0.45
                self.plan_all(np.array([tb[0], tb[1], zc]), task['block_yaw'][self.tgt],
                              np.array([db[0], db[1], zc]), task['block_yaw'][self.dst])
                if self.plan_cost >= 1e8:
                    self.ub[1] = self.UB1
                    self.psi, self.psi2, self.plan_q = first
        except Exception as e:  # pragma: no cover
            print('plan error', e)
            self.psi = 0.0
            self.psi2 = 0.0
            self.plan_q = None
        self.psi_cur = self.psi

    # ---------- kinematics helpers
    def rot(self, psi):
        x = np.array([np.cos(psi), np.sin(psi), 0.0])
        y = np.array([0, 0, -1.0])
        z = np.cross(x, y)
        return np.stack([x, y, z], 1)

    def ik(self, q0, p, R, iters=8, w=0.02):
        r = self.r
        q = q0.copy()
        d = self.d
        bad0 = self.torso_pen(q) if self.avoid else 0.0
        for _ in range(iters):
            d.qpos[r.arm_qpos_adr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.sid)
            cp = d.site_xpos[self.sid]
            cR = d.site_xmat[self.sid].reshape(3, 3)
            ep = p - cp
            eo = 0.5 * (np.cross(cR[:, 0], R[:, 0]) + np.cross(cR[:, 1], R[:, 1]) + np.cross(cR[:, 2], R[:, 2]))
            J = np.vstack([self.jp[:, r.arm_dof_adr], self.jr[:, r.arm_dof_adr]])
            e = np.concatenate([ep, eo])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (w * (self.qhome - q))
            n = np.abs(dq).max()
            if n > 0.15:
                dq *= 0.15 / n
            qc = np.clip(q + dq, r.arm_lower, self.ub)
            if self.avoid:
                lim = max(bad0, 0.0003)
                for _ in range(3):
                    if self.torso_pen(qc) <= lim:
                        break
                    qc = q + 0.5 * (qc - q)
                else:
                    qc = q
            q = qc
        return q

    def torso_pen(self, q):
        d = self.d
        d.qpos[self.r.arm_qpos_adr] = q
        mujoco.mj_forward(self.m, d)
        w = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.torso_geoms or c.geom2 in self.torso_geoms:
                if c.geom1 in self.block_geoms or c.geom2 in self.block_geoms:
                    continue
                if c.geom1 in self.table_geoms or c.geom2 in self.table_geoms:
                    continue
                w = max(w, -c.dist)
        return w

    def pose_err(self, q, p):
        d = self.d
        d.qpos[self.r.arm_qpos_adr] = q
        mujoco.mj_kinematics(self.m, d)
        return np.linalg.norm(d.site_xpos[self.sid] - p)

    def contact_pen(self, q):
        d = self.d
        d.qpos[self.r.arm_qpos_adr] = q
        mujoco.mj_forward(self.m, d)
        bad = 0.0
        tab = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.block_geoms or g2 in self.block_geoms:
                continue
            if g1 in self.hand_geoms and g2 in self.hand_geoms:
                continue
            if g1 in self.table_geoms or g2 in self.table_geoms:
                tab = max(tab, -c.dist)
            else:
                bad = max(bad, -c.dist)
        return bad, tab

    # ---------- planning
    def feasible(self, q, p, psi):
        for _ in range(10):
            q = self.ik(q, p, self.rot(psi), iters=10)
            if self.pose_err(q, p) < 0.0015:
                break
        if self.pose_err(q, p) > 0.003:
            return None
        bad, tab = self.contact_pen(q)
        if bad > 0.001 or tab > 0.012:
            return None
        return q

    def wrap(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def pick_options(self, pick, cy, starts):
        r = self.r
        zc = r.table_height + r.block_half_size
        out = []
        for k in range(4):
            for dl in (0.0, 0.2, -0.2):
                psi = self.wrap(cy + k * np.pi / 2 + dl)
                pk = pick[:2] + self.rot(psi)[:2, 0] * self.GX
                pa = np.array([pk[0], pk[1], zc + H_ABOVE])
                pg = np.array([pk[0], pk[1], zc + self.GH])
                bestq = None
                bc = 1e9
                for q0 in starts:
                    qa = self.feasible(q0, pa, psi)
                    if qa is None:
                        continue
                    qg = self.feasible(qa, pg, psi)
                    if qg is None:
                        continue
                    c = np.abs(qa - self.qhome).sum()
                    if c < bc:
                        bc = c
                        bestq = (qa, qg)
                if bestq is not None:
                    out.append((bc + self.DLP * abs(dl), psi, bestq[0], bestq[1]))
        out.sort(key=lambda t: t[0])
        return out

    def transfer(self, qa, psi0, pick, place, psi1):
        r = self.r
        hs = r.block_half_size
        zc = r.table_height + hs
        q = qa.copy()
        pk = pick[:2] + self.rot(psi0)[:2, 0] * self.GX
        cur = np.array([pk[0], pk[1], zc + H_ABOVE])
        psi = psi0
        cost = 0.0
        pl = place[:2] + self.rot(psi1)[:2, 0] * self.GX
        goal1 = np.array([pl[0], pl[1], zc + H_ABOVE])
        goal2 = np.array([pl[0], pl[1], r.table_height + 3 * hs + 0.006])
        k = 0
        for stage in (0, 1):
            goal = goal1 if stage == 0 else goal2
            for _ in range(200):
                diff = goal - cur
                n = np.linalg.norm(diff)
                sp = 0.015
                cur = goal.copy() if n < sp else cur + diff / n * sp
                if stage == 0:
                    dpsi = self.wrap(psi1 - psi)
                    psi = psi + np.clip(dpsi, -0.025, 0.025)
                else:
                    psi = psi1
                qn = self.ik(q, cur, self.rot(psi), iters=6)
                cost += np.abs(qn - q).sum()
                q = qn
                k += 1
                done = (n < sp) and (stage == 1 or abs(self.wrap(psi1 - psi)) < 1e-6)
                if k % 3 == 0 or done:
                    if self.pose_err(q, cur) > 0.006:
                        return 1e6, None
                    bad, tab = self.contact_pen(q)
                    if bad > 0.002:
                        return 1e6, None
                if done:
                    break
        return cost, q

    def plan_all(self, pick, cy, place, dy):
        r = self.r
        rng = np.random.RandomState(1)
        starts = [self.qhome.copy()]
        for _ in range(3):
            starts.append(np.clip(self.qhome + rng.uniform(-0.5, 0.5, 7), r.arm_lower, r.arm_upper))
        opts = self.pick_options(pick, cy, starts)
        best = (1e9, None, None, None)
        for oc, psi, qa, qg in opts[:4]:
            cands = [psi] + [self.wrap(dy + k * np.pi / 2) for k in range(4)]
            for psi1 in cands:
                c, qp = self.transfer(qa, psi, pick, place, psi1)
                tot = oc + c + 8.0 * abs(self.wrap(psi1 - psi))
                if c < 1e6 and tot < best[0]:
                    best = (tot, psi, psi1, qa)
        self.plan_cost = best[0]
        if best[1] is None:
            if opts:
                best = (0, opts[0][1], opts[0][1], opts[0][2])
            else:
                best = (0, self.wrap(cy), self.wrap(cy), None)
        self.psi, self.psi2, self.plan_q = best[1], best[2], best[3]
        self.planned = True

    # ---------- execution
    def step_toward(self, goal, speed):
        if self.lag > 0.03:
            return False
        diff = goal - self.cur
        n = np.linalg.norm(diff)
        if n < speed:
            self.cur = goal.copy()
            return True
        self.cur = self.cur + diff / n * speed
        return False

    def act(self, obs):
        r = self.r
        dt = r.control_dt
        blocks = obs['blocks']
        if self.q is None:
            self.q = np.array(obs['arm_target'], float)
            self.cur = np.array(obs['grasp_site_pos'], float)
            self.phase = 0
            print('plan cost', round(self.plan_cost, 2), 'psi', round(self.psi, 2), round(self.psi2, 2))
        tb = blocks[self.tgt]['pos']
        db = blocks[self.dst]['pos']
        hs = r.block_half_size
        self.lag = float(np.linalg.norm(np.array(obs['grasp_site_pos']) - self.cur))
        self.t_phase += dt
        ph = self.phase
        speed = 0.012
        closure = self.closure
        zc = r.table_height + hs
        qn = None
        if ph == 0:  # joint-space move to above pick
            if self.plan_q is None:
                self.cur = np.array([tb[0], tb[1], zc + H_ABOVE])
                if np.linalg.norm(obs['grasp_site_pos'] - self.cur) < 0.01 and self.t_phase > 1:
                    self.next(1)
            else:
                dq = np.clip(self.plan_q - self.q, -0.07, 0.07)
                self.q = self.q + dq
                qn = self.q
                if np.abs(self.plan_q - self.q).max() < 1e-4 and self.t_phase > 0.5:
                    self.cur = np.array([tb[0], tb[1], zc + H_ABOVE])
                    self.next(1)
        elif ph == 1:  # descend
            Rm = self.rot(self.psi_cur)
            gp = tb[:2] + Rm[:2, 0] * self.GX + Rm[:2, 2] * self.GZ
            goal = np.array([gp[0], gp[1], zc + self.GH])
            self.cur[:2] = gp
            if self.step_toward(goal, speed * 0.7):
                self.next(2)
        elif ph == 2:  # close
            closure = min(1.0, closure + 0.04)
            if self.t_phase > 1.2:
                self.next(3)
        elif ph == 3:  # lift
            goal = np.array([self.cur[0], self.cur[1], zc + H_ABOVE])
            if self.step_toward(goal, speed):
                self.next(4)
        elif ph == 4:  # move over dest, rotate wrist
            off = self.offset(obs)
            goal = np.array([db[0] + off[0], db[1] + off[1], zc + H_ABOVE])
            a = self.step_toward(goal, speed * 1.2)
            dpsi = self.wrap(self.psi2 - self.psi_cur)
            self.psi_cur += np.clip(dpsi, -0.02, 0.02)
            if a and abs(dpsi) < 0.02:
                self.psi_cur = self.psi2
                self.next(5)
        elif ph == 5:  # descend onto dest
            off = self.offset(obs)
            goal = np.array([db[0] + off[0], db[1] + off[1], r.table_height + 3 * hs + 0.006])
            self.cur[:2] = goal[:2]
            if self.step_toward(goal, speed * 0.6):
                self.next(6)
        elif ph == 6:  # release
            closure = max(0.0, closure - 0.08)
            if self.t_phase > 1.0:
                self.next(7)
        elif ph == 7:  # retreat up
            goal = np.array([self.cur[0], self.cur[1], zc + H_ABOVE + 0.05])
            self.step_toward(goal, speed)
        self.closure = closure
        if qn is None:
            qn = self.ik(self.q, self.cur, self.rot(self.psi_cur))
            dq = np.clip(qn - self.q, -0.07, 0.07)
            self.q = self.q + dq
        return list(self.q) + [closure]

    def offset(self, obs):
        o = np.array(obs['grasp_site_pos'][:2]) - np.array(obs['blocks'][self.tgt]['pos'][:2])
        n = np.linalg.norm(o)
        if n > 0.04:
            o = o / n * 0.04
        return o

    def next(self, p):
        self.phase = p
        self.t_phase = 0.0
