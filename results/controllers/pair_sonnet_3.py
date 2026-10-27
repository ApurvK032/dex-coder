import numpy as np, mujoco


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.adr = np.array(robot.arm_qpos_adr)
        self.dof = np.array(robot.arm_dof_adr)
        self.order = [task['target'], task['second']]
        self.dest = robot.bin_xy[task['destination']]
        self.left = task['destination'] == 'left'
        self.slot = 0
        self.phase = 'above'
        self.pt = 0.0
        self.p_cmd = None
        self.yaw_cmd = 0.0
        self.closure = 0.0
        self.H = 0.97
        self.GZ = 0.0
        self.last_print = -1
        self.cap = -0.1
        self.w = 0.5
        self.gy = None
        self.delta = None
        self.delta0 = np.array([0.0, 0.0])
        self.off = None
        self.rel_pos = None
        self.drop_yaw = 0.03
        self.release_z = robot.table_height + 0.08

    # ---------- IK ----------
    def ik_step(self, q, pos, yaw, iters=4, w=None, full=False):
        if w is None:
            w = self.w
        c, s = np.cos(yaw), np.sin(yaw)
        xd = np.array([c, s, 0.0]); yd = np.array([0, 0, -1.0]); zd = np.cross(xd, yd)
        R = np.stack([xd, yd, zd], 1)
        m, d = self.m, self.d
        hi = np.array(self.r.arm_upper).copy(); hi[1] = self.cap
        lo = np.array(self.r.arm_lower)
        qn = np.array(self.r.home_arm)
        for _ in range(iters):
            d.qpos[self.adr] = q
            mujoco.mj_forward(m, d)
            p = d.site_xpos[self.sid]; Rc = d.site_xmat[self.sid].reshape(3, 3)
            ep = pos - p
            eo = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dof], w * jr[:, self.dof]])
            e = np.concatenate([ep, w * eo])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.05 * (qn - q))
            q = np.clip(q + dq, lo, hi)
        if full:
            d.qpos[self.adr] = q
            mujoco.mj_forward(m, d)
            return q, np.linalg.norm(pos - d.site_xpos[self.sid])
        return q

    def pick_yaw(self, pos, cy):
        best = None
        for k in range(-1, 4):
            y = cy + k * np.pi / 2
            if y < -1.0 or y > 2.0:
                continue
            qq, e = self.ik_step(np.array(self.r.home_arm), pos, y, iters=150, w=0.5, full=True)
            score = e + 0.02 * abs(y - 0.3)
            if best is None or score < best[0]:
                best = (score, y, e)
        return best[1]

    def move_to(self, goal, gyaw, vmax=0.25, wmax=1.0):
        dt = 0.04
        dp = goal - self.p_cmd
        n = np.linalg.norm(dp)
        if n > vmax * dt:
            dp = dp / n * vmax * dt
        self.p_cmd = self.p_cmd + dp
        dy = gyaw - self.yaw_cmd
        dy = np.clip(dy, -wmax * dt, wmax * dt)
        self.yaw_cmd += dy

    def act(self, obs):
        r = self.r
        t = obs['t']
        site = np.array(obs['grasp_site_pos'])
        q = np.array(obs['arm_target'])
        if self.p_cmd is None:
            self.p_cmd = site.copy()
            Rm = np.array(obs['grasp_site_mat'])
            self.yaw_cmd = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
        if self.slot >= 2:
            self.closure = 0.0
            goal = np.array([self.dest[0] - 0.1, self.dest[1], self.H + 0.05])
            self.move_to(goal, 0.0)
            q2 = self.ik_step(q, self.p_cmd, self.yaw_cmd)
            return self.out(q, q2)

        name = self.order[self.slot]
        bp = np.array(obs['blocks'][name]['pos'])
        byaw = obs['blocks'][name]['yaw']
        if self.gy is None:
            self.gy = self.pick_yaw(np.array([bp[0], bp[1], bp[2]]), byaw)
        gy = self.gy
        bx, by = self.dest
        k = 0.02 if self.left else 0.025
        sgn = 1 if self.slot == 0 else -1
        a_low = 0.016 if self.left else 0.02
        a_push = 0.028 if self.left else 0.032
        xs = -1 if self.slot == 0 else 1
        if self.phase in ('carry', 'lower', 'slide', 'release', 'retreat'):
            self.cap = 0.05
        else:
            self.cap = -0.05
        if self.phase in ('lift', 'carry', 'lower', 'slide', 'release'):
            if self.phase in ('lift', 'carry') or self.off is None:
                self.off = bp[:2] - site[:2]
        else:
            self.off = None
        offv = self.off if self.off is not None else np.zeros(2)
        tgt_low = np.array([bx + xs * 0.025, by - 0.02])
        if self.slot == 0:
            tgt_push = np.array([bx - 0.036, by - 0.030])
            tgt_low = np.array([bx - 0.026, by - 0.02])
        else:
            c1 = np.array(obs['blocks'][self.order[0]]['pos'])
            best = None
            for gx in np.linspace(-0.03, 0.027, 15):
                for gy2 in np.linspace(-0.028, -0.01, 7):
                    cheb = max(abs(bx + gx - c1[0]), abs(by + gy2 - c1[1]))
                    sc = min(cheb, 0.062) - 0.05 * abs(gy2 + 0.02) - 0.1 * max(0.0, abs(gx) - 0.025)
                    if best is None or sc > best[0]:
                        best = (sc, gx, gy2)
            tgt_push = np.array([bx + best[1], by + best[2]])
            tgt_low = tgt_push.copy()
        if self.phase in ('carry', 'lower'):
            drop_xy = site[:2] + (tgt_low - bp[:2])
            push_xy = drop_xy
        elif self.phase == 'slide':
            drop_xy = site[:2] + (tgt_push - bp[:2])
            push_xy = drop_xy
        else:
            drop_xy = tgt_low - offv
            push_xy = tgt_push - offv
        ph = self.phase
        self.pt += 0.04
        goal = self.p_cmd; gyaw = self.yaw_cmd; vmax = 0.25

        def go(p):
            self.phase = p; self.pt = 0.0

        if ph == 'above':
            self.closure = 0.0
            self.w = 0.5
            goal = np.array([bp[0], bp[1], self.H]); gyaw = gy
            if np.linalg.norm(site[:2] - bp[:2]) < 0.01 and abs(site[2] - self.H) < 0.02 and self.pt > 0.3:
                go('down')
        elif ph == 'down':
            self.closure = 0.0
            goal = np.array([bp[0], bp[1], bp[2] + self.GZ]); gyaw = gy; vmax = 0.12
            if abs(site[2] - goal[2]) < 0.006 and np.linalg.norm(site[:2] - bp[:2]) < 0.01:
                go('close')
        elif ph == 'close':
            goal = self.p_cmd.copy(); gyaw = self.yaw_cmd
            self.closure = min(1.0, self.pt / 1.0)
            if self.pt < 0.05:
                Rm = np.array(obs['grasp_site_mat'])
                print('GRASP yaw_cube %.2f gy %.2f hand %.2f' % (byaw, self.gy, np.arctan2(Rm[1, 0], Rm[0, 0])), 'ydir', Rm[:, 1].round(2))
            if self.pt > 1.6:
                go('lift')
        elif ph == 'lift':
            self.closure = 1.0
            goal = self.p_cmd.copy(); goal[2] = self.H; gyaw = self.yaw_cmd
            if abs(site[2] - self.H) < 0.02:
                go('carry')
        elif ph == 'carry':
            self.closure = 1.0
            self.w = 0.4
            goal = np.array([drop_xy[0], drop_xy[1], self.H]); gyaw = self.drop_yaw
            if np.linalg.norm(bp[:2] - tgt_low) < 0.008 and np.linalg.norm(site[:2] - self.p_cmd[:2]) < 0.01 and self.pt > 0.3:
                go('lower')
        elif ph == 'lower':
            self.closure = 1.0
            rz = self.release_z if self.slot == 0 else self.release_z + 0.03
            goal = np.array([drop_xy[0], drop_xy[1], rz]); gyaw = self.drop_yaw; vmax = 0.12
            if abs(site[2] - goal[2]) < 0.01:
                go('slide')
        elif ph == 'slide':
            self.closure = 1.0
            rz = self.release_z if self.slot == 0 else self.release_z + 0.03
            goal = np.array([push_xy[0], push_xy[1], rz]); gyaw = self.drop_yaw; vmax = 0.04
            if self.pt > (1.2 if self.slot == 0 else 0.3):
                go('release')
        elif ph == 'release':
            if self.pt <= 0.05:
                self.rel_pos = bp[:2].copy()
            self.closure = max(0.0, 1.0 - self.pt / 0.6)
            goal = self.p_cmd.copy()

            if self.pt > 1.0:
                self.delta = bp[:2] - self.rel_pos
                print('LAND slot', self.slot, 'rel', (bp[:2] - np.array(self.dest)).round(3), 'z', round(bp[2], 3), 'yaw', round(byaw, 2))
                go('retreat')
        elif ph == 'retreat':
            self.closure = 0.0
            goal = np.array([self.p_cmd[0], self.p_cmd[1], self.H]); gyaw = self.drop_yaw
            if abs(site[2] - self.H) < 0.02:
                self.slot += 1
                self.gy = None
                go('above')

        self.move_to(goal, gyaw, vmax)
        q2 = self.ik_step(q, self.p_cmd, self.yaw_cmd)
        return self.out(q, q2)

    def out(self, q, q2):
        dq = q2 - q
        mx = np.max(np.abs(dq))
        if mx > 0.07:
            dq *= 0.07 / mx
        return list(q + dq) + [float(self.closure)]
