
import numpy as np
import mujoco

PRECURL = 0.35
GRASP_SHIFT = 0.006   # aim the hand a little toward the fingers (thumb side has the least clearance)
HOLD_OFFSET = np.array([-0.010, 0.0, -0.006])  # typical cube centre in grasp-site frame while held
BIN_EXPECT = {'left': (-0.0035, -0.0024), 'right': (-0.0025, 0.0011)}
COLORS = {'red': 0.0, 'green': 120.0, 'blue': 230.0}


def backproject(dep, pos, mat, fovy):
    H, W = dep.shape
    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    u = np.arange(W) + 0.5 - W / 2
    v = np.arange(H) + 0.5 - H / 2
    U, V = np.meshgrid(u, v)
    P = np.stack([U * dep / f, -V * dep / f, -dep], -1) @ np.asarray(mat).T + np.asarray(pos)
    return P


def hsv(rgb):
    c = rgb.astype(np.float32) / 255
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(-1)
    mn = c.min(-1)
    df = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / df) % 6, np.where(mx == g, (b - r) / df + 2, (r - g) / df + 4)) * 60
    s = df / (mx + 1e-6)
    return h, s, mx


def yaw_est(xy):
    c = xy.mean(0)
    best = None
    for a in np.deg2rad(np.arange(0, 90, 1.0)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = (xy - c) @ R.T
        ext = (np.percentile(q[:, 0], 97) - np.percentile(q[:, 0], 3)) + \
              (np.percentile(q[:, 1], 97) - np.percentile(q[:, 1], 3))
        if best is None or ext < best[0]:
            best = (ext, a)
    return best[1]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper, dtype=float)
        # shoulder roll above ~0.15 makes the upper arm hit the torso
        self.hi[1] = min(self.hi[1], 0.08)
        self.home = np.array(robot.home_arm, dtype=float)
        try:
            self.fovy = {k: float(v['fovy_deg']) for k, v in robot.cameras.items()}
        except Exception:
            self.fovy = {'head': 70.0, 'wrist': 80.0}
        self.target = task['target']
        self.dest = task['destination']
        self.bin = np.array(robot.bin_xy[self.dest], dtype=float)
        self.table = robot.table_height
        self.hs = robot.block_half_size
        self.cubes = {}
        self.calib = None
        self.qcmd = None
        self.closure = 0.0
        self.attempts = 0
        self.start_perceive(0.0)

    # ---------------- kinematics ----------------
    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, p, xdir, q0, iters=60, ydown_only=False):
        q = np.array(q0, dtype=float)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        yt = np.array([0, 0, -1.0])
        ep = er = None
        for _ in range(iters):
            pc, Rc = self.fk(q)
            ep = p - pc
            er = np.cross(Rc[:, 1], yt)
            if not ydown_only:
                er = er + np.cross(Rc[:, 0], xdir)
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.r_[jp[:, self.da], 0.3 * jr[:, self.da]]
            e = np.r_[ep, 0.3 * er]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lo, self.hi)
        return q, np.linalg.norm(ep), np.linalg.norm(er)

    # ---------------- perception ----------------
    def perceive(self, obs):
        cam = obs['cameras']['head']
        dep = np.asarray(cam['depth'], dtype=np.float32)
        P = backproject(dep, cam['pos'], cam['mat'], self.fovy.get('head', 70.0))
        h, s, v = hsv(np.asarray(cam['rgb']))
        z = P[..., 2]
        valid = dep > 0
        if self.calib is None:
            self.calib = self.calibrate(P, h, s, v, valid)
        ztab, off = self.calib
        base = valid & (s > 0.3) & (v > 0.08) & (z > ztab + 0.01) & (z < ztab + 0.09)
        res = {}
        for c, h0 in COLORS.items():
            dh = np.abs((h - h0 + 180) % 360 - 180)
            m = base & (dh < (20 if c == 'red' else 28))
            if m.sum() < 15:
                continue
            pts = P[m].copy()
            pts[:, 2] += self.table - ztab
            pts[:, :2] -= off
            ztop = np.percentile(pts[:, 2], 90)
            top = pts[pts[:, 2] > ztop - 0.012]
            if len(top) < 10:
                top = pts
            xy = top[:, :2].mean(0)
            yaw = yaw_est(top[:, :2])
            res[c] = (xy, yaw, ztop, len(pts))
        return res

    def calibrate(self, P, h, s, v, valid):
        """Table height and xy offset of the head camera measured on the known table and bins."""
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        ztab = self.table
        try:
            tab = valid & (x > 0.15) & (x < 0.32) & \
                (y > -0.4) & (y < 0.0) & (np.abs(z - self.table) < 0.03)
            if tab.sum() > 500:
                ztab = float(np.median(z[tab]))
            offs = []
            for name, (bx, by) in self.robot.bin_xy.items():
                m = valid & (z > ztab + 0.015) & (z < ztab + 0.065) & \
                    (np.abs(x - bx) < 0.09) & (np.abs(y - by) < 0.09)
                if m.sum() > 300:
                    exp = BIN_EXPECT.get(name, (0.0, 0.0))
                    offs.append(P[m][:, :2].mean(0) - np.array([bx, by]) - np.array(exp))
            off = np.mean(offs, 0) if offs else np.zeros(2)
            if np.linalg.norm(off) > 0.04:
                off = np.zeros(2)
        except Exception:
            off = np.zeros(2)
        print('calib ztab', round(ztab, 4), 'off', np.round(off, 4))
        return ztab, off

    # ---------------- planning ----------------
    def sweep_cost(self, xy, a):
        c, s = np.cos(a), np.sin(a)
        xd = np.array([c, s])
        yd = np.array([-s, c])
        pts = []
        for dd in np.linspace(0.035, 0.08, 6):
            for ll in np.linspace(-0.045, 0.045, 5):
                pts.append(xy + dd * xd + ll * yd)
        for dd in np.linspace(-0.07, -0.03, 4):
            for ll in np.linspace(-0.015, 0.015, 3):
                pts.append(xy + dd * xd + ll * yd)
        pts = np.array(pts)
        n = 0
        for b in self.robot.bin_xy.values():
            lim = self.robot.bin_inner_half_size + self.robot.bin_wall_thickness + 0.008
            n += np.sum((np.abs(pts[:, 0] - b[0]) < lim) & (np.abs(pts[:, 1] - b[1]) < lim))
        for cc, (oxy, _, _, _) in self.cubes.items():
            if cc == self.target:
                continue
            n += np.sum(np.linalg.norm(pts - oxy, axis=1) < 0.047)
        return int(n)

    def choose_grasp(self, xy, yaw, zg):
        best = None
        seeds = [self.home,
                 np.array([0.44, 0.0, -0.26, -0.33, -1.46, -0.36, -0.07]),
                 np.array([0.2, -0.56, -0.25, 0.12, -1.29, -1.1, -0.35]),
                 np.array([-0.34, 0.53, -0.39, 0.23, -1.62, 1.43, -0.55]),
                 np.array([0.31, 0.31, 0.06, -0.23, -1.83, 0.69, -0.07])]
        for k in range(4):
            a = wrap(yaw + k * np.pi / 2)
            xdir = np.array([np.cos(a), np.sin(a), 0.0])
            bq = None
            for q0 in seeds:
                qg, ep, er = self.ik(np.array([xy[0], xy[1], zg]), xdir, q0, iters=150)
                c0 = 10 * ep + er
                if bq is None or c0 < bq[0]:
                    bq = (c0, qg, ep, er)
                if ep < 1e-3 and er < 0.02:
                    break
            _, qg, ep, er = bq
            qp, ep2, er2 = self.ik(np.array([xy[0], xy[1], zg + 0.10]), xdir, qg, iters=150)
            ncol = self.sweep_cost(xy, a)
            margin = np.min(np.minimum(qg - self.lo, self.hi - qg))
            cost = 10 * (ep + ep2) + (er + er2) + 0.05 * np.linalg.norm(qp - self.home) + 0.06 * ncol
            cost += 1.0 * max(0.0, 0.15 - margin)
            if np.cos(a) < -0.2:
                cost += 2.0
            if ep + ep2 > 0.006 or er + er2 > 0.25:
                cost += 5
            print('  cand a', round(np.rad2deg(a)), 'ep', round(ep + ep2, 4), 'er', round(er + er2, 3), 'ncol', ncol, 'cost', round(cost, 3))
            if best is None or cost < best[0]:
                best = (cost, a, xdir, qg, qp)
        return best

    def plan_place(self, q_lift):
        goal = np.array([self.bin[0], self.bin[1], self.table + 0.13])
        best = None
        for a in np.deg2rad(np.arange(-180, 180, 30)):
            xdir = np.array([np.cos(a), np.sin(a), 0.0])
            for q0 in (q_lift, self.home):
                q, ep, er = self.ik(goal, xdir, q0, iters=120)
                margin = np.min(np.minimum(q - self.lo, self.hi - q))
                cost = 20 * ep + 2 * er + 0.1 * np.linalg.norm(q - q_lift) + 0.5 * max(0.0, 0.25 - margin)
                if best is None or cost < best[0]:
                    best = (cost, q, ep, er)
        print('place cost', round(best[0], 3), 'ep', round(best[2], 4), 'er', round(best[3], 4))
        return best[1]

    def plan_path(self, q_lift, shift=None):
        p0, _ = self.fk(q_lift)
        p0 = p0.copy()
        zt = max(p0[2], self.table + 0.165)
        bxy = self.bin.copy() if shift is None else self.bin + shift
        p1 = np.array([bxy[0], bxy[1], zt])
        p2 = np.array([bxy[0], bxy[1], self.table + 0.12])
        wps = [p0 + (np.array([p0[0], p0[1], zt]) - p0) * k / 2 for k in range(1, 3)]
        wps += [np.array([p0[0], p0[1], zt]) + (p1 - np.array([p0[0], p0[1], zt])) * k / 10 for k in range(1, 11)]
        wps += [p1 + (p2 - p1) * k / 3 for k in range(1, 4)]
        path = [q_lift.copy()]
        q = q_lift.copy()
        ep = 0
        for w in wps:
            q, ep, er = self.ik(w, self.xdir, q, iters=80, ydown_only=True)
            path.append(q.copy())
        print('path final ep', round(ep, 4))
        if shift is None and ep <= 0.01:
            _, Rf = self.fk(path[-1])
            sh = -(Rf @ HOLD_OFFSET)[:2]
            path2 = self.plan_path(q_lift, shift=sh)
            pf, _ = self.fk(path2[-1])
            if np.linalg.norm(pf[:2] - (self.bin + sh)) < 0.01:
                return path2
            return path
        if ep > 0.01:
            qp = self.plan_place(q_lift)
            qv, _, _ = self.ik((p0 + p2) / 2 + np.array([0, 0, 0.1]), self.xdir, qp, iters=100, ydown_only=True)
            path = [q_lift, qv, qp]
        return path

    def step_towards(self, qdes, maxstep=0.07):
        dq = np.clip(qdes - self.qcmd, -maxstep, maxstep)
        self.qcmd = self.qcmd + dq
        return np.max(np.abs(qdes - self.qcmd))

    def out(self, closure):
        self.closure = closure
        return [float(v) for v in self.qcmd] + [float(closure)]

    def set_phase(self, p, t):
        self.phase = p
        self.t0 = t

    def start_perceive(self, t):
        self.obs_acc = {c: [] for c in COLORS}
        self.nobs = 0
        self.set_phase('perceive', t)

    def target_on_table(self, obs):
        res = self.perceive(obs)
        if self.target in res:
            xy, yaw, ztop, n = res[self.target]
            for b in self.robot.bin_xy.values():
                if np.all(np.abs(xy - np.array(b)) < self.robot.bin_inner_half_size + 0.01):
                    return False, res
            if ztop < self.table + 0.075 and n > 30:
                return True, res
        return False, res

    def act(self, obs):
        t = obs['t']
        if self.qcmd is None:
            self.qcmd = np.array(obs['arm_target'], dtype=float)
        q_meas = np.array(obs['qpos'][:7])
        hq = np.array(obs['qpos'][7:])
        ph = self.phase

        if ph == 'perceive':
            res = self.perceive(obs)
            for c, r in res.items():
                self.obs_acc[c].append(r)
            self.nobs += 1
            if self.nobs >= 3:
                self.cubes = {}
                for c, lst in self.obs_acc.items():
                    if lst:
                        xy = np.median([r[0] for r in lst], 0)
                        self.cubes[c] = (xy, lst[-1][1], np.median([r[2] for r in lst]), lst[-1][3])
                print(round(t, 2), 'cubes', {c: (np.round(v[0], 3), round(np.rad2deg(v[1]), 1)) for c, v in self.cubes.items()})
                if self.target not in self.cubes:
                    self.start_perceive(t)
                    return self.out(0.0)
                xy, yaw, _, _ = self.cubes[self.target]
                self.zg = self.table + self.hs + 0.003
                cost, a, xdir, qg, qp = self.choose_grasp(xy, yaw, self.zg)
                print('grasp a', round(np.rad2deg(a), 1), 'cost', round(cost, 3))
                self.attempts += 1
                self.gxy = xy + GRASP_SHIFT * xdir[:2]
                self.xdir = xdir
                self.qg = qg
                self.qp = qp
                self.set_phase('pregrasp', t)
            return self.out(0.0)

        if ph == 'pregrasp':
            rem = self.step_towards(self.qp)
            if rem < 1e-3 and (np.max(np.abs(q_meas - self.qp)) < 0.03 or t - self.t0 > 4.0):
                self.set_phase('descend', t)
            return self.out(min(PRECURL, self.closure + 0.05))

        if ph == 'descend':
            s = min(1.0, (t - self.t0) / 1.0)
            z = self.zg + 0.10 * (1 - s)
            q, _, _ = self.ik(np.array([self.gxy[0], self.gxy[1], z]), self.xdir, self.qcmd)
            self.step_towards(q)
            if s >= 1.0 and (np.max(np.abs(q_meas - self.qcmd)) < 0.02 or t - self.t0 > 2.0):
                self.set_phase('close', t)
            return self.out(PRECURL)

        if ph == 'close':
            s = min(1.0, (t - self.t0) / 0.7)
            if t - self.t0 > 0.9:
                self.set_phase('lift', t)
            return self.out(PRECURL + (1.0 - PRECURL) * s)

        if ph == 'lift':
            s = min(1.0, (t - self.t0) / 1.0)
            z = self.zg + 0.14 * s
            q, _, _ = self.ik(np.array([self.gxy[0], self.gxy[1], z]), self.xdir, self.qcmd)
            self.step_towards(q)
            if s >= 1.0 and t - self.t0 > 1.2:
                fail, res = self.target_on_table(obs)
                empty = hq[1] < -0.2 and hq[2] < -0.3
                print(round(t, 2), 'lifted: hand q', np.round(hq, 2), 'on table', fail, 'empty', empty)
                fail = fail or empty
                if fail:
                    self.set_phase('abort', t)
                else:
                    self.q_lift = self.qcmd.copy()
                    self.path = self.plan_path(self.q_lift)
                    self.set_phase('transfer', t)
            return self.out(1.0)

        if ph == 'abort':
            if t - self.t0 > 0.6:
                self.start_perceive(t)
            return self.out(0.0)

        if ph == 'transfer':
            T = 2.4
            s = min(1.0, (t - self.t0) / T)
            s2 = 0.5 - 0.5 * np.cos(np.pi * s)
            n = len(self.path) - 1
            x = s2 * n
            i = min(int(x), n - 1)
            q = self.path[i] + (self.path[i + 1] - self.path[i]) * (x - i)
            self.step_towards(q)
            if s >= 1.0 and (np.max(np.abs(q_meas - self.qcmd)) < 0.02 or t - self.t0 > T + 1.5):
                p_meas = obs['grasp_site_pos']
                print(round(t, 2), 'at release site', np.round(p_meas, 3))
                self.set_phase('release', t)
            return self.out(1.0)

        if ph == 'release':
            s = min(1.0, (t - self.t0) / 0.5)
            if t - self.t0 > 0.8:
                self.set_phase('retreat', t)
            return self.out(1.0 - s)

        if ph == 'retreat':
            if t - self.t0 < 0.05:
                p, _ = self.fk(self.qcmd)
                self.q_ret, _, _ = self.ik(p + np.array([0, 0, 0.06]), self.xdir, self.qcmd, ydown_only=True, iters=60)
            if t - self.t0 < 1.0:
                self.step_towards(self.q_ret, 0.03)
            elif t - self.t0 > 1.5:
                fail, res = self.target_on_table(obs)
                print(round(t, 2), 'after release on table', fail)
                if fail:
                    self.start_perceive(t)
                else:
                    self.set_phase('done', t)
            return self.out(0.0)
        return self.out(0.0)
