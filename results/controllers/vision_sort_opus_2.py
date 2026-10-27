
import numpy as np
import mujoco

HALF = 0.025
PRECURL = 0.25
GRASP_DX = 0.004


def deproject(depth, pos, mat, fovy):
    H, W = depth.shape
    f = (H / 2) / np.tan(np.radians(fovy) / 2)
    u = np.arange(W) + 0.5 - W / 2
    v = -(np.arange(H) + 0.5 - H / 2)
    U, V = np.meshgrid(u, v)
    pc = np.stack([U * depth / f, V * depth / f, -depth], -1)
    return pc @ np.asarray(mat).T + np.asarray(pos)


def color_masks(rgb):
    c = rgb.astype(np.float32)
    s = c.sum(-1) + 1e-6
    rn, gn, bn = c[..., 0] / s, c[..., 1] / s, c[..., 2] / s
    bright = s > 40
    return {'red': (rn > 0.55) & (gn < 0.25) & bright,
            'green': (gn > 0.45) & (rn < 0.3) & bright,
            'blue': (bn > 0.5) & (rn < 0.25) & bright}


def fit_square(pts):
    """pts: Nx3 world points of a cube. returns (cxy, ztop, yaw mod 90deg, n)"""
    zt = np.percentile(pts[:, 2], 85)
    top = pts[pts[:, 2] > zt - 0.008]
    xy = top[:, :2]
    best = None
    for a in np.radians(np.arange(0, 90, 2.0)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = xy @ R.T
        lo = np.percentile(q, 2, axis=0)
        hi = np.percentile(q, 98, axis=0)
        ext = (hi - lo).sum()
        if best is None or ext < best[0]:
            best = (ext, a, R, lo, hi)
    _, a, R, lo, hi = best
    cen = (lo + hi) / 2
    cxy = cen @ R
    return np.array([cxy[0], cxy[1]]), zt, a, len(top)


def find_cubes(cam, fovy, table=0.78, minpts=15):
    P = deproject(cam['depth'], cam['pos'], cam['mat'], fovy)
    h = P[..., 2] - table
    valid = (cam['depth'] > 0) & (h > 0.004) & (h < 0.08) & (P[..., 0] > -0.1) & (P[..., 0] < 0.8) \
        & (P[..., 1] > -0.8) & (P[..., 1] < 0.5)
    out = {}
    for k, mk in color_masks(cam['rgb']).items():
        pts = P[mk & valid]
        if len(pts) < minpts:
            out[k] = None
            continue
        med = np.median(pts[:, :2], axis=0)
        dd = np.linalg.norm(pts[:, :2] - med, axis=1)
        pts = pts[dd < 0.05]
        if len(pts) < minpts:
            out[k] = None
            continue
        out[k] = fit_square(pts)
    return out


def rot_palm_down(psi):
    x = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def wrapa(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.fov = {c: robot.cameras[c]['fovy_deg'] for c in robot.cameras}
        self.target = task['target']
        self.dest = task['destination']
        self.bin = np.array(robot.bin_xy[self.dest], dtype=float)
        self.table = robot.table_height
        self.phase = 'look'
        self.t_phase = 0.0
        self.looks = []
        self.qcmd = None
        self.cart = None
        self.psi_cmd = 0.0
        self.closure = 0.0
        self.cube = None
        self.others = []
        self.attempts = 0
        self.failed = []
        self.path = None
        arm_pref = ('right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow', 'right_wrist', 'right_hand')
        self.is_arm = np.array([self.m.body(i).name.startswith(arm_pref) for i in range(self.m.nbody)])
        self.grasp_dz = 0.006
        self.place = np.array([self.bin[0], self.bin[1], self.table + HALF + 0.035])
        self.psi_place = None
        rng = np.random.default_rng(0)
        self.rand_seeds = [self.lo + (self.hi - self.lo) * (0.2 + 0.6 * rng.random(7)) for _ in range(6)]
        self.walls = []
        hs = robot.bin_inner_half_size + robot.bin_wall_thickness / 2
        for b in robot.bin_xy.values():
            for s_ in np.linspace(-hs, hs, 9):
                self.walls += [np.array([b[0] + s_, b[1] - hs]), np.array([b[0] + s_, b[1] + hs]),
                               np.array([b[0] - hs, b[1] + s_]), np.array([b[0] + hs, b[1] + s_])]

    # ---------------- kinematics
    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pt, Rt, iters=15, wrot=0.6):
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pt - p
            er = 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))
            e = np.concatenate([ep, wrot * er])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], wrot * jr[:, self.da]])
            lam = 0.02
            JJ = J @ J.T + lam ** 2 * np.eye(6)
            Jp = J.T @ np.linalg.inv(JJ)
            dq = Jp @ e
            N = np.eye(7) - Jp @ J
            dq += N @ (0.05 * (self.home - q))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        return q

    def limit_cost(self, q):
        m = np.minimum(q - self.lo, self.hi - q)
        return float(np.sum(np.maximum(0.0, 0.25 - m) ** 2) * 50)

    def solve(self, pos, psi, seeds=(), qref=None):
        Rt = rot_palm_down(psi)
        allseeds = list(seeds) + [self.home] + self.rand_seeds
        best = None
        for s0 in allseeds:
            q = self.ik(np.array(s0, dtype=float), pos, Rt, iters=40)
            p, R = self.fk(q)
            err = np.linalg.norm(p - pos) + 0.05 * np.linalg.norm(R - Rt)
            sc = err * 200 + self.limit_cost(q) + self.collision_cost(q)
            if qref is not None:
                sc += 0.05 * np.linalg.norm(q - qref)
            if best is None or sc < best[0]:
                best = (sc, q, err)
        return best[1], best[2]

    def jmove(self, obs, qgoal, vmax=0.05):
        dq = qgoal - self.qcmd
        s = np.max(np.abs(dq)) / vmax
        self.qcmd = self.qcmd + dq / max(1.0, s)
        p, R = self.fk(self.qcmd)
        self.cart = p
        self.psi_cmd = float(np.arctan2(R[1, 0], R[0, 0]))
        return float(np.max(np.abs(np.asarray(obs['qpos'])[:7] - qgoal)))

    # ---------------- helpers
    def goto(self, obs, pos, psi, speed=0.3, wspeed=1.2):
        dpsi = wrapa(psi - self.psi_cmd)
        ws = wspeed * self.r.control_dt
        self.psi_cmd += float(np.clip(dpsi, -ws, ws))
        R = rot_palm_down(self.psi_cmd)
        d = pos - self.cart
        n = np.linalg.norm(d)
        stepl = speed * self.r.control_dt
        if n > stepl:
            self.cart = self.cart + d * stepl / n
        else:
            self.cart = pos.copy()
        self.qcmd = self.ik(self.qcmd, self.cart, R)
        return np.linalg.norm(np.asarray(obs['grasp_site_pos']) - pos)

    def set_phase(self, p, t):
        self.phase = p
        self.t_phase = t
        print(f"t={t:.2f} phase {p}")

    def clearance_cost(self, psi, gp):
        c = 0.0
        x = np.array([np.cos(psi), np.sin(psi)])
        z = np.array([-np.sin(psi), np.cos(psi)])
        obst = [(o, 0.025) for o in self.others] + [(w, 0.006) for w in self.walls]
        for o, r in obst:
            dv = o - gp[:2]
            ax, az = dv @ x, dv @ z
            wl = 0.05 + r       # lateral half width of hand + obstacle
            fx = 0.10 + r       # finger reach (+x) with pre-curl
            tx = 0.07 + r       # thumb reach (-x)
            if abs(az) < wl:
                if 0 < ax < fx:
                    c += min(fx - ax, 0.03) * (wl - abs(az)) * 1500
                elif -tx < ax <= 0:
                    c += min(tx + ax, 0.03) * (wl - abs(az)) * 1500
        return c

    def choose_psi(self, yaw, pos):
        cands = [yaw + k * np.pi / 2 for k in range(-4, 5)]
        cands = [a for a in cands if -2.2 < a < 1.9]
        pref = -0.35
        sols = {}

        def cost(a):
            qp, e1 = self.solve(pos + np.array([0, 0, 0.12]), a, seeds=[self.qcmd])
            qg, e2 = self.solve(pos, a, seeds=[qp], qref=qp)
            sols[a] = (qp, qg)
            fp = sum(4.0 for f in self.failed if abs((a - f + np.pi) % (2 * np.pi) - np.pi) < 0.45)
            return fp + (e1 + e2) * 100 + 0.3 * abs(a - pref) + self.clearance_cost(a, pos) \
                + self.collision_cost(qp) + self.collision_cost(qg) \
                + self.limit_cost(qp) + self.limit_cost(qg) + 0.2 * np.linalg.norm(qg - qp)
        costs = [(cost(a), a) for a in cands]
        print('psi costs', [(round(float(np.degrees(a))), round(float(c), 2)) for c, a in costs])
        a = min(costs)[1]
        self.q_pre, self.q_grasp = sols[a]
        return a

    def choose_place_psi(self, pos):
        cands = np.radians(np.arange(-70, 41, 10))
        pref = -0.35
        best = None
        for a in cands:
            qa, e1 = self.solve(np.array([pos[0], pos[1], 0.96]), a, seeds=[self.home])
            ql, e2 = self.solve(pos, a, seeds=[qa], qref=qa)
            c = (e1 + e2) * 100 + 0.2 * abs(a - pref) + self.limit_cost(qa) + self.limit_cost(ql) \
                + self.collision_cost(qa) + self.collision_cost(ql) \
                + 0.2 * np.linalg.norm(ql - qa)
            if best is None or c < best[0]:
                best = (c, float(a), qa, ql)
        self.q_above = best[2]
        print('place psi', np.degrees(best[1]), 'cost', best[0])
        return best[1]

    def held(self, obs):
        if 'cameras' not in obs:
            return True
        cam = obs['cameras']['wrist']
        mk = color_masks(cam['rgb'])[self.target]
        dep = cam['depth']
        near = int((mk & (dep > 0) & (dep < 0.13)).sum())
        thumb = float(np.asarray(obs['qpos'])[8])
        res = find_cubes(obs['cameras']['head'], self.fov['head'], self.table)
        on_table = res.get(self.target) is not None
        ok = near > 800 or (thumb > -0.3 and not on_table)
        print(f"held check: near={near} thumb={thumb:.2f} on_table={on_table} -> {ok}")
        return ok

    def refine(self, obs):
        """refine target cube pose with the wrist camera (close range)."""
        try:
            cam = obs['cameras']['wrist']
            r = find_cubes(cam, self.fov['wrist'], self.table).get(self.target)
        except Exception as e:
            print('refine error', e)
            return
        if r is None or r[3] < 300:
            print('refine: not seen')
            return
        cxy, cz, yaw = self.cube
        dxy = r[0] - cxy
        if np.linalg.norm(dxy) > 0.03:
            print('refine: inconsistent', dxy)
            return
        dyaw = (r[2] - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        if abs(dyaw) < np.radians(12):
            self.psi += dyaw
            yaw = yaw + dyaw
        self.cube = (r[0].copy(), cz, yaw)
        print('refine: dxy', np.round(dxy, 4), 'dyaw', round(float(np.degrees(dyaw)), 1))

    def collision_cost(self, q):
        d = self.d
        d.qpos[self.qa] = q
        mujoco.mj_fwdPosition(self.m, d)
        c = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            a1 = self.is_arm[self.m.geom_bodyid[con.geom1]]
            a2 = self.is_arm[self.m.geom_bodyid[con.geom2]]
            if a1 != a2 and con.dist < 0.01:
                c += (0.01 - con.dist)
        return c * 300

    def start_path(self, q1, p1, psi1):
        p0, R0 = self.fk(self.qcmd)
        self.path = dict(q0=self.qcmd.copy(), q1=np.array(q1), p0=p0, p1=np.array(p1),
                         psi0=float(np.arctan2(R0[1, 0], R0[0, 0])), psi1=float(psi1), s=0.0)
        P = self.path
        L = np.linalg.norm(P['p1'] - P['p0']) / 0.3 + abs(wrapa(P['psi1'] - P['psi0'])) / 1.5
        J = np.max(np.abs(P['q1'] - P['q0'])) / 1.4
        P['T'] = max(L, J, 0.2)

    def path_step(self):
        P = self.path
        P['s'] = min(1.0, P['s'] + self.r.control_dt / P['T'])
        s = P['s']
        s2 = 0.5 - 0.5 * np.cos(np.pi * s)  # smooth
        ql = P['q0'] + s2 * (P['q1'] - P['q0'])
        p = P['p0'] + s2 * (P['p1'] - P['p0'])
        psi = P['psi0'] + s2 * wrapa(P['psi1'] - P['psi0'])
        q = self.ik(ql, p, rot_palm_down(psi), iters=12)
        dq = q - self.qcmd
        self.qcmd = self.qcmd + np.clip(dq, -0.075, 0.075)
        self.cart = p
        self.psi_cmd = psi
        return s >= 1.0

    # ---------------- main
    def act(self, obs):
        t = obs['t']
        if self.qcmd is None:
            self.qcmd = np.array(obs['arm_target'], dtype=float)
            p, R = self.fk(self.qcmd)
            self.cart = p
            self.psi_cmd = float(np.arctan2(R[1, 0], R[0, 0]))
        if self.psi_place is None:
            self.psi_place = self.choose_place_psi(self.place)
        tp = t - self.t_phase
        ph = self.phase

        if ph == 'look':
            if self.attempts > 0:
                self.closure = 0.0
                if tp < 0.4:
                    self.goto(obs, self.cart + np.array([0, 0, 0.02]), self.psi_cmd, 0.2)
                    return list(self.qcmd) + [0.0]
                jd = self.jmove(obs, self.home)
                if jd > 0.05 and tp < 3.0:
                    return list(self.qcmd) + [0.0]
            res = find_cubes(obs['cameras']['head'], self.fov['head'], self.table)
            if res.get(self.target) is not None:
                self.looks.append(res)
            if len(self.looks) >= 3:
                tl = [l[self.target] for l in self.looks]
                cxy = np.median([l[0] for l in tl], axis=0)
                zt = np.median([l[1] for l in tl])
                yaw = tl[-1][2]
                self.others = []
                for k in ['red', 'green', 'blue']:
                    if k != self.target and self.looks[-1].get(k) is not None:
                        self.others.append(self.looks[-1][k][0])
                cz = max(zt - HALF, self.table + HALF - 0.003)
                self.cube = (cxy, cz, yaw)
                gp = np.array([cxy[0], cxy[1], cz + self.grasp_dz])
                self.psi = self.choose_psi(yaw, gp)
                print('cube', self.target, cxy, zt, 'yaw', np.degrees(yaw), 'psi', np.degrees(self.psi),
                      'psi_place', np.degrees(self.psi_place))
                self.looks = []
                self.set_phase('pre', t)
            elif tp > 4.0:
                print('target not found')
                self.looks = []
                self.set_phase('look', t)
            return list(self.qcmd) + [0.0]

        cxy, cz, yaw = self.cube
        grasp = np.array([cxy[0], cxy[1], cz + self.grasp_dz])
        grasp[:2] -= GRASP_DX * np.array([np.cos(self.psi), np.sin(self.psi)])
        if ph == 'pre':
            tgt = grasp + [0, 0, 0.12]
            if tp < 1e-6 or self.path is None:
                self.start_path(self.q_pre, tgt, self.psi)
            done = self.path_step()
            dist = np.linalg.norm(np.asarray(obs['grasp_site_pos']) - tgt)
            self.closure = min(PRECURL, tp * 0.5)
            if (done and dist < 0.01) or tp > 5.0:
                self.path = None
                self.refine(obs)
                self.set_phase('descend', t)
        elif ph == 'descend':
            dist = self.goto(obs, grasp, self.psi, 0.12)
            self.closure = PRECURL
            if dist < 0.006 or tp > 3.0:
                self.set_phase('close', t)
        elif ph == 'close':
            self.goto(obs, grasp, self.psi, 0.15)
            self.closure = min(1.0, PRECURL + (1 - PRECURL) * tp / 0.7)
            if tp > 1.2:
                self.set_phase('lift', t)
        elif ph == 'lift':
            tgt = grasp + [0, 0, 0.15]
            tgt[2] = max(tgt[2], 0.96)
            dist = self.goto(obs, tgt, self.psi, 0.25)
            if dist < 0.02 or tp > 3.0:
                if not self.held(obs) and self.attempts < 4:
                    self.attempts += 1
                    self.failed.append(self.psi)
                    self.set_phase('look', t)
                else:
                    self.path = None
                    self.set_phase('carry', t)
        elif ph == 'carry':
            tgt = np.array([self.bin[0], self.bin[1], 0.96])
            if self.path is None:
                self.start_path(self.q_above, tgt, self.psi_place)
            done = self.path_step()
            dist = np.linalg.norm(np.asarray(obs['grasp_site_pos']) - tgt)
            if (done and dist < 0.015) or tp > 5.0:
                self.path = None
                if not self.held(obs) and self.attempts < 4:
                    self.attempts += 1
                    self.set_phase('look', t)
                else:
                    self.set_phase('lower', t)
        elif ph == 'lower':
            dist = self.goto(obs, self.place, self.psi_place, 0.15)
            if dist < 0.01 or tp > 3.0:
                self.set_phase('release', t)
        elif ph == 'release':
            self.goto(obs, self.place, self.psi_place, 0.15)
            self.closure = max(0.0, 1.0 - tp / 0.5)
            if tp > 0.8:
                self.set_phase('retreat', t)
        elif ph == 'retreat':
            tgt = np.array([self.bin[0], self.bin[1], 1.0])
            self.goto(obs, tgt, self.psi_place, 0.2)
            self.closure = 0.0
        return list(self.qcmd) + [float(self.closure)]
