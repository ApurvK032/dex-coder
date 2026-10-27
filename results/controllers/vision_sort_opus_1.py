
import numpy as np
import mujoco

COLORS = ("red", "green", "blue")


def backproject(dep, pos, mat, fovy):
    H, W = dep.shape
    f = (H / 2) / np.tan(np.deg2rad(fovy) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u + 0.5 - W / 2) * dep / f
    y = -(v + 0.5 - H / 2) * dep / f
    P = np.asarray(pos) + np.stack([x, y, -dep], -1) @ np.asarray(mat).T
    P[dep <= 0] = np.nan
    return P


def colmasks(rgb):
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / (mx + 1)
    ok = (sat > 0.35) & (mx > 50)
    return {
        "red": ok & (r >= mx) & (r > 1.4 * g) & (r > 1.4 * b),
        "green": ok & (g >= mx) & (g > 1.25 * r) & (g > 1.25 * b),
        "blue": ok & (b >= mx) & (b > 1.25 * r) & (b > 1.25 * g),
    }


def estimate_cube(P, mask, table=0.78, hs=0.025, min_pts=15):
    z = P[..., 2]
    with np.errstate(invalid="ignore"):
        m = mask & (z > table + 0.01) & (z < table + 0.12)
    pts = P[m]
    if len(pts) < min_pts:
        return None
    zt = np.percentile(pts[:, 2], 90)
    top = pts[np.abs(pts[:, 2] - zt) < 0.007][:, :2]
    if len(top) < 8:
        return None
    med = np.median(top, 0)
    top = top[np.linalg.norm(top - med, axis=1) < 0.05]
    if len(top) < 8:
        return None
    best = None
    for a in np.deg2rad(np.arange(0, 90, 2.0)):
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = top @ R.T
        lo = np.percentile(q, 2, 0)
        hi = np.percentile(q, 98, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, a, (lo + hi) / 2, R)
    area, a, cq, R = best
    ctr = R.T @ cq
    return dict(pos=np.array([ctr[0], ctr[1], zt - hs]), yaw=a, n=len(top))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    GRASP_FWD = 0.0    # site ahead of cube center along fingers
    GRASP_DZ = 0.015   # site above cube center
    PRE_DZ = 0.08

    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = self.m.site(robot.grasp_site).id
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.d.qpos[self.qa] = self.home
        mujoco.mj_forward(self.m, self.d)
        self.R_home = self.d.site_xmat[self.sid].reshape(3, 3).copy()
        self.home_heading = np.arctan2(self.R_home[1, 0], self.R_home[0, 0])
        self.home_pos = self.d.site_xpos[self.sid].copy()
        self.q_ik = self.home.copy()
        self.th = robot.table_height
        self.hs = robot.block_half_size
        self.fovy = {"head": float(self.m.cam_fovy[self.m.cam("head").id]),
                     "wrist": float(self.m.cam_fovy[self.m.cam("wrist").id])}
        self.phase = "perceive"
        self.phase_t = 0.0
        self.cmd_pos = None
        self.cmd_psi = 0.0
        self.closure = 0.0
        self.target = task["target"]
        self.dest = task["destination"]
        self.cubes = {}
        self.t = 0.0
        self.bias = np.zeros(3)
        self.attempts = 0
        self.checked = False
        self._col_setup()

    # ---------------- kinematics ----------------
    def _col_setup(self):
        m = self.m

        def gid(body):
            b = m.body(body).id
            return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.colA = []
        for b in ("right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link"):
            self.colA += gid(b)
        self.colT = gid("torso_link") + gid("pelvis")
        self.ft = np.zeros(6)
        self.jc = np.zeros((3, m.nv))

    DSAFE = 0.03
    ROLL_W = 0.15  # weight of rotation about the finger axis (soft)

    def ik(self, p_des, R_des, iters=12, q0=None, store=True):
        q = (self.q_ik if q0 is None else q0).copy()
        m, d = self.m, self.d
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        DS = self.DSAFE
        for _ in range(iters):
            d.qpos[self.qa] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.sid]
            R = d.site_xmat[self.sid].reshape(3, 3)
            ep = p_des - p
            Re = R_des @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            mujoco.mj_jacSite(m, d, jp, jr, self.sid)
            # rotation rows expressed in the desired frame; finger-axis rotation weighted low
            W = np.diag([0.5 * self.ROLL_W, 0.5, 0.5]) @ R_des.T
            rows = [jp[:, self.da], W @ jr[:, self.da]]
            es = [ep, W @ er]
            for a in self.colA:
                for tg in self.colT:
                    dist = mujoco.mj_geomDistance(m, d, a, tg, DS, self.ft)
                    if dist < DS:
                        pa = self.ft[:3].copy()
                        n = pa - self.ft[3:]
                        nn = np.linalg.norm(n)
                        if nn < 1e-7:
                            continue
                        n /= nn
                        if dist < 0:
                            n = -n
                        mujoco.mj_jac(m, d, self.jc, None, pa, m.geom_bodyid[a])
                        rows.append(3.0 * (n @ self.jc[:, self.da])[None])
                        es.append(np.array([3.0 * (DS - dist)]))
            J = np.vstack(rows)
            e = np.concatenate(es)
            JJ = J @ J.T + 1e-3 * np.eye(len(e))
            dq = J.T @ np.linalg.solve(JJ, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
            dq += N @ (0.03 * (self.home - q))
            q = np.clip(q + dq, self.lo, self.hi)
        if store:
            self.q_ik = q
        return q

    def ik_err(self, p_des, R_des, q0):
        q = q0.copy()
        for _ in range(6):
            q = self.ik(p_des, R_des, iters=12, q0=q, store=False)
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        p = self.d.site_xpos[self.sid]
        R = self.d.site_xmat[self.sid].reshape(3, 3)
        ax = np.arccos(np.clip(R[:, 0] @ R_des[:, 0], -1, 1))
        ay = np.arccos(np.clip(R[:, 1] @ R_des[:, 1], -1, 1))
        return np.linalg.norm(p - p_des) + 0.05 * ax + 0.01 * max(0.0, ay - 0.3), q

    # ---------------- perception ----------------
    def perceive_head(self, obs):
        cam = obs["cameras"]["head"]
        P = backproject(cam["depth"], cam["pos"], cam["mat"], self.fovy["head"])
        M = colmasks(cam["rgb"])
        out = {}
        for c in COLORS:
            e = estimate_cube(P, M[c], self.th, self.hs)
            if e is not None:
                out[c] = e
        return out

    def plan_grasp(self, tc):
        """Choose hand heading (a cube face direction) by reachability, rotation and clearance."""
        best = None
        for k in range(4):
            h = wrap(tc["yaw"] + k * np.pi / 2)
            psi = wrap(h - self.home_heading)
            if abs(psi) > np.deg2rad(135):
                continue
            R = rotz(psi) @ self.R_home
            fx = np.array([np.cos(h), np.sin(h), 0.0])
            gp = tc["pos"] + self.GRASP_FWD * fx + np.array([0, 0, self.GRASP_DZ])
            e1, q1 = self.ik_err(gp + np.array([0, 0, self.PRE_DZ]), R, self.home)
            e2, _ = self.ik_err(gp, R, q1)
            cost = abs(psi) * 0.02 + 5 * max(0, e1 - 0.003) + 5 * max(0, e2 - 0.003)
            # clearance from other cubes (fingers sweep in front, thumb behind)
            for c, oc in self.cubes.items():
                if c == self.target:
                    continue
                dv = oc["pos"][:2] - gp[:2]
                ax = dv @ fx[:2]
                lat = dv @ np.array([-fx[1], fx[0]])
                ox = max(0.0, abs(ax - 0.0) - 0.085)
                oy = max(0.0, abs(lat) - 0.065)
                gap = np.hypot(ox, oy)
                cost += 0.5 * max(0.0, 0.03 - gap)
            # fingers sweep up to ~13 cm ahead: keep them away from bin walls
            lat_v = np.array([-fx[1], fx[0]])
            pen = 0.0
            for bx, by in self.robot.bin_xy.values():
                hb = self.robot.bin_inner_half_size + self.robot.bin_wall_thickness + 0.012
                for sx in np.linspace(0.03, 0.14, 8):
                    for sl in (-0.04, 0.0, 0.04):
                        pt = gp[:2] + sx * fx[:2] + sl * lat_v
                        ox = hb - abs(pt[0] - bx)
                        oy = hb - abs(pt[1] - by)
                        if ox > 0 and oy > 0:
                            pen = max(pen, min(ox, oy))
            cost += 3.0 * pen
            if best is None or cost < best[0]:
                best = (cost, psi, gp, h, e1, e2)
        return best

    # ---------------- control ----------------
    def set_phase(self, p):
        self.phase = p
        self.phase_t = self.t
        print(f"t={self.t:.2f} phase {p}")

    def move_cmd(self, goal, speed):
        dp = goal - self.cmd_pos
        n = np.linalg.norm(dp)
        step = speed * 0.04
        if n <= step:
            self.cmd_pos = goal.copy()
            return True
        self.cmd_pos = self.cmd_pos + dp / n * step
        return False

    def turn_cmd(self, psi, rate=0.05):
        dpsi = wrap(psi - self.cmd_psi)
        self.cmd_psi += np.clip(dpsi, -rate, rate)
        return abs(dpsi) <= rate

    def act(self, obs):
        self.t = float(obs["t"])
        if self.cmd_pos is None:
            self.cmd_pos = np.array(obs["grasp_site_pos"], dtype=float)
            self.prev_cmd = self.cmd_pos.copy()
            self.q_ik = np.array(obs["qpos"][:7], dtype=float)
        site = np.array(obs["grasp_site_pos"])
        hq = np.array(obs["qpos"][7:])
        el = self.t - self.phase_t
        ph = self.phase
        if ph == "perceive":
            self.cubes = self.perceive_head(obs)
            print("cubes", {c: (np.round(v["pos"], 3).tolist(), round(float(np.rad2deg(v["yaw"])), 1)) for c, v in self.cubes.items()})
            if self.target in self.cubes:
                tc = self.cubes[self.target]
                cost, self.gpsi, self.gpos, h, e1, e2 = self.plan_grasp(tc)
                print(f"plan psi={np.rad2deg(self.gpsi):.0f} cost={cost:.3f} e={e1:.4f},{e2:.4f}")
                self.attempts += 1
                self.set_phase("approach")
            else:
                self.move_cmd(self.home_pos, 0.3)
                self.turn_cmd(0.0, 0.06)
        elif ph == "reset":
            self.closure = max(0.0, self.closure - 0.1)
            up = np.array([self.cmd_pos[0], self.cmd_pos[1], max(self.cmd_pos[2], self.home_pos[2])])
            if self.cmd_pos[2] < self.home_pos[2] - 0.01:
                self.move_cmd(up, 0.25)
            else:
                done = self.move_cmd(self.home_pos, 0.35)
                tdone = self.turn_cmd(0.0, 0.06)
                if done and tdone and (np.linalg.norm(site - self.home_pos) < 0.02 or el > 4.0):
                    self.set_phase("perceive")
        elif ph == "approach":
            self.checked = False
            goal = self.gpos + np.array([0, 0, self.PRE_DZ])
            done = self.move_cmd(goal, 0.35)
            tdone = self.turn_cmd(self.gpsi, 0.06)
            if done and tdone and (np.linalg.norm(site - goal) < 0.005 or el > 3.0):
                self.set_phase("descend")
        elif ph == "descend":
            done = self.move_cmd(self.gpos, 0.15)
            if done and (np.linalg.norm(site - self.gpos) < 0.006 or el > 1.6):
                self.set_phase("close")
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.06)
            if el > 0.9:
                print("hand q closed", np.round(hq, 2))
                if hq[1] < 0.0 and self.attempts < 4:
                    # thumb closed too far: cube not pinched squarely (finger on top / missed)
                    print("bad grasp; retry")
                    self.set_phase("reset")
                else:
                    self.set_phase("lift")
        elif ph == "lift":
            goal = np.array([self.cmd_pos[0], self.cmd_pos[1], self.th + 0.20])
            if self.move_cmd(goal, 0.25):
                print("hand q lifted", np.round(hq, 2))
                if hq[1] < -0.38 and self.attempts < 4:
                    print("grasp failed (thumb closed on nothing); retry")
                    self.set_phase("reset")
                else:
                    self.set_phase("transport")
        elif ph == "transport" and hq[1] < -0.4 and self.attempts < 4:
            print("cube lost during transport; retry")
            self.set_phase("reset")
        elif ph == "transport":
            bx, by = self.robot.bin_xy[self.dest]
            goal = np.array([bx, by, self.th + 0.20])
            lo, hi = (-0.2, 1.2) if self.dest == "left" else (-0.6, 0.45)
            tdone = self.turn_cmd(float(np.clip(self.gpsi, lo, hi)), 0.025)
            if self.move_cmd(goal, 0.25) and tdone and (np.linalg.norm(site - goal) < 0.01 or el > 4.0):
                self.set_phase("lower")
        elif ph == "lower":
            bx, by = self.robot.bin_xy[self.dest]
            goal = np.array([bx, by, self.th + self.robot.bin_wall_height + self.hs + 0.025])
            if self.move_cmd(goal, 0.15) and (np.linalg.norm(site - goal) < 0.008 or el > 3.0):
                self.set_phase("release")
        elif ph == "release":
            self.closure = max(0.0, self.closure - 0.1)
            if el > 0.8:
                self.set_phase("retreat")
        elif ph == "retreat":
            goal = np.array([self.cmd_pos[0], self.cmd_pos[1], self.th + 0.25])
            self.move_cmd(goal, 0.2)
            if el > 0.8 and not self.checked:
                self.checked = True
                cubes = self.perceive_head(obs)
                bx, by = self.robot.bin_xy[self.dest]
                if self.target in cubes:
                    p = cubes[self.target]["pos"]
                    inside = abs(p[0] - bx) < self.robot.bin_inner_half_size and abs(p[1] - by) < self.robot.bin_inner_half_size
                    print("post-check target at", np.round(p, 3), "inside" if inside else "NOT inside")
                    if not inside and self.attempts < 4:
                        self.set_phase("reset")
        # outer-loop integral correction for servo sag / model mismatch
        err = self.cmd_pos - site
        moving = np.linalg.norm(self.cmd_pos - self.prev_cmd) > 1e-4 or np.max(np.abs(obs["qvel"][:7])) > 0.05
        self.prev_cmd = self.cmd_pos.copy()
        if not moving and self.phase in ("approach", "lift", "transport", "lower", "retreat", "perceive", "reset") and np.linalg.norm(err) < 0.06:
            self.bias = np.clip(self.bias + 0.1 * err, -0.04, 0.04)
        R_des = rotz(self.cmd_psi) @ self.R_home
        q = self.ik(self.cmd_pos + self.bias, R_des)
        return np.concatenate([q, [self.closure]])
