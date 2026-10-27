
import numpy as np
import mujoco

COLORS = ("red", "green", "blue")


def color_masks(rgb):
    c = rgb.astype(np.float32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    s = r + g + b + 1e-3
    rn, gn, bn = r / s, g / s, b / s
    bright = s > 60
    return {
        "red": bright & (rn > 0.55) & (gn < 0.25) & (bn < 0.25),
        "green": bright & (gn > 0.5) & (rn < 0.3) & (bn < 0.35),
        "blue": bright & (bn > 0.5) & (rn < 0.3) & (gn < 0.4),
    }


def backproject(cam, fovy_deg):
    dep = cam["depth"].astype(np.float64)
    H, W = dep.shape
    f = (H / 2) / np.tan(np.radians(fovy_deg) / 2)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    P = np.stack([(u + 0.5 - W / 2) * dep / f, -(v + 0.5 - H / 2) * dep / f, -dep], -1)
    Pw = P @ np.asarray(cam["mat"]).T + np.asarray(cam["pos"])
    return Pw, dep > 0


def largest_component(mask):
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    lab = -np.ones(mask.shape, np.int32)
    best, bestn = -1, 0
    cur = 0
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0] >= 0:
            continue
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if n > bestn:
            best, bestn = cur, n
        cur += 1
    return lab == best


def estimate_cube(Pw, mask, table_z, half):
    pts = Pw[mask]
    if len(pts) < 8:
        return None
    pts = pts[(pts[:, 2] > table_z + 0.005) & (pts[:, 2] < table_z + 0.25)]
    if len(pts) < 8:
        return None
    ztop = np.percentile(pts[:, 2], 90)
    top = pts[pts[:, 2] > ztop - 0.008]
    if len(top) < 5:
        top = pts
    xy = top[:, :2]
    c = xy.mean(0)
    best = (1e9, 0.0)
    for th in np.radians(np.arange(0, 90, 2.0)):
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = (xy - c) @ R.T
        ext = (q.max(0) - q.min(0))
        a = ext[0] * ext[1]
        if a < best[0]:
            best = (a, th)
    th = best[1]
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (xy - c) @ R.T
    mid = (q.max(0) + q.min(0)) / 2
    c = c + R.T @ mid
    return np.array([c[0], c[1], ztop - half]), th, len(pts)


def rotz_frame(psi):
    # grasp frame: x along fingers (horizontal, heading psi), y out of palm = down
    x = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rlog(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def rslerp(Ra, Rb, s):
    return Ra @ rodrigues(s * rlog(Ra.T @ Rb))


def tilted(psi, pitch, roll):
    # pitch about the site z (lateral) axis: + tips fingers downward; roll about fingers
    R = rotz_frame(psi)
    return R @ rodrigues(np.array([0, 0, pitch])) @ rodrigues(np.array([roll, 0, 0]))


class Controller:
    PRECURL = 0.3
    PRE_H = 0.065

    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, float)
        self.tz = robot.table_height
        self.half = robot.block_half_size
        self.target = task["target"]
        self.bin = np.array(robot.bin_xy[task["destination"]], float)
        self.bin0 = self.bin.copy()
        self.banned = []
        self.psi_g = 0.0
        self.fov = {k: v["fovy_deg"] for k, v in robot.cameras.items()}
        self.phase = "look"
        self.t0 = 0.0
        self.qcmd = None
        self.closure = 0.0
        self.cube = None
        self.yaw = 0.0
        self.jpath = []
        self.est = []
        self.others = {}
        self.attempts = 0
        self.rng = np.random.default_rng(0)
        self.arm_bodies = set()
        for b in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith("right_") and any(k in nm for k in ("shoulder", "elbow", "wrist", "hand")):
                self.arm_bodies.add(b)
        self.table_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "table")
        self.hand_bodies = set()
        self.ignore_bodies = set()
        for b in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith("right_hand"):
                self.hand_bodies.add(b)
            if "block" in nm or "cube" in nm:
                self.ignore_bodies.add(b)
        # hand roughly in the pre-curl pose for collision checks
        hq = {"right_hand_thumb_1_joint": 0.25, "right_hand_index_0_joint": 0.9, "right_hand_index_1_joint": 0.24,
              "right_hand_middle_0_joint": 0.9, "right_hand_middle_1_joint": 0.24}
        for nm, adr in zip(robot.hand_joints, robot.hand_qpos_adr):
            self.d.qpos[adr] = hq.get(nm, 0.0)

    # ---------------- kinematics -----------------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, q0, p, R, iters=60, wrot=0.5, null=0.05):
        q = np.array(q0, float).copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pc = self.d.site_xpos[self.site]
            Rc = self.d.site_xmat[self.site].reshape(3, 3)
            ep = p - pc
            er = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], wrot * jr[:, self.dadr]])
            e = np.concatenate([ep, wrot * er])
            JJ = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if null > 0:
                N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
                # pull toward home and away from joint limits
                mid = 0.5 * (self.lo + self.hi)
                dq += N @ (null * (self.home - q))
            n = np.abs(dq).max()
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo + 0.02, self.hi - 0.02)
        return q

    def pose_err(self, q, p, R):
        pc, Rc = self.fk(q)
        return np.linalg.norm(pc - p) + 0.03 * np.linalg.norm(Rc - R)

    def seeds(self, n_rand):
        s = [self.qcmd.copy(), self.home.copy()]
        for i in range(n_rand):
            sc = 0.5 if i < n_rand // 2 else 1.0
            s.append(np.clip(self.home + self.rng.normal(0, sc, 7), self.lo, self.hi))
        return s

    def collides(self, q):
        """self / environment collision of the arm (except hand-table light touches)."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            a1, a2 = b1 in self.arm_bodies, b2 in self.arm_bodies
            if a1 == a2:
                continue
            other = b2 if a1 else b1
            mine = b1 if a1 else b2
            if other in self.ignore_bodies:
                continue
            if other == self.table_body and mine in self.hand_bodies:
                continue
            if c.dist < -0.001:
                return True
        return False

    def chain(self, q, pts, R):
        """track a dense cartesian path with warm-started IK; returns list of q and max err.
        R may be a single rotation or a list (one per point)."""
        qs = []
        emax = 0.0
        Rs = R if isinstance(R, list) else [R] * len(pts)
        for i, p in enumerate(pts):
            R = Rs[i]
            q = self.ik(q, p, R, iters=40, null=0.0)
            e = self.pose_err(q, p, R)
            if i % 3 == 2 or i == len(pts) - 1:
                if self.collides(q):
                    e += 0.02
            emax = max(emax, e)
            qs.append(q)
        return qs, emax

    @staticmethod
    def line(a, b, step=0.005):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        return [a + (b - a) * (i + 1) / n for i in range(n)]

    def plan_segments(self, R, p_pre, p_low, p_lift, n_rand=10, tol=0.003):
        """find a start config at p_pre from which descending to p_low and lifting to
        p_lift are all trackable.  Returns (err, q_pre, desc_chain, lift_chain)."""
        best = None
        for s in self.seeds(n_rand):
            q0 = self.ik(s, p_pre, R, iters=150)
            e0 = self.pose_err(q0, p_pre, R)
            if self.collides(q0):
                e0 += 0.02
            if e0 > tol:
                if best is None or e0 + 1 < best[0]:
                    best = (e0 + 1, q0, [], [])
                continue
            d, e1 = self.chain(q0, self.line(p_pre, p_low), R)
            l, e2 = self.chain(d[-1], self.line(p_low, p_lift), R) if p_lift is not None else ([], 0.0)
            e = max(e0, e1, e2)
            sc = e + 0.0005 * np.abs(q0 - self.qcmd).sum()
            if best is None or sc < best[0]:
                best = (sc, q0, d, l)
            if e < tol:
                break
        return best

    # ---------------- perception -----------------
    def perceive(self, obs, cam="head"):
        c = obs["cameras"][cam]
        Pw, valid = backproject(c, self.fov[cam])
        masks = color_masks(c["rgb"])
        res = {}
        for col in COLORS:
            mk = masks[col] & valid
            if mk.sum() < 8:
                continue
            mk = largest_component(mk)
            r = estimate_cube(Pw, mk, self.tz, self.half)
            if r is not None:
                res[col] = r
        return res

    # ---------------- motion helpers -----------------
    def jmove(self, qgoal, speed=1.2):
        qgoal = np.clip(qgoal, self.lo, self.hi)
        dq = qgoal - self.qcmd
        n = max(1, int(np.ceil(1.5 * np.abs(dq).max() / (speed * self.robot.control_dt))))
        s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
        self.jpath = [self.qcmd + dq * si for si in s]

    def cmove(self, qs, step=0.005, speed=0.12):
        """follow a joint chain whose nodes are `step` m apart at `speed` m/s."""
        qs = [self.qcmd.copy()] + list(qs)
        L = len(qs) - 1
        n = max(1, int(np.ceil(1.3 * L * step / (speed * self.robot.control_dt))))
        s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
        path = []
        for si in s:
            x = si * L
            i = min(int(x), L - 1)
            a = x - i
            path.append((1 - a) * qs[i] + a * qs[i + 1])
        self.jpath = path

    def step_j(self):
        if self.jpath:
            self.qcmd = self.jpath.pop(0)
        return not self.jpath

    def plan_grasp(self):
        best = None
        p_low = self.cube.copy()
        p_pre = self.cube + np.array([0, 0, self.PRE_H])
        for k in range(-3, 4):
            psi = self.yaw + k * np.pi / 2
            if psi < np.radians(-100) or psi > np.radians(100):
                continue
            cl = self.clearance(psi)
            pen = 0.004 * abs(psi - np.radians(-20)) + 2.0 * max(0.0, 0.012 - cl)
            for b in self.banned:
                if abs((psi - b + np.pi) % (2 * np.pi) - np.pi) < np.radians(10):
                    pen += 0.5
            if best is not None and pen > best[0]:
                continue
            for pitch, roll in ((0, 0), (0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3),
                                (-0.5, 0), (0.5, 0), (0, 0.5), (0, -0.5)):
                R = tilted(psi, pitch, roll)
                tpen = 0.01 * (abs(pitch) + abs(roll))
                for hl in (0.13, 0.10, 0.08):
                    p_lift = np.array([self.cube[0], self.cube[1], self.tz + hl])
                    sc, q0, d, l = self.plan_segments(R, p_pre, p_low, p_lift, n_rand=8)
                    if sc < 0.003 or sc > 0.9:
                        break
                score = sc + pen + tpen + (1.0 if sc > 0.006 else 0.0)

                if best is None or score < best[0]:
                    best = (score, psi, R, q0, d, l)
                if sc < 0.003:
                    break
        return best

    def plan_place(self, p_start):
        """cartesian carry from p_start up/over to the bin, then down. Returns
        (score, psi, carry_chain, lower_chain)."""
        best = None
        psi_g = self.psi_g
        cands = list(np.radians(np.arange(-60, 61, 15)))
        for zh in (0.15, 0.13):
            p_hi = np.array([self.bin[0], self.bin[1], self.tz + zh])
            p_lo = np.array([self.bin[0], self.bin[1], self.tz + 0.085])
            pts = self.line(p_start, p_hi, 0.01)
            for psi in cands:
                Rs = [rslerp(self.R, rotz_frame(psi), (i + 1) / len(pts)) for i in range(len(pts))]
                c1, e1 = self.chain(self.qcmd, pts, Rs)
                c2, e2 = self.chain(c1[-1], self.line(p_hi, p_lo), rotz_frame(psi))
                sc = max(e1 - 0.003, 0) + e2 + 0.0005 * abs(psi - psi_g)
                if best is None or sc < best[0]:
                    best = (sc, psi, c1, c2)
            if best[0] < 0.004:
                break
        return best

    def clearance(self, psi):
        """min 2D distance from hand footprint (fingers / thumb) to obstacles."""
        c, s = np.cos(psi), np.sin(psi)
        ax = np.array([c, s]); lat = np.array([-s, c])
        pts = []
        for x in np.linspace(0.03, 0.08, 5):
            for l in np.linspace(-0.045, 0.045, 5):
                pts.append(self.cube[:2] + x * ax + l * lat)
        for x in np.linspace(-0.06, -0.035, 3):
            for l in np.linspace(-0.012, 0.012, 3):
                pts.append(self.cube[:2] + x * ax + l * lat)
        pts = np.array(pts)
        dmin = 1.0
        for col, (p, yaw) in self.others.items():
            R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            loc = np.abs((pts - p[:2]) @ R.T) - self.half
            d = np.linalg.norm(np.maximum(loc, 0), axis=1)
            dmin = min(dmin, d.min())
        hw = self.robot.bin_inner_half_size + self.robot.bin_wall_thickness
        for b in self.robot.bin_xy.values():
            loc = np.abs(pts - np.array(b)) - hw
            d = np.linalg.norm(np.maximum(loc, 0), axis=1)
            dmin = min(dmin, d.min())
        return dmin

    def start_grasp(self):
        g = self.plan_grasp()
        print("grasp plan psi %.1f score %.4f" % (np.degrees(g[1]), g[0]))
        _, psi, self.R, q0, self.desc, self.lift = g
        self.psi_g = psi
        self.jmove(q0)
        self.phase = "pre"

    def act(self, obs):
        t = obs["t"]
        if self.qcmd is None:
            self.qcmd = np.array(obs["arm_target"], float)
        ph = self.phase
        if ph == "look":
            res = self.perceive(obs)
            if self.target in res:
                self.est.append(res[self.target])
            if len(self.est) >= 2 or (t > 1.0 and self.est):
                ps = np.array([e[0] for e in self.est])
                self.cube = np.median(ps, 0)
                self.cube[2] = self.tz + self.half
                self.yaw = self.est[-1][1]
                self.others = {k: (v[0], v[1]) for k, v in res.items() if k != self.target}
                print("cube est", self.cube.round(4), "yaw", np.degrees(self.yaw).round(1), "all", {k: v[0].round(3) for k, v in res.items()})
                self.start_grasp()
        elif ph == "pre":
            self.closure = min(self.PRECURL, self.closure + 0.03) if self.closure <= self.PRECURL else max(self.PRECURL, self.closure - 0.1)
            if self.step_j():
                self.phase = "settle_pre"; self.t0 = t
        elif ph == "settle_pre":
            self.closure = self.PRECURL
            if t - self.t0 > 0.25:
                self.cmove(self.desc, speed=0.10)
                self.phase = "descend"
        elif ph == "descend":
            if self.step_j():
                self.phase = "settle_low"; self.t0 = t
        elif ph == "settle_low":
            if t - self.t0 > 0.25:
                err = np.array(obs["grasp_site_pos"]) - self.cube
                print("at grasp: site err", err.round(4))
                if err[2] > 0.006 and self.attempts < 3:
                    # hand blocked on the way down (neighbour / obstacle): back off, try another heading
                    print("descent blocked -> back off")
                    self.attempts += 1
                    self.banned.append(self.psi_g)
                    self.cmove(list(reversed(self.desc)), speed=0.1)
                    self.phase = "backoff"
                else:
                    self.phase = "close"; self.t0 = t
        elif ph == "backoff":
            if self.step_j():
                self.start_grasp()
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.06)
            if t - self.t0 > 1.0:
                self.cmove(self.lift, speed=0.15)
                self.phase = "lift"
        elif ph == "lift":
            if self.step_j():
                self.phase = "check"; self.t0 = t
        elif ph == "check":
            res = self.perceive(obs)
            ok = True
            off = np.zeros(2)
            if self.target in res:
                c = res[self.target][0]
                off = c[:2] - np.array(obs["grasp_site_pos"])[:2]
                if c[2] < self.tz + self.half + 0.03:
                    ok = False
                    print("grasp failed: cube still on table at", c.round(3))
                elif np.linalg.norm(off) > 0.032 and self.attempts < 3:
                    print("held off-centre by", off.round(3), "-> put down and regrasp")
                    self.attempts += 1
                    self.cmove(list(reversed(self.lift))[:-2] if len(self.lift) > 3 else [], speed=0.1)
                    self.phase = "putdown"
                    return np.concatenate([self.qcmd, [self.closure]])
            if ok or self.attempts >= 3:
                p_start, _ = self.fk(self.qcmd)
                # the held-cube estimate is biased by hand occlusion (~1-2 cm), so only
                # compensate the part of the offset beyond that
                n = np.linalg.norm(off)
                off = off * (max(0.0, n - 0.02) / n) if n > 1e-6 else off * 0
                self.bin = self.bin0 - off
                print("held offset", off.round(3))
                p = self.plan_place(p_start)
                print("place plan psi %.1f score %.4f" % (np.degrees(p[1]), p[0]))
                _, psi, self.carry, self.lower = p
                self.cmove(self.carry, step=0.01, speed=0.2)
                self.phase = "carry"
            else:
                self.attempts += 1
                self.closure = 0.0
                self.cube = c.copy()
                self.cube[2] = self.tz + self.half
                self.yaw = res[self.target][1]
                self.others = {k: (v[0], v[1]) for k, v in res.items() if k != self.target}
                self.phase = "reopen"; self.t0 = t
        elif ph == "reopen":
            if t - self.t0 > 0.5:
                self.start_grasp()
        elif ph == "putdown":
            if self.step_j():
                self.closure = max(0.0, self.closure - 0.1)
                if self.closure <= 0.0:
                    self.phase = "putdown_wait"; self.t0 = t
        elif ph == "putdown_wait":
            if t - self.t0 > 0.4:
                self.cmove(self.lift, speed=0.1)
                self.phase = "putdown_up"
        elif ph == "putdown_up":
            if self.step_j():
                self.jmove(self.home, 1.0)
                self.phase = "go_home"
        elif ph == "go_home":
            if self.step_j():
                self.est = []
                self.phase = "look"
        elif ph == "carry":
            if self.step_j():
                self.cmove(self.lower, speed=0.08)
                self.phase = "lower"
        elif ph == "lower":
            if self.step_j():
                self.phase = "release"; self.t0 = t
        elif ph == "release":
            self.closure = max(0.0, self.closure - 0.1)
            if t - self.t0 > 0.8:
                self.cmove(list(reversed(self.lower)), speed=0.12)
                self.phase = "retreat"
        elif ph == "retreat":
            self.step_j()
        return np.concatenate([self.qcmd, [self.closure]])
