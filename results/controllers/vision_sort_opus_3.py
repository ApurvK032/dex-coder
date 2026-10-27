
import numpy as np, mujoco, math

TABLE_Z = 0.78


def rotmat_from_xy(xax, yax):
    xax = xax / np.linalg.norm(xax)
    yax = yax - xax * (xax @ yax)
    yax = yax / np.linalg.norm(yax)
    return np.stack([xax, yax, np.cross(xax, yax)], 1)


def rot_err(Rt, R):
    E = Rt @ R.T
    c = np.clip((np.trace(E) - 1) / 2, -1, 1)
    ang = math.acos(c)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s = np.linalg.norm(v)
    if s < 1e-9:
        if ang < 1e-6:
            return np.zeros(3)
        w, V = np.linalg.eigh(E + E.T)
        return V[:, -1] * ang
    return v / s * ang


def R_of(a):
    return rotmat_from_xy(np.array([math.cos(a), math.sin(a), 0.0]), np.array([0, 0, -1.0]))


class Perception:
    def __init__(self, robot):
        self.robot = robot
        m = robot.model
        self.fovy = {}
        for c in ("head", "wrist"):
            try:
                fovy = robot.cameras[c]["fovy_deg"]
            except Exception:
                cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
                fovy = m.cam_fovy[cid]
            self.fovy[c] = fovy

    def points(self, cam, name):
        rgb = cam["rgb"].astype(np.int16)
        dep = cam["depth"].astype(np.float64)
        H, W = dep.shape
        f = (H / 2) / math.tan(math.radians(self.fovy[name]) / 2)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u + 0.5 - W / 2) * dep / f
        y = -(v + 0.5 - H / 2) * dep / f
        P = np.stack([x, y, -dep], -1) @ np.asarray(cam["mat"]).T + np.asarray(cam["pos"])
        return rgb, dep, P

    def detect(self, cam, name):
        rgb, dep, P = self.points(cam, name)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = (mx - mn) > 0.35 * np.maximum(mx, 1)
        masks = {
            "red": (r >= mx) & (r - np.maximum(g, b) > np.maximum(25, 0.4 * r)),
            "green": (g >= mx) & (g - np.maximum(r, b) > np.maximum(25, 0.3 * g)),
            "blue": (b >= mx) & (b - np.maximum(r, g) > np.maximum(25, 0.3 * b)),
        }
        valid = dep > 0
        z = P[..., 2]
        out = {}
        for c, mk in masks.items():
            mk = mk & sat & valid & (z > TABLE_Z + 0.015) & (z < TABLE_Z + 0.12)
            pz = z[mk]
            if len(pz) < 8:
                out[c] = None
                continue
            ztop = np.percentile(pz, 85)
            top = mk & (z > ztop - 0.012) & (z < ztop + 0.012)
            pts = P[top]
            if len(pts) < 8:
                out[c] = None
                continue
            med = np.median(pts[:, :2], 0)
            d = np.linalg.norm(pts[:, :2] - med, axis=1)
            pts = pts[d < 0.05]
            if len(pts) < 8:
                out[c] = None
                continue
            xy = pts[:, :2]
            best = None
            for th in np.radians(np.arange(0, 90, 2.0)):
                cth, sth = math.cos(th), math.sin(th)
                a = xy @ np.array([cth, sth])
                bb = xy @ np.array([-sth, cth])
                a2, a98 = np.percentile(a, [2, 98])
                b2, b98 = np.percentile(bb, [2, 98])
                ext = (a98 - a2) + (b98 - b2)
                if best is None or ext < best[0]:
                    ca = 0.5 * (a98 + a2)
                    cb = 0.5 * (b98 + b2)
                    best = (ext, th, ca * np.array([cth, sth]) + cb * np.array([-sth, cth]))
            ext, th, ctr = best
            out[c] = dict(xy=0.5 * (ctr + xy.mean(0)), yaw=th, n=len(pts), ztop=float(np.median(pts[:, 2])))
        return out


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.per = Perception(robot)
        self.lo = np.asarray(robot.arm_lower, dtype=float)
        self.hi = np.asarray(robot.arm_upper, dtype=float)
        self.qa = np.asarray(robot.arm_qpos_adr)
        self.da = np.asarray(robot.arm_dof_adr)
        self.home = np.asarray(robot.home_arm, dtype=float)
        self.q_rest = self.home.copy()
        self.ns_gain = 0.0
        self.arm_bodies = set()
        for b in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith("right_shoulder") or nm.startswith("right_elbow") or nm.startswith("right_wrist") or nm.startswith("right_hand"):
                self.arm_bodies.add(b)
        self.q_drop = None
        self.q_via = None
        self.phase = "perceive"
        self.t_phase = 0.0
        self.cmd_q = None
        self.closure = 0.0
        self.target = task["target"]
        self.dest = task.get("destination")
        self.cube = None
        self.wp = None
        self.R_grasp = None
        self.corr = np.zeros(3)
        self.site = None
        self.hz = TABLE_Z + robot.block_half_size
        self.Z_ABOVE = self.hz + 0.08
        self.Z_GRASP = self.hz + 0.01
        self.dets = None
        self.bin_shift = 0.015
        self.Z_CARRY = self.hz + 0.11
        self.Z_DROP = self.hz + 0.07

    # ---------- kinematics ----------
    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pt, Rt, iters=30, wrot=0.3):
        """Rt=None: only constrain palm facing down (yaw free)."""
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self.d.qpos[self.qa] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.sid]
            R = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = pt - p
            if Rt is None:
                er = np.cross(R[:, 1], np.array([0, 0, -1.0]))
            else:
                er = rot_err(Rt, R)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], wrot * jr[:, self.da]])
            e = np.r_[ep, wrot * er]
            JJ = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            if self.ns_gain > 0:
                z = self.ns_gain * (self.q_rest - q)
                dq += z - J.T @ np.linalg.solve(JJ, J @ z)
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def chain_err(self, q, pts, R):
        errs = []
        qs = []
        for p in pts:
            for _ in range(6):
                q = self.ik(q, p, R, iters=30)
            pp, RR = self.fk(q)
            errs.append(np.linalg.norm(pp - p) + 0.05 * np.linalg.norm(rot_err(R, RR)))
            qs.append(q.copy())
        return max(errs), qs

    def limit_pen(self, qs):
        pen = 0.0
        for q in qs:
            mm = np.minimum(q - self.lo, self.hi - q)
            pen += np.sum(np.clip(0.2 - mm, 0, None))
        return pen

    def grasp_obstacle_pen(self, xy, xa):
        xa = xa / np.linalg.norm(xa)
        la = np.array([-xa[1], xa[0]])
        pen = 0.0
        # neighbouring cubes
        for c, v in (self.dets or {}).items():
            if c == self.target or v is None:
                continue
            dv = np.asarray(v["xy"]) - xy
            fx, fl = dv @ xa, dv @ la
            if -0.005 < fx < 0.125 and abs(fl) < 0.075:
                pen += 1.0
            if -0.085 < fx <= -0.005 and abs(fl) < 0.05:
                pen += 1.0
        # bin walls
        binpen = 0.0
        hb = self.robot.bin_inner_half_size + self.robot.bin_wall_thickness + 0.008
        for bc in self.robot.bin_xy.values():
            bc = np.asarray(bc, dtype=float)
            for fx in (0.04, 0.06, 0.08, -0.05):
                for fl in (-0.035, 0.0, 0.035):
                    p = xy + fx * xa + fl * la
                    if np.max(np.abs(p - bc)) < hb:
                        binpen += 0.2
        return pen, binpen

    def plan_grasp_gen(self, q_now, n_rand=4):
        """Generator: yields between chunks of work; sets self.R_grasp, q_above, q_carry."""
        xy = self.cube["xy"]
        best = None
        rng = np.random.default_rng(1)
        for k in range(4):
            a = self.cube["yaw"] + k * math.pi / 2
            a = math.atan2(math.sin(a), math.cos(a))
            R = R_of(a)
            cpen, bpen = self.grasp_obstacle_pen(xy, R[:2, 0])
            gxy = xy.copy()
            if (bpen > 0 or cpen > 0) and self.bin_shift > 0:
                # grasp off-centre (hand shifted away from the bin) so the curling fingers miss the wall
                gxy = xy - self.bin_shift * R[:2, 0]
                c2, b2 = self.grasp_obstacle_pen(gxy, R[:2, 0])
                cpen, bpen = 0.5 * c2, 0.5 * b2
            obst = cpen + bpen
            pts = [np.r_[gxy, self.Z_ABOVE], np.r_[gxy, self.Z_GRASP], np.r_[gxy, self.Z_CARRY]]
            seeds = [q_now.copy()] + [np.clip(q_now + rng.normal(0, 1.0, 7), self.lo, self.hi) for _ in range(n_rand)]
            for si, q0 in enumerate(seeds):
                e, qs = self.chain_err(q0, pts, R)
                if e > 0.01:
                    yield
                    continue
                travel = np.max(np.abs(qs[0] - q_now))
                apath = 0.0 if self.path_ok(q_now, qs[0], self.Z_ABOVE - 0.03) else 1.0
                lim = self.limit_pen(qs)
                score = e + 0.05 * obst + 0.1 * apath + 0.002 * abs(a - math.radians(15)) + 0.02 * lim + 0.004 * travel
                print(f"  cand yaw {math.degrees(a):.0f} seed {si}: ik {e*1000:.1f}mm obst {obst:.2f} path {apath} lim {lim:.2f} travel {travel:.2f} score {score:.4f}")
                if best is None or score < best[0]:
                    best = (score, a, R, e, qs, gxy)
                yield
        if best is None:
            # fallback: whatever the plain chain gives
            a = self.cube["yaw"]
            R = R_of(a)
            e, qs = self.chain_err(q_now.copy(), [np.r_[xy, self.Z_ABOVE], np.r_[xy, self.Z_GRASP], np.r_[xy, self.Z_CARRY]], R)
            best = (1.0, a, R, e, qs, xy.copy())
        print(f"grasp yaw {math.degrees(best[1]):.0f} err {best[3]*1000:.1f}mm score {best[0]:.4f}")
        self.q_above = best[4][0]
        self.q_carry = best[4][2]
        self.R_grasp = best[2]
        self.grasp_xy = best[5]

    def collides(self, q):
        d = self.d
        d.qpos[self.qa] = q
        mujoco.mj_forward(self.m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            if (b1 in self.arm_bodies) != (b2 in self.arm_bodies):
                if c.dist < 0.005:
                    return True
        return False

    def path_ok(self, qa, qb, zmin, n=16):
        for s in np.linspace(0, 1, n):
            q = qa + (qb - qa) * s
            if self.collides(q):
                return False
            if zmin is not None and self.fk(q)[0][2] < zmin:
                return False
        return True

    def plan_drop(self, q_carry, bxy):
        rng = np.random.default_rng(0)
        goal = np.r_[bxy, self.Z_CARRY]
        seeds = [q_carry.copy()] + [rng.uniform(self.lo, self.hi) for _ in range(40)]
        cands = []
        for q0 in seeds:
            q = q0
            for _ in range(8):
                q = self.ik(q, goal, None, iters=30)
            p, R = self.fk(q)
            if np.linalg.norm(p - goal) > 2e-3 or np.linalg.norm(np.cross(R[:, 1], [0, 0, -1.0])) > 0.03:
                continue
            if self.limit_pen([q]) > 0.05 or self.collides(q):
                continue
            cands.append((np.max(np.abs(q - q_carry)), q))
        cands.sort(key=lambda c: c[0])
        for dist, q in cands:
            if self.path_ok(q_carry, q, self.Z_CARRY - 0.012):
                ql = q
                for _ in range(6):
                    ql = self.ik(ql, np.r_[bxy, self.Z_DROP], None, iters=30)
                print(f"drop plan: {len(cands)} cands, dist {dist:.2f}")
                return q, ql
        # try a via pose: midway, higher
        pc = self.fk(q_carry)[0]
        for dist, q in cands[:6]:
            pg = self.fk(q)[0]
            for hz_extra in (0.04, 0.08):
                mid = 0.5 * (pc + pg)
                mid[2] = self.Z_CARRY + hz_extra
                qv = 0.5 * (q_carry + q)
                for _ in range(6):
                    qv = self.ik(qv, mid, None, iters=30)
                if np.linalg.norm(self.fk(qv)[0] - mid) > 0.01 or self.collides(qv):
                    continue
                if self.path_ok(q_carry, qv, self.Z_CARRY - 0.012) and self.path_ok(qv, q, self.Z_CARRY - 0.012):
                    ql = q
                    for _ in range(6):
                        ql = self.ik(ql, np.r_[bxy, self.Z_DROP], None, iters=30)
                    self.q_via = qv
                    print(f"drop plan: via pose (+{hz_extra}), dist {dist:.2f}")
                    return q, ql
        # fallback: closest candidate
        print("drop plan: no direct path", len(cands))
        if cands:
            q = cands[0][1]
            ql = q
            for _ in range(6):
                ql = self.ik(ql, np.r_[bxy, self.Z_DROP], None, iters=30)
            return q, ql
        return None, None

    def jmove(self, qg, speed):
        dq = qg - self.cmd_q
        mx = np.max(np.abs(dq))
        step = speed * 0.04
        if mx <= step:
            self.cmd_q = qg.copy()
            return True
        self.cmd_q = self.cmd_q + dq * (step / mx)
        return False

    def set_phase(self, p, t):
        print(f"t={t:.2f} phase {p}")
        self.phase = p
        self.t_phase = t

    def move_toward(self, goal, speed, R, wrot=0.3):
        dv = goal - self.wp
        n = np.linalg.norm(dv)
        step = speed * 0.04
        if n > step:
            self.wp = self.wp + dv / n * step
        else:
            self.wp = goal.copy()
        err = self.wp - self.site
        if n <= step:
            if np.linalg.norm(err) < 0.05:
                self.corr += 0.15 * err
        else:
            self.corr[2] += 0.08 * np.clip(err[2], -0.02, 0.02)
        cn = np.linalg.norm(self.corr)
        if cn > 0.04:
            self.corr *= 0.04 / cn
        self.cmd_q = self.ik(self.cmd_q, self.wp + self.corr, R, wrot=wrot)
        return n <= step

    # ---------- main ----------
    def act(self, obs):
        t = float(obs["t"])
        if self.cmd_q is None:
            self.cmd_q = np.asarray(obs["arm_target"], dtype=float).copy()
            self.wp, self.R0 = self.fk(self.cmd_q)
        site = np.asarray(obs["grasp_site_pos"], dtype=float)
        self.site = site
        ph = self.phase
        el = t - self.t_phase
        bxy = np.asarray(self.robot.bin_xy[self.dest], dtype=float)

        if ph == "perceive":
            det = self.per.detect(obs["cameras"]["head"], "head")
            print("det", {k: (None if v is None else (np.round(v["xy"], 3), round(math.degrees(v["yaw"]), 1), v["n"])) for k, v in det.items()})
            if det[self.target] is not None:
                self.dets = det
                self.cube = det[self.target]
                self.planner = self.plan_grasp_gen(self.cmd_q.copy())
                self.set_phase("plan", t)
        elif ph == "plan":
            try:
                next(self.planner)
            except StopIteration:
                self.q_drop, self.q_drop_lo = self.plan_drop(self.q_carry, bxy)
                self.set_phase("approach", t)
        elif ph == "approach":
            goal = np.r_[self.grasp_xy, self.Z_ABOVE]
            done = self.jmove(self.q_above, 1.2)
            if (done and np.linalg.norm(site - goal) < 0.015) or el > 4:
                self.wp = self.fk(self.cmd_q)[0]
                self.set_phase("descend", t)
        elif ph == "descend":
            goal = np.r_[self.grasp_xy, self.Z_GRASP]
            if (self.move_toward(goal, 0.12, self.R_grasp) and np.linalg.norm(site - goal) < 0.006) or el > 3:
                self.set_phase("close", t)
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.06)
            if el > 1.0:
                self.set_phase("lift", t)
        elif ph == "lift":
            goal = np.r_[self.grasp_xy, self.Z_CARRY]
            if (self.move_toward(goal, 0.2, self.R_grasp) and np.linalg.norm(site - goal) < 0.02) or el > 3:
                self.set_phase("transport", t)
        elif ph == "transport":
            goal = np.r_[bxy, self.Z_CARRY]
            if self.q_via is not None:
                if self.jmove(self.q_via, 1.0):
                    self.q_via = None
            elif self.q_drop is not None:
                done = self.jmove(self.q_drop, 1.0)
                if (done and np.linalg.norm(site - goal) < 0.03) or el > 5:
                    self.wp = self.fk(self.cmd_q)[0]
                    self.set_phase("lower", t)
            elif (self.move_toward(goal, 0.3, None) and np.linalg.norm(site - goal) < 0.02) or el > 5:
                self.set_phase("lower", t)
        elif ph == "lower":
            goal = np.r_[bxy, self.Z_DROP]
            if (self.move_toward(goal, 0.12, None) and np.linalg.norm(site - goal) < 0.012) or el > 2.5:
                self.set_phase("release", t)
        elif ph == "release":
            self.closure = max(0.0, self.closure - 0.1)
            self.move_toward(self.wp, 0.1, None)
            if el > 0.8:
                self.set_phase("retreat", t)
        elif ph == "retreat":
            goal = np.r_[bxy, self.hz + 0.15]
            self.move_toward(goal, 0.2, None)
        return np.r_[self.cmd_q, self.closure]
