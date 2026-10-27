
import numpy as np
import mujoco


def _rotvec_from_R(Re):
    ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def target_R(yaw, tilt=0.0, pitch=0.0):
    """Site frame: x = pinch axis (at yaw, pitched down by pitch), y = palm normal (down),
    tilt rotates the palm normal about the pinch axis."""
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cos(pitch) * h - np.sin(pitch) * np.array([0, 0, 1.0])
    y = -np.sin(pitch) * h - np.cos(pitch) * np.array([0, 0, 1.0])
    z = np.cross(x, y)
    y2 = np.cos(tilt) * y + np.sin(tilt) * z
    z2 = np.cross(x, y2)
    return np.stack([x, y2, z2], axis=1)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        m = self.m
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.d = mujoco.MjData(m)
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.arm_q = np.array(robot.arm_qpos_adr)
        self.arm_v = np.array(robot.arm_dof_adr)
        self.lower = np.array(robot.arm_lower)
        self.upper = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm)
        self.q_cmd = self.home.copy()
        self.q_goal = self.home.copy()
        self.queue = []
        self.closure = 0.0
        self.step_lim = robot.max_joint_speed * robot.control_dt * 0.95
        self.dt = robot.control_dt
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.bin_xy = np.array(robot.bin_xy[task["destination"]])
        self.bin_in = robot.bin_inner_half_size
        self.rim_z = self.table + robot.bin_wall_height + 0.006
        self.cubes = [task["target"], task["second"]]
        self.other = [c for c in ["red", "green", "blue"] if c not in self.cubes][0]
        self.carry_z = 0.90
        self.pick_z = self.table + self.half + 0.003
        self.phase = "start"
        self.idx = 0
        self.attempt = 0
        self.timer = 0.0
        self.goal_pos = None
        self.kin_initialized = False
        self.rng = np.random.default_rng(0)
        self.speed_scale = 1.0
        self.grasp_offset = np.zeros(3)  # cube - site, in site frame
        self.place_yaw = 0.0
        self.place_pitch = 0.0
        self.pitch = 0.0
        self.tilt = 0.0
        self.yaw = 0.0
        self.slot = self.bin_xy.copy()
        self.good_seeds = []
        # bodies for collision checking
        self.arm_bodies = set()
        self.block_bodies = set()
        for b in range(m.nbody):
            n = m.body(b).name
            if n.startswith("right_") and any(s in n for s in ("shoulder", "elbow", "wrist", "hand")):
                self.arm_bodies.add(b)
            if n.startswith("block"):
                self.block_bodies.add(b)

    # ---------- kinematics ----------
    def fk(self, q):
        self.d.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def collides(self, q):
        self.d.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        m = self.m
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            a1 = b1 in self.arm_bodies
            a2 = b2 in self.arm_bodies
            if a1 == a2:
                continue
            other = b2 if a1 else b1
            if other in self.block_bodies:
                continue
            if con.dist < -0.001:
                return True
        return False

    def ik_iter(self, q, pos, R, iters=100, wrot=0.5):
        q = q.copy()
        ep = er = np.zeros(3)
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = _rotvec_from_R(R @ Rc.T)
            if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 5e-3:
                break
            e = np.concatenate([ep, wrot * er])
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp, self.jacr])[:, self.arm_v]
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            n = 0.02 * (self.home - q)
            dq += n - J.T @ np.linalg.solve(JJt, J @ n)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), self.lower, self.upper)
        return q, np.linalg.norm(ep), np.linalg.norm(er)

    def ik_candidates(self, pos, R, q_init, n_random=6, need=3):
        """Collision-free converged IK solutions sorted by distance from q_init."""
        pos = np.asarray(pos, dtype=float)
        starts = [q_init, self.home] + list(self.good_seeds[-3:])
        for _ in range(n_random):
            starts.append(self.rng.uniform(self.lower, self.upper))
        sols = []
        for q0 in starts:
            q, ep, er = self.ik_iter(q0, pos, R)
            if ep < 4e-3 and er < 3e-2 and not self.collides(q):
                if all(np.max(np.abs(q - s[1])) > 0.05 for s in sols):
                    margin = np.min(np.minimum(q - self.lower, self.upper - q))
                    score = np.linalg.norm(q - q_init) + 2.0 * max(0.0, 0.2 - margin)
                    sols.append((score, q, ep, er))
            if len(sols) >= need:
                break
        sols.sort(key=lambda s: s[0])
        return sols

    def path_ok(self, q0, q1, zmin, R=None, dev=0.04, n=12):
        p0, _ = self.fk(q0)
        p1, _ = self.fk(q1)
        seg = p1 - p0
        L = np.linalg.norm(seg)
        for i in range(1, n + 1):
            q = q0 + (q1 - q0) * i / n
            p, Rc = self.fk(q)
            if zmin is not None and p[2] < zmin - 0.005:
                return False
            if L > 1e-6:
                s = np.clip(np.dot(p - p0, seg) / (L * L), 0, 1)
                if np.linalg.norm(p - (p0 + s * seg)) > dev:
                    return False
            if R is not None and np.linalg.norm(_rotvec_from_R(R @ Rc.T)) > 0.6:
                return False
            if self.collides(q):
                return False
        return True

    def plan(self, q_start, pos, R, zmin=None, depth=0, dev=0.04):
        """Returns list of joint waypoints from q_start to pose, or None."""
        pos = np.asarray(pos, dtype=float)
        cands = self.ik_candidates(pos, R, q_start)
        if not cands:
            return None
        for sc, q, ep, er in cands:
            if self.path_ok(q_start, q, zmin, R, dev):
                return [q]
        if depth >= 2:
            return [cands[0][1]]
        # subdivide in Cartesian space
        p0, _ = self.fk(q_start)
        mid = 0.5 * (p0 + pos)
        if zmin is not None:
            mid[2] = max(mid[2], zmin + 0.01)
        first = self.plan(q_start, mid, R, zmin, depth + 1, dev)
        if first is None:
            return [cands[0][1]]
        second = self.plan(first[-1], pos, R, zmin, depth + 1, dev)
        if second is None:
            return [cands[0][1]]
        return first + second

    # ---------- motion helpers ----------
    def set_goal(self, pos, yaw, speed=1.0, zmin=None, tilt=0.0, dev=0.04, pitch=0.0):
        self.goal_pos = np.array(pos, dtype=float)
        self.yaw = yaw
        R = target_R(yaw, tilt, pitch)
        path = self.plan(self.q_cmd, self.goal_pos, R, zmin, 0, dev)
        if path is None:
            print(f"IK failed for {np.round(self.goal_pos,3)} yaw {yaw:.2f} pitch {pitch:.2f}")
            q, ep, er = self.ik_iter(self.q_cmd, self.goal_pos, R)
            path = [q]
            ok = False
        else:
            ok = True
            self.good_seeds.append(path[-1].copy())
        self.queue = path
        self.q_goal = self.queue.pop(0)
        self.speed_scale = speed
        return ok

    def reachable(self, pos, yaw, tilt=0.0, pitch=0.0):
        cands = self.ik_candidates(pos, target_R(yaw, tilt, pitch), self.q_cmd, n_random=4, need=1)
        return len(cands) > 0

    def servo(self):
        dq = self.q_goal - self.q_cmd
        mx = np.max(np.abs(dq))
        lim = self.step_lim * self.speed_scale
        if mx > lim:
            dq = dq * (lim / mx)
        self.q_cmd = np.clip(self.q_cmd + dq, self.lower, self.upper)
        if mx <= lim and self.queue:
            self.q_goal = self.queue.pop(0)

    def arrived(self, tol_q=0.03):
        if self.queue:
            return False
        qerr = np.max(np.abs(self.q_goal - self.q_cmd))
        qmeas = np.max(np.abs(self.q_goal - self.obs["qpos"][:7]))
        return qerr < 1e-4 and qmeas < tol_q

    def cube_pos(self, name):
        return np.array(self.obs["blocks"][name]["pos"], dtype=float)

    def choose_grasp(self, name):
        """Pick a hand yaw (pinch axis aligned with cube faces) that is reachable at the
        pre-grasp and grasp poses."""
        cy = float(self.obs["blocks"][name]["yaw"])
        cp = self.cube_pos(name)
        opts = []
        for k in range(4):
            yaw = wrap_angle(cy + k * np.pi / 2)
            opts.append(yaw)
        # prefer yaws in a comfortable range
        opts.sort(key=lambda y: abs(y - 0.3))
        for tilt, pitch in ((0.0, 0.0), (0.0, 0.3), (0.25, 0.0), (-0.25, 0.0), (0.0, 0.5)):
            for yaw in opts:
                if abs(yaw) > np.deg2rad(110):
                    continue
                c1 = self.ik_candidates([cp[0], cp[1], self.pick_z], target_R(yaw, tilt, pitch), self.q_cmd, n_random=4, need=1)
                if not c1:
                    continue
                c0 = self.ik_candidates([cp[0], cp[1], self.carry_z], target_R(yaw, tilt, pitch), c1[0][1], n_random=4, need=1)
                if not c0:
                    continue
                return yaw, tilt, pitch
        print(f"no reachable grasp for {name}; using default")
        return opts[0], 0.0, 0.0

    def slot_for(self, idx):
        """Target cube position in the bin."""
        bx, by = self.bin_xy
        if idx == 0:
            return np.array([bx, by + 0.027])
        # place second cube beside the first one (axis-aligned separation >= 5.4 cm)
        p1 = self.cube_pos(self.cubes[0])[:2]
        lim = self.bin_in - self.half - 0.002
        cands = [np.array([bx, p1[1] - 0.054]), np.array([bx, p1[1] + 0.054]),
                 np.array([p1[0] + 0.054, p1[1]])]
        for c in cands:
            if abs(c[0] - bx) <= lim + 0.004 and abs(c[1] - by) <= lim + 0.004:
                c = np.array([np.clip(c[0], bx - lim, bx + lim), np.clip(c[1], by - lim, by + lim)])
                return c
        # fallback: farthest slot
        best = None
        for dx in (0.0,):
            for dy in (-0.028, 0.028):
                c = np.array([bx + dx, by + dy])
                dist = np.linalg.norm(c - p1)
                if best is None or dist > best[0]:
                    best = (dist, c)
        print("no clean slot for second cube; using fallback", best[1])
        return best[1]

    def rel_z(self, pitch):
        return self.rim_z - 0.003 + 0.02 * pitch

    def place_site_pos(self, slot, z):
        """Site position so that the held cube ends at slot."""
        R = target_R(self.place_yaw, 0.0, self.place_pitch)
        off = R @ self.grasp_offset
        return np.array([slot[0] - off[0], slot[1] - off[1], z])

    # ---------- main ----------
    def act(self, obs):
        self.obs = obs
        t = obs["t"]
        self.site_pos = np.array(obs["grasp_site_pos"], dtype=float)
        self.site_R = np.array(obs["grasp_site_mat"], dtype=float)
        if not self.kin_initialized:
            self.d.qpos[:] = obs["kin"].qpos
            self.kin_initialized = True
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            self.q_goal = self.q_cmd.copy()
        try:
            self.fsm(t)
        except Exception as e:  # keep running
            print("fsm error", repr(e))
        self.servo()
        return np.concatenate([self.q_cmd, [float(np.clip(self.closure, 0.0, 1.0))]])

    def go(self, phase):
        self.phase = phase
        self.timer = self.obs["t"]

    def fsm(self, t):
        ph = self.phase
        if self.idx >= len(self.cubes):
            if ph != "done":
                self.closure = 0.0
                self.go("done")
            return
        name = self.cubes[self.idx]
        cp = self.cube_pos(name)
        el = t - self.timer

        if ph == "start":
            self.closure = 0.0
            self.go("pre")
            ph = "pre"
        if ph == "pre":
            yaw, tilt, pitch = self.choose_grasp(name)
            self.tilt = tilt
            self.pitch = pitch
            print(f"t={t:.1f} grasp {name} at {np.round(cp,3)} yaw {yaw:.2f} tilt {tilt:.2f} pitch {pitch:.2f}")
            self.set_goal([cp[0], cp[1], self.carry_z], yaw, zmin=self.carry_z - 0.02, tilt=tilt, pitch=pitch)
            self.closure = 0.0
            self.go("to_pre")
        elif ph == "to_pre":
            if (self.arrived() and el > 0.2) or el > 5.0:
                self.set_goal([cp[0], cp[1], self.pick_z], self.yaw, speed=0.6, tilt=self.tilt, pitch=self.pitch, dev=0.025)
                self.go("descend")
        elif ph == "descend":
            if (self.arrived() and el > 0.2) or el > 3.0:
                self.go("close")
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.08)
            if self.closure >= 1.0 and el > 0.9:
                self.set_goal([self.site_pos[0], self.site_pos[1], self.carry_z], self.yaw, speed=0.6, tilt=self.tilt, pitch=self.pitch, dev=0.025)
                self.go("lift")
        elif ph == "lift":
            if (self.arrived() and el > 0.2) or el > 2.5:
                if cp[2] > self.table + self.half + 0.04:
                    # record grasp offset in site frame
                    self.grasp_offset = self.site_R.T @ (cp - self.site_pos)
                    self.grasp_offset[2] = 0.0
                    slot = self.slot_for(self.idx)
                    # choose placement orientation (yaw, pitch) reachable at the release pose
                    found = False
                    for pp in (0.0, 0.3, 0.5, 0.7):
                        for py in (0.0, 0.3, -0.3):
                            self.place_yaw = py
                            self.place_pitch = pp
                            p = self.place_site_pos(slot, self.rel_z(pp))
                            if self.reachable(p, py, 0.0, pp):
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        print("no reachable place pose; using yaw 0 pitch 0.5")
                        self.place_yaw, self.place_pitch = 0.0, 0.5
                    print(f"place {name} at slot {np.round(slot,3)} yaw {self.place_yaw:.2f} pitch {self.place_pitch:.2f}")
                    self.slot = slot
                    p = self.place_site_pos(slot, self.carry_z)
                    self.set_goal(p, self.place_yaw, zmin=self.carry_z - 0.02, pitch=self.place_pitch)
                    self.go("carry")
                else:
                    print(f"grasp failed on {name}, attempt {self.attempt}")
                    self.attempt += 1
                    self.closure = 0.0
                    if self.attempt >= 6:
                        self.idx += 1
                        self.attempt = 0
                    self.go("pre")
        elif ph == "carry":
            if cp[2] < self.table + self.half + 0.03 and el > 0.3:
                print(f"dropped {name} during carry")
                self.closure = 0.0
                self.attempt += 1
                if self.attempt >= 6:
                    self.idx += 1
                    self.attempt = 0
                self.go("pre")
            elif (self.arrived() and el > 0.2) or el > 5.0:
                # refine offset with current observation
                off = self.site_R.T @ (cp - self.site_pos)
                off[2] = 0.0
                self.grasp_offset = off
                p = self.place_site_pos(self.slot, self.rel_z(self.place_pitch))
                self.set_goal(p, self.place_yaw, speed=0.5, pitch=self.place_pitch, dev=0.025)
                self.go("lower")
        elif ph == "lower":
            if (self.arrived() and el > 0.2) or el > 3.0:
                self.go("release")
        elif ph == "release":
            # open the thumb partially (fingers stay curled so they clear the bin walls)
            self.closure = max(0.65, self.closure - 0.05)
            if self.closure <= 0.65 and el > 0.8:
                self.set_goal([self.site_pos[0], self.site_pos[1], self.carry_z], self.place_yaw, speed=0.4, pitch=self.place_pitch, dev=0.025)
                self.go("retreat")
        elif ph == "retreat":
            if self.site_pos[2] > self.rim_z + 0.03:
                self.closure = max(0.0, self.closure - 0.08)
            if (self.arrived() and el > 0.2) or el > 2.5:
                self.closure = 0.0
                self.idx += 1
                self.attempt = 0
                self.go("pre")
