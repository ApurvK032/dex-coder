
import numpy as np
import mujoco


def rot_yaw_down(yaw):
    """Site rotation: x axis (fingers) horizontal at `yaw`, y axis (palm normal) pointing down."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def rotvec(Rerr):
    v = np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
    c = np.clip((np.trace(Rerr) - 1.0) / 2.0, -1.0, 1.0)
    s = np.linalg.norm(v) / 2.0
    ang = np.arctan2(s, c)
    if ang < 1e-9:
        return np.zeros(3), 0.0
    return v / (2.0 * s) * ang, ang


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.dt = robot.control_dt
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.bin_xy = {k: np.array(v, dtype=float) for k, v in robot.bin_xy.items()}
        self.bin_half = robot.bin_inner_half_size
        self.wall_t = robot.bin_wall_thickness
        self.rim = self.table + robot.bin_wall_height
        self.order = [task["target"], task["second"]]
        self.dest = task["destination"]
        self.all_names = ["red", "green", "blue"]
        self.q_cmd = self.home.copy()
        self.closure = 0.0
        self.obs = None
        self.gen = None
        self.last_action = np.concatenate([self.home, [0.0]])
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.z_carry = self.table + 0.18
        # grasp parameters (found empirically)
        self.g_dx = 0.0
        self.g_dz = -0.01
        self.g_close = 0.6
        self.slot_off = 0.028
        # obstacle points for the bin walls (top of the walls)
        self.wall_pts = []
        for k, c in self.bin_xy.items():
            r = self.bin_half + self.wall_t / 2.0
            for s in np.linspace(-r, r, 13):
                self.wall_pts.append(np.array([c[0] + s, c[1] - r, self.rim]))
                self.wall_pts.append(np.array([c[0] + s, c[1] + r, self.rim]))
                self.wall_pts.append(np.array([c[0] - r, c[1] + s, self.rim]))
                self.wall_pts.append(np.array([c[0] + r, c[1] + s, self.rim]))

    # ------------------------------------------------------------ kinematics
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, pos, R, q0, iters=30, damp=0.01):
        q = np.array(q0, dtype=float).copy()
        ep = np.zeros(3)
        ang = 0.0
        for _ in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            p = self.d.site_xpos[self.sid]
            Rc = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = pos - p
            er, ang = rotvec(R @ Rc.T)
            if np.linalg.norm(ep) < 2e-4 and ang < 2e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            e = np.concatenate([ep, 0.5 * er])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
            q = np.clip(q + 0.6 * dq, self.lo, self.hi)
        return q, float(np.linalg.norm(ep)), float(ang)

    def reach_err(self, pos, yaw, q0=None):
        q, ep, ea = self.ik(np.asarray(pos, float), rot_yaw_down(yaw), self.home if q0 is None else q0, iters=80)
        return ep, ea, q

    # ------------------------------------------------------------ interface
    def act(self, obs):
        self.obs = obs
        if self.gen is None:
            self.gen = self.run()
        try:
            a = next(self.gen)
            self.last_action = a
        except StopIteration:
            a = self.last_action
        except Exception as e:  # pragma: no cover
            print("controller exception:", repr(e))
            a = self.last_action
            self.gen = iter(())
        return np.array(a, dtype=float)

    def action(self):
        return np.concatenate([self.q_cmd, [self.closure]])

    # ------------------------------------------------------------ primitives
    def move_to(self, pos, yaw, speed=0.4, timeout=5.0, tol=0.004, settle=True, accel=1.0,
                correct=False, measure=None, corr_tol=0.002, max_corr=4, shift=None):
        """Cartesian straight-line move of the (commanded) grasp site, palm down.
        With correct=True the commanded target is shifted after settling so that the
        measured position (default: grasp site; or measure()) ends up at `pos`."""
        pos = np.asarray(pos, float)
        goal = pos.copy() if shift is None else pos + np.asarray(shift, float)
        R = rot_yaw_down(yaw)
        t0 = self.obs["t"]
        rot_max = 1.5 * self.dt
        v = 0.0
        n_corr = 0
        while True:
            p_cmd, R_cmd = self.fk(self.q_cmd)
            dp = goal - p_cmd
            dist = np.linalg.norm(dp)
            er, ang = rotvec(R @ R_cmd.T)
            v = min(v + accel * self.dt, speed, np.sqrt(2.0 * accel * max(dist - 0.002, 0.0)) + 0.02)
            step_max = v * self.dt
            if dist > step_max:
                dp = dp / dist * step_max
            sub_p = p_cmd + dp
            if ang > rot_max:
                er = er / ang * rot_max
            Rsub = rodrigues(er) @ R_cmd if ang > 1e-6 else R_cmd
            q, ep, ea = self.ik(sub_p, Rsub, self.q_cmd, iters=15)
            self.q_cmd = q
            yield self.action()
            t = self.obs["t"]
            if dist < tol and ang < 0.02:
                if not settle:
                    return True
                qv = np.max(np.abs(self.obs["qvel"][:7]))
                if qv < 0.25:
                    if not correct or n_corr >= max_corr:
                        return True
                    meas = np.array(self.obs["grasp_site_pos"], float) if measure is None else np.asarray(measure(), float)
                    err = pos - meas
                    if np.linalg.norm(err) < corr_tol:
                        return True
                    goal = goal + np.clip(err, -0.02, 0.02)
                    n_corr += 1
                    v = 0.0
            if t - t0 > timeout:
                return False

    def wait(self, dur):
        t0 = self.obs["t"]
        while self.obs["t"] - t0 < dur:
            yield self.action()

    def set_closure(self, c, dur):
        self.closure = float(c)
        yield from self.wait(dur)

    # ------------------------------------------------------------ task logic
    def block(self, name):
        return np.array(self.obs["blocks"][name]["pos"], float), float(self.obs["blocks"][name]["yaw"])

    def in_bin(self, pos, which, margin=0.005):
        c = self.bin_xy[which]
        return (abs(pos[0] - c[0]) < self.bin_half - margin and abs(pos[1] - c[1]) < self.bin_half - margin
                and pos[2] < self.rim - margin)

    def obstacle_penalty(self, pos, yaw, name):
        R = rot_yaw_down(yaw)
        pts = [np.array(self.obs["blocks"][n]["pos"], float) for n in self.all_names if n != name]
        pts += self.wall_pts
        pen = 0.0
        for o in pts:
            r = R.T @ (o - pos)
            if abs(r[1]) > 0.08:
                continue
            # finger sweep zone (+x side) and thumb zone (-x side)
            if 0.0 < r[0] < 0.125 and abs(r[2]) < 0.065:
                pen += 1.0 + (0.125 - r[0]) * 10
            elif -0.095 < r[0] <= 0.0 and abs(r[2]) < 0.05:
                pen += 1.0 + (r[0] + 0.095) * 10
            elif np.linalg.norm(r[[0, 2]]) < 0.07:
                pen += 0.5
        return pen

    def choose_grasp_yaw(self, pos, cube_yaw, name):
        cands = []
        for k in range(-4, 5):
            y = cube_yaw + k * np.pi / 2
            if -2.0 < y < 2.0:
                cands.append(y)
        pref = -0.2
        scored = []
        for y in cands:
            ep1, ea1, q1 = self.reach_err([pos[0], pos[1], pos[2] + self.g_dz], y)
            ep2, ea2, q2 = self.reach_err([pos[0], pos[1], pos[2] + 0.12], y, q1)
            reach_ok = ep1 < 0.003 and ea1 < 0.03 and ep2 < 0.004 and ea2 < 0.05
            pen = self.obstacle_penalty(pos, y, name)
            score = (0 if reach_ok else 100 + 1000 * (ep1 + ep2)) + pen * 10 + abs(y - pref)
            scored.append((score, y))
        scored.sort(key=lambda s: s[0])
        return scored[0][1]

    def choose_drop(self, cube_xy, z_site):
        for yaw in [0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3]:
            pos = np.array([cube_xy[0], cube_xy[1], z_site])
            ep, ea, q = self.reach_err(pos, yaw)
            if ep < 0.003 and ea < 0.03:
                return pos, yaw
        for dz in [0.02, 0.04, 0.06]:
            for yaw in [0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3]:
                pos = np.array([cube_xy[0], cube_xy[1], z_site + dz])
                ep, ea, q = self.reach_err(pos, yaw)
                if ep < 0.003 and ea < 0.03:
                    return pos, yaw
        return np.array([cube_xy[0], cube_xy[1], z_site]), -0.2

    def pick(self, name):
        pos, cyaw = self.block(name)
        yaw = self.choose_grasp_yaw(pos, cyaw, name)
        above = np.array([pos[0], pos[1], pos[2] + 0.12])
        self.closure = 0.0
        yield from self.move_to(above, yaw, speed=0.5, timeout=6.0, tol=0.006)
        pos, cyaw = self.block(name)
        xdir = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        grasp = np.array([pos[0], pos[1], pos[2] + self.g_dz]) + self.g_dx * xdir
        # descend to just above the grasp height with xy correction, then down
        pre = grasp + np.array([0, 0, 0.03])
        yield from self.move_to(pre, yaw, speed=0.25, timeout=3.0, tol=0.003, correct=True,
                                measure=lambda: self.obs["grasp_site_pos"][:2].tolist() + [pre[2]], corr_tol=0.003)
        yield from self.move_to(grasp, yaw, speed=0.15, timeout=2.0, tol=0.003, correct=True,
                                measure=lambda: self.obs["grasp_site_pos"][:2].tolist() + [grasp[2]], corr_tol=0.004, max_corr=2)
        yield from self.set_closure(self.g_close, 0.6)
        lift = np.array([grasp[0], grasp[1], self.z_carry])
        yield from self.move_to(lift, yaw, speed=0.3, timeout=4.0, tol=0.01, settle=False, accel=0.6)
        bp, _ = self.block(name)
        site = np.array(self.obs["grasp_site_pos"])
        ok = bp[2] > self.table + 0.06 and np.linalg.norm(bp - site) < 0.05
        print(f"pick {name} ok={ok} yaw={yaw:.2f} t={self.obs['t']:.1f}")
        return ok, yaw

    def place(self, name, slot, yaw_hold):
        c = self.bin_xy[self.dest]
        goal_xy = np.array([c[0], c[1] + (self.slot_off if slot == 0 else -self.slot_off)])
        above = np.array([goal_xy[0], goal_xy[1], self.z_carry])
        yield from self.move_to(above, 0.0, speed=0.4, timeout=6.0, tol=0.006, accel=0.8)
        yield from self.wait(0.3)
        # compensate for where the cube actually sits in the hand
        bp, _ = self.block(name)
        site = np.array(self.obs["grasp_site_pos"])
        off = bp - site
        if np.linalg.norm(off) > 0.06:
            print(f"place {name}: cube not in hand (off={np.round(off,3)})")
            return False
        z_site = self.rim + 0.012 + self.half - off[2]
        dpos, dyaw = self.choose_drop(goal_xy - off[:2], z_site)
        yield from self.move_to([dpos[0], dpos[1], self.z_carry], dyaw, speed=0.3, timeout=3.0, tol=0.004, accel=0.8)
        # desired: the *cube* at goal_xy, site at dpos z; correct using the measured cube position
        want = np.array([goal_xy[0], goal_xy[1], dpos[2]])
        yield from self.move_to(want, dyaw, speed=0.2, timeout=4.0, tol=0.003, accel=0.6, correct=True,
                                shift=[-off[0], -off[1], 0.0],
                                measure=lambda: list(self.obs["blocks"][name]["pos"][:2]) + [self.obs["grasp_site_pos"][2]],
                                corr_tol=0.003)
        yield from self.set_closure(0.0, 0.6)
        yield from self.move_to([dpos[0], dpos[1], self.z_carry], dyaw, speed=0.3, timeout=3.0, tol=0.01, settle=False)
        bp, _ = self.block(name)
        ok = self.in_bin(bp, self.dest)
        print(f"place {name} slot={slot} ok={ok} pos={np.round(bp,3)} t={self.obs['t']:.1f}")
        return ok

    def run(self):
        yield self.action()
        for i, name in enumerate(self.order):
            force = False
            for attempt in range(5):
                bp, _ = self.block(name)
                if self.in_bin(bp, self.dest) and not force:
                    break
                force = False
                ok, yaw = yield from self.pick(name)
                if not ok:
                    self.closure = 0.0
                    yield from self.wait(0.3)
                    p_cmd, _ = self.fk(self.q_cmd)
                    yield from self.move_to([p_cmd[0], p_cmd[1], self.z_carry], yaw, speed=0.4, timeout=3.0, tol=0.01, settle=False)
                    continue
                ok = yield from self.place(name, i, yaw)
                if ok:
                    bp, _ = self.block(name)
                    c = self.bin_xy[self.dest]
                    slot_y = c[1] + (self.slot_off if i == 0 else -self.slot_off)
                    if i == 0 and attempt < 3 and abs(bp[1] - slot_y) > 0.017:
                        print(f"re-placing {name}: y off by {bp[1]-slot_y:.3f}")
                        continue
                    break
        # retreat
        self.closure = 0.0
        p_cmd, _ = self.fk(self.q_cmd)
        yield from self.move_to([p_cmd[0], p_cmd[1], self.z_carry + 0.05], 0.0, speed=0.4, timeout=3.0, tol=0.01, settle=False)
        while True:
            dq = np.clip(self.home - self.q_cmd, -0.06, 0.06)
            self.q_cmd = self.q_cmd + dq
            yield self.action()
