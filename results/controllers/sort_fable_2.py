
import numpy as np
import mujoco


def R_yaw(a):
    x = np.array([np.cos(a), np.sin(a), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    return np.stack([x, y, np.cross(x, y)], 1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


JUMP = 0.35  # max joint change (rad) allowed between consecutive path knots


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower); self.hi = np.array(robot.arm_upper)
        self.hi[1] = min(self.hi[1], 0.3)   # keep the shoulder from crossing far into the torso (collision check guards the rest)
        self.home = np.array(robot.home_arm)
        self.nominal = self.home.copy()
        self.geom_kind = np.full(self.m.ngeom, 2, dtype=int)
        for g in range(self.m.ngeom):
            bname = self.m.body(self.m.geom_bodyid[g]).name
            gname = self.m.geom(g).name or ""
            if gname.startswith("table"):
                self.geom_kind[g] = 1
            elif bname.startswith("block") or bname.startswith("bin") or gname.startswith("bin") or bname == "world":
                self.geom_kind[g] = 2
            else:
                self.geom_kind[g] = 0
        self.jacp = np.zeros((3, self.m.nv)); self.jacr = np.zeros((3, self.m.nv))
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest], float)
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.rng = np.random.RandomState(0)
        self.state = "init"
        self.t_state = 0.0
        self.qt = self.home.copy()
        self.closure = 0.0
        self.knots = []
        self.knot_i = 0
        self.attempts = 0
        self.pre_h = 0.10
        self.carry_z = self.table + 0.14
        self.place_z = self.table + 0.068
        self.max_step = 0.078
        self.grasp_yaw = 0.0
        self.place_yaw = 0.0
        self.place_xy = self.bin.copy()
        self.fixes = 0
        self.cur_yaw = 0.0
        self.grasp_off = 0.005

    # ---------- kinematics ----------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pos, R_des, iters=60, wrot=0.5, bias=0.02):
        q = q0.copy()
        I7 = np.eye(7)
        for _ in range(iters):
            p, R = self.fk(q)
            ep = pos - p
            Re = R_des @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp[:, self.dadr], wrot * self.jacr[:, self.dadr]])
            e = np.concatenate([ep, wrot * er])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = I7 - J.T @ np.linalg.solve(JJt, J)
            dq += bias * (N @ (self.nominal - q))
            q = np.clip(q + 0.7 * dq, self.lo, self.hi)
        p, R = self.fk(q)
        Re = R_des @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        return q, np.linalg.norm(pos - p), np.linalg.norm(er)

    def in_collision(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        for j in range(self.d.ncon):
            con = self.d.contact[j]
            if con.dist > 0.002:
                continue
            k1 = self.geom_kind[con.geom1]; k2 = self.geom_kind[con.geom2]
            if k1 <= 1 and k2 <= 1:
                return True
        return False

    def solve(self, pos, R, seed, extra_seeds=True, tol=0.004):
        """Robust IK: multiple seeds, pick a converged collision-free solution nearest to seed."""
        best = None
        seeds = [seed]
        if extra_seeds:
            seeds += [self.home]
            for _ in range(5):
                seeds.append(self.lo + self.rng.rand(7) * (self.hi - self.lo))
        for i, s in enumerate(seeds):
            q, ep, er = self.ik(s, pos, R)
            ok = ep < tol and er < 0.05
            if ok and self.in_collision(q):
                ok = False
                ep += 0.05
            dist = np.max(np.abs(q - seed))
            key = (0 if ok else 1, 0.0 if ok else ep + 0.02 * er, dist)
            if best is None or key < best[0]:
                best = (key, q, ep, ok)
            if ok and i == 0:
                break
        return best[1], best[2], best[3]

    # ---------- path planning ----------
    def plan(self, q0, yaw0, waypoints, spacing=0.03):
        """Sequential IK along straight Cartesian segments. Returns (knots, ok)."""
        p0, _ = self.fk(q0)
        knots = []
        q = q0.copy()
        prev_p, prev_yaw = p0, yaw0
        ok_all = True
        for (pw, yw) in waypoints:
            pw = np.array(pw, float)
            dist = np.linalg.norm(pw - prev_p)
            dyaw = wrap(yw - prev_yaw)
            n = max(1, int(np.ceil(max(dist / spacing, abs(dyaw) / 0.25))))
            for k in range(1, n + 1):
                s = k / n
                pk = prev_p + s * (pw - prev_p)
                yk = prev_yaw + s * dyaw
                qn, e, ok = self.solve(pk, R_yaw(yk), q, extra_seeds=False)
                if not ok or np.max(np.abs(qn - q)) > JUMP:
                    qn2, e2, ok2 = self.solve(pk, R_yaw(yk), q, extra_seeds=True)
                    if ok2 and np.max(np.abs(qn2 - q)) <= JUMP:
                        qn, e, ok = qn2, e2, ok2
                    elif ok and np.max(np.abs(qn - q)) <= JUMP:
                        pass
                    else:
                        # accept the better of the two but flag the path as awkward
                        ok_all = False
                        if ok2 and (not ok or np.max(np.abs(qn2 - q)) < np.max(np.abs(qn - q))):
                            qn = qn2
                q = qn
                knots.append(q.copy())
            prev_p, prev_yaw = pw, yw
        return knots, ok_all

    def start_path(self, waypoints, spacing=0.03):
        knots, ok = self.plan(self.qt, self.cur_yaw, waypoints, spacing)
        self.knots = knots
        self.knot_i = 0
        self.cur_yaw = waypoints[-1][1]
        return ok

    def follow(self):
        while self.knot_i < len(self.knots):
            goal = self.knots[self.knot_i]
            dq = goal - self.qt
            mx = np.max(np.abs(dq))
            if mx <= self.max_step:
                self.qt = goal.copy()
                self.knot_i += 1
                if mx < 1e-6:
                    continue
                return self.knot_i >= len(self.knots)
            self.qt = self.qt + dq * (self.max_step / mx)
            return False
        return True

    def arrived(self, obs, tol_q=0.04):
        return np.max(np.abs(np.array(obs["qpos"][:7]) - self.qt)) < tol_q

    # ---------- grasp / place selection ----------
    def grasp_pose(self, bp, yaw):
        return np.array([bp[0], bp[1], self.table + self.half]) - R_yaw(yaw)[:, 0] * self.grasp_off

    def choose_grasp(self, obs):
        blocks = obs["blocks"]
        bp = np.array(blocks[self.target]["pos"])
        yaw = float(blocks[self.target]["yaw"])
        p0, _ = self.fk(self.qt)
        pre_z = self.table + self.half + self.pre_h
        cands = []
        for k in range(4):
            a = wrap(yaw + k * np.pi / 2)
            R = R_yaw(a)
            cost = abs(a) * 1.0
            for c, b in blocks.items():
                if c == self.target:
                    continue
                rel = R.T @ (np.array(b["pos"]) - bp)
                if -0.12 < rel[0] < -0.02 and abs(rel[2]) < 0.07:
                    cost += 5.0
                if 0.02 < rel[0] < 0.13 and abs(rel[2]) < 0.07:
                    cost += 0.5
            cands.append((cost, a))
        cands.sort()
        best = None
        for cost, a in cands:
            wps = []
            if p0[2] < self.table + 0.12:
                wps.append((np.array([p0[0], p0[1], self.carry_z]), self.cur_yaw))
            wps.append((np.array([bp[0], bp[1], pre_z]), a))
            wps.append((self.grasp_pose(bp, a), a))
            knots, ok = self.plan(self.qt, self.cur_yaw, wps, spacing=0.03)
            if ok:
                qg = knots[-1]
                margin = np.min(np.minimum(qg - self.lo, self.hi - qg))
                if margin < 0.05:
                    cost += 1.0
            else:
                cost += 10.0
            if best is None or cost < best[0]:
                best = (cost, a)
            if ok and cost < 1.0:
                break
        return best[1]

    def choose_place(self, from_yaw, rel):
        """Pick a place yaw for which carry + lower paths plan OK."""
        order = sorted(np.linspace(-1.2, 1.2, 9), key=lambda a: abs(wrap(a - from_yaw)) * 0.3 + abs(a) * 0.1)
        order = [from_yaw] + list(order)
        fallback = None
        for a in order:
            xy = self.bin - (R_yaw(a) @ rel)[:2]
            wps = [(np.array([xy[0], xy[1], self.carry_z]), a), (np.array([xy[0], xy[1], self.place_z]), a)]
            knots, ok = self.plan(self.qt, self.cur_yaw, wps, spacing=0.03)
            if ok:
                return a, xy
            if fallback is None:
                fallback = (a, xy)
        return fallback

    def set_state(self, s, t):
        self.state = s
        self.t_state = t

    # ---------- main ----------
    def act(self, obs):
        t = obs["t"]
        blocks = obs["blocks"]
        bp = np.array(blocks[self.target]["pos"])
        site = np.array(obs["grasp_site_pos"])
        st = self.state
        el = t - self.t_state

        if st == "init":
            _, R0 = self.fk(self.qt)
            self.cur_yaw = np.arctan2(R0[1, 0], R0[0, 0])
            self.set_state("plan_pre", t)
            st = "plan_pre"

        if st == "plan_pre":
            self.fixes = 0
            self.closure = 0.0
            self.grasp_yaw = self.choose_grasp(obs)
            pre = np.array([bp[0], bp[1], self.table + self.half + self.pre_h])
            p0, _ = self.fk(self.qt)
            wps = []
            if p0[2] < self.table + 0.12:
                wps.append((np.array([p0[0], p0[1], self.carry_z]), self.cur_yaw))
            wps.append((pre, self.grasp_yaw))
            self.start_path(wps, spacing=0.03)
            self.set_state("pre", t)
        elif st == "pre":
            if (self.follow() and self.arrived(obs)) or el > 4.0:
                self.start_path([(self.grasp_pose(bp, self.grasp_yaw), self.grasp_yaw)], spacing=0.02)
                self.set_state("descend", t)
        elif st == "descend":
            if (self.follow() and self.arrived(obs, 0.03)) or el > 2.5:
                want = self.grasp_pose(bp, self.grasp_yaw)
                if np.linalg.norm(site[:2] - want[:2]) > 0.012 and self.fixes < 2 and el <= 2.5:
                    self.fixes += 1
                    self.start_path([(want, self.grasp_yaw)], spacing=0.01)
                    self.set_state("descend", t)
                else:
                    self.set_state("close", t)
        elif st == "close":
            self.closure = 1.0
            if el > 0.6:
                p0, _ = self.fk(self.qt)
                self.start_path([(np.array([p0[0], p0[1], self.carry_z]), self.grasp_yaw)], spacing=0.02)
                self.set_state("lift", t)
        elif st == "lift":
            self.closure = 1.0
            if (self.follow() and self.arrived(obs, 0.05)) or el > 2.0:
                if bp[2] > self.table + self.half + 0.04:
                    Rs = np.array(obs["grasp_site_mat"])
                    rel = Rs.T @ (bp - site)
                    rel[1] = 0.0
                    rel = np.clip(rel, -0.03, 0.03)
                    self.place_yaw, self.place_xy = self.choose_place(self.grasp_yaw, rel)
                    wps = [(np.array([self.place_xy[0], self.place_xy[1], self.carry_z]), self.place_yaw)]
                    self.start_path(wps, spacing=0.03)
                    self.fixes = 0
                    self.set_state("carry", t)
                else:
                    self.attempts += 1
                    self.closure = 0.0
                    self.set_state("reopen", t)
        elif st == "reopen":
            self.closure = 0.0
            if el > 0.5:
                self.set_state("plan_pre", t)
        elif st == "carry":
            self.closure = 1.0
            if (self.follow() and self.arrived(obs, 0.05)) or el > 5.0:
                goal = np.array([self.place_xy[0], self.place_xy[1], self.place_z])
                self.start_path([(goal, self.place_yaw)], spacing=0.02)
                self.set_state("lower", t)
        elif st == "lower":
            self.closure = 1.0
            if (self.follow() and self.arrived(obs, 0.04)) or el > 2.0:
                err = self.bin - bp[:2]
                if bp[2] > self.table + 0.04 and np.linalg.norm(err) > 0.012 and self.fixes < 2 and el <= 2.0:
                    # cube not centred over the bin: shift the hand by the error
                    self.fixes += 1
                    self.place_xy = self.place_xy + np.clip(err, -0.04, 0.04)
                    goal = np.array([self.place_xy[0], self.place_xy[1], self.place_z])
                    self.start_path([(goal, self.place_yaw)], spacing=0.01)
                    self.set_state("lower", t)
                else:
                    self.set_state("release", t)
        elif st == "release":
            self.closure = 0.45 if el < 0.3 else 0.0
            if el > 0.7:
                goal = np.array([self.place_xy[0], self.place_xy[1], self.carry_z])
                self.start_path([(goal, self.place_yaw)], spacing=0.02)
                self.set_state("retreat", t)
        elif st == "retreat":
            self.closure = 0.0
            if self.follow() or el > 2.0:
                self.set_state("done", t)
        elif st == "done":
            self.closure = 0.0
            self.follow()
            inb = (abs(bp[0] - self.bin[0]) < 0.05 and abs(bp[1] - self.bin[1]) < 0.05 and bp[2] < self.table + 0.05)
            if not inb and el > 1.0 and self.attempts < 3 and t < 17.0:
                self.attempts += 1
                self.set_state("plan_pre", t)

        return np.concatenate([self.qt, [self.closure]])
