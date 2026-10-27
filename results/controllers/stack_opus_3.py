
import numpy as np
import mujoco

HALF = 0.025
GOFF = 0.018  # site offset along fingers at pick (fingers pre-curled)


def yaw_rot(heading):
    """Desired site frame: x along heading (horizontal), y down (palm down)."""
    c, s = np.cos(heading), np.sin(heading)
    x = np.array([c, s, 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.task = task
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.target = task["target"]
        self.dest = task["destination"]
        self.phase = "init"
        self.t_phase = 0.0
        self.q_cmd = None
        self.cmd_pos = None
        self.closure = 0.0
        self.heading = 0.0
        self.qref = self.home.copy()
        self.q_goal = None
        rng = np.random.default_rng(1)
        self.seeds = [self.home.copy()] + [
            np.clip(self.home + rng.normal(0, 0.6, 7), self.lo, self.hi) for _ in range(7)]
        self.log = True
        self.carry_z = robot.table_height + 0.11  # held cube center height during carry
        m = self.m
        names = [m.body(i).name for i in range(m.nbody)]
        self.arm_bodies = set(i for i, n in enumerate(names) if n.startswith("right_") and any(
            k in n for k in ("shoulder", "elbow", "wrist", "hand")))
        self.body_bodies = set(i for i, n in enumerate(names) if i > 0 and i not in self.arm_bodies
                               and not n.startswith("block") and not n.startswith("bin") and n != "table")
        self.arm_check = set(i for i in self.arm_bodies if not any(
            k in names[i] for k in ("shoulder_pitch", "shoulder_roll")))
        self.hand_bodies = set(i for i in self.arm_bodies if "hand" in names[i])
        self.block_body = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block_" + c)
                           for c in ("red", "green", "blue")}
        self.hadr = np.array(robot.hand_qpos_adr)
        self.hand_open = np.array([0, 0.25, 0, 0, 0, 0, 0.0])
        self.hand_mid = np.array([0, 0.25, 0, 1.5, 0.4, 1.5, 0.4])
        self.hand_closed = np.array([0, 0.4, -0.4, 1.5, 0.4, 1.5, 0.4])
        self.hand_seq = [self.hand_open,
                         np.array([0, 0.25, 0, 0.4, 0.12, 0.4, 0.12]),
                         np.array([0, 0.25, 0, 0.8, 0.25, 0.8, 0.25]),
                         np.array([0, 0.25, 0, 1.2, 0.35, 1.2, 0.35]),
                         self.hand_mid, self.hand_closed]
        self.hand_pick = [self.hand_mid, np.array([0, 0.32, -0.2, 1.5, 0.4, 1.5, 0.4]), self.hand_closed]
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in self.arm_bodies or b in self.body_bodies:
                m.geom_margin[g] = 0.01

    # ---------------- kinematics / collision ----------------
    def collides(self, q):
        d = self.d
        d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        worst = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            b1 = self.m.geom_bodyid[c.geom1]
            b2 = self.m.geom_bodyid[c.geom2]
            if (b1 in self.arm_check and b2 in self.body_bodies) or (
                    b2 in self.arm_check and b1 in self.body_bodies):
                worst = max(worst, 0.011 - c.dist)
        return worst

    def hand_block_pen(self, q, blocks, hands=None):
        """Penetration (1 cm margin) of hand vs given blocks for open/mid/closed hand."""
        d = self.d
        saved = d.qpos[self.hadr].copy()
        worst = 0.0
        bset = set(self.block_body[b] for b in blocks)
        for hq in (hands if hands is not None else self.hand_seq):
            d.qpos[self.qa] = q
            d.qpos[self.hadr] = hq
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_collision(self.m, d)
            for k in range(d.ncon):
                c = d.contact[k]
                b1 = self.m.geom_bodyid[c.geom1]
                b2 = self.m.geom_bodyid[c.geom2]
                if (b1 in self.hand_bodies and b2 in bset) or (b2 in self.hand_bodies and b1 in bset):
                    worst = max(worst, 0.011 - c.dist)
        d.qpos[self.hadr] = saved
        return worst

    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pos, R, iters=15, wrot=0.3, qref=None, tol=2e-4, kn=0.05):
        if qref is None:
            qref = self.qref
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        err = 1e9
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
            err = np.linalg.norm(ep) + 0.05 * np.linalg.norm(er)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 10 * tol:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.da], wrot * jr[:, self.da]])
            e = np.concatenate([ep, wrot * er])
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (kn * (qref - q))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo + 1e-3, self.hi - 1e-3)
        return q, err

    def solve_multi(self, pos, R, extra_seeds=()):
        best = None
        for s in list(extra_seeds) + self.seeds:
            q, e = self.ik(s, pos, R, iters=80, qref=self.home, tol=3e-4)
            pen = self.collides(q)
            e = e + (0.05 + pen if pen > 0 else 0.0)
            if best is None or e < best[1]:
                best = (q, e)
            if e < 0.002:
                break
        return best

    def pick_poses(self, tp, h):
        xh = np.array([np.cos(h), np.sin(h), 0.0])
        p = tp + GOFF * xh
        return [p + [0, 0, 0.08], p]

    def plan_heading(self, yaw, poses, q_from, avoid=(), place_block=None, hands=None):
        cands = []
        for k in range(4):
            h = wrap(yaw + k * np.pi / 2)
            R = yaw_rot(h)
            sols = []
            tot = 0.0
            prev = q_from
            for p in (poses(h) if callable(poses) else poses):
                q, e = self.solve_multi(p, R, extra_seeds=[prev])
                sols.append(q)
                tot += e
                prev = q
            pen = self.hand_block_pen(sols[-1], avoid, hands=hands) if avoid else 0.0
            if place_block is not None:
                pen = max(pen, self.hand_block_pen(sols[-1], [place_block], hands=[self.d.qpos[self.hadr].copy()]))
            tot += 2.0 * pen
            cost = tot * 100 + 0.1 * np.linalg.norm(sols[0] - q_from) + 0.2 * abs(wrap(h - 0.3))
            cands.append((cost, h, sols, tot, pen))
        cands.sort(key=lambda c: c[0])
        if self.log:
            print("heading plan:", [(round(c[1], 2), round(c[3] * 100, 2), round(c[4] * 100, 2)) for c in cands])
        return cands[0][1], cands[0][2]

    def set_phase(self, name, t):
        if self.log:
            print(f"t={t:.2f} phase {name}")
        self.phase = name
        self.t_phase = t

    # ---------------- main ----------------
    def act(self, obs):
        t = obs["t"]
        kin = obs["kin"]
        self.d.qpos[:] = kin.qpos
        if self.q_cmd is None:
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            self.cmd_pos = np.array(obs["grasp_site_pos"], dtype=float)
        blocks = obs["blocks"]
        tp = np.array(blocks[self.target]["pos"])
        dp = np.array(blocks[self.dest]["pos"])
        site = np.array(obs["grasp_site_pos"])
        qmeas = np.array(obs["qpos"][:7])
        el = t - self.t_phase
        dt = self.r.control_dt

        mode = "cart"
        goal = self.cmd_pos.copy()
        speed = 0.2
        kn = 0.05

        if self.phase == "init":
            self.__dict__.pop("t_dur", None)
            self.heading, sols = self.plan_heading(
                blocks[self.target]["yaw"], lambda h: self.pick_poses(tp, h), self.q_cmd,
                avoid=[b for b in blocks if b != self.target], hands=self.hand_pick)
            self.pick_pos = self.pick_poses(tp, self.heading)[1]
            self.pick_dxy = self.pick_pos[:2] - tp[:2]
            self.q_goal = sols[0]
            self.qref = sols[1]
            self.set_phase("to_pick", t)

        if self.phase == "to_pick":
            mode = "joint"
            self.closure = min(0.5, self.closure + 0.05)
            if (np.max(np.abs(qmeas - self.q_goal)) < 0.05 and np.allclose(self.q_cmd, self.q_goal)) \
                    or el > 3.0:
                self.cmd_pos = self.fk(self.q_cmd)[0]
                self.set_phase("descend", t)
        elif self.phase == "descend":
            goal = tp.copy()
            goal[:2] += self.pick_dxy
            speed = 0.12
            self.closure = 0.5
            if np.linalg.norm(site - goal) < 0.006 or el > 2.5:
                self.set_phase("close", t)
        elif self.phase == "close":
            self.closure = min(1.0, 0.5 + el / 0.6)
            if el > 0.8:
                self.set_phase("lift", t)
        elif self.phase == "lift":
            goal = self.cmd_pos.copy()
            goal[2] = self.carry_z + (site[2] - tp[2])
            speed = 0.15
            if site[2] - tp[2] > 0.035 and site[2] > self.r.table_height + 0.06:
                # grasp failed: open, back off and retry
                self.retries = getattr(self, "retries", 0) + 1
                self.set_phase("regrasp", t)
            elif abs(site[2] - goal[2]) < 0.015:
                off = tp - site
                above = dp - off
                above[2] = self.carry_z - off[2]
                place = dp + [0, 0, 2 * HALF] - off
                self.h0 = self.heading
                self.h1, sols = self.plan_heading(
                    blocks[self.dest]["yaw"], [above, place], self.q_cmd,
                    avoid=[b for b in blocks if b != self.target and b != self.dest],
                    place_block=self.dest)
                self.dh = wrap(self.h1 - self.h0)
                self.q_goal = sols[0]
                self.qref = sols[1]
                self.set_phase("transfer", t)
        elif self.phase == "transfer":
            err = (dp - tp)[:2]
            goal = self.cmd_pos.copy()
            goal[:2] = self.cmd_pos[:2] + err
            goal[2] = self.carry_z - (tp - site)[2]
            speed = float(np.clip(1.5 * np.linalg.norm(err) + 0.03, 0.03, 0.2))
            if not hasattr(self, "t_dur"):
                self.t_dur = max(0.6, 1.3 * np.linalg.norm(err) / 0.15)
            frac = min(1.0, el / self.t_dur)
            self.heading = self.h0 + frac * self.dh
            kn = 0.15
            self.qref = self.q_goal
            if (np.linalg.norm(err) < 0.008 and frac >= 1.0) or el > 5.0:
                self.heading = self.h1
                self.set_phase("lower", t)
        elif self.phase == "lower":
            err = (dp - tp)[:2]
            goal = self.cmd_pos.copy()
            goal[:2] = self.cmd_pos[:2] + err
            dz = (dp[2] + 2 * HALF - 0.004) - tp[2]
            if np.linalg.norm(err) < 0.012:
                goal[2] = self.cmd_pos[2] + dz
            speed = 0.05
            if (tp[2] - dp[2] < 2 * HALF + 0.004 and el > 0.3) or el > 4.0:
                self.set_phase("release", t)
        elif self.phase == "regrasp":
            self.closure = max(0.0, 1.0 - el / 0.3)
            goal = self.cmd_pos.copy()
            goal[2] = self.r.table_height + 0.14
            speed = 0.1
            if el > 1.0 and np.linalg.norm(np.array(blocks[self.target]["linvel"])) < 0.02:
                self.phase = "init"
                self.q_cmd = np.array(obs["arm_target"], dtype=float)
        elif self.phase == "release":
            self.closure = max(0.0, 1.0 - el / 0.3)
            if el > 0.7:
                self.set_phase("retreat", t)
        elif self.phase == "retreat":
            goal = self.cmd_pos.copy()
            goal[2] = dp[2] + 0.15
            speed = 0.1
            self.closure = 0.0
            # placement failed (target fell back to the table): retry whole pick & place
            if el > 1.2 and tp[2] < dp[2] + 0.03 and tp[2] > self.r.table_height \
                    and np.linalg.norm(np.array(blocks[self.target]["linvel"])) < 0.02 \
                    and getattr(self, "replace_tries", 0) < 3 and t < 18.0:
                self.replace_tries = getattr(self, "replace_tries", 0) + 1
                self.phase = "init"
                self.q_cmd = np.array(obs["arm_target"], dtype=float)

        if mode == "joint":
            dq = self.q_goal - self.q_cmd
            mx = np.max(np.abs(dq))
            lim = 1.2 * dt
            if mx > lim:
                dq *= lim / mx
            self.q_cmd = self.q_cmd + dq
        else:
            dvec = goal - self.cmd_pos
            n = np.linalg.norm(dvec)
            step = speed * dt
            if n > step:
                dvec *= step / n
            self.cmd_pos = self.cmd_pos + dvec
            R = yaw_rot(self.heading)
            self.q_cmd, _ = self.ik(self.q_cmd, self.cmd_pos, R, kn=kn)
        return np.concatenate([self.q_cmd, [self.closure]])
