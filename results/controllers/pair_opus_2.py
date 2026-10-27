
import numpy as np
import mujoco


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class IK:
    def __init__(self, robot):
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = self.m.site(robot.grasp_site).id
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper, dtype=float).copy()
        self.lo = self.lo + 0.05
        self.hi = self.hi - 0.05
        self.hi[1] = min(self.hi[1], -0.05)  # shoulder roll: keep upper arm off the torso
        self.home = np.array(robot.home_arm)
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        rng = np.random.default_rng(1)
        self.seeds = [self.home] + [np.clip(self.home + rng.normal(0, 0.6, 7), self.lo, self.hi) for _ in range(6)]

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def solve(self, q0, p, R, iters=30, wrot=0.3, wtilt=0.1):
        W = np.array([wtilt, wtilt, wrot])
        q = np.array(q0, dtype=float).copy()
        err = 1e9
        for it in range(iters + 1):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pc = self.d.site_xpos[self.sid]
            Rc = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = p - pc
            Re = R @ Rc.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            # error measure: position (m) + tilt/yaw (rad scaled)
            err = np.linalg.norm(ep) + 0.02 * np.linalg.norm(er[:2]) + 0.05 * abs(er[2])
            if it == iters or (np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3):
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], W[:, None] * self.jr[:, self.dadr]])
            e = np.concatenate([ep, W * er])
            JJ = J @ J.T + 2e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJ, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJ, J)
            dq += N @ (0.05 * (self.home - q))
            q = np.clip(q + np.clip(dq, -0.25, 0.25), self.lo, self.hi)
        return q, err

    def collides(self, q):
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        for k in range(d.ncon):
            c = d.contact[k]
            if c.dist > -0.002:
                continue
            n1 = self.m.body(self.m.geom_bodyid[c.geom1]).name
            n2 = self.m.body(self.m.geom_bodyid[c.geom2]).name
            if "block" in n1 or "block" in n2:
                continue
            if ("right_" in n1 and "hand" not in n1) or ("right_" in n2 and "hand" not in n2):
                if "right_hand" in n1 and "right_hand" in n2:
                    continue
                return True
        return False

    def solve_multi(self, q0, p, R, **kw):
        best = None
        for s in [q0] + self.seeds:
            q, e = self.solve(s, p, R, iters=80, **kw)
            if self.collides(q):
                e += 0.05
            if best is None or e < best[1] - 1e-4:
                best = (q, e)
            if e < 0.002 and s is q0:
                break
        return best


class Controller:
    CARRY_Z = 0.97
    REL_Z = 0.862      # cube center height at release (bottom just above the rim)

    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.ik = IK(robot)
        self.home = np.array(robot.home_arm)
        _, self.R0 = self.ik.fk(self.home)
        self.fy0 = np.arctan2(self.R0[1, 0], self.R0[0, 0])  # finger yaw at psi=0
        self.cubes = [task["target"], task["second"]]
        self.other = [c for c in ("red", "green", "blue") if c not in self.cubes][0]
        bx, by = robot.bin_xy[task["destination"]]
        self.bin = np.array([bx, by])
        self.slots = [np.array([bx - 0.013, by + 0.0275]), np.array([bx - 0.013, by - 0.0275])]
        self.q_cmd = None
        self.corr = np.zeros(3)
        self.idx = 0
        self.phase = "start"
        self.t_phase = 0.0
        self.closure = 0.0
        self.mode = "hold"
        self.retries = 0

    def R(self, psi):
        return rotz(psi) @ self.R0

    # motion primitives ------------------------------------------------
    def move_joint(self, p, psi, t, speed=0.075):
        """joint-space move to IK solution of (p, psi)"""
        q, e = self.ik.solve_multi(self.q_cmd, p, self.R(psi))
        self.goal_p = np.array(p, float)
        self.goal_psi = psi
        self.q_goal = q
        self.jspeed = speed
        self.mode = "joint"
        self.p_cmd = None
        self.t_move = t
        return e

    def move_line(self, p, psi, t, speed=0.12):
        self.goal_p = np.array(p, float)
        self.goal_psi = psi
        self.lspeed = speed
        self.mode = "line"
        p_now, _ = self.ik.fk(self.q_cmd)
        self.p_cmd = p_now - self.corr
        self.psi_cmd = psi
        self.t_move = t

    def arrived(self, tol=0.006):
        return self.mode == "hold" and np.linalg.norm(self.site - self.goal_p) < tol

    def goto(self, name, t):
        self.phase = name
        self.t_phase = t

    def choose_grasp_psi(self, cpos, cyaw, blocks):
        # 4 symmetric hand yaws; prefer reachable, clamp axis away from neighbours
        best = None
        for k in range(4):
            psi = wrap(cyaw - self.fy0 + k * np.pi / 2)
            R = self.R(psi)
            q1, e1 = self.ik.solve_multi(self.q_cmd, cpos + [0, 0, 0.10], R)
            q2, e2 = self.ik.solve(q1, cpos, R, iters=80)
            fx = R[:2, 0]
            pen = 0.0
            for c, b in blocks.items():
                d = np.array(b["pos"][:2]) - cpos[:2]
                dist = np.linalg.norm(d)
                if dist < 1e-6 or dist > 0.12:
                    continue
                along = abs(fx @ d) / dist
                pen += along * (0.12 - dist) * 0.3
            sc = e1 + e2 + pen + 0.002 * abs(wrap(psi - 0.35))
            if best is None or sc < best[0]:
                best = (sc, psi)
        return best[1]

    # main -------------------------------------------------------------
    def act(self, obs):
        t = obs["t"]
        self.site = np.array(obs["grasp_site_pos"])
        Rs = np.array(obs["grasp_site_mat"])
        blocks = obs["blocks"]
        if self.q_cmd is None:
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            self.goal_p = self.site.copy()
            self.goal_psi = 0.0
        ph = self.phase
        el = t - self.t_phase
        name = self.cubes[min(self.idx, 1)]
        cpos = np.array(blocks[name]["pos"])
        cyaw = blocks[name]["yaw"]

        if ph == "start":
            self.closure = 0.0
            self.grasp_psi = self.choose_grasp_psi(cpos, cyaw, blocks)
            self.move_joint([cpos[0], cpos[1], max(self.site[2], cpos[2] + 0.10)], self.grasp_psi, t)
            self.goto("above", t)
        elif ph == "above":
            if self.arrived(0.01) or el > 4:
                self.move_line(cpos + [0, 0, 0.0], self.grasp_psi, t, 0.12)
                self.goto("descend", t)
        elif ph == "descend":
            if self.arrived(0.004) or el > 3:
                self.goto("close", t)
        elif ph == "close":
            self.closure = min(1.0, el / 0.5)
            if el > 0.8:
                self.move_line([self.goal_p[0], self.goal_p[1], self.CARRY_Z], self.goal_psi, t, 0.15)
                self.goto("lift", t)
        elif ph == "lift":
            self.closure = 1.0
            if self.site[2] > cpos[2] + 0.05 and abs(cpos[2] - self.site[2]) > 0.06:
                # cube not following: failed grasp
                self.goto("regrasp", t)
                self.closure = 0.0
            elif self.arrived(0.012) or el > 3:
                if cpos[2] < 0.86:
                    self.goto("regrasp", t)
                    self.closure = 0.0
                else:
                    self.plan_place(cpos, cyaw, Rs, t, blocks)
                    self.goto("carry", t)
        elif ph == "regrasp":
            self.closure = 0.0
            if el > 0.6:
                self.retries += 1
                self.goto("start", t)
        elif ph == "carry":
            if cpos[2] < 0.86:
                self.closure = 0.0
                self.goto("regrasp", t)
            elif self.arrived(0.01) or el > 5:
                # refine offset using current measurement
                off = cpos - self.site
                slot = self.slot
                self.p_down = np.array([slot[0], slot[1], self.rel_z]) - off
                self.move_line(self.p_down, self.place_psi, t, 0.15)
                self.goto("lower", t)
        elif ph == "lower":
            blocked = el > 0.6 and self.site[2] > self.goal_p[2] + 0.004 and abs(obs["qvel"][:7]).max() < 0.03
            if self.arrived(0.005) or el > 3 or blocked:
                self.goto("release", t)
        elif ph == "release":
            # open the thumb slowly, then slide the curled fingers up off the cube
            self.closure = max(0.45, 1.0 - el / 0.5)
            if el > 0.7:
                self.move_line([self.goal_p[0], self.goal_p[1], self.goal_p[2] + 0.05], self.goal_psi, t, 0.10)
                self.goto("release2", t)
        elif ph == "release2":
            self.closure = 0.45
            if self.arrived(0.01) or el > 1.0:
                self.closure = 0.0
                self.move_line([self.goal_p[0], self.goal_p[1], self.CARRY_Z], self.goal_psi, t, 0.15)
                self.goto("retreat", t)
        elif ph == "retreat":
            self.closure = 0.0
            if self.arrived(0.015) or el > 3:
                self.idx += 1
                if self.idx >= 2 and t < 29.0:
                    def ok(c):
                        p = np.array(blocks[c]["pos"])
                        return np.all(np.abs(p[:2] - self.bin) < 0.05) and p[2] < 0.822
                    a, b = self.cubes
                    if not ok(b):
                        self.idx = 1
                    elif not ok(a):
                        self.cubes = [b, a]
                        self.idx = 1
                    if self.idx == 1:
                        print("fix", round(t, 2), self.cubes)
                if self.idx >= 2:
                    self.goto("park", t)
                    self.mode = "home"
                else:
                    self.goto("start", t)
        elif ph == "park":
            self.closure = 0.0

        self.update_command(t)
        return np.concatenate([self.q_cmd, [self.closure]])

    def slot_offset(self, theta):
        th = abs(wrap(theta * 4) / 4)  # fold to [0, pi/4]
        s = np.cos(th) + np.sin(th)
        hi = 0.055 - 0.025 * s - 0.0015
        return hi

    def plan_place(self, cpos, cyaw, Rs, t, blocks):
        off_local = Rs.T @ (cpos - self.site)
        psi_now = np.arctan2(Rs[1, 0], Rs[0, 0]) - self.fy0
        cands = []
        if self.idx == 0:
            corners = [(1, -1), (1, 1), (-1, 1), (-1, -1)]
        else:
            first = np.array(blocks[self.cubes[0]]["pos"][:2]) - self.bin
            sx = -1 if first[0] > 0 else 1
            sy = -1 if first[1] > 0 else 1
            corners = [(sx, sy)]
        best = None
        for (sx, sy) in corners:
            for psi in np.radians(np.arange(-45, 106, 15)):
                cyaw_new = cyaw + wrap(psi - psi_now)
                o = self.slot_offset(cyaw_new)
                slot = self.bin + [sx * o, sy * o]
                th = abs(wrap(cyaw_new * 4) / 4)
                off = self.R(psi) @ off_local
                p1 = np.array([slot[0], slot[1], self.CARRY_Z]) - [off[0], off[1], 0]
                q1, e1 = self.ik.solve(self.q_cmd, p1, self.R(psi), iters=60)
                if self.ik.collides(q1):
                    e1 += 0.05
                if e1 > 0.02:
                    continue
                e_opp = 0.0
                if self.idx == 0:
                    slot_o = self.bin - [sx * o, sy * o]
                    best_o = 1.0
                    for dpsi in (-0.26, 0.0, 0.26):
                        po = np.array([slot_o[0], slot_o[1], 0.83]) - self.R(psi + dpsi) @ off_local
                        _, eo = self.ik.solve(q1, po, self.R(psi + dpsi), iters=60)
                        best_o = min(best_o, eo)
                    e_opp = best_o
                for rz in (0.835, self.REL_Z):
                    p2 = np.array([slot[0], slot[1], rz]) - off
                    q, e2 = self.ik.solve(q1, p2, self.R(psi), iters=60)
                    if self.ik.collides(q):
                        e2 += 0.05
                    sc = e1 + 3 * e2 + 3 * e_opp + 0.002 * abs(wrap(psi - psi_now)) + 0.012 * th + 0.1 * (rz - 0.818)
                    if best is None or sc < best[0]:
                        best = (sc, psi, p1, slot, e1, e2, rz)
        sc, self.place_psi, p_over, self.slot, e1, e2, self.rel_z = best
        print("place", round(t, 2), "psi", round(np.degrees(self.place_psi)), "slot", (self.slot - self.bin).round(3), round(e1, 4), round(e2, 4), self.rel_z)
        self.move_joint(p_over, self.place_psi, t, 0.08)

    def update_command(self, t):
        if self.mode == "home":
            q_t = self.home
            self.q_cmd = self.q_cmd + np.clip(q_t - self.q_cmd, -0.05, 0.05)
            return
        if self.mode == "joint":
            d = self.q_goal - self.q_cmd
            self.q_cmd = self.q_cmd + np.clip(d, -self.jspeed, self.jspeed)
            if np.max(np.abs(d)) < 1e-3:
                self.mode = "hold"
                self.p_cmd = self.goal_p.copy()
                self.psi_cmd = self.goal_psi
            return
        if self.mode == "line":
            step = self.lspeed * 0.04
            dvec = self.goal_p - self.p_cmd
            n = np.linalg.norm(dvec)
            if n <= step:
                self.p_cmd = self.goal_p.copy()
                self.mode = "hold"
            else:
                self.p_cmd = self.p_cmd + dvec * step / n
        if self.mode == "hold":
            e = self.goal_p - self.site
            if np.linalg.norm(e) < 0.03:
                self.corr = np.clip(self.corr + 0.2 * e, -0.03, 0.03)
        q_t, _ = self.ik.solve(self.q_cmd, self.p_cmd + self.corr, self.R(self.psi_cmd), iters=15)
        self.q_cmd = self.q_cmd + np.clip(q_t - self.q_cmd, -0.08, 0.08)
