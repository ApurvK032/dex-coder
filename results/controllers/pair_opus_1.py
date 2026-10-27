
import numpy as np, mujoco

COLORS = ("red", "green", "blue")


def rot_heading(h, tilt=0.0):
    """Site frame: x horizontal at heading h (pitched down by tilt), y down-ish, z = x cross y."""
    c, s = np.cos(h), np.sin(h)
    x = np.array([c * np.cos(tilt), s * np.cos(tilt), -np.sin(tilt)])
    y = np.array([-c * np.sin(tilt), -s * np.sin(tilt), -np.cos(tilt)])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def axrot(a, ang):
    a = np.array(a, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def rot(h, roll=0.0):
    return rot_heading(h) @ axrot([1, 0, 0], roll)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class IK:
    def __init__(self, robot):
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.dadr = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower) + 0.02
        self.hi = np.array(robot.arm_upper) - 0.02
        self.home = np.array(robot.home_arm)
        self.qn = np.array([-0.08, -0.6, 0.24, -0.3, -1.04, 0.38, 0.21])
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    @staticmethod
    def rot_err(Rc, R):
        return 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))

    def solve(self, q0, p, R, iters=30, wrot=0.3, tol=5e-4, qpref=None):
        if qpref is None:
            qpref = self.qn
        q = np.array(q0, float).copy()
        for it in range(iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pc = self.d.site_xpos[self.sid]
            Rc = self.d.site_xmat[self.sid].reshape(3, 3)
            ep = p - pc
            er = self.rot_err(Rc, R)
            if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 0.005:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.dadr], wrot * self.jr[:, self.dadr]])
            e = np.r_[ep, wrot * er]
            JJt = J @ J.T + 2e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.1 * (qpref - q))
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, self.lo, self.hi)
        pc, Rc = self.fk(q)
        return q, np.linalg.norm(p - pc), np.linalg.norm(self.rot_err(Rc, R))


def robust_solve(ik, q0, p, R, iters=15):
    q, e, r = ik.solve(q0, p, R, iters=iters)
    if e > 0.004 or r > 0.03:
        q2, e2, r2 = ik.solve(ik.qn, p, R, iters=100)
        if e2 + 0.1 * r2 < e + 0.1 * r - 0.002:
            return q2, e2, r2
    return q, e, r


ROLLS = (0.0, -0.2, -0.4, -0.6)
REL_Z = 0.826  # cube center height at release


def rot_to(Rcur, Rgoal, maxang):
    Rrel = Rgoal @ Rcur.T
    c = np.clip((np.trace(Rrel) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang <= maxang or ang < 1e-6:
        return Rgoal.copy(), True
    ax = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    n = np.linalg.norm(ax)
    if n < 1e-9:
        ax = np.array([0, 0, 1.0]); n = 1.0
    return axrot(ax / n, maxang) @ Rcur, False


def snap_rot(R):
    A = np.abs(R).copy()
    S = np.zeros((3, 3))
    for _ in range(3):
        i, j = np.unravel_index(np.argmax(A), A.shape)
        S[i, j] = np.sign(R[i, j])
        A[i, :] = -1; A[:, j] = -1
    if np.linalg.det(S) < 0:
        j = np.argmin([abs(R[:, k] @ S[:, k]) for k in range(3)])
        S[:, j] *= -1
    return S


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.ik = IK(robot)
        self.dt = robot.control_dt
        self.q_cmd = np.array(robot.home_arm, float)
        self.closure = 0.0
        bx, by = robot.bin_xy[task["destination"]]
        self.bin = np.array([bx, by])
        self.queue = [task["target"], task["second"]]
        # right bin: the -y side is hard to reach low, so fill it second (dropped from a little higher)
        self.s0 = 1.0 if by < -0.2 else -1.0
        self.slots = [np.array([bx - 0.008, by + self.s0 * 0.0285]), np.array([bx - 0.004, by - self.s0 * 0.0285])]
        self.rel_z = [0.826, 0.845]
        self.press_tries = 0
        self.press_xy = np.zeros(2)
        self.placed = 0
        self.phase = "prep"
        self.pt = 0.0
        self.fresh = True
        self.attempts = 0
        self.goal = None
        self.log = False
        self.hover_z = 0.90
        self.bad_h = []
        self.zmin = {}
        for si, s in enumerate(self.slots):
            for roll in ROLLS:
                self.zmin[(si, roll)] = self.min_z(np.r_[s, 0.0], np.zeros(3), rot(0.0, roll), 0.815)

    def min_z(self, slot3, c_s, R, z0):
        for zc in np.arange(z0, 0.90, 0.01):
            p = np.r_[slot3[:2], zc] - R @ c_s
            for q0 in (self.ik.qn, self.ik.home):
                q, e, r = self.ik.solve(q0, p, R, iters=100)
                if e < 0.002 and r < 0.02:
                    return zc
        return 0.95

    # ---- helpers
    def set_goal(self, p, R=None, v=0.25, w=1.5):
        self.goal = (np.array(p, float), self.cmd_R.copy() if R is None else R, v, w)

    def step_goal(self):
        p, R, v, w = self.goal
        d = p - self.cmd_p
        n = np.linalg.norm(d)
        st = v * self.dt
        self.cmd_p = p.copy() if n <= st else self.cmd_p + d * st / n
        self.cmd_R, rdone = rot_to(self.cmd_R, R, w * self.dt)
        return n <= st and rdone

    def choose_grasp(self, pos, yaw, pref=-10.0):
        best = None
        si = self.placed
        for roll in ROLLS:
            zm = self.zmin[(si, roll)]
            zcost = 8.0 * max(0.0, zm - 0.825) + 0.1 * abs(roll)
            if best is not None and zcost > best[0]:
                continue
            for k in range(4):
                h = wrap(yaw + k * np.pi / 2)
                cost = zcost + 0.3 * abs(wrap(h - np.radians(pref)))
                cost += 0.5 * sum(1 for hb in self.bad_h if abs(wrap(h - hb)) < 0.2)
                if best is not None and cost > best[0]:
                    continue
                for z in (pos[2], pos[2] + 0.04, self.hover_z):
                    q, e, r = self.ik.solve(self.ik.qn, np.r_[pos[:2], z], rot(h, roll), iters=80)
                    cost += e * 20 + r
                if best is None or cost < best[0]:
                    best = (cost, h, roll)
        if self.log:
            print("grasp choice cost %.3f heading %.0f roll %.1f" % (best[0], np.degrees(best[1]), best[2]))
        return best[1], best[2]

    def plan_place(self, obs):
        b = np.array(obs["blocks"][self.cur]["pos"])
        quat = np.array(obs["blocks"][self.cur]["quat"], float)
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, quat)
        Rc = Rc.reshape(3, 3)
        sp = np.array(obs["grasp_site_pos"])
        Rs = np.array(obs["grasp_site_mat"]).reshape(3, 3)
        c_s = Rs.T @ (b - sp)
        Rsnap = snap_rot(Rc)
        slot = self.slots[self.placed]
        best = None
        for k in range(4):
            Rnew = rotz(k * np.pi / 2) @ Rsnap @ Rc.T @ Rs
            ang = np.arccos(np.clip((np.trace(Rnew @ Rs.T) - 1) / 2, -1, 1))
            if ang > 1.6:
                continue
            q, e, r = self.ik.solve(self.ik.qn, np.r_[slot, self.hover_z] - Rnew @ c_s, Rnew, iters=100)
            if e > 0.003 or r > 0.03:
                continue
            zm = self.min_z(np.r_[slot, 0.0], c_s, Rnew, REL_Z)
            cost = zm + 0.01 * ang
            if best is None or cost < best[0]:
                best = (cost, Rnew, zm, ang)
        if best is None:
            Rnew = rot(0.0, 0.0)
            zm = 0.87
        else:
            _, Rnew, zm, ang = best
        if self.log:
            print("plan place: c_s", np.round(c_s, 3),
                  "cube tilt %.1f deg" % np.degrees(np.arccos(np.clip(np.max(np.abs(Rc[2])), -1, 1))),
                  "zc %.3f" % zm, "found", best is not None)
        self.place_R = Rnew
        self.c_s = c_s
        self.place_zc = max(self.rel_z[min(self.placed, 1)], zm + 0.003)

    def enter(self, ph, t):
        if self.log:
            print("t=%.2f phase %s" % (t, ph))
        self.phase = ph
        self.pt = t
        self.fresh = True

    # ---- main
    def act(self, obs):
        t = obs["t"]
        blocks = obs["blocks"]
        sp = np.array(obs["grasp_site_pos"])
        el = t - self.pt
        ph = self.phase

        if ph == "prep":
            dq = self.ik.qn - self.q_cmd
            self.q_cmd = self.q_cmd + np.clip(dq, -0.06, 0.06)
            if np.max(np.abs(dq)) < 1e-3:
                p, R = self.ik.fk(self.q_cmd)
                self.cmd_p = p
                self.cmd_R = R
                self.set_goal(p, v=0.2)
                self.enter("start", t)
            return np.r_[self.q_cmd, self.closure]

        # ---- set goals for this phase
        if ph == "to_hover":
            b = np.array(blocks[self.cur]["pos"])
            self.set_goal(np.r_[b[:2], self.hover_z], rot(self.grasp_h, self.grasp_roll), v=0.3)
        elif ph == "descend":
            b = np.array(blocks[self.cur]["pos"])
            self.set_goal(np.r_[b[:2], b[2] + 0.005], rot(self.grasp_h, self.grasp_roll), v=0.12)
        elif ph == "lower":
            b = np.array(blocks[self.cur]["pos"])
            slot = self.slots[self.placed]
            err = slot - b[:2]
            if el > 0.3:
                self.corr += 0.2 * err
                self.corr = np.clip(self.corr, -0.03, 0.03)
            yerr = blocks[self.cur]["yaw"]
            yerr = (yerr + np.pi / 4) % (np.pi / 2) - np.pi / 4
            if el > 0.3:
                self.corr_yaw = np.clip(self.corr_yaw - 0.25 * yerr, -0.3, 0.3)
            Rp = rotz(self.corr_yaw) @ self.place_R
            ok = np.linalg.norm(err) < 0.006 and abs(yerr) < 0.05
            zc = self.place_zc if ok else max(self.place_zc, min(b[2], 0.87))
            tgt = np.r_[slot + self.corr, zc] - Rp @ self.c_s
            self.set_goal(tgt, Rp, v=0.06)
            self.yerr = yerr
        elif ph == "idle":
            self.set_goal(np.r_[0.2, -0.35, 0.95], rot(0.0, 0.0), v=0.2)

        done = self.step_goal() and not self.fresh
        self.fresh = False

        # ---- transitions
        if ph == "start":
            if self.placed >= 2 or not self.queue:
                self.enter("idle", t)
            else:
                self.cur = self.queue[0]
                b = np.array(blocks[self.cur]["pos"])
                self.grasp_h, self.grasp_roll = self.choose_grasp(b + [0, 0, 0.005], blocks[self.cur]["yaw"])
                self.closure = 0.0
                self.enter("to_hover", t)
        elif ph == "to_hover":
            if done and np.linalg.norm(sp - self.cmd_p) < 0.01:
                self.enter("descend", t)
            elif el > 6.0:
                self.attempts += 1
                self.bad_h.append(self.grasp_h)
                self.enter("start", t)
        elif ph == "descend":
            if done and (np.linalg.norm(sp - self.cmd_p) < 0.006 or el > 2.5):
                self.enter("close", t)
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.06)
            if el > 1.0:
                self.enter("lift", t)
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.15)
        elif ph == "lift":
            b = np.array(blocks[self.cur]["pos"])
            if done and el > 0.3:
                quat = np.array(blocks[self.cur]["quat"], float)
                Rc = np.zeros(9)
                mujoco.mju_quat2Mat(Rc, quat)
                tilt = np.arccos(np.clip(np.max(np.abs(Rc.reshape(3, 3)[2])), -1, 1))
                if b[2] > 0.85 and np.linalg.norm(b - sp) < 0.04 and tilt > 0.2:
                    if self.log:
                        print("grasp tilted %.2f rad; putting back" % tilt)
                    self.attempts += 1
                    self.set_goal(np.r_[self.cmd_p[:2], 0.825], v=0.1)
                    self.enter("putback", t)
                elif b[2] > 0.85 and np.linalg.norm(b - sp) < 0.04:
                    self.plan_place(obs)
                    slot = self.slots[self.placed]
                    self.set_goal(np.r_[slot, self.hover_z] - self.place_R @ self.c_s, self.place_R, v=0.2)
                    self.enter("to_bin", t)
                else:
                    self.closure = 0.0
                    self.attempts += 1
                    self.enter("regrasp_up", t)
                    self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.2)
        elif ph == "regrasp_up":
            if done:
                self.enter("start", t)
        elif ph == "putback":
            b = np.array(blocks[self.cur]["pos"])
            if (done and el > 0.5) or b[2] < 0.815 or el > 3.0:
                self.closure = 0.0
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.15)
                self.enter("regrasp_up", t)
        elif ph == "to_bin":
            if done and el > 0.3:
                self.corr = np.zeros(2)
                self.corr_yaw = 0.0
                self.yerr = 0.0
                self.enter("lower", t)
        elif ph == "lower":
            b = np.array(blocks[self.cur]["pos"])
            err = self.slots[self.placed] - b[:2]
            if (done and np.linalg.norm(err) < 0.002 and abs(self.yerr) < 0.03 and b[2] < self.place_zc + 0.004
                    and el > 0.8) or el > 4.0:
                if self.log:
                    print("release err", np.round(err, 4), "yaw err %.3f" % self.yerr, "cube", np.round(b, 3))
                self.enter("release", t)
        elif ph == "release":
            # low release: open the thumb only; fingers stay curled so they don't flick the cube.
            # high release (drop): open fully so the cube falls straight.
            floor_c = 0.42
            self.closure = max(floor_c, self.closure - 0.08)
            if el > 0.7:
                self.set_goal(self.cmd_p + np.array([0, 0, 0.045]), v=0.08)
                self.enter("release_up", t)
        elif ph == "release_up":
            if done:
                self.enter("release_open", t)
        elif ph == "release_open":
            self.closure = max(0.0, self.closure - 0.08)
            if el > 0.6:
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z + 0.02], v=0.15)
                self.enter("retreat", t)
        elif ph == "press_hover":
            self.closure = min(0.7, self.closure + 0.08)
            if done and el > 0.5:
                self.set_goal(np.r_[self.press_xy, 0.856], v=0.05)
                self.enter("press_down", t)
        elif ph == "press_down":
            if (done and el > 0.6) or el > 2.5:
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z + 0.02], v=0.12)
                self.enter("press_up", t)
        elif ph == "press_up":
            self.closure = max(0.0, self.closure - 0.08)
            if done and self.closure <= 0.0:
                self.enter("retreat", t)
        elif ph == "retreat" and done and self.press_tries < 2 and \
                np.array(blocks[self.cur]["pos"])[2] > 0.818 and \
                np.max(np.abs(np.array(blocks[self.cur]["pos"])[:2] - self.bin)) < 0.06:
            # cube not seated (bridging/leaning): push it down with the closed hand
            b = np.array(blocks[self.cur]["pos"])
            self.press_tries += 1
            self.press_xy = b[:2].copy()
            if self.log:
                print("cube not seated at", np.round(b, 3), "-> press")
            self.set_goal(np.r_[self.press_xy, 0.92], v=0.15)
            self.enter("press_hover", t)
        elif ph == "retreat":
            if done:
                self.press_tries = 0
                self.placed += 1
                self.queue.pop(0)
                if self.placed == 1:
                    # adapt second slot to where the first cube actually is
                    b1 = np.array(blocks[self.cur]["pos"])
                    s0 = self.slots[0]
                    if np.max(np.abs(b1[:2] - s0)) < 0.03:
                        y2 = b1[1] - self.s0 * 0.0545
                        y2 = np.clip(y2, self.bin[1] - 0.0285, self.bin[1] + 0.0285)
                        x2 = np.clip(b1[0] + 0.008, self.bin[0] - 0.01, self.bin[0] + 0.002)
                        self.slots[1] = np.array([x2, y2])
                        if self.log:
                            print("first cube at", np.round(b1, 3), "second slot", np.round(self.slots[1], 3))
                self.enter("start", t)

        q, e, r = robust_solve(self.ik, self.q_cmd, self.cmd_p, self.cmd_R)
        self.q_cmd = q
        return np.r_[q, self.closure]


class _OldController:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.ik = IK(robot)
        self.dt = robot.control_dt
        self.q_cmd = np.array(robot.home_arm, float)
        self.closure = 0.0
        bx, by = robot.bin_xy[task["destination"]]
        self.bin = np.array([bx, by])
        self.queue = [task["target"], task["second"]]
        self.slots = [np.array([bx - 0.006, by - 0.0285]), np.array([bx - 0.006, by + 0.0285])]
        self.placed = 0
        self.phase = "prep"
        self.pt = 0.0
        self.fresh = True
        self.attempts = 0
        self.goal = None
        self.cmd_roll = 0.0
        self.log = True
        self.hover_z = 0.90
        self.bad_h = []
        # lowest reachable site z for each slot and roll (place heading 0)
        self.zmin = {}
        for si, s in enumerate(self.slots):
            for roll in ROLLS:
                zm = None
                for z in np.arange(0.815, 0.90, 0.01):
                    q, e, r = self.ik.solve(self.ik.qn, np.r_[s, z], rot(0.0, roll), iters=100)
                    if e < 0.002 and r < 0.02:
                        zm = z
                        break
                self.zmin[(si, roll)] = zm if zm is not None else 0.95

    # ---- helpers
    def set_goal(self, p, h=None, roll=None, v=0.25, w=1.5):
        self.goal = (np.array(p, float), self.cmd_h if h is None else h,
                     self.cmd_roll if roll is None else roll, v, w)

    def step_goal(self):
        p, h, roll, v, w = self.goal
        d = p - self.cmd_p
        n = np.linalg.norm(d)
        st = v * self.dt
        self.cmd_p = p.copy() if n <= st else self.cmd_p + d * st / n
        dh = wrap(h - self.cmd_h)
        self.cmd_h += np.clip(dh, -w * self.dt, w * self.dt)
        self.cmd_roll += np.clip(roll - self.cmd_roll, -w * self.dt, w * self.dt)
        return n <= st and abs(dh) < 1e-3 and abs(roll - self.cmd_roll) < 1e-3

    def choose_grasp(self, pos, yaw, pref=-10.0):
        best = None
        si = self.placed
        for roll in ROLLS:
            zm = self.zmin[(si, roll)]
            zcost = 8.0 * max(0.0, zm - 0.82) + 0.1 * abs(roll)
            if best is not None and zcost > best[0]:
                continue
            for k in range(4):
                h = wrap(yaw + k * np.pi / 2)
                cost = zcost + 0.3 * abs(wrap(h - np.radians(pref)))
                cost += 0.5 * sum(1 for hb in self.bad_h if abs(wrap(h - hb)) < 0.2)
                if best is not None and cost > best[0]:
                    continue
                for z in (pos[2], pos[2] + 0.04, self.hover_z):
                    q, e, r = self.ik.solve(self.ik.qn, np.r_[pos[:2], z], rot(h, roll), iters=80)
                    cost += e * 20 + r
                if best is None or cost < best[0]:
                    best = (cost, h, roll)
        if self.log:
            print("grasp choice cost %.3f heading %.0f roll %.1f" % (best[0], np.degrees(best[1]), best[2]))
        return best[1], best[2]

    def enter(self, ph, t):
        if self.log:
            print("t=%.2f phase %s" % (t, ph))
        self.phase = ph
        self.pt = t
        self.fresh = True

    # ---- main
    def act(self, obs):
        t = obs["t"]
        blocks = obs["blocks"]
        sp = np.array(obs["grasp_site_pos"])
        el = t - self.pt
        ph = self.phase

        if ph == "prep":
            dq = self.ik.qn - self.q_cmd
            self.q_cmd = self.q_cmd + np.clip(dq, -0.06, 0.06)
            if np.max(np.abs(dq)) < 1e-3:
                p, R = self.ik.fk(self.q_cmd)
                self.cmd_p = p
                self.cmd_h = np.arctan2(R[1, 0], R[0, 0])
                self.set_goal(p, v=0.2)
                self.enter("start", t)
            return np.r_[self.q_cmd, self.closure]

        # ---- set goals for this phase
        if ph == "to_hover":
            b = np.array(blocks[self.cur]["pos"])
            self.set_goal(np.r_[b[:2], self.hover_z], self.grasp_h, self.grasp_roll, v=0.3)
        elif ph == "descend":
            b = np.array(blocks[self.cur]["pos"])
            self.set_goal(np.r_[b[:2], b[2] + 0.005], self.grasp_h, self.grasp_roll, v=0.12)
        elif ph == "lower":
            b = np.array(blocks[self.cur]["pos"])
            slot = self.slots[self.placed]
            err = slot - b[:2]
            if el > 0.3:
                self.corr += 0.2 * err
                self.corr = np.clip(self.corr, -0.03, 0.03)
            self.set_goal(np.r_[slot - self.offset[:2] + self.corr, self.place_z], v=0.08)
        elif ph == "idle":
            self.set_goal(np.r_[0.2, -0.35, 0.95], 0.0, 0.0, v=0.2)

        done = self.step_goal() and not self.fresh
        self.fresh = False

        # ---- transitions
        if ph == "start":
            if self.placed >= 2 or not self.queue:
                self.enter("idle", t)
            else:
                self.cur = self.queue[0]
                b = np.array(blocks[self.cur]["pos"])
                self.grasp_h, self.grasp_roll = self.choose_grasp(b + [0, 0, 0.005], blocks[self.cur]["yaw"])
                self.closure = 0.0
                self.enter("to_hover", t)
        elif ph == "to_hover":
            if done and np.linalg.norm(sp - self.cmd_p) < 0.01:
                self.enter("descend", t)
            elif el > 6.0:
                self.attempts += 1
                self.bad_h.append(self.grasp_h)
                self.enter("start", t)
        elif ph == "descend":
            if done and (np.linalg.norm(sp - self.cmd_p) < 0.006 or el > 2.5):
                self.enter("close", t)
        elif ph == "close":
            self.closure = min(1.0, self.closure + 0.06)
            if el > 1.0:
                self.enter("lift", t)
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.15)
        elif ph == "lift":
            b = np.array(blocks[self.cur]["pos"])
            if done:
                if b[2] > 0.85 and np.linalg.norm(b - sp) < 0.04:
                    slot = self.slots[self.placed]
                    self.offset = b - sp
                    yaw = blocks[self.cur]["yaw"]
                    dy = wrap(np.round(yaw / (np.pi / 2)) * np.pi / 2 - yaw)
                    ph0 = wrap(self.cmd_h + dy)
                    cands = [wrap(ph0 + k * np.pi / 2) for k in range(4)]
                    self.place_h = min(cands, key=lambda h: abs(wrap(h)))
                    zm = self.zmin[(self.placed, self.grasp_roll)]
                    self.place_z = max(REL_Z - self.offset[2], zm + 0.003)
                    if self.log:
                        print("offset", np.round(self.offset, 3), "place_h %.1f deg" % np.degrees(self.place_h),
                              "place_z %.3f" % self.place_z)
                    self.set_goal(np.r_[slot - self.offset[:2], self.hover_z], self.place_h, self.grasp_roll, v=0.25)
                    self.enter("to_bin", t)
                else:
                    self.closure = 0.0
                    self.attempts += 1
                    self.enter("regrasp_up", t)
                    self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.2)
        elif ph == "regrasp_up":
            if done:
                self.enter("start", t)
        elif ph == "to_bin":
            if done and el > 0.3:
                self.corr = np.zeros(2)
                self.enter("lower", t)
        elif ph == "lower":
            b = np.array(blocks[self.cur]["pos"])
            err = self.slots[self.placed] - b[:2]
            if (done and np.linalg.norm(err) < 0.002 and el > 0.8) or el > 4.0:
                if self.log:
                    print("release err", np.round(err, 4), "cube", np.round(b, 3))
                self.enter("release", t)
        elif ph == "release":
            self.closure = max(0.0, self.closure - 0.1)
            if el > 0.6:
                self.set_goal(np.r_[self.cmd_p[:2], self.hover_z], v=0.15)
                self.enter("retreat", t)
        elif ph == "retreat":
            if done:
                self.placed += 1
                self.queue.pop(0)
                self.enter("start", t)

        q, e, r = robust_solve(self.ik, self.q_cmd, self.cmd_p, rot(self.cmd_h, self.cmd_roll))
        self.q_cmd = q
        return np.r_[q, self.closure]
