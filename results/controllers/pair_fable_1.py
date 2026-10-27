
import numpy as np
import mujoco


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def mat_for(yaw):
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


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
        self.home = np.array(robot.home_arm)
        self.dt = robot.control_dt
        self.step_max = robot.max_joint_speed * self.dt * 0.95
        rng = np.random.default_rng(0)
        self.seeds = [self.home] + [self.lo + (self.hi - self.lo) * rng.random(7) for _ in range(8)]
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.bin_xy = {k: np.array(v) for k, v in robot.bin_xy.items()}
        self.bin_half = robot.bin_inner_half_size
        self.rim = self.table + robot.bin_wall_height
        self.order = [task["target"], task["second"]]
        self.dest = task["destination"]
        self.obs = None
        self.q_cmd = self.home.copy()
        self.clo = 0.0
        self.gen = self.script()
        self.log = []

    # ---------------- kinematics ----------------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik1(self, q0, tp, tm, iters=60, wr=1.0):
        q = q0.copy()
        m, d, sid = self.m, self.d, self.sid
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        ep = er = None
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
            ep = tp - p
            Rerr = tm @ R.T
            er = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            err = np.concatenate([ep, wr * er])
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([jp[:, self.dadr], wr * jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, np.linalg.norm(ep), np.linalg.norm(er)

    def ik(self, q0, tp, tm, quick=True):
        """returns q, cost (position error + orientation error)"""
        best = None
        for i, s in enumerate([q0] + self.seeds):
            q, ep, er = self.ik1(s, tp, tm)
            cost = ep + 0.05 * er + 0.003 * np.linalg.norm(q - q0)
            if best is None or cost < best[0]:
                best = (cost, q, ep, er)
            if quick and ep < 1e-3 and er < 1e-2:
                break
        return best[1], best[2], best[3]

    # ---------------- motion primitives (generators) ----------------
    def act(self, obs):
        self.obs = obs
        try:
            a = next(self.gen)
        except StopIteration:
            a = np.concatenate([self.q_cmd, [self.clo]])
        return a

    def emit(self):
        return np.concatenate([self.q_cmd, [self.clo]])

    def goto_q(self, qgoal, dur=None, settle=0.2):
        qgoal = np.clip(np.asarray(qgoal, float), self.lo, self.hi)
        q0 = self.q_cmd.copy()
        n = int(np.ceil(np.max(np.abs(qgoal - q0)) / self.step_max))
        if dur is not None:
            n = max(n, int(dur / self.dt))
        n = max(n, 1)
        for i in range(1, n + 1):
            self.q_cmd = q0 + (qgoal - q0) * (i / n)
            yield self.emit()
        ns = int(settle / self.dt)
        for i in range(ns):
            yield self.emit()
            if np.max(np.abs(self.obs["qpos"][:7] - qgoal)) < 0.02:
                break

    def goto_pose(self, pos, mat, n_way=1, settle=0.2, speed=0.3, tol=0.003):
        """straight-line Cartesian servo toward pos/mat, nullspace pull toward global IK solution"""
        pos = np.asarray(pos, float)
        q_goal, ep_g, er_g = self.ik(self.q_cmd, pos, mat)
        m, d, sid = self.m, self.d, self.sid
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        p0, _ = self.fk(self.q_cmd)
        dist = np.linalg.norm(pos - p0)
        max_steps = int(dist / (speed * self.dt)) + 60
        wr = 0.5
        last_err = None; stall = 0
        for step in range(max_steps):
            d.qpos[self.qadr] = self.q_cmd
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
            ep = pos - p
            Rerr = mat @ R.T
            er = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
            ne = np.linalg.norm(ep); nr = np.linalg.norm(er)
            if ne < tol and nr < 0.02:
                break
            # limit cartesian step
            dp = ep if ne < speed * self.dt else ep * (speed * self.dt / ne)
            dr = er if nr < 0.1 else er * (0.1 / nr)
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([jp[:, self.dadr], wr * jr[:, self.dadr]])
            err = np.concatenate([dp, wr * dr])
            JJt = J @ J.T + 2e-3 * np.eye(6)
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            dq = Jpinv @ err
            N = np.eye(7) - Jpinv @ J
            dq += N @ (0.1 * (q_goal - self.q_cmd))
            mx = np.max(np.abs(dq))
            if mx > self.step_max:
                dq *= self.step_max / mx
            self.q_cmd = np.clip(self.q_cmd + dq, self.lo, self.hi)
            yield self.emit()
            # stall detection
            e_tot = ne + 0.1 * nr
            if last_err is not None and e_tot > last_err - 1e-4:
                stall += 1
            else:
                stall = 0
            last_err = e_tot
            if stall > 12:
                print(f"[{self.obs['t']:.1f}] servo stalled err {ne:.4f}/{nr:.3f}; jumping to IK goal (ep {ep_g:.4f})")
                yield from self.goto_q(q_goal, settle=0.0)
                break
        ns = int(settle / self.dt)
        for i in range(ns):
            yield self.emit()
            if np.max(np.abs(self.obs["qpos"][:7] - self.q_cmd)) < 0.02 and np.max(np.abs(self.obs["qvel"][:7])) < 0.1:
                break
        return self.q_cmd

    def set_closure(self, c, dur=0.8):
        c0 = self.clo
        n = max(int(dur / self.dt), 1)
        for i in range(1, n + 1):
            self.clo = c0 + (c - c0) * (i / n)
            yield self.emit()

    def wait(self, dur):
        for i in range(int(dur / self.dt)):
            yield self.emit()

    # ---------------- helpers ----------------
    def hand_yaw(self):
        R = self.obs["grasp_site_mat"]
        return np.arctan2(R[1, 0], R[0, 0])

    def choose_grasp(self, p, yaw):
        best = None
        for k in range(4):
            gy = wrap(yaw + k * np.pi / 2)
            M = mat_for(gy)
            fdir = M[:, 0]
            pre = np.array([p[0], p[1], p[2] + 0.10])
            gp = np.array([p[0], p[1], self.table + self.half + 0.008]) + 0.01 * fdir
            q1, ep1, er1 = self.ik(self.q_cmd, pre, M)
            q2, ep2, er2 = self.ik(q1, gp, M)
            cost = ep1 + ep2 + 0.05 * (er1 + er2) + 0.002 * np.linalg.norm(q1 - self.q_cmd)
            if best is None or cost < best[0]:
                best = (cost, gy, q1, q2, pre, gp)
        return best

    def choose_drop(self, cube_target, delta_site, dyaw, z):
        """choose hand yaw so cube is axis-aligned; returns (yaw, q, site_target)"""
        best = None
        for k in range(4):
            hy = wrap(k * np.pi / 2 - dyaw)
            M = mat_for(hy)
            st = np.array([cube_target[0], cube_target[1], z]) - M @ delta_site
            q, ep, er = self.ik(self.q_cmd, st, M)
            cost = ep + 0.05 * er + 0.003 * np.linalg.norm(q - self.q_cmd)
            if best is None or cost < best[0]:
                best = (cost, hy, q, st, ep, er)
        return best

    # ---------------- task helpers ----------------
    def in_bin(self, p, name=None):
        name = name or self.dest
        c = self.bin_xy[name]
        return abs(p[0] - c[0]) < self.bin_half and abs(p[1] - c[1]) < self.bin_half and p[2] < self.rim

    def settled_flat(self, blk):
        p = self.obs["blocks"][blk]["pos"]
        return self.in_bin(p) and p[2] < self.table + self.half + 0.012

    def grasp(self, blk):
        """returns (delta_site, hand_yaw_for_aligned_cube) or None"""
        p = np.array(self.obs["blocks"][blk]["pos"])
        yaw = self.obs["blocks"][blk]["yaw"]
        cost, gy, q1, q2, pre, gp = self.choose_grasp(p, yaw)
        print(f"[{self.obs['t']:.1f}] grasp {blk} at {np.round(p,3)} yaw {gy:.2f} cost {cost:.4f}")
        ps, _ = self.fk(self.q_cmd)
        if ps[2] < 0.94:
            yield from self.goto_pose(np.array([ps[0], ps[1], 0.96]), mat_for(self.hand_yaw()), settle=0.0)
        yield from self.goto_pose(np.array([pre[0], pre[1], 0.96]), mat_for(gy), settle=0.0)
        yield from self.goto_pose(pre, mat_for(gy), settle=0.2)
        p = np.array(self.obs["blocks"][blk]["pos"])
        gp2 = np.array([p[0], p[1], max(p[2], self.table + self.half) + 0.008]) + 0.01 * mat_for(gy)[:, 0]
        yield from self.goto_pose(gp2, mat_for(gy), settle=0.3, speed=0.2)
        yield from self.set_closure(1.0, 1.0)
        yield from self.wait(0.4)
        yield from self.goto_pose(np.array([gp2[0], gp2[1], 0.96]), mat_for(gy), settle=0.3, speed=0.2)
        cp = np.array(self.obs["blocks"][blk]["pos"])
        sp = np.array(self.obs["grasp_site_pos"])
        if not (cp[2] > self.table + 0.08 and np.linalg.norm(cp - sp) < 0.05):
            print(f"[{self.obs['t']:.1f}] grasp failed, cube z {cp[2]:.3f} dist {np.linalg.norm(cp-sp):.3f}")
            yield from self.set_closure(0.0, 0.5)
            yield from self.wait(0.2)
            return None
        R = np.array(self.obs["grasp_site_mat"])
        delta = R.T @ (cp - sp)
        dyaw = wrap(self.obs["blocks"][blk]["yaw"] - self.hand_yaw())
        dyaw = (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return delta, wrap(-dyaw)

    def carry_to(self, blk, tgt_xy, zc, delta, hy, speed=0.12):
        M = mat_for(hy)
        high = np.array([tgt_xy[0], tgt_xy[1], 0.96]) - M @ delta
        yield from self.goto_pose(high, M, settle=0.2)
        cp = np.array(self.obs["blocks"][blk]["pos"]); sp = np.array(self.obs["grasp_site_pos"])
        if cp[2] < self.table + 0.08 or np.linalg.norm(cp - sp) > 0.06:
            return False
        low = np.array([tgt_xy[0], tgt_xy[1], zc]) - M @ delta
        yield from self.goto_pose(low, M, settle=0.3, speed=speed)
        # closed-loop yaw correction so the cube is axis-aligned with the bin
        yerr = (wrap(self.obs["blocks"][blk]["yaw"]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        if 0.03 < abs(yerr) < 0.5:
            hy = wrap(hy - yerr)
            M = mat_for(hy)
            cp = np.array(self.obs["blocks"][blk]["pos"])
            sp = np.array(self.obs["grasp_site_pos"])
            R = np.array(self.obs["grasp_site_mat"])
            d_now = R.T @ (cp - sp)
            tgt3 = np.array([tgt_xy[0], tgt_xy[1], cp[2]])
            yield from self.goto_pose(tgt3 - M @ d_now, M, settle=0.3, speed=0.05)
        self.hy_cur = hy
        cp = np.array(self.obs["blocks"][blk]["pos"])
        err = np.array([tgt_xy[0] - cp[0], tgt_xy[1] - cp[1], 0.0])
        if 0.003 < np.linalg.norm(err) < 0.03:
            sp = np.array(self.obs["grasp_site_pos"])
            yield from self.goto_pose(sp + err, M, settle=0.3, speed=0.05)
        return True

    def release_and_lift(self, hy):
        yield from self.set_closure(0.0, 0.6)
        yield from self.wait(0.4)
        sp = np.array(self.obs["grasp_site_pos"])
        yield from self.goto_pose(np.array([sp[0], sp[1], 0.96]), mat_for(hy), settle=0.2, speed=0.2)

    # ---------------- main script ----------------
    def script(self):
        yield self.emit()
        bin_c = self.bin_xy[self.dest]
        b1, b2 = self.order
        tlimit = self.robot.time_limit_s - 6.0
        # push axis: +x (toward back wall) for the left bin, +y for the right bin (reach)
        a = np.array([1.0, 0.0]) if self.dest == "left" else np.array([0.0, 1.0])
        a3 = np.array([a[0], a[1], 0.0])

        def coord(blk):
            p = self.obs["blocks"][blk]["pos"]
            return (p[0] - bin_c[0]) * a[0] + (p[1] - bin_c[1]) * a[1]

        # ---- cube 1: drop at the far half of the bin ----
        for attempt in range(3):
            g = yield from self.grasp(b1)
            if g is None:
                continue
            delta, hy = g
            ok = yield from self.carry_to(b1, bin_c + 0.03 * a, 0.85, delta, hy)
            yield from self.release_and_lift(hy)
            p = self.obs["blocks"][b1]["pos"]
            print(f"[{self.obs['t']:.1f}] cube1 at {np.round(p - np.array([bin_c[0], bin_c[1], 0]),3)}")
            if self.in_bin(p):
                break
        # ---- cube 2: push cube 1 to the far wall, then release at the near side ----
        for attempt in range(4):
            if self.obs["t"] > tlimit:
                break
            g = yield from self.grasp(b2)
            if g is None:
                continue
            delta, hy = g
            M = mat_for(hy)
            p1 = self.obs["blocks"][b1]["pos"]
            if self.in_bin(p1) and coord(b1) < 0.027 and attempt == 0:
                self.hy_cur = hy
                ok = yield from self.carry_to(b2, bin_c - 0.036 * a, 0.836, delta, hy)
                hy = self.hy_cur
                M = mat_for(hy)
                if ok:
                    t0 = self.obs["t"]
                    best = coord(b1); t_best = t0
                    while self.obs["t"] - t0 < 4.0:
                        x1 = coord(b1); x2 = coord(b2)
                        z2 = self.obs["blocks"][b2]["pos"][2]
                        if x1 > best + 0.002:
                            best = x1; t_best = self.obs["t"]
                        if x1 >= 0.027 or x2 >= -0.012 or z2 < self.table + 0.05 or self.obs["t"] - t_best > 1.2:
                            break
                        sp = np.array(self.obs["grasp_site_pos"])
                        yield from self.goto_pose(sp + 0.006 * a3, M, settle=0.0, speed=0.04, tol=0.002)
                    yield from self.wait(0.2)
                    x1 = coord(b1); x2 = coord(b2)
                    print(f"[{self.obs['t']:.1f}] push done: x1 {x1:.3f} x2 {x2:.3f}")
                    # back off so cube 2 clears cube 1's face, then release
                    x2_goal = max(min(x1 - 0.054, -0.020), -0.028)
                    back = x2_goal - x2
                    if back < -0.002:
                        sp = np.array(self.obs["grasp_site_pos"])
                        yield from self.goto_pose(sp + back * a3, M, settle=0.3, speed=0.06, tol=0.002)
                yield from self.release_and_lift(hy)
            else:
                # plain drop into the free half (release slightly toward cube 1: release flick goes the other way)
                x1 = coord(b1) if self.in_bin(p1) else 0.03
                tx = min(-0.018, x1 - 0.05)
                tx = max(tx, -0.028)
                self.hy_cur = hy
                ok = yield from self.carry_to(b2, bin_c + tx * a, 0.845, delta, hy)
                yield from self.release_and_lift(self.hy_cur)
            p2 = self.obs["blocks"][b2]["pos"]; p1 = self.obs["blocks"][b1]["pos"]
            print(f"[{self.obs['t']:.1f}] cube1 {np.round(p1 - np.array([bin_c[0], bin_c[1], 0]),3)} cube2 {np.round(p2 - np.array([bin_c[0], bin_c[1], 0]),3)}")
            if self.settled_flat(b1) and self.settled_flat(b2):
                break
            if not self.in_bin(p1) or (p1[2] > self.table + self.half + 0.012 and p2[2] < p1[2]):
                # cube 1 is the problem: re-drop it
                g = yield from self.grasp(b1)
                if g is not None:
                    delta, hy = g
                    yield from self.carry_to(b1, bin_c + np.array([0.03, 0.0]), 0.85, delta, hy)
                    yield from self.release_and_lift(hy)
        # retreat
        yield from self.goto_pose(np.array([0.25, -0.3, 0.98]), mat_for(0.0), settle=0.0)
        while True:
            yield self.emit()

    def script_old(self):
        yield self.emit()
        bin_c = self.bin_xy[self.dest]
        sign = 1.0  # first cube goes to +y side
        placed = []
        for idx, blk in enumerate(self.order):
            held = False
            for attempt in range(4):
                p = np.array(self.obs["blocks"][blk]["pos"])
                yaw = self.obs["blocks"][blk]["yaw"]
                cost, gy, q1, q2, pre, gp = self.choose_grasp(p, yaw)
                print(f"[{self.obs['t']:.1f}] grasp {blk} attempt {attempt} yaw {gy:.2f} cost {cost:.4f}")
                # rise to safe height first
                ps, _ = self.fk(self.q_cmd)
                if ps[2] < 0.93:
                    yield from self.goto_pose(np.array([ps[0], ps[1], 0.95]), mat_for(self.hand_yaw()), 2, settle=0.0)
                # go above cube via cartesian waypoints at high z
                yield from self.goto_pose(np.array([pre[0], pre[1], 0.95]), mat_for(gy), 3, settle=0.0)
                yield from self.goto_pose(pre, mat_for(gy), 2, settle=0.2)
                # re-observe and descend
                p = np.array(self.obs["blocks"][blk]["pos"])
                gp2 = np.array([p[0], p[1], gp[2]]) + 0.01 * mat_for(gy)[:, 0]
                yield from self.goto_pose(gp2, mat_for(gy), 3, settle=0.3)
                yield from self.set_closure(1.0, 1.0)
                yield from self.wait(0.4)
                lift = np.array([gp2[0], gp2[1], 0.95])
                yield from self.goto_pose(lift, mat_for(gy), 3, settle=0.3)
                cp = np.array(self.obs["blocks"][blk]["pos"])
                sp = np.array(self.obs["grasp_site_pos"])
                if cp[2] > self.table + 0.08 and np.linalg.norm(cp - sp) < 0.05:
                    held = True
                    break
                print(f"[{self.obs['t']:.1f}] grasp failed, cube z {cp[2]:.3f} dist {np.linalg.norm(cp-sp):.3f}")
                yield from self.set_closure(0.0, 0.5)
                yield from self.wait(0.3)
            if not held:
                continue
            # ---- carry & place ----
            R = np.array(self.obs["grasp_site_mat"])
            sp = np.array(self.obs["grasp_site_pos"])
            cp = np.array(self.obs["blocks"][blk]["pos"])
            delta_site = R.T @ (cp - sp)
            dyaw = wrap(self.obs["blocks"][blk]["yaw"] - self.hand_yaw())
            dyaw = (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            # target cube position
            if idx == 0 or not placed:
                tgt = np.array([bin_c[0], bin_c[1] + sign * 0.0275])
            else:
                p1 = np.array(self.obs["blocks"][placed[0]]["pos"])
                # free space on the -sign side
                tgt = np.array([bin_c[0], p1[1] - sign * (2 * self.half + 0.004)])
                tgt[1] = np.clip(tgt[1], bin_c[1] - 0.03, bin_c[1] + 0.03)
            zdrop = self.rim + 0.004 + self.half - delta_site[2] * 0  # site z s.t. cube bottom ~ rim+4mm
            zdrop = self.rim + 0.006 + self.half + (sp[2] - cp[2])
            cost, hy, qd, st, ep, er = self.choose_drop(tgt, delta_site, dyaw, zdrop)
            print(f"[{self.obs['t']:.1f}] drop {blk} at {np.round(tgt,3)} hand yaw {hy:.2f} ep {ep:.4f} er {er:.4f} delta {np.round(delta_site,3)} dyaw {dyaw:.2f}")
            high = np.array([st[0], st[1], 0.96])
            yield from self.goto_pose(high, mat_for(hy), 4, settle=0.0)
            yield from self.goto_pose(st, mat_for(hy), 3, settle=0.4)
            # closed-loop correction on observed cube position
            for it in range(2):
                cp = np.array(self.obs["blocks"][blk]["pos"])
                err = np.array([tgt[0] - cp[0], tgt[1] - cp[1], 0.0])
                if np.linalg.norm(err) < 0.003:
                    break
                sp = np.array(self.obs["grasp_site_pos"])
                st = sp + err
                st[2] = zdrop
                q, ep, er = self.ik(self.q_cmd, st, mat_for(hy))
                yield from self.goto_q(q, settle=0.3)
            cp = np.array(self.obs["blocks"][blk]["pos"])
            print(f"[{self.obs['t']:.1f}] releasing {blk}; cube at {np.round(cp,3)} target {np.round(tgt,3)} yaw {self.obs['blocks'][blk]['yaw']:.2f}")
            yield from self.set_closure(0.0, 0.6)
            yield from self.wait(0.3)
            sp = np.array(self.obs["grasp_site_pos"])
            yield from self.goto_pose(np.array([sp[0], sp[1], 0.96]), mat_for(hy), 2, settle=0.0)
            placed.append(blk)
            cp = np.array(self.obs["blocks"][blk]["pos"])
            print(f"[{self.obs['t']:.1f}] placed {blk} at {np.round(cp,3)} yaw {self.obs['blocks'][blk]['yaw']:.2f}")
        # retreat
        ps, _ = self.fk(self.q_cmd)
        yield from self.goto_pose(np.array([0.25, -0.3, 0.98]), mat_for(0.0), 3, settle=0.0)
        while True:
            yield self.emit()
