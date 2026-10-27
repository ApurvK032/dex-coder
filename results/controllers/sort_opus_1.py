
import numpy as np
import mujoco


def _rot_err(R, Rd):
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


def _target_R(psi):
    x = np.array([np.cos(psi), np.sin(psi), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    GRASP_DX = 0.012   # cube center sits this far ahead (site x) of the grasp site at grasp time
    GRASP_DZ = 0.005

    def __init__(self, robot, task):
        self.r = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.qa = np.array(robot.arm_qpos_adr)
        self.da = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm, dtype=float)
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest], dtype=float)
        self.th = robot.table_height
        self.h = robot.block_half_size
        self.rel_psi = 0.15 if self.bin[1] > -0.18 else -0.4
        self.q_cmd = None
        self.phase = "init"
        self.t_phase = 0.0
        self.seg = None
        self.closure = 0.0
        self.psi = 0.0
        self.tries = 0
        self.rng = np.random.default_rng(0)
        self.debug = False
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        names = [self.m.body(i).name for i in range(self.m.nbody)]
        self.armb = set(i for i, n in enumerate(names) if n.startswith("right_") and
                        any(k in n for k in ("shoulder", "elbow", "wrist", "hand")))
        self.tgt_body = names.index("block_" + self.target)
        self.ignore_target = True
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            n = names[b]
            if b in self.armb and "shoulder_pitch" in n:
                self.armb.discard(b)   # always near torso; not a meaningful collision
            elif b in self.armb and "shoulder" in n:
                self.m.geom_margin[g] = 0.01
            elif b in self.armb and ("elbow" in n or "wrist_roll" in n or "wrist_pitch" in n):
                self.m.geom_margin[g] = 0.015
            elif b in self.armb:
                self.m.geom_margin[g] = 0.002

    # ---------------- kinematics -----------------
    def fk(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def colpen(self, q):
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        pen = 0.0
        gb = self.m.geom_bodyid
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = gb[c.geom1], gb[c.geom2]
            a1, a2 = b1 in self.armb, b2 in self.armb
            if a1 == a2:
                continue
            other = b2 if a1 else b1
            if other == self.tgt_body or other == 0:
                continue
            pen += max(0.0, c.includemargin - c.dist)
        return pen

    def ik(self, q0, p_des, R_des, iters=30, wrot=0.5, null=0.02, qnull=None):
        q = q0.copy()
        qn = self.home if qnull is None else qnull
        for _ in range(iters):
            p, R = self.fk(q)
            ep = p_des - p
            er = _rot_err(R, R_des)
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.sid)
            J = np.vstack([self.jp[:, self.da], wrot * self.jr[:, self.da]])
            e = np.concatenate([ep, wrot * er])
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (null * (qn - q))
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        p, R = self.fk(q)
        err = np.linalg.norm(p_des - p) + 0.05 * np.linalg.norm(_rot_err(R, R_des))
        return q, err

    def ik_global(self, p_des, R_des, seeds_extra=()):
        seeds = [self.q_cmd, self.home] + list(seeds_extra)
        best = None
        k = 0
        while True:
            if k >= len(seeds):
                seeds.append(self.lo + (self.hi - self.lo) * self.rng.uniform(0.2, 0.8, 7))
            q, e = self.ik(seeds[k], p_des, R_des, iters=120)
            tot = e + 2.0 * self.colpen(q)
            cost = tot + 0.01 * np.linalg.norm(q - self.q_cmd)
            if best is None or cost < best[0]:
                best = (cost, q, tot)
            k += 1
            if k >= 1 and tot < 0.002 and k == 1:
                break
            if k >= 4 and best[2] < 0.002:
                break
            if k >= 14:
                break
        return best[1], best[2]

    # ---------------- planning helpers -----------------
    def grasp_point(self, bp, psi):
        R = _target_R(psi)
        return bp - self.GRASP_DX * R[:, 0] + [0, 0, self.GRASP_DZ]

    def choose_psi(self, obs):
        blk = obs["blocks"][self.target]
        bp = np.array(blk["pos"])
        best = None
        for k in range(4):
            a = _wrap(blk["yaw"] + k * np.pi / 2)
            R = _target_R(a)
            q, e = self.ik_global(self.grasp_point(bp, a), R)
            q2, e2 = self.ik_global(self.grasp_point(bp, a) + [0, 0, 0.12], R, seeds_extra=[q])
            # the lift must be reachable continuously from the grasp configuration
            q3, e3 = self.ik(q, self.grasp_point(bp, a) + [0, 0, 0.17], R, iters=150)
            e3 += 2.0 * self.colpen(q3) + 0.05 * max(0.0, np.linalg.norm(q3 - q) - 1.0)
            cost = (50.0 * max(0.0, e - 0.002) + 20.0 * max(0.0, e2 - 0.002) +
                    20.0 * max(0.0, e3 - 0.003) + 0.3 * abs(_wrap(a + 0.35)))
            for name, b in obs["blocks"].items():
                if name == self.target:
                    continue
                rel = R.T @ (np.array(b["pos"]) - bp)
                lx, lz = rel[0], rel[2]
                if -0.10 < lx < 0.115 and abs(lz) < 0.08:
                    cost += 1.0
            if self.debug > 1:
                print(f"   psi cand {a:.2f}: e={e:.4f} e2={e2:.4f} cost={cost:.3f}")
            if best is None or cost < best[0]:
                best = (cost, a, q)
        return best[1]

    def choose_rel_psi(self):
        best = None
        for a in np.linspace(-0.8, 0.8, 9):
            R = _target_R(a)
            q, e = self.ik_global(np.array([self.bin[0], self.bin[1], self.th + 0.17]), R)
            q2, e2 = self.ik_global(np.array([self.bin[0], self.bin[1], self.th + 0.095]), R, seeds_extra=[q])
            cost = 50.0 * max(0.0, e - 0.002) + 50.0 * max(0.0, e2 - 0.002) + 0.2 * abs(_wrap(a - self.cur_psi))
            if best is None or cost < best[0]:
                best = (cost, a)
        return best[1]

    def start_seg(self, t, p1, psi1, dur, local=False):
        R1 = _target_R(psi1)
        qg = None
        if local:
            qg, e = self.ik(self.q_cmd, p1, R1, iters=150)
            if e + 2.0 * self.colpen(qg) > 0.003:
                qg = None
                local = False
        if qg is None:
            qg, e = self.ik_global(p1, R1)
        self.seg = dict(t0=t, dur=dur, p0=self.cur_p.copy(), p1=np.array(p1, float),
                        psi0=self.cur_psi, psi1=psi1, q0=self.q_cmd.copy(), q1=qg, local=local)
        self.cur_p, self.cur_psi = np.array(p1, float), psi1

    def seg_pose(self, t):
        s_ = self.seg
        s = np.clip((t - s_["t0"]) / s_["dur"], 0, 1)
        s = s * s * (3 - 2 * s)
        p = s_["p0"] + s * (s_["p1"] - s_["p0"])
        psi = s_["psi0"] + s * _wrap(s_["psi1"] - s_["psi0"])
        qs = s_["q0"] + s * (s_["q1"] - s_["q0"])
        return p, psi, qs, (t - s_["t0"]) >= s_["dur"]

    # ---------------- main -----------------
    def act(self, obs):
        t = obs["t"]
        self.d.qpos[:] = obs["kin"].qpos
        if self.q_cmd is None:
            self.q_cmd = np.array(obs["arm_target"], dtype=float)
            p, R = self.fk(self.q_cmd)
            self.cur_p = p
            self.cur_psi = np.arctan2(R[1, 0], R[0, 0])
        blk = obs["blocks"][self.target]
        bp = np.array(blk["pos"])
        site = np.array(obs["grasp_site_pos"])
        dist = np.linalg.norm(bp - site)
        held = dist < 0.045 or (bp[2] > self.th + self.h + 0.03 and dist < 0.08)
        ph = self.phase
        done = self.seg_pose(t)[3] if self.seg is not None else True
        tp = t - self.t_phase

        def go(newphase):
            if self.debug:
                print(f"{t:5.2f} {ph}->{newphase} site={np.round(site,3)} blk={np.round(bp,3)} psi={self.psi:.2f} yaw={blk['yaw']:.2f} cl={self.closure:.2f}")
            self.phase = newphase
            self.t_phase = t

        if ph == "init":
            self.closure = 0.0
            self.psi = self.choose_psi(obs)
            gp = self.grasp_point(bp, self.psi)
            hover = max(self.cur_p[2], gp[2] + 0.12) if self.tries > 0 else gp[2] + 0.12
            self.start_seg(t, [gp[0], gp[1], hover], self.psi, 1.5)
            go("pre")
        elif ph == "pre" and done:
            self.psi = self.choose_psi(obs)
            self.start_seg(t, self.grasp_point(bp, self.psi) + [0, 0, 0.12], self.psi, 0.4)
            go("align")
        elif ph == "align" and done:
            self.start_seg(t, self.grasp_point(bp, self.psi), self.psi, 1.2, local=True)
            go("descend")
        elif ph == "descend" and done:
            go("settle")
        elif ph == "settle" and tp > 0.25:
            go("close")
        elif ph == "close":
            self.closure = min(1.0, 0.5 * tp / 0.35) if tp < 0.35 else min(1.0, 0.5 + 0.5 * (tp - 0.35) / 0.35)
            if tp > 1.0:
                self.closure = 1.0
                self.start_seg(t, self.cur_p + [0, 0, 0.17], self.cur_psi, 1.0, local=True)
                go("lift")
        elif ph == "lift" and done:
            if not held:
                self.tries += 1
                go("init")
            else:
                self.rel_psi = self.choose_rel_psi()
                tgt = np.array([self.bin[0], self.bin[1], self.th + 0.17])
                self.start_seg(t, tgt, self.rel_psi, 1.6)
                go("carry")
        elif ph == "carry0":
            if not held:
                self.tries += 1
                self.closure = 0.0
                go("init")
            elif done:
                tgt = np.array([self.bin[0], self.bin[1], self.th + 0.17])
                self.start_seg(t, tgt, self.rel_psi, 1.0)
                go("carry")
        elif ph == "carry":
            if not held:
                self.tries += 1
                self.closure = 0.0
                go("init")
            elif done:
                tgt = np.array([self.bin[0], self.bin[1], self.th + 0.095])
                self.start_seg(t, tgt, self.cur_psi, 0.8, local=True)
                go("lower")
        elif ph == "lower" and done:
            go("release")
        elif ph == "release":
            self.closure = max(0.4, 1.0 - tp / 0.5)
            if tp > 0.8:
                self.start_seg(t, self.cur_p + [0, 0, 0.12], self.cur_psi, 0.8, local=True)
                go("retreat")
        elif ph == "retreat":
            if done:
                self.closure = 0.0
                go("rest")

        if self.seg is not None:
            p_des, psi_des, qs, _ = self.seg_pose(t)
            Rd = _target_R(psi_des)
            q, e = self.ik(self.q_cmd, p_des, Rd, iters=20, null=0.1, qnull=qs)
            e += 2.0 * self.colpen(q)
            if e > 0.003 and not self.seg["local"]:
                q2, e2 = self.ik(qs, p_des, Rd, iters=40, null=0.1, qnull=qs)
                e2 += 2.0 * self.colpen(q2)
                if e2 < e:
                    q = q2
        else:
            q = self.q_cmd
        dq = np.clip(q - self.q_cmd, -0.08, 0.08)
        self.q_cmd = self.q_cmd + dq
        return np.concatenate([self.q_cmd, [self.closure]])
