import numpy as np
import mujoco

HOVER = 0.92      # site height while travelling
GRASP_Z = 0.805   # site height at grasp
DROP_Z = 0.90


class Controller:
    def __init__(self, robot, task):
        self.r = robot
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.task = task
        self.cubes = [task["target"], task["second"]]
        self.dest = task["destination"]
        self.idx = 0
        self.phase = "above"
        self.pt = 0.0
        self.closure = 0.0
        self.psi = 0.0
        self.q = np.array(robot.home_arm, dtype=float)
        self.qref = np.array(robot.home_arm, dtype=float)
        self.lo = np.array(robot.arm_lower)
        self.hi = np.array(robot.arm_upper)
        self.adr = np.array(robot.arm_qpos_adr)
        self.dof = np.array(robot.arm_dof_adr)
        self.first = True
        self.gtgt = None
        self.off = np.zeros(3)
        self.bname = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.m.nbody)]

    def ik(self, pos, psi, max_step=0.08, wr=1.0, k=0.3):
        d = self.d
        d.qpos[self.adr] = self.q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        p = d.site_xpos[self.sid].copy()
        R = d.site_xmat[self.sid].reshape(3, 3).copy()
        xd = np.array([np.cos(psi), np.sin(psi), 0.0])
        yd = np.array([0.0, 0.0, -1.0])
        zd = np.cross(xd, yd)
        Rd = np.stack([xd, yd, zd], axis=1)
        ep = pos - p
        n = np.linalg.norm(ep)
        if n > 0.04:
            ep *= 0.04 / n
        er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, jr, self.sid)
        J = np.vstack([jp[:, self.dof], wr * jr[:, self.dof]])
        e = np.concatenate([ep, wr * er])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(6))
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (k * (self.qref - self.q))
        mx = np.max(np.abs(dq))
        if mx > max_step:
            dq *= max_step / mx
        qn = np.clip(self.q + dq, self.lo, self.hi)
        if self.selfcol(qn):
            q2 = qn.copy()
            q2[1] = min(qn[1], self.q[1])
            if self.selfcol(q2):
                q2[1] = self.q[1] - 0.03
            qn = q2
        self.q = qn
        return p

    def selfcol(self, q):
        d = self.d
        qq = q.copy()
        qq[1] += 0.02
        d.qpos[self.adr] = qq
        mujoco.mj_forward(self.m, d)
        for k in range(d.ncon):
            c = d.contact[k]
            n1 = self.bname[self.m.geom_bodyid[c.geom1]]
            n2 = self.bname[self.m.geom_bodyid[c.geom2]]
            if self.skip(n1) or self.skip(n2):
                continue
            if n1 == n2:
                continue
            return True
        return False

    @staticmethod
    def skip(n):
        return ("block" in n) or ("table" in n) or ("bin" in n) or n == "world"

    def pick_psi(self, yaw):
        return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def act(self, obs):
        site = np.array(obs["grasp_site_pos"])
        if self.first:
            self.q = np.array(obs["arm_target"], dtype=float)
            self.first = False
        if self.idx >= 2:
            self.ik(np.array([site[0], site[1], HOVER]), self.psi)
            return list(self.q) + [0.0]
        name = self.cubes[self.idx]
        b = obs["blocks"][name]
        bp = np.array(b["pos"])
        bx, by = self.r.bin_xy[self.dest]
        if self.idx == 0:
            bx = bx + 0.028
            by = by - 0.026
        else:
            c1 = obs["blocks"][self.cubes[0]]["pos"]
            if abs(c1[1] - by) < 0.07 and abs(c1[0] - bx) < 0.07:
                sgn = -1.0 if c1[0] > bx else 1.0
                cand = bx + sgn * 0.028
                by = by + (0.026 if self.dest == "right" else 0.0)
                bx = cand
            else:
                bx = bx - 0.028
                by = by - 0.026
        self.pt += self.r.control_dt
        ph = self.phase
        if ph in ("above", "descend"):
            self.psi = self.pick_psi(b["yaw"])
        if ph in ("carry", "lower"):
            yw = (b["yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
            self.psi = float(np.clip(self.psi - 0.25 * yw, -1.4, 1.4))
        psi = self.psi

        if ph == "above":
            self.closure = 0.0
            tgt = np.array([bp[0], bp[1], HOVER])
            self.ik(tgt, psi)
            if np.linalg.norm(site - tgt) < 0.012 and self.pt > 0.5:
                self.phase, self.pt = "descend", 0.0
            elif self.pt > 8:
                self.phase, self.pt = "descend", 0.0
        elif ph == "descend":
            self.closure = 0.0
            tgt = np.array([bp[0], bp[1], GRASP_Z])
            self.ik(tgt, psi)
            if (np.linalg.norm(site - tgt) < 0.008 and self.pt > 0.3) or self.pt > 6.0:
                self.phase, self.pt = "close", 0.0
                self.gtgt = tgt.copy()
        elif ph == "close":
            self.ik(self.gtgt, psi)
            self.closure = min(1.0, self.pt / 1.0)
            if self.pt > 1.6:
                self.phase, self.pt = "lift", 0.0
        elif ph == "lift":
            self.closure = 1.0
            tgt = np.array([self.gtgt[0], self.gtgt[1], HOVER])
            self.ik(tgt, psi)
            if site[2] > HOVER - 0.015:
                self.phase, self.pt = "carry", 0.0
        elif ph in ("carry", "lower"):
            self.closure = 1.0
            off = bp - site
            self.off = off.copy()
            zc = HOVER if ph == "carry" else DROP_Z
            tgt = np.array([bx - off[0], by - off[1], zc])
            self.ik(tgt, psi)
            cerr = np.linalg.norm(bp[:2] - np.array([bx, by]))
            if ph == "carry":
                if (cerr < 0.015 and self.pt > 0.3) or self.pt > 4:
                    self.phase, self.pt = "lower", 0.0
            else:
                if (abs(site[2] - tgt[2]) < 0.012 and cerr < 0.012) or self.pt > 3:
                    self.phase, self.pt = "release", 0.0
        elif ph == "release":
            self.closure = max(0.0, self.closure - 0.08)
            tgt = np.array([bx - self.off[0], by - self.off[1], DROP_Z])
            self.ik(tgt, psi)
            if self.pt > 1.0:
                self.phase, self.pt = "retreat", 0.0
        elif ph == "retreat":
            self.closure = 0.0
            tgt = np.array([bx - self.off[0], by - self.off[1], HOVER])
            self.ik(tgt, psi)
            if site[2] > HOVER - 0.015:
                self.idx += 1
                self.phase, self.pt = "above", 0.0
        return list(self.q) + [float(self.closure)]
