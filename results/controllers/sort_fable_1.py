
import numpy as np
import mujoco


def _rotvec(Rerr):
    ang = np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array(
        [Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])


def R_for(yaw, tilt=0.0):
    """site x = fingers direction (horizontal at angle yaw), y = palm normal (down)."""
    x = np.array([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), -np.sin(tilt)])
    y0 = np.array([0, 0, -1.0])
    z = np.cross(x, y0)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, robot, task):
        self.robot = robot
        self.task = task
        self.m = robot.model
        self.d = mujoco.MjData(self.m)
        self.dcol = mujoco.MjData(self.m)
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.dofs = np.array(robot.arm_dof_adr)
        self.qadr = np.array(robot.arm_qpos_adr)
        self.lower = np.array(robot.arm_lower)
        self.upper = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm)
        self.target = task["target"]
        self.dest = task["destination"]
        self.bin = np.array(robot.bin_xy[self.dest])
        self.table = robot.table_height
        self.half = robot.block_half_size
        self.phase = "plan"
        self.phase_t0 = 0.0
        self.closure = 0.0
        self.cmd_pos = None
        self.cmd_R = None
        self.yaw = 0.0
        self.carry_yaw = 0.0
        self.attempts = 0
        self.lower_tries = 0
        self.grasp_z = self.table + self.half + 0.003
        self.pre_dz = 0.12
        self.lift_z = self.table + 0.21
        self.carry_z = self.table + 0.21
        self.place_z = self.table + 0.095
        self.grasp_ramp = 0.6
        self.grasp_hold = 1.0
        self.grasp_dx = -0.01  # site slightly toward the thumb: cube sits deeper in the fingers
        self.q_pre = None
        self.q_start = None
        self.s = 0.0
        self.block_bodies = {c: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "block_" + c)
                             for c in ("red", "green", "blue")}
        self.carry_yaw_cache = {}

    # ---------- kinematics ----------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.sid].copy(), self.d.site_xmat[self.sid].reshape(3, 3).copy()

    def ik(self, q0, pos, R, iters=30, wr=0.3):
        q = q0.copy()
        for it in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = _rotvec(R @ Rc.T)
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            e = np.concatenate([ep, wr * er])
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.sid)
            J = np.vstack([jp[:, self.dofs], wr * jr[:, self.dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + 0.6 * dq, self.lower, self.upper)
        return q

    def pose_ok(self, pos, R, q0=None, iters=120):
        """IK from home; return (feasible, q). Feasible = converged and no self/table collision."""
        if q0 is None:
            q0 = self.home
        q = self.ik(q0, pos, R, iters=iters)
        p, Rc = self.fk(q)
        if np.linalg.norm(pos - p) > 0.006 or np.linalg.norm(_rotvec(R @ Rc.T)) > 0.06:
            return False, q
        self.dcol.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, self.dcol)
        for i in range(self.dcol.ncon):
            con = self.dcol.contact[i]
            b1 = self.m.body(self.m.geom_bodyid[con.geom1]).name
            b2 = self.m.body(self.m.geom_bodyid[con.geom2]).name
            if "block" in b1 or "block" in b2:
                continue
            if "table" in (b1, b2) and ("hand" in b1 or "hand" in b2) and con.dist > -0.008:
                continue
            return False, q
        return True, q

    def prepare_col_data(self, obs):
        self.dcol.qpos[:] = obs["kin"].qpos
        # move blocks far away so they do not count as collisions
        for c, bid in self.block_bodies.items():
            jid = self.m.body_jntadr[bid]
            if jid >= 0:
                adr = self.m.jnt_qposadr[jid]
                self.dcol.qpos[adr:adr + 3] = [2.0, 2.0, 2.0]

    # ---------- planning ----------
    def pref_yaw(self, x, y):
        return -0.4 + 5.5 * (y + 0.35) - 2.0 * (x - 0.2)

    def plan_grasp(self, obs):
        blk = obs["blocks"][self.target]
        bpos = np.array(blk["pos"])
        others = [np.array(obs["blocks"][c]["pos"]) for c in obs["blocks"] if c != self.target]
        pref = self.pref_yaw(bpos[0], bpos[1])
        byaw = blk["yaw"]
        cands = []
        for k in range(-2, 3):
            y = wrap(byaw + k * np.pi / 2)
            cands.append((y, 0.0))
        for k in range(-2, 3):  # diagonal fallbacks
            y = wrap(byaw + np.pi / 4 + k * np.pi / 2)
            cands.append((y, 1.0))
        best = None
        for yaw, pen in cands:
            cost = abs(wrap(yaw - pref)) + pen
            # penalty if another cube lies along the finger / thumb footprint
            dirv = np.array([np.cos(yaw), np.sin(yaw)])
            side = np.array([-np.sin(yaw), np.cos(yaw)])
            for o in others:
                r = o[:2] - bpos[:2]
                a = r @ dirv
                b = r @ side
                if -0.08 < a < 0.13 and abs(b) < 0.06:
                    cost += 0.8
            if best is not None and cost >= best[0] + 2.0:
                continue
            Rg = R_for(yaw)
            ok1, q1 = self.pose_ok(np.array([bpos[0], bpos[1], self.grasp_z]), Rg)
            ok2, q2 = self.pose_ok(np.array([bpos[0], bpos[1], self.grasp_z + self.pre_dz]), Rg, q0=q1)
            if not (ok1 and ok2):
                cost += 3.0
            if best is None or cost < best[0]:
                best = (cost, yaw, q2)
        self.yaw = best[1]
        self.q_pre = best[2]
        print("plan: cube %s yaw %.2f pref %.2f chosen %.2f cost %.2f" % (
            np.round(bpos, 3), byaw, pref, self.yaw, best[0]))

    def plan_carry(self):
        best = None
        for yaw in np.arange(-1.2, 1.21, 0.1):
            cost = abs(wrap(yaw - self.yaw))
            if best is not None and cost >= best[0]:
                continue
            ok1, q1 = self.pose_ok(np.array([self.bin[0], self.bin[1], self.place_z]), R_for(yaw))
            ok2, q2 = self.pose_ok(np.array([self.bin[0], self.bin[1], self.carry_z]), R_for(yaw), q0=q1)
            if ok1 and ok2:
                best = (cost, yaw, q1, q2)
        if best is None:
            yaw = 0.3 if self.dest == "left" else -0.4
            ok1, q1 = self.pose_ok(np.array([self.bin[0], self.bin[1], self.place_z]), R_for(yaw))
            ok2, q2 = self.pose_ok(np.array([self.bin[0], self.bin[1], self.carry_z]), R_for(yaw), q0=q1)
            best = (0, yaw, q1, q2)
        self.carry_yaw = best[1]
        self.q_place = best[2]
        self.q_carry = best[3]
        print("carry yaw %.2f" % self.carry_yaw)

    def joint_move(self, q_goal, rate=0.07):
        dq = q_goal - self.q_start
        mx = np.max(np.abs(dq)) + 1e-9
        self.s = min(1.0, self.s + rate / mx)
        q_cmd = self.q_start + self.s * dq
        self.cmd_pos, self.cmd_R = self.fk(q_cmd)
        return q_cmd

    # ---------- helpers ----------
    def set_phase(self, name, t):
        self.phase = name
        self.phase_t0 = t

    def move_towards(self, goal_pos, goal_R, speed, dt, ang_speed=1.5):
        dp = goal_pos - self.cmd_pos
        dist = np.linalg.norm(dp)
        step = speed * dt
        if dist > step:
            self.cmd_pos = self.cmd_pos + dp / dist * step
        else:
            self.cmd_pos = goal_pos.copy()
        er = _rotvec(goal_R @ self.cmd_R.T)
        ang = np.linalg.norm(er)
        astep = ang_speed * dt
        if ang > astep:
            axis = er / ang
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, axis, astep)
            Rd = np.zeros(9)
            mujoco.mju_quat2Mat(Rd, quat)
            self.cmd_R = Rd.reshape(3, 3) @ self.cmd_R
        else:
            self.cmd_R = goal_R.copy()
        return dist < 1e-6 and ang < 1e-6

    def act(self, obs):
        t = obs["t"]
        dt = self.robot.control_dt
        q_now = np.array(obs["arm_target"])
        q_meas = np.array(obs["qpos"][:7])
        self.d.qpos[:] = obs["kin"].qpos
        if self.cmd_pos is None:
            self.cmd_pos, self.cmd_R = self.fk(q_now)
        blk = obs["blocks"][self.target]
        bpos = np.array(blk["pos"])
        site = np.array(obs["grasp_site_pos"])
        ph = self.phase
        el = t - self.phase_t0
        closure = self.closure
        q_cmd = None

        if ph == "plan":
            self.prepare_col_data(obs)
            self.plan_grasp(obs)
            self.plan_carry()
            self.q_start = q_now.copy()
            self.s = 0.0
            self.set_phase("pre", t)
            ph = "pre"
            el = 0.0

        if ph == "pre":
            closure = 0.0
            q_cmd = self.joint_move(self.q_pre)
            if (self.s >= 1.0 and np.max(np.abs(q_meas - self.q_pre)) < 0.05) or el > 4.0:
                self.set_phase("descend", t)
        elif ph == "descend":
            closure = 0.0
            goal = np.array([bpos[0] + self.grasp_dx * np.cos(self.yaw),
                             bpos[1] + self.grasp_dx * np.sin(self.yaw), self.grasp_z])
            R = R_for(self.yaw)
            self.move_towards(goal, R, 0.3, dt)
            err = np.linalg.norm(site - goal)
            if (err < 0.008 and el > 0.3) or el > 2.5:
                self.set_phase("grasp", t)
        elif ph == "grasp":
            closure = min(1.0, el / self.grasp_ramp)
            if el > self.grasp_hold:
                self.set_phase("lift", t)
        elif ph == "lift":
            closure = 1.0
            goal = self.cmd_pos.copy()
            goal[2] = self.lift_z
            self.move_towards(goal, self.cmd_R, 0.3, dt)
            if site[2] > self.lift_z - 0.03 or el > 2.0:
                if np.linalg.norm(bpos - site) < 0.05 and bpos[2] > self.table + 0.08:
                    self.q_start = q_now.copy()
                    self.s = 0.0
                    self.set_phase("carry", t)
                else:
                    self.attempts += 1
                    print("grasp failed, retry", self.attempts)
                    self.set_phase("plan", t)
        elif ph == "carry":
            closure = 1.0
            q_cmd = self.joint_move(self.q_carry, rate=0.06)
            goal = np.array([self.bin[0], self.bin[1], self.carry_z])
            err = np.linalg.norm(site - goal)
            if (self.s >= 1.0 and err < 0.02) or el > 4.0:
                self.q_start = q_now.copy()
                self.s = 0.0
                self.set_phase("lower", t)
            if bpos[2] < self.table + 0.05 and el > 0.3:
                print("lost cube during carry")
                self.attempts += 1
                self.set_phase("plan", t)
        elif ph == "lower":
            closure = 1.0
            q_cmd = self.joint_move(self.q_place, rate=0.04)
            goal = np.array([self.bin[0], self.bin[1], self.place_z])
            err_xy = np.linalg.norm(site[:2] - goal[:2])
            err = np.linalg.norm(site - goal)
            if self.s >= 1.0 and err < 0.012 and el > 0.3:
                self.set_phase("release", t)
            elif el > 2.5:
                if err_xy < 0.03 or self.lower_tries >= 2:
                    self.set_phase("release", t)
                else:
                    self.lower_tries += 1
                    print("lower misaligned by %.3f, retry" % err_xy)
                    self.q_start = q_now.copy()
                    self.s = 0.0
                    self.set_phase("carry", t)
        elif ph == "release":
            closure = max(0.0, 1.0 - el / 0.4)
            if el > 0.8:
                self.set_phase("retract", t)
        elif ph == "retract":
            closure = 0.0
            goal = np.array([self.bin[0] - 0.05, self.bin[1], self.table + 0.22])
            self.move_towards(goal, self.cmd_R, 0.3, dt)
            # if the cube is not in the bin, try again
            if el > 1.5 and bpos[2] < self.table + 0.05 and (
                    abs(bpos[0] - self.bin[0]) > 0.05 or abs(bpos[1] - self.bin[1]) > 0.05):
                print("cube not in bin, retry")
                self.attempts += 1
                self.lower_tries = 0
                self.set_phase("plan", t)

        self.closure = closure
        if q_cmd is None:
            q_cmd = self.ik(q_now, self.cmd_pos, self.cmd_R)
        return np.concatenate([q_cmd, [closure]])
