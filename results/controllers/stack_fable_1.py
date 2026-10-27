
import numpy as np, mujoco

HALF = 0.025

class Controller:
    PRECURL = 0.38        # finger pre-curl closure on approach
    X_OFF = 0.015         # cube center offset along site x (toward fingers) at grasp
    Z_OFF = 0.0           # site height relative to cube center at grasp
    TILT = 0.2            # fingers-down tilt of the hand (rad)
    V_CART = 0.35         # m/s cartesian speed limit
    W_YAW = 1.5           # rad/s yaw target speed limit
    Z_CARRY = 0.92
    Z_PRE = 0.10
    SEEDS = [np.array([0.875, -0.751, -1.223, -0.331, -0.24, -0.28, 0.391]),
             np.array([0.4, 0.0, -0.1, -0.3, -1.5, 0.1, -0.1]),
             np.array([0.2, -0.6, -0.2, -0.6, -0.8, 0.3, 0.3]),
             np.array([0.3, -0.3, 0.3, 0.3, -1.2, 0.5, 0.0])]

    def __init__(self, robot, task):
        self.robot = robot; self.task = task
        self.m = robot.model
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, robot.grasp_site)
        self.aq = np.array(robot.arm_qpos_adr); self.ad = np.array(robot.arm_dof_adr)
        self.lo = np.array(robot.arm_lower); self.hi = np.array(robot.arm_upper)
        self.home = np.array(robot.home_arm)
        self.jacp = np.zeros((3, self.m.nv)); self.jacr = np.zeros((3, self.m.nv))
        self.dt = robot.control_dt
        self.phase = -1; self.t_phase_start = 0.0
        self.yaw = None; self.yaw_cmd = None
        self.p_cmd = None
        self.closure = 0.0
        self.hold_off = None
        self.q_ref = self.home.copy()
        self.attempt = 0
        self.frozen_goal = None
        self.stall_count = 0
        self.prev_site = None
        self.sag = np.zeros(7)

    # ---------- kinematics ----------
    def ik(self, d, q0, p_des, R_des, iters=25, w_rot=0.5, q_ref=None):
        q = q0.copy(); m = self.m; sid = self.sid
        if q_ref is None: q_ref = self.q_ref
        for it in range(iters):
            d.qpos[self.aq] = q
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
            ep = p_des - p
            Re = R_des @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, sid)
            J = np.vstack([self.jacp[:, self.ad], w_rot * self.jacr[:, self.ad]])
            e = np.concatenate([ep, w_rot * er])
            Jp = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.eye(6))
            dq = Jp @ e
            N = np.eye(7) - Jp @ J
            dq += N @ (0.1 * (q_ref - q))
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def fk_err(self, d, q, p_des, R_des):
        d.qpos[self.aq] = q
        mujoco.mj_kinematics(self.m, d)
        p = d.site_xpos[self.sid]; R = d.site_xmat[self.sid].reshape(3, 3)
        Re = R_des @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        return np.linalg.norm(p_des - p), np.linalg.norm(er)

    def global_ik(self, d, p_des, R_des):
        """multi-seed IK; returns (score, q). lower score = better."""
        best = None
        for q0 in self.SEEDS:
            q = self.ik(d, q0, p_des, R_des, iters=60, q_ref=q0)
            ep, er = self.fk_err(d, q, p_des, R_des)
            margin = np.min(np.minimum(q - self.lo, self.hi - q))
            sc = ep * 100 + er * 5 - min(margin, 0.3)
            if best is None or sc < best[0]:
                best = (sc, q, ep, er)
        return best

    def R_from_yaw(self, yaw):
        th = self.TILT
        x = np.array([np.cos(yaw) * np.cos(th), np.sin(yaw) * np.cos(th), -np.sin(th)])
        z = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def choose_yaw(self, d, cube_yaw, positions, near=None):
        base = cube_yaw + np.pi / 4
        best = None
        for k in range(4):
            yw = (base + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
            R = self.R_from_yaw(yw)
            sc = 0.0; qs = []
            for p in positions:
                s, q, ep, er = self.global_ik(d, p, R)
                sc += s; qs.append(q)
            if near is not None:
                sc += 0.1 * abs((yw - near + np.pi) % (2 * np.pi) - np.pi)
            if best is None or sc < best[0]:
                best = (sc, yw, qs)
        return best[1], best[2]

    def grasp_site_target(self, cube_pos, R):
        return cube_pos - R @ np.array([self.X_OFF, 0.0, 0.0]) + np.array([0, 0, self.Z_OFF])

    def place_site_target(self, dpos, R):
        off = self.hold_off if self.hold_off is not None else np.array([self.X_OFF, -self.Z_OFF, 0.0])
        cube_goal = dpos + np.array([0, 0, 2 * HALF + 0.006])
        return cube_goal - R @ off

    # ---------- phases ----------
    def set_phase(self, k, t, obs):
        self.phase = k; self.t_phase_start = t; self.stall_count = 0
        print("t=%.2f -> phase %d site %s tgt %s c=%.2f" % (t, k, obs["grasp_site_pos"].round(3),
              obs["blocks"][self.task["target"]]["pos"].round(3), self.closure))

    def plan_grasp(self, obs):
        tgt = obs["blocks"][self.task["target"]]
        dst = obs["blocks"][self.task["destination"]]
        d = obs["kin"]
        Rg = self.R_from_yaw(0.0)
        pg = self.grasp_site_target(tgt["pos"], Rg)
        yaw, qs = self.choose_yaw(d, tgt["yaw"], [pg, pg + [0, 0, self.Z_PRE]], near=self.yaw)
        self.yaw = yaw
        if self.yaw_cmd is None: self.yaw_cmd = yaw
        self.q_ref = qs[0]
        print("grasp yaw %.2f (cube yaw %.2f)" % (yaw, tgt["yaw"]))

    def plan_place(self, obs):
        dst = obs["blocks"][self.task["destination"]]
        d = obs["kin"]
        pp = self.place_site_target(dst["pos"], self.R_from_yaw(self.yaw))
        yaw, qs = self.choose_yaw(d, self.yaw - np.pi / 4, [pp, pp + [0, 0, 0.06]], near=self.yaw)
        self.yaw = yaw
        self.q_ref = qs[0]
        print("place yaw %.2f" % yaw)

    def act(self, obs):
        t = obs["t"]
        tgt = obs["blocks"][self.task["target"]]
        dst = obs["blocks"][self.task["destination"]]
        site = obs["grasp_site_pos"]; Rs = obs["grasp_site_mat"]
        if self.phase == -1:
            self.p_cmd = site.copy()
            self.plan_grasp(obs)
            self.set_phase(0, t, obs)
        dtp = t - self.t_phase_start
        ph = self.phase
        tpos = tgt["pos"]; dpos = dst["pos"]
        c = self.closure
        R = self.R_from_yaw(self.yaw_cmd)
        if ph == 0:  # move above target
            goal = self.grasp_site_target(tpos, R) + np.array([0, 0, self.Z_PRE])
            c = self.PRECURL
            if np.linalg.norm(site - goal) < 0.012 and dtp > 0.5:
                self.set_phase(1, t, obs)
            elif dtp > 4.0:
                self.plan_grasp(obs); self.set_phase(0, t, obs)
        elif ph == 1:  # descend
            goal = self.grasp_site_target(tpos, R)
            c = self.PRECURL
            moving = self.prev_site is not None and np.linalg.norm(site - self.prev_site) > 0.0015
            if not moving and dtp > 0.4: self.stall_count += 1
            else: self.stall_count = 0
            if (np.linalg.norm(site - goal) < 0.006 and dtp > 0.3) or dtp > 2.5 or self.stall_count > 8:
                self.frozen_goal = self.p_cmd.copy()
                self.set_phase(2, t, obs)
        elif ph == 2:  # close
            goal = self.frozen_goal
            c = min(1.0, self.PRECURL + dtp / 0.6 * (1.0 - self.PRECURL))
            if dtp > 1.0:
                self.set_phase(3, t, obs)
        elif ph == 3:  # lift
            goal = self.p_cmd.copy(); goal[2] = self.Z_CARRY
            c = 1.0
            if dtp > 0.6 and self.hold_off is None:
                off = Rs.T @ (tpos - site)
                if np.linalg.norm(off) < 0.05 and tpos[2] > 0.83:
                    self.hold_off = off
                    print("hold offset", off.round(3))
                    self.plan_place(obs)
                else:
                    print("grasp failed, off", off.round(3), "cube z", tpos[2])
                    self.attempt += 1
                    self.hold_off = None
                    self.plan_grasp(obs)
                    self.set_phase(0, t, obs)
            if self.hold_off is not None and abs(site[2] - self.Z_CARRY) < 0.015 and dtp > 0.5:
                self.set_phase(4, t, obs)
        elif ph == 4:  # move above destination
            goal = self.place_site_target(dpos, R) + np.array([0, 0, 0.06])
            goal[2] = max(goal[2], self.Z_CARRY)
            c = 1.0
            still = self.prev_site is not None and np.linalg.norm(site - self.prev_site) < 0.001
            if np.linalg.norm(site - goal) < 0.008 and abs(self.yaw_cmd - self.yaw) < 0.02 and dtp > 0.5 and still:
                self.set_phase(5, t, obs)
            if tpos[2] < 0.83 and dtp > 0.5:  # dropped it
                print("lost cube during carry")
                self.hold_off = None; self.plan_grasp(obs); self.set_phase(0, t, obs)
        elif ph == 5:  # descend to place
            goal = self.place_site_target(dpos, R)
            c = 1.0
            moving = self.prev_site is not None and np.linalg.norm(site - self.prev_site) > 0.0015
            if not moving and dtp > 0.4: self.stall_count += 1
            else: self.stall_count = 0
            if (np.linalg.norm(site - goal) < 0.006 and dtp > 0.3) or dtp > 3.0 or self.stall_count > 8:
                self.frozen_goal = self.p_cmd.copy()
                self.set_phase(6, t, obs)
        elif ph == 6:  # release
            goal = self.frozen_goal
            c = max(0.0, 1.0 - dtp / 0.4)
            if dtp > 0.8:
                self.set_phase(7, t, obs)
        elif ph == 7:  # retreat up
            goal = self.frozen_goal + np.array([0, 0, 0.08])
            c = 0.0
            if dtp > 1.5:
                # check result
                dxy = np.linalg.norm(tpos[:2] - dpos[:2]); dz = tpos[2] - dpos[2]
                if dxy < 0.02 and abs(dz - 2 * HALF) < 0.01:
                    self.set_phase(8, t, obs)
                else:
                    print("place failed dxy %.3f dz %.3f, retry" % (dxy, dz))
                    self.hold_off = None; self.attempt += 1
                    self.plan_grasp(obs); self.set_phase(0, t, obs)
        else:  # done, hold
            goal = self.p_cmd.copy()
            c = 0.0
        # cartesian rate limit (slower for contact-sensitive moves)
        v = self.V_CART
        if ph == 1:
            v = 0.25
        elif ph == 5:
            v = 0.2 if (self.p_cmd[2] - goal[2]) > 0.025 else 0.07
        elif ph == 7:
            v = 0.08 if dtp < 0.6 else self.V_CART
        step = goal - self.p_cmd
        n = np.linalg.norm(step)
        if n > v * self.dt:
            step *= v * self.dt / n
        self.p_cmd = self.p_cmd + step
        # yaw rate limit
        dy = (self.yaw - self.yaw_cmd + np.pi) % (2 * np.pi) - np.pi
        self.yaw_cmd += np.clip(dy, -self.W_YAW * self.dt, self.W_YAW * self.dt)
        self.closure = c
        self.prev_site = site.copy()
        # steady-state servo error (sag under weak motors): low-pass of target - measured
        qerr = obs["arm_target"] - obs["qpos"][:7]
        if ph not in (2, 5, 6):
            self.sag = 0.9 * self.sag + 0.1 * qerr
        sag = np.clip(self.sag, -0.2, 0.2)
        q0 = obs["arm_target"] - sag
        q = self.ik(obs["kin"], q0, self.p_cmd, self.R_from_yaw(self.yaw_cmd))
        q = np.clip(q + sag, self.lo, self.hi)
        return np.concatenate([q, [c]])
