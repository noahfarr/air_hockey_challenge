import threading
import time
import numpy as np
from scipy.interpolate import CubicSpline

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker


def build_agent(env_info, **kwargs):
    """
    Return the unified hybrid agent.
    """
    return ChickenHockeyAgent(env_info, **kwargs)


class ChickenHockeyAgent(AgentBase):
    """
    Hybrid planner that:
      - Predicts puck motion (PuckTracker)
      - Chooses HIT when a safe/smart shot is available
      - Chooses DEFEND to block incoming threats
      - Otherwise RECOVERS to anchor/hold and stays ready
    """
    def __init__(self, env_info, agent_id=1, **kwargs):
        super(ChickenHockeyAgent, self).__init__(env_info, agent_id, **kwargs)
        self.last_cmd = None
        self.joint_trajectory = None
        self._obs = None

        self.restart = True
        self.tactic_finished = False
        self.optimization_failed = False
        self.plan_new_trajectory = True
        self.replan_time = 0.0
        self.replan_hit = False

        self.x_cmd = None
        self.v_cmd = None
        self.q_cmd = None
        self.dq_cmd = None
        self.x_init = None
        self.q_init = None
        self.q_anchor_pos = None

        self.dt = 1.0 / self.env_info['robot']['control_frequency']
        self.ee_height = self.env_info['robot']["ee_desired_height"]
        self.predicted_time = 0.75
        self.hit_vel = 1.0
        self.hit_finished = False

        self.bound_points = np.array([
            [-(self.env_info['table']['length'] / 2 - 0.05),
             -(self.env_info['table']['width'] / 2 - 0.05)],
            [-(self.env_info['table']['length'] / 2 - 0.05),
             (self.env_info['table']['width'] / 2 - 0.05)],
            [-0.25, (self.env_info['table']['width'] / 2 - 0.05)],
            [-0.25, -(self.env_info['table']['width'] / 2 - 0.05)],
        ])
        self.bound_points = self.bound_points + np.tile([1.51, 0.0], (4, 1))

        table_bounds = np.array([
            [self.bound_points[0], self.bound_points[1]],
            [self.bound_points[1], self.bound_points[2]],
            [self.bound_points[2], self.bound_points[3]],
            [self.bound_points[3], self.bound_points[0]],
        ])

        self.bezier_planner = BezierPlanner(table_bounds, self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)
        self.puck_tracker = PuckTracker(self.env_info, agent_id)

        if self.env_info['robot']['n_joints'] == 3:
            self.joint_anchor_pos = np.array([-1.08, 1.15, 1.49])
        else:
            self.joint_anchor_pos = np.array([
                6.28479822e-11, 7.13520517e-01, -2.96302903e-11,
                -5.02477487e-01, -7.67250279e-11, 1.92566224e+00,
                -2.34645597e-11
            ])

        goal_world = np.array([0.98, 0.0, 0.0])
        goal_pos_robot = world_to_robot(self.env_info["robot"]["base_frame"][0], goal_world)
        self.goal_pos_2d = goal_pos_robot[0][:2]

        self.agent_params = {
            'hit_range': [0.8, 1.3],
            'max_plan_steps': 10,  
            'defend_line': 0.80,  
            'defend_uncertainty_det': 1e-3,
        }

        self.plan_thread = None

    def reset(self):
        self.restart = True
        if self.plan_thread is not None:
            self.plan_thread.join()

        self.last_cmd = None
        self.joint_trajectory = []
        self.tactic_finished = False
        self.optimization_failed = False
        self.plan_new_trajectory = True
        self.replan_time = 0.0
        self.replan_hit = False
        self._obs = None

        self.x_cmd = None
        self.v_cmd = None
        self.q_cmd = None
        self.dq_cmd = None
        self.x_init = None
        self.q_init = None
        self.hit_finished = False
        self.predicted_time = 0.75
        self.hit_vel = 1.0

        self.plan_thread = threading.Thread(target=self._plan_trajectory_thread, daemon=True)

    def draw_action(self, obs):
        """
        Consumer: pops from trajectory queue and returns low-level [q, dq].
        Producer is the planner thread.
        """
        if self.restart:
            self.restart = False
            puck_pos = self.get_puck_pos(obs)

            # initialize state
            self.x_cmd = self.get_ee_pose(obs)[0][:2]
            self.v_cmd = np.zeros(2)
            self.q_cmd = self.get_joint_pos(obs)
            self.dq_cmd = self.get_joint_vel(obs)
            self.x_init = self.x_cmd.copy()
            self.q_init = self.q_cmd.copy()

            self.puck_tracker.reset(puck_pos)
            self.last_cmd = np.vstack([self.q_cmd, self.dq_cmd])

            self.q_anchor_pos = self.q_cmd.copy()
            self.joint_trajectory = np.array([[self.q_cmd, self.dq_cmd]])

            self.plan_thread.start()

        self.puck_tracker.step(self.get_puck_pos(obs))
        self._obs = obs.copy()

        if len(self.joint_trajectory) > 0:
            joint_pos_des, joint_vel_des = self.joint_trajectory[0]
            self.joint_trajectory = self.joint_trajectory[1:]
            self.last_cmd[1] = joint_vel_des
            self.last_cmd[0] = joint_pos_des
        else:
            self.last_cmd[1] = np.zeros(self.env_info['robot']['n_joints'])
        return self.last_cmd

    def stop(self):
        self.restart = True

    def _plan_trajectory_thread(self):
        """
        Producer: evaluates puck prediction, chooses mode, pushes joint waypoints.
        """
        while not self.restart:
            time.sleep(0.01)
            opt_trial = 0

            for _ in range(5):
                if len(self.joint_trajectory) >= self.agent_params['max_plan_steps']:
                    break

                state, P, t_predict = self.puck_tracker.get_prediction(self.predicted_time,
                                                                       defend_line=self.agent_params['defend_line'])

                mode = self._choose_mode(state, P)

                if mode == "HIT" and not self.hit_finished:
                    if self.replan_hit:
                        self.plan_new_trajectory = True
                        self.predicted_time = 0.75
                        self.replan_hit = False
                    ee_traj, q_anchor = self.plan_hit_trajectory(state, self.hit_vel, self.predicted_time)
                elif mode == "DEFEND":
                    ee_traj, q_anchor = self.plan_defend_trajectory(state, t_predict)
                else:
                    self.plan_new_trajectory = True
                    if self.predicted_time <= 1e-3:
                        self.predicted_time = 0.5
                    ee_traj, q_anchor = self.plan_stop_trajectory(self.predicted_time)

                success, joint_pos_traj = self.optimizer.optimize_trajectory(
                    ee_traj, self.q_cmd, self.dq_cmd, q_anchor
                )
                if not success or len(joint_pos_traj) == 0:
                    self.hit_vel *= 0.85
                    self.hit_finished = False
                    self.predicted_time = min(self.predicted_time + 0.10, 1.50)
                    self.replan_time = 0.0
                    opt_trial += 1
                    continue

                traj_len = min(joint_pos_traj.shape[0],
                               self.agent_params['max_plan_steps'] if not self.hit_finished else joint_pos_traj.shape[0])
                joint_pos_traj = joint_pos_traj[:traj_len]
                self.predicted_time = max(self.bezier_planner.t_final - joint_pos_traj.shape[0] * self.dt, 0.0)

                self.joint_trajectory = np.vstack([
                    self.joint_trajectory,
                    self.cubic_spline_interpolation(joint_pos_traj)
                ])

                self.x_cmd = ee_traj[traj_len - 1][:3].copy()
                self.v_cmd = ee_traj[traj_len - 1][3:6].copy()
                self.q_cmd = self.joint_trajectory[-1][0].copy()
                self.dq_cmd = self.joint_trajectory[-1][1].copy()

                self.plan_new_trajectory = False
                self.replan_time = self.agent_params['max_plan_steps'] * self.dt

            if opt_trial >= 5:
                self.hit_finished = True
                break

    def _choose_mode(self, predicted_state, P):
        """
        Simple arbitration:

        HIT if puck is hittable and is safe.
        DEFEND if uncertainty small and puck is inbound near defend line.
        Else RECOVER.
        """
        if self.should_hit(predicted_state):
            return "HIT"

        cov_det_ok = np.linalg.det(P[:2, :2]) < self.agent_params['defend_uncertainty_det']
        inbound = predicted_state[3] < -1e-3
        near_line = predicted_state[0] <= (self.agent_params['defend_line'] + 0.05)
        within_y = abs(predicted_state[1]) < (self.env_info['table']['width'] / 2
                                              - self.env_info['puck']['radius']
                                              - 2 * self.env_info['mallet']['radius'])
        if cov_det_ok and inbound and near_line and within_y:
            return "DEFEND"

        return "RECOVER"

    def should_hit(self, state):
        x_ok = self.agent_params['hit_range'][0] < state[0] < self.agent_params['hit_range'][1]
        y_ok = np.abs(state[1]) < (self.env_info['table']['width'] / 2
                                   - self.env_info['puck']['radius']
                                   - 2 * self.env_info['mallet']['radius'])
        vx = state[3]
        safe_inbound = vx > -0.4
        return x_ok and y_ok and safe_inbound

    def plan_hit_trajectory(self, predicted_state, hit_vel, t_predict):
        puck_pos = predicted_state[:2]

        hit_dir_2d = self.goal_pos_2d - puck_pos
        hit_dir_2d = hit_dir_2d / (np.linalg.norm(hit_dir_2d) + 1e-12)

        hit_pos_2d = puck_pos - hit_dir_2d * (self.env_info['puck']['radius'] + self.env_info['mallet']['radius'])
        hit_vel_2d = hit_dir_2d * hit_vel

        if self.plan_new_trajectory:
            self.q_anchor_pos = self.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_init, solve_max_time=5e-3)
            self.bezier_planner.compute_control_point(self.x_cmd[:2], self.v_cmd[:2],
                                                      hit_pos_2d, hit_vel_2d, t_predict)
            cart_traj = self._bezier_to_cart(self.agent_params['max_plan_steps'])
        else:
            if self.bezier_planner.t_final > self.replan_time:
                self.q_anchor_pos = self.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_init,
                                                                  solve_max_time=0.1)
                self.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d, hit_vel_2d, t_predict)
            cart_traj = self._bezier_to_cart()

        if self.bezier_planner.t_final <= 2 * self.agent_params['max_plan_steps'] * self.dt:
            self.hit_finished = True
        return cart_traj, self.q_anchor_pos

    def plan_defend_trajectory(self, puck_state, t_predict):
        """
        Block on a vertical line (defend_line) at puck's y with zero approach velocity,
        then smoothly return to center.
        """
        start_pos_2d = self.x_cmd[:2]
        defend_y = np.clip(
            puck_state[1],
            self.bound_points[0, 1] + 0.05,
            self.bound_points[2, 1] - 0.05
        )
        defend_x = self.agent_params['defend_line']
        hit_dir_2d = np.array([0.0, np.sign(defend_y - start_pos_2d[1])])
        target_2d = np.array([defend_x, defend_y])

        self.bezier_planner.compute_control_point(start_pos_2d, self.v_cmd[:2],
                                                  target_2d,
                                                  np.zeros(2), t_predict)
        res1 = self._bezier_to_cart_raw()

        last_p = res1[0][-1]
        last_v = res1[1][-1]
        standby = np.array([0.65, 0.0])
        self.bezier_planner.compute_control_point(last_p, last_v, standby, np.zeros(2), 1.5)
        res2 = self._bezier_to_cart_raw()

        ee_traj = self._stack_cart(res1, res2)
        return ee_traj, None

    def plan_stop_trajectory(self, t_predict):
        """
        If moving, coast to a comfortable stop; else drift back to x_init.
        """
        if np.linalg.norm(self.v_cmd) > 1e-2:
            x_stop = (self.x_cmd[:2] + self.v_cmd[:2] / 5.0)
            self.hit_finished = True
        elif np.linalg.norm(self.x_cmd[:2] - self.x_init) > 1e-3:
            x_stop = self.x_init.copy()
            t_predict = 1.5
            self.hit_finished = True
        else:
            x_stop = self.x_init.copy()
            self.hit_finished = False
            self.replan_hit = True

        x_stop = np.clip(x_stop, self.bound_points[0] + 0.05, self.bound_points[2] - 0.05)
        self.bezier_planner.compute_control_point(self.x_cmd[:2], self.v_cmd[:2], x_stop,
                                                  np.zeros_like(self.x_init), t_predict)
        cart_traj = self._bezier_to_cart()
        return cart_traj, self.q_init

    def _bezier_to_cart(self, max_steps=-1):
        if max_steps > 0:
            t_plan = min(self.bezier_planner.t_final, max_steps * self.dt)
        else:
            t_plan = self.bezier_planner.t_final
        res = np.array([self.bezier_planner.get_point(t_i) for t_i in
                        np.arange(self.dt, t_plan + 1e-6, self.dt)])
        p = res[:, 0]
        dp = res[:, 1]
        ddp = res[:, 2]
        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return np.hstack([p, dp, ddp])

    def _bezier_to_cart_raw(self):
        res = np.array([self.bezier_planner.get_point(t_i) for t_i in
                        np.arange(0, self.bezier_planner.t_final + 1e-6, self.dt)])
        p = res[1:, 0].squeeze()
        dp = res[1:, 1].squeeze()
        ddp = res[1:, 2].squeeze()
        return p, dp, ddp

    def _stack_cart(self, res1, res2):
        p1, dp1, ddp1 = res1
        p2, dp2, ddp2 = res2
        p = np.vstack([p1, p2])
        dp = np.vstack([dp1, dp2])
        ddp = np.vstack([ddp1, ddp2])
        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return np.hstack([p, dp, ddp])

    def get_joint_trajectory(self, ee_traj):
        """
        Pure IK back-up (kept for completeness; optimizer is primary).
        """
        init_q = self.last_cmd[0]
        joint_pos_traj = list()
        while len(ee_traj) > 0:
            ee_pos_des = ee_traj[0][:3]
            ee_traj = ee_traj[1:]
            success, joint_pos_des = inverse_kinematics(self.robot_model, self.robot_data,
                                                        ee_pos_des, initial_q=init_q)
            if not success:
                joint_pos_traj.clear()
                return joint_pos_traj
            init_q = joint_pos_des
            joint_pos_traj.append(joint_pos_des[:self.env_info['robot']['n_joints']])
        return joint_pos_traj

    def cubic_spline_interpolation(self, joint_pos_traj):
        joint_pos_traj = np.array(joint_pos_traj)
        t = np.linspace(1, joint_pos_traj.shape[0], joint_pos_traj.shape[0]) * 0.02
        f = CubicSpline(t, joint_pos_traj, axis=0)
        df = f.derivative(1)
        return np.stack([f(t), df(t)]).swapaxes(0, 1)

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0, solve_max_time):
        hit_pos = np.concatenate([hit_pos_2d, [self.ee_height]])
        hit_dir = np.concatenate([hit_dir_2d, [0.0]])
        success, q_star = self.optimizer.solve_hit_config_ik_null(
            hit_pos, hit_dir, q_0, max_time=solve_max_time
        )
        return q_star

