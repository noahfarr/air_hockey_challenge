"""Agent builder implementing the "Mirror-Goalie + Two-Touch Offense" policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import forward_kinematics, inverse_kinematics, jacobian


class AgentState(Enum):
    """Finite state machine for the deterministic policy."""

    EMERGENCY_CLEAR = auto()
    SAVE = auto()
    PREPARE = auto()
    STRIKE = auto()


@dataclass
class StrikePlan:
    """Container describing a planned strike."""

    approach_point: np.ndarray
    impact_point: np.ndarray
    shot_direction: np.ndarray
    goal_point: np.ndarray
    mallet_speed: float
    follow_distance: float
    follow_time: float
    stage: str = "approach"
    elapsed: float = 0.0


class PuckPredictor:
    """Simple 2-D puck predictor with specular wall reflections."""

    def __init__(self, x_limits: Tuple[float, float], y_limits: Tuple[float, float], radius: float, dt: float = 0.002):
        self.x_min, self.x_max = x_limits
        self.y_min, self.y_max = y_limits
        self.radius = radius
        self.dt = dt

    def predict_line(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target_x: float,
        max_time: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """Predict the first intercept with a vertical line ``x = target_x``."""

        pos = np.asarray(position[:2], dtype=float).copy()
        vel = np.asarray(velocity[:2], dtype=float).copy()
        time_passed = 0.0

        if np.linalg.norm(vel) < 1e-6 or max_time <= 0.0:
            return None

        steps = int(max_time / self.dt)
        if steps <= 0:
            return None

        for _ in range(steps):
            next_pos = pos + vel * self.dt

            if next_pos[1] < self.y_min:
                next_pos[1] = 2 * self.y_min - next_pos[1]
                vel[1] = abs(vel[1])
            elif next_pos[1] > self.y_max:
                next_pos[1] = 2 * self.y_max - next_pos[1]
                vel[1] = -abs(vel[1])

            if next_pos[0] < self.x_min:
                next_pos[0] = 2 * self.x_min - next_pos[0]
                vel[0] = abs(vel[0])
            elif next_pos[0] > self.x_max:
                next_pos[0] = 2 * self.x_max - next_pos[0]
                vel[0] = -abs(vel[0])

            if (pos[0] - target_x) * (next_pos[0] - target_x) <= 0:
                if abs(next_pos[0] - pos[0]) < 1e-8:
                    alpha = 0.0
                else:
                    alpha = (target_x - pos[0]) / (next_pos[0] - pos[0])
                    alpha = float(np.clip(alpha, 0.0, 1.0))

                impact = pos + alpha * (next_pos - pos)
                impact_vel = vel.copy()
                intercept_time = time_passed + alpha * self.dt
                return intercept_time, impact, impact_vel

            pos = next_pos
            time_passed += self.dt

        return None


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """
    return MirrorGoalieTwoTouchAgent(env_info, **kwargs)


class MirrorGoalieTwoTouchAgent(AgentBase):
    """Implementation of the "Mirror-Goalie + Two-Touch Offense" policy."""

    def __init__(self, env_info, agent_id: int = 1, **kwargs):
        super().__init__(env_info, agent_id, **kwargs)

        self.dt = 1.0 / float(self.env_info["robot"]["control_frequency"])
        self.n_joints = self.env_info["robot"]["n_joints"]
        self.ee_height = self.env_info["robot"]["ee_desired_height"]
        self.puck_radius = self.env_info["puck"]["radius"]
        self.mallet_radius = self.env_info["mallet"]["radius"]

        vel_limits = self.env_info["robot"]["joint_vel_limit"].copy()
        self.joint_vel_limit = vel_limits * 0.25
        self.joint_pos_limit = self.env_info["robot"]["joint_pos_limit"].copy()
        self.max_joint_step = self.joint_vel_limit[1] * self.dt
        self.min_joint_step = self.joint_vel_limit[0] * self.dt

        base_frame = np.array(self.env_info["robot"]["base_frame"][self.agent_id - 1])
        self.base_to_world = base_frame.copy()
        self.world_to_base = np.linalg.inv(self.base_to_world)

        self._init_table_geometry()
        self._init_parameters()

        x_limits = (self.table_x_min + self.puck_radius, self.table_x_max - self.puck_radius)
        y_limits = (self.table_y_min + self.puck_radius, self.table_y_max - self.puck_radius)
        self.predictor = PuckPredictor(x_limits, y_limits, self.puck_radius)

        self.home_position = np.array([self.defend_line_x, 0.0, self.ee_height])
        success, q_home = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            self.home_position,
            initial_q=None,
        )
        if success:
            self.home_q = q_home[: self.n_joints].copy()
        else:
            self.home_q = np.zeros(self.n_joints)

        self.reset()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_table_geometry(self):
        length = float(self.env_info["table"]["length"])
        width = float(self.env_info["table"]["width"])

        corners = [
            np.array([-length / 2, -width / 2, 0.0, 1.0]),
            np.array([-length / 2, width / 2, 0.0, 1.0]),
            np.array([length / 2, -width / 2, 0.0, 1.0]),
            np.array([length / 2, width / 2, 0.0, 1.0]),
        ]
        corners_base = [(self.world_to_base @ corner)[:3] for corner in corners]
        xs = np.array([corner[0] for corner in corners_base])
        ys = np.array([corner[1] for corner in corners_base])

        self.table_x_min = float(xs.min())
        self.table_x_max = float(xs.max())
        self.table_y_min = float(ys.min())
        self.table_y_max = float(ys.max())

        self.safe_x_min = min(self.table_x_min, self.table_x_max) + self.mallet_radius
        self.safe_x_max = max(self.table_x_min, self.table_x_max) - self.mallet_radius
        self.safe_y_min = min(self.table_y_min, self.table_y_max) + self.mallet_radius
        self.safe_y_max = max(self.table_y_min, self.table_y_max) - self.mallet_radius

        center_world = np.array([0.0, 0.0, 0.0, 1.0])
        own_goal_world = np.array([-length / 2, 0.0, 0.0, 1.0])
        opp_goal_world = np.array([length / 2, 0.0, 0.0, 1.0])

        self.center_line_x = float((self.world_to_base @ center_world)[0])
        self.goal_line_x = float((self.world_to_base @ own_goal_world)[0])
        self.opponent_goal_line_x = float((self.world_to_base @ opp_goal_world)[0])

    def _init_parameters(self):
        home_sign = np.sign(self.center_line_x - self.goal_line_x)
        if home_sign == 0:
            home_sign = 1.0
        attack_sign = np.sign(self.opponent_goal_line_x - self.center_line_x)
        if attack_sign == 0:
            attack_sign = 1.0

        self.home_side_sign = home_sign
        self.attack_sign = attack_sign

        self.emergency_timeout = 10.0
        self.clear_speed = 1.0
        self.save_forward_speed = 0.2
        self.save_prediction_time = 1.5
        self.save_velocity_threshold = 0.05

        defend_offset = 0.14
        defend_guess = self.goal_line_x + self.home_side_sign * defend_offset
        self.defend_line_x = float(self._clip_x(defend_guess, stay_on_half=True))

        strike_offsets = np.sort(self.center_line_x - self.home_side_sign * np.array([0.32, 0.12]))
        self.strike_box_x_range = (float(strike_offsets[0]), float(strike_offsets[1]))
        self.strike_box_half_y = 0.2
        self.strike_lateral_speed_threshold = 0.25

        self.strike_speed = 1.2
        self.strike_follow_time = 0.15
        self.strike_approach_offset = 0.08

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------
    def reset(self):
        self.state = AgentState.SAVE
        self.state_timer = 0.0
        self.time_on_own_half = 0.0
        self.force_emergency_clear = False
        self.post_strike_cooldown = 0.0
        self.strike_plan: Optional[StrikePlan] = None
        self.last_q_cmd: Optional[np.ndarray] = None
        self.last_dq_cmd: Optional[np.ndarray] = None
        self.prepare_stage = "approach"
        self.current_ee = np.zeros(3)
        self.save_intercept: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    def draw_action(self, obs):
        joint_pos = self.get_joint_pos(obs)
        joint_vel = self.get_joint_vel(obs)
        puck_pos = self.get_puck_pos(obs)
        puck_vel = self.get_puck_vel(obs)
        opponent = self._get_opponent(obs)

        self.current_ee, _ = forward_kinematics(self.robot_model, self.robot_data, joint_pos)

        if self._on_our_half(puck_pos[0]):
            self.time_on_own_half += self.dt
        else:
            self.time_on_own_half = max(0.0, self.time_on_own_half - self.dt)

        if self.post_strike_cooldown > 0.0:
            self.post_strike_cooldown = max(0.0, self.post_strike_cooldown - self.dt)

        next_state = self._select_state(puck_pos, puck_vel)
        if next_state != self.state:
            self._on_exit_state(self.state)
            self.state = next_state
            self.state_timer = 0.0
            self._on_enter_state(next_state, puck_pos, opponent)
        else:
            self.state_timer += self.dt

        target_pos, target_vel = self._state_action(puck_pos, puck_vel)

        q_target, success = self._solve_ik(target_pos, joint_pos)
        if not success:
            q_target = self.home_q.copy()
            target_vel = np.zeros(3)
            self.force_emergency_clear = True

        q_cmd, dq_cmd = self._compute_joint_velocities(q_target, joint_pos, target_vel)

        self.last_q_cmd = q_cmd.copy()
        self.last_dq_cmd = dq_cmd.copy()

        return np.vstack([q_cmd, dq_cmd])

    # ------------------------------------------------------------------
    # State machine management
    # ------------------------------------------------------------------
    def _select_state(self, puck_pos, puck_vel):
        if self.time_on_own_half > self.emergency_timeout or self.force_emergency_clear:
            self.force_emergency_clear = False
            return AgentState.EMERGENCY_CLEAR

        if self._should_save(puck_pos, puck_vel):
            return AgentState.SAVE

        if self.post_strike_cooldown <= 0.0 and self._can_strike(puck_pos, puck_vel):
            return AgentState.STRIKE

        if self._should_prepare(puck_pos, puck_vel):
            return AgentState.PREPARE

        return AgentState.SAVE

    def _on_enter_state(self, state, puck_pos, opponent):
        if state == AgentState.PREPARE:
            self.prepare_stage = "approach"
        elif state == AgentState.STRIKE:
            self.strike_plan = self._plan_strike(puck_pos, opponent)
            if self.strike_plan is None:
                self.post_strike_cooldown = 0.05
                self.state = AgentState.SAVE
        elif state == AgentState.EMERGENCY_CLEAR:
            self.force_emergency_clear = False

    def _on_exit_state(self, state):
        if state == AgentState.STRIKE:
            self.strike_plan = None

    def _state_action(self, puck_pos, puck_vel):
        if self.state == AgentState.EMERGENCY_CLEAR:
            return self._emergency_action(puck_pos)
        if self.state == AgentState.SAVE:
            return self._save_action(puck_pos)
        if self.state == AgentState.PREPARE:
            return self._prepare_action(puck_pos, puck_vel)
        if self.state == AgentState.STRIKE and self.strike_plan is not None:
            return self._strike_action(puck_pos)
        return self._home_target()

    # ------------------------------------------------------------------
    # State logic implementations
    # ------------------------------------------------------------------
    def _emergency_action(self, puck_pos):
        hit_dir = np.array([self.attack_sign, 0.0])
        offset = self.puck_radius + 0.5 * self.mallet_radius
        impact_xy = puck_pos[:2] - hit_dir * offset
        impact_xy[1] = puck_pos[1]
        impact_xy = self._clip_xy(impact_xy, stay_on_half=True)

        target_pos = np.array([impact_xy[0], impact_xy[1], self.ee_height])
        target_vel = np.array([self.attack_sign * self.clear_speed, -0.3 * puck_pos[1], 0.0])
        return target_pos, target_vel

    def _save_action(self, puck_pos):
        intercept = self.save_intercept
        if intercept is not None:
            _, impact_pos, _ = intercept
            y_target = impact_pos[1]
        else:
            y_target = 0.0

        defend_xy = self._clip_xy(np.array([self.defend_line_x, y_target]), stay_on_half=True)
        target_pos = np.array([defend_xy[0], defend_xy[1], self.ee_height])
        target_vel = np.array([self.attack_sign * self.save_forward_speed, 0.0, 0.0])
        return target_pos, target_vel

    def _prepare_action(self, puck_pos, puck_vel):
        strike_center = np.array([
            np.mean(self.strike_box_x_range),
            0.0,
        ])
        strike_center = self._clip_xy(strike_center, stay_on_half=True)

        direction = strike_center - puck_pos[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([self.attack_sign, 0.0])
        direction /= np.linalg.norm(direction)

        behind_offset = self.puck_radius + 0.8 * self.mallet_radius
        approach_point = puck_pos[:2] - direction * behind_offset
        approach_point = self._clip_xy(approach_point, stay_on_half=True)

        if self.prepare_stage == "approach":
            target_xy = approach_point
            if np.linalg.norm(self.current_ee[:2] - approach_point) < 0.04:
                self.prepare_stage = "drag"
        else:
            target_xy = strike_center - direction * (self.puck_radius + 0.01)
            target_xy = self._clip_xy(target_xy, stay_on_half=True)

        target_pos = np.array([target_xy[0], target_xy[1], self.ee_height])
        lateral_comp = -np.clip(puck_vel[1], -0.5, 0.5)
        push_speed = 0.45 if self.prepare_stage == "drag" else 0.3
        target_vel = np.array([self.attack_sign * push_speed, lateral_comp, 0.0])
        return target_pos, target_vel

    def _strike_action(self, puck_pos):
        plan = self.strike_plan
        target_vel = plan.shot_direction * plan.mallet_speed

        if plan.stage == "approach":
            target_xy = self._clip_xy(plan.approach_point, stay_on_half=True)
            target_vel = plan.shot_direction * 0.4
            if np.linalg.norm(self.current_ee[:2] - plan.approach_point) < 0.03:
                plan.stage = "impact"
        elif plan.stage == "impact":
            target_xy = self._clip_xy(plan.impact_point, stay_on_half=True)
            if np.linalg.norm(self.current_ee[:2] - plan.impact_point) < 0.02:
                plan.stage = "follow"
                plan.elapsed = 0.0
        else:
            plan.elapsed += self.dt
            progress = min(plan.follow_distance, plan.mallet_speed * plan.elapsed)
            target_xy = plan.impact_point + plan.shot_direction * progress
            target_xy = self._clip_xy(target_xy, stay_on_half=False)
            if plan.elapsed >= plan.follow_time:
                self.post_strike_cooldown = 0.1
                self.strike_plan = None
                return self._save_action(puck_pos)

        target_pos = np.array([target_xy[0], target_xy[1], self.ee_height])
        target_vel3 = np.array([target_vel[0], target_vel[1], 0.0])
        return target_pos, target_vel3

    def _home_target(self):
        return self.home_position.copy(), np.zeros(3)

    # ------------------------------------------------------------------
    # Decision helper functions
    # ------------------------------------------------------------------
    def _should_save(self, puck_pos, puck_vel):
        moving_to_goal = puck_vel[0] * self.home_side_sign < -self.save_velocity_threshold
        puck_inside_band = (puck_pos[0] - self.defend_line_x) * self.home_side_sign <= 0

        self.save_intercept = None

        if not moving_to_goal and not puck_inside_band:
            return False

        intercept = self.predictor.predict_line(puck_pos, puck_vel, self.defend_line_x, self.save_prediction_time)
        if intercept is None:
            return puck_inside_band

        self.save_intercept = intercept
        return True

    def _can_strike(self, puck_pos, puck_vel):
        if not self._in_strike_box(puck_pos):
            return False
        if np.abs(puck_vel[1]) > self.strike_lateral_speed_threshold:
            return False
        if not self._on_our_half(puck_pos[0], margin=0.02):
            return False
        return True

    def _should_prepare(self, puck_pos, puck_vel):
        if not self._on_our_half(puck_pos[0]):
            return False
        if self._in_strike_box(puck_pos):
            return np.abs(puck_vel[1]) > self.strike_lateral_speed_threshold
        return True

    def _in_strike_box(self, puck_pos):
        x_lower, x_upper = self.strike_box_x_range
        x_min, x_max = sorted([x_lower, x_upper])
        if not (x_min <= puck_pos[0] <= x_max):
            return False
        if np.abs(puck_pos[1]) > self.strike_box_half_y:
            return False
        return True

    def _plan_strike(self, puck_pos, opponent):
        goal_y_world = self._select_goal_lane(opponent, puck_pos)
        goal_point = self._goal_point(goal_y_world)

        direction = goal_point[:2] - puck_pos[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([self.attack_sign, 0.0])
            norm = 1.0
        direction /= norm

        impact_point = puck_pos[:2] - direction * (self.puck_radius + 0.01)
        impact_point = self._clip_xy(impact_point, stay_on_half=True)

        approach_point = impact_point - direction * self.strike_approach_offset
        approach_point = self._clip_xy(approach_point, stay_on_half=True)

        follow_distance = 0.12
        return StrikePlan(
            approach_point=approach_point,
            impact_point=impact_point,
            shot_direction=direction,
            goal_point=goal_point,
            mallet_speed=self.strike_speed,
            follow_distance=follow_distance,
            follow_time=self.strike_follow_time,
        )

    # ------------------------------------------------------------------
    # Geometry & kinematics helpers
    # ------------------------------------------------------------------
    def _clip_xy(self, xy, stay_on_half=False):
        x_val = self._clip_x(xy[0], stay_on_half=stay_on_half)
        y_val = self._clip_y(xy[1])
        return np.array([x_val, y_val])

    def _clip_x(self, value, stay_on_half=False):
        x_val = float(np.clip(value, self.safe_x_min, self.safe_x_max))
        if stay_on_half:
            if self.home_side_sign > 0:
                x_val = min(x_val, self.center_line_x - self.mallet_radius)
            else:
                x_val = max(x_val, self.center_line_x + self.mallet_radius)
        return x_val

    def _clip_y(self, value):
        return float(np.clip(value, self.safe_y_min, self.safe_y_max))

    def _on_our_half(self, x_value, margin=0.0):
        if self.home_side_sign >= 0:
            return x_value <= self.center_line_x - margin
        return x_value >= self.center_line_x + margin

    def _goal_point(self, y_world):
        attack_dir = np.sign(self.attack_sign) if self.attack_sign != 0 else 1.0
        x_world = attack_dir * (self.env_info["table"]["length"] / 2)
        world = np.array([x_world, float(y_world), 0.0, 1.0])
        return (self.world_to_base @ world)[:3]

    def _select_goal_lane(self, opponent, puck_pos):
        goal_width = float(self.env_info["table"]["goal_width"])
        if opponent is not None and opponent.size >= 2:
            opp_y = float(opponent[1])
            if opp_y > 0.05:
                return -goal_width / 3
            if opp_y < -0.05:
                return goal_width / 3
            return -np.sign(puck_pos[1]) * goal_width / 3 if abs(puck_pos[1]) > 0.05 else goal_width / 4
        return -np.sign(puck_pos[1]) * goal_width / 2 if abs(puck_pos[1]) > 0.05 else goal_width / 2

    def _solve_ik(self, target_pos, initial_q):
        success, q_sol = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            target_pos,
            initial_q=initial_q,
        )
        if success:
            return q_sol[: self.n_joints].copy(), True
        if self.last_q_cmd is not None:
            return self.last_q_cmd.copy(), False
        return initial_q.copy(), False

    def _compute_joint_velocities(self, q_target, q_current, cart_vel):
        desired_delta = q_target - q_current
        step = np.clip(desired_delta, self.min_joint_step, self.max_joint_step)

        q_cmd = q_current + step
        q_cmd = np.clip(q_cmd, self.joint_pos_limit[0], self.joint_pos_limit[1])
        step = q_cmd - q_current
        dq_cmd = step / self.dt

        if cart_vel is not None and cart_vel.shape[0] == 3:
            try:
                jac = jacobian(self.robot_model, self.robot_data, q_current)[:3, : self.n_joints]
                dq_task, *_ = np.linalg.lstsq(jac, cart_vel, rcond=1e-4)
                dq_task = np.clip(dq_task, self.joint_vel_limit[0], self.joint_vel_limit[1])
                dq_cmd = 0.5 * dq_cmd + 0.5 * dq_task
            except np.linalg.LinAlgError:
                pass

        dq_cmd = np.clip(dq_cmd, self.joint_vel_limit[0], self.joint_vel_limit[1])

        delta_cmd = dq_cmd * self.dt
        sign_mismatch = np.sign(delta_cmd) != np.sign(desired_delta)
        delta_cmd = np.where(sign_mismatch, 0.0, delta_cmd)
        overshoot = np.abs(delta_cmd) > np.abs(desired_delta)
        delta_cmd = np.where(overshoot, np.sign(delta_cmd) * np.abs(desired_delta), delta_cmd)
        delta_cmd = np.clip(delta_cmd, self.min_joint_step, self.max_joint_step)

        q_cmd = q_current + delta_cmd
        q_cmd = np.clip(q_cmd, self.joint_pos_limit[0], self.joint_pos_limit[1])
        dq_cmd = (q_cmd - q_current) / self.dt

        return q_cmd, dq_cmd

    def _get_opponent(self, obs):
        ids = self.env_info.get("opponent_ee_ids", [])
        if ids:
            return obs[ids]
        return None
