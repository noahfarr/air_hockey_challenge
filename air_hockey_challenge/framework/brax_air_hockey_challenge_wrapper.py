from copy import deepcopy
import jax
import jax.numpy as jnp

from air_hockey_challenge.constraints.brax_constraints import *
from air_hockey_challenge.environments import brax_position_control_wrapper as position
from air_hockey_challenge.utils.mjx.transformations import robot_to_world

from brax.envs.base import Wrapper


class AirHockeyChallengeWrapper(Wrapper):
    def __init__(
        self, env, custom_reward_function=None, interpolation_order=3, **kwargs
    ):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [hit, defend, prepare, tournament].
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,
            "hit": position.IiwaPositionHit,
            "hit_single": position.IiwaPositionHitSingle,
            "defend": position.IiwaPositionDefend,
            "prepare": position.IiwaPositionPrepare,
        }

        if env == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        base_env = env_dict[env](interpolation_order=interpolation_order)
        self.env_name = env
        self.env_info = base_env.env_info

        if custom_reward_function:
            base_env._reward = lambda state: custom_reward_function(state)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        constraint_list.add(LinkConstraint(self.env_info))

        self.env_info["constraints"] = constraint_list
        self.env_info["env_name"] = self.env_name

        super().__init__(base_env)  # base_env is now self.env

    def reset(self, state):
        state = super().reset(state)

        if "tournament" in self.env_name:
            state.info.update(
                constraints_value=list(), success=jnp.zeros((), dtype=jnp.int32)
            )
        else:
            state.info.update(
                constraints_value=dict(
                    joint_vel_constr=jnp.zeros(14),
                    joint_pos_constr=jnp.zeros(14),
                    ee_constr=jnp.zeros(5),
                    link_constr=jnp.zeros(2),
                ),
                success=jnp.zeros((), dtype=jnp.int32),
            )

        return state

    def step(self, state, action):
        state = super().step(state, action)

        obs, info = state.obs, state.info

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2) : (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(
                    deepcopy(
                        self.env_info["constraints"].fun(
                            obs_agent[jnp.asarray(self.env_info["joint_pos_ids"])],
                            obs_agent[jnp.asarray(self.env_info["joint_vel_ids"])],
                        )
                    )
                )

        else:
            info["constraints_value"] = deepcopy(
                self.env_info["constraints"].fun(
                    obs[jnp.asarray(self.env_info["joint_pos_ids"])],
                    obs[jnp.asarray(self.env_info["joint_vel_ids"])],
                )
            )
            info["success"] = self.check_success(obs)

        return state.replace(info=info)

    def check_success(self, obs):
        puck_pos, puck_vel = self.env.get_puck(obs)

        puck_pos, _ = robot_to_world(
            self.env_info["robot"]["base_frame"][0], translation=puck_pos
        )
        success = 0

        def hit_case():
            cond_x = puck_pos[0] - self.env_info["table"]["length"] / 2 > 0
            cond_y = jnp.abs(puck_pos[1]) - self.env_info["table"]["goal_width"] / 2 < 0
            return jnp.logical_and(cond_x, cond_y)

        def defend_case():
            cond_x = jnp.logical_and(puck_pos[0] > -0.8, puck_pos[0] <= -0.2)
            cond_vel = puck_vel[0] < 0.1
            return jnp.logical_and(cond_x, cond_vel)

        def prepare_case():
            cond_x = jnp.logical_and(puck_pos[0] > -0.8, puck_pos[0] <= -0.2)
            cond_y = jnp.abs(puck_pos[1]) < 0.39105
            cond_vel = puck_vel[0] < 0.1
            return jnp.logical_and(jnp.logical_and(cond_x, cond_y), cond_vel)

        success = 0

        if "hit" in self.env_name:
            success = jnp.where(hit_case(), 1, 0)
        elif "defend" in self.env_name:
            success = jnp.where(defend_case(), 1, 0)
        elif "prepare" in self.env_name:
            success = jnp.where(prepare_case(), 1, 0)

        return success
