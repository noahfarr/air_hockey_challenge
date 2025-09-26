from brax.envs.base import Wrapper, State
from jax import numpy as jnp
import jax
from air_hockey_challenge.utils.mjx.kinematics import forward_kinematics

from air_hockey_challenge.framework.brax_air_hockey_challenge_wrapper import (
    AirHockeyChallengeWrapper,
)


class EnvBase(Wrapper):
    def __init__(self, env_name, custom_reward_fn, **kwargs):
        env = AirHockeyChallengeWrapper(
            env=env_name, custom_reward_function=custom_reward_fn, **kwargs
        )
        super().__init__(env)

        self.env_info = env.env_info

        self.act_low = jnp.repeat(-30.0, 6)
        self.act_high = jnp.repeat(30.0, 6)

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        last_acceleration = jnp.zeros(6)
        interp_pos = state.obs[jnp.asarray(self.env_info["joint_pos_ids"][:-1])]
        interp_vel = state.obs[jnp.asarray(self.env_info["joint_vel_ids"][:-1])]
        planned_world_pos = self._fk(interp_pos)

        # state.info.update(
        #     last_acceleration=last_acceleration,
        #     interp_pos=interp_pos,
        #     interp_vel=interp_vel,
        #     planned_world_pos=planned_world_pos,
        #     step=-1,
        # )
        info = {
            **state.info,
            "last_acceleration": last_acceleration,
            "interp_pos": interp_pos,
            "interp_vel": interp_vel,
            "planned_world_pos": planned_world_pos,
            "step": -1,
        }
        state = state.replace(info=info)

        state = self.modify_obs(state)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        action /= 10

        new_vel = state.info["interp_vel"] + action
        jerk = (
            2
            * (
                new_vel
                - state.info["interp_vel"]
                - state.info["last_acceleration"] * 0.02
            )
            / (0.02**2)
        )
        new_pos = (
            state.info["interp_pos"]
            + state.info["interp_vel"] * 0.02
            + (1 / 2) * state.info["last_acceleration"] * (0.02**2)
            + (1 / 6) * jerk * (0.02**3)
        )
        abs_action = jnp.vstack([jnp.hstack([new_pos, 0]), jnp.hstack([new_vel, 0])])

        # state.info.update(
        #     last_acceleration=state.info["last_acceleration"] + jerk * 0.02,
        #     interp_pos=new_pos,
        #     interp_vel=new_vel,
        #     planned_world_pos=self._fk(new_pos),
        # )
        info = {
            **state.info,
            "last_acceleration": state.info["last_acceleration"] + jerk * 0.02,
            "interp_pos": new_pos,
            "interp_vel": new_vel,
            "planned_world_pos": self._fk(new_pos),
        }
        state = state.replace(info=info)

        state = super().step(state, abs_action)

        fatal_rew = self.check_fatal(state.obs)

        state = self.modify_obs(state)

        state = jax.lax.cond(
            fatal_rew != 0,
            lambda s: s.replace(reward=fatal_rew, done=jnp.ones(())),
            lambda s: s,
            state,
        )

        return state

    def filter_obs(self, obs):
        obs = jnp.hstack([obs[0:2], obs[3:5], obs[6:12], obs[13:19], obs[20:23]])
        return obs

    def check_fatal(self, obs):
        fatal_rew = 0

        q = obs[jnp.asarray(self.env_info["joint_pos_ids"])]
        qd = obs[jnp.asarray(self.env_info["joint_vel_ids"])]
        constraint_values_obs = self.env_info["constraints"].fun(q, qd)
        fatal_rew += self.check_constraints(constraint_values_obs)

        return fatal_rew

    def check_constraints(self, constraint_values):
        fatal_rew = 0

        j_pos_constr = constraint_values["joint_pos_constr"]
        # if j_pos_constr.max() > 0:
        #     fatal_rew += j_pos_constr.max()

        j_vel_constr = constraint_values["joint_vel_constr"]
        # if j_vel_constr.max() > 0:
        #     fatal_rew += j_vel_constr.max()

        ee_constr = constraint_values["ee_constr"]
        # if ee_constr.max() > 0:
        #     fatal_rew += ee_constr.max()

        link_constr = constraint_values["link_constr"]
        # if link_constr.max() > 0:
        #     fatal_rew += link_constr.max()

        fatal_rew = jnp.where(
            j_pos_constr.max() > 0, fatal_rew + j_pos_constr.max(), fatal_rew
        )
        fatal_rew = jnp.where(
            j_vel_constr.max() > 0, fatal_rew + j_vel_constr.max(), fatal_rew
        )
        fatal_rew = jnp.where(
            ee_constr.max() > 0, fatal_rew + ee_constr.max(), fatal_rew
        )
        fatal_rew = jnp.where(
            link_constr.max() > 0, fatal_rew + link_constr.max(), fatal_rew
        )

        return -fatal_rew

    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        state, base_absorbing = super()._is_absorbing(state)

        puck_in_goal = self.is_puck_in_goal(state.obs)

        done = jnp.logical_or.reduce(jnp.array([base_absorbing, puck_in_goal]))
        return state, done * 1.0

    def _fk(self, pos):
        return forward_kinematics(
            self.env_info["robot"]["robot_mjx_model"],
            self.env_info["robot"]["robot_mjx_data"],
            pos,
        )[0]

    # def is_puck_in_goal(self, obs):
    #     puck_pos, _ = self.get_puck(obs)
    #     table_length = self.env_info["table"]["length"]
    #     goal_width = self.env_info["table"]["goal_width"]
    #     return (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0) & (
    #         jnp.abs(puck_pos[0]) > table_length / 2
    #     )

    def modify_obs(self, state: State) -> State:
        obs = self.filter_obs(state.obs)

        obs = jnp.hstack(
            [
                obs,
                state.info["interp_pos"],
                state.info["interp_vel"],
                state.info["last_acceleration"],
                state.info["planned_world_pos"],
            ]
        )

        return state.replace(obs=obs)

    @property
    def action_size(self) -> int:
        return 6

    @property
    def observation_size(self) -> int:
        obs = self.reset(jax.random.PRNGKey(0)).obs
        return obs.shape[-1]
