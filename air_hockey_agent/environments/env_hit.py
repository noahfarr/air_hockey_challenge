import jax
from jax import numpy as jnp

from .env_base import EnvBase


class EnvHit(EnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            env_name="hit_single",
            custom_reward_fn=lambda *args: self.reward(*args),
            **kwargs,
        )

    def reset(self, rng: jax.Array) -> jax.Array:
        state = super().reset(rng)

        last_ee_pos: jax.Array = state.info["planned_world_pos"].copy()
        last_ee_pos = last_ee_pos.at[0].subtract(1.51)

        # state.info.update(
        #     last_ee_pos=last_ee_pos,
        # )
        info = {
            **state.info,
            "last_ee_pos": last_ee_pos,
        }
        state = state.replace(info=info)

        return state

    def reward(self, state: jax.Array) -> jax.Array:
        obs = state.info["internal_obs"]

        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, _ = self.get_ee(state.pipeline_state)
        ee_vel = (ee_pos - state.info["last_ee_pos"]) / self.dt

        # state.info["last_ee_pos"] = ee_pos
        info = {
            **state.info,
            "last_ee_pos": ee_pos,
        }
        state = state.replace(info=info)

        is_puck_not_hit = jnp.logical_and(puck_vel[0] < 0.25, puck_pos[0] < 0)

        def proximity_reward():
            ee_puck_dir = (puck_pos - ee_pos)[:2]
            ee_puck_dir /= jnp.linalg.norm(ee_puck_dir)
            return jnp.maximum(0, jnp.dot(ee_puck_dir, ee_vel[:2]))

        def hit_reward():
            return 10 * jnp.linalg.norm(puck_vel[:2])

        rew = jnp.where(is_puck_not_hit, proximity_reward(), hit_reward())

        rew = jnp.where(
            self.check_success(state.obs),
            rew + 2000 + 5000 * jnp.linalg.norm(puck_vel[:2]),
            rew,
        )

        return state.replace(reward=rew)


if __name__ == "__main__":
    env = EnvHit()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(10000)

    key, rng = jax.random.split(rng)

    state = jit_reset(key)

    for i in range(100):
        action = jnp.zeros(6)
        state = jit_step(state, action)
