import jax
from brax.envs.base import State
from jax import numpy as jnp

from .env_base import EnvBase


class EnvPrepare(EnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            env_name="prepare",
            custom_reward_fn=lambda *args: self.reward(*args),
            **kwargs,
        )

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)

        puck_pos, puck_vel = self.get_puck(state.info["internal_obs"])

        state = jax.lax.cond(
            self.puck_in_success_region(puck_pos, puck_vel),
            lambda s: s.replace(done=jnp.ones(())),
            lambda s: s,
            state,
        )

        return state

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        last_ee_pos: jax.Array = state.info["planned_world_pos"].copy()
        last_ee_pos = last_ee_pos.at[0].subtract(1.51)

        state.info.update(
            last_ee_pos=last_ee_pos,
        )

        return state

    def puck_in_success_region(self, puck_pos, puck_vel) -> jax.Array:
        x = puck_pos[0]
        y = puck_pos[1]

        return jnp.logical_and.reduce(
            jnp.array(
                [
                    x > -0.65,
                    x < -0.35,
                    jnp.abs(y) < 0.4,
                    jnp.linalg.norm(puck_vel[0:2]) < 0.5,
                ]
            )
        )

    def reward(self, state: State) -> State:
        obs = state.info["internal_obs"]

        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, _ = self.get_ee(state.pipeline_state)
        ee_vel = (ee_pos - state.info["last_ee_pos"]) / self.dt

        state.info["last_ee_pos"] = ee_pos

        is_puck_not_hit = jnp.logical_and(puck_vel[0] < 0.25, puck_pos[0] < 0)

        def proximity_reward():
            ee_puck_dir = (puck_pos - ee_pos)[:2]
            ee_puck_dir /= jnp.linalg.norm(ee_puck_dir)
            return jnp.maximum(0, jnp.dot(ee_puck_dir, ee_vel[:2]))

        def puck_target_reward():
            target = jnp.array([-0.5, 0])
            puck_target = target - puck_pos[:2]
            puck_target = (puck_target) / jnp.linalg.norm(puck_target)
            puck_target_vel = jnp.minimum(0.5, jnp.dot(puck_target[:2], puck_vel[:2]))
            return 10 * jnp.maximum(0, puck_target_vel)

        rew = jnp.where(is_puck_not_hit, proximity_reward(), 0) + puck_target_reward()

        rew += jnp.where(self.puck_in_success_region(puck_pos, puck_vel), 2000, 0)

        return state.replace(reward=rew)


if __name__ == "__main__":
    from sbx import PPO

    # from stable_baselines3 import PPO
    import pickle
    from pathlib import Path
    import cv2

    path = Path(
        "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/prepare/prepare-8"
    )
    with open(path / "best_model.zip", "rb") as f:
        model = PPO.load(f)
    with open(path / "vecnormalize.pkl", "rb") as f:
        normalizer = pickle.load(f)

    # path = Path(
    #     "/home/donat/projects/air-hockit/airhockey/air_hockey/air_hockey_agent/agents/combined_agent/models"
    # # )
    # with open(path / "ppo_prepare_low.zip", "rb") as f:
    #     model = PPO.load(f, device="cpu")
    # with open(path / "ppo_prepare_low.pkl", "rb") as f:
    #     normalizer = pickle.load(f)

    env = EnvPrepare()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)

    key, rng = jax.random.split(rng)

    state = jit_reset(key)
    norm_obs = normalizer.normalize_obs(state.obs)
    # print(norm_obs)
    # exit()
    # agent.episode_start()

    done = False

    states = [state.pipeline_state]

    reward = 0.0

    episode_step = 0
    for i in range(1000):
        action, _ = model.predict(norm_obs, deterministic=True)
        state = jit_step(state, action)
        # img = env.render(state.pipeline_state)
        # cv2.imshow("Air Hockey", img[:, :, ::-1])
        # cv2.waitKey(1)
        states.append(state.pipeline_state)
        norm_obs = normalizer.normalize_obs(state.obs)
        reward += state.reward

        if state.reward >= 2000:
            print(f"Goal scored: {state.reward}")

        episode_step += 1
        if state.done:
            print(f"Episode finished in step {episode_step}")
            print(f"Final reward: {state.reward}")
            break

            key, rng = jax.random.split(rng)
            state = jit_reset(key)
            norm_obs = normalizer.normalize_obs(state.obs)
            states.append(state.pipeline_state)
            reward = 0.0
            episode_step = 0

    # jit_reset = jax.jit(env.reset)
    # jit_step = jax.jit(env.step)
    # jit_reset = env.reset
    # jit_step = env.step

    # rng = random.PRNGKey(0)

    # state = jit_reset(rng)
    # print(state.obs)
    # action = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # state = jit_step(state, action)
    # print(state.obs)

    # states = [state.pipeline_state]

    # for i in range(150):
    #     # action = random.uniform(rng, shape=(6,), minval=-1, maxval=1)
    #     action = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #     state = jit_step(state, action)
    #     states.append(state.pipeline_state)

    #     print(state.reward)

    #     if state.done:
    #         # print(f"Episode finished in step {i}")
    #         # print(f"Final reward: {state.reward}")
    #         break

    imgs = env.render(states)
    input("Press Enter to watch...")
    for img in imgs:
        cv2.imshow("Air Hockey", img[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
