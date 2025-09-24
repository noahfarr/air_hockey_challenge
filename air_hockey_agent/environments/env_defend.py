import jax
from jax import numpy as jnp
from brax.envs.base import State

from .env_base import EnvBase


class EnvDefend(EnvBase):
    def __init__(self, **kwargs):
        super().__init__(
            env_name="defend",
            custom_reward_fn=lambda *args: self.reward(*args),
            **kwargs,
        )

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        state.info.update(
            hit_step_flag=False,
            hit_step=False,
            give_reward_next=False,
            received_hit_reward=False,
            hit_this_step=False,
        )

        return state

    def reward(self, state: State) -> State:
        obs = state.info["internal_obs"]

        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, _ = self.get_ee(state.pipeline_state)
        rew = 0.01

        def f(puck_vel):
            return 30.0 + 100.0 * (100.0 ** (-0.25 * jnp.linalg.norm(puck_vel[:2])))

        # Reward for stopping puck in defend zone
        cond1 = (
            (-0.7 < puck_pos[0])
            & (puck_pos[0] <= -0.2)
            & (jnp.linalg.norm(puck_vel[:2]) < 0.1)
        )
        rew = jnp.where(cond1, rew + 70.0, rew)

        hit_step_flag, hit_step = jax.lax.cond(
            jnp.logical_and(
                state.info["has_hit_robot1"],
                jnp.logical_not(state.info["hit_step_flag"]),
            ),
            lambda: (True, True),
            lambda: (state.info["hit_step_flag"], False),
        )

        state.info["hit_step_flag"] = hit_step_flag
        state.info["hit_step"] = hit_step

        # Compute reward for hitting puck
        cond2 = jnp.logical_and.reduce(
            jnp.array(
                [
                    jnp.logical_not(state.info["give_reward_next"]),
                    jnp.logical_not(state.info["received_hit_reward"]),
                    state.info["hit_step"],
                    ee_pos[0] < puck_pos[0],
                ]
            )
        )
        cond3 = jnp.logical_and(
            jnp.logical_not(state.info["received_hit_reward"]),
            state.info["give_reward_next"],
        )

        def reward_hit():
            return jnp.where(
                jnp.linalg.norm(puck_vel[:2]) < 0.1, rew + f(puck_vel), rew
            )

        def reward_next():
            return jnp.where(puck_vel[0] >= -0.2, rew + f(puck_vel), rew)

        rew = jnp.where(cond2, reward_hit(), jnp.where(cond3, reward_next(), rew))

        hit_this_step = jax.lax.cond(
            cond2, lambda: True, lambda: state.info["hit_this_step"]
        )

        give_reward_next = jax.lax.cond(
            cond2 & (jnp.linalg.norm(puck_vel[:2]) >= 0.1),
            lambda: True,
            lambda: state.info["give_reward_next"],
        )

        received_hit_reward = jax.lax.cond(
            jnp.logical_and(jnp.logical_not(cond2), cond3),
            lambda: True,
            lambda: state.info["received_hit_reward"],
        )

        state.info.update(
            give_reward_next=give_reward_next,
            received_hit_reward=received_hit_reward,
            hit_this_step=hit_this_step,
        )

        return state.replace(reward=rew)

        # rew = jnp.where(cond2, reward_hit(), rew)
        # rew = jnp.where(cond3, reward_next(), rew)

        # return state.replace(reward=rew)

        # # table_length = self.env_info["table"]["length"]
        # # goal_width = self.env_info["table"]["goal_width"]
        # # is_goal = (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0) & (
        # #     puck_pos[0] <= - table_length / 2
        # # )

        # # return jnp.where(
        # #     is_goal,
        # #     -1000.0,  # Negative reward for conceding a goal
        # #     0.01,   # Neutral reward otherwise
        # # )


if __name__ == "__main__":
    from sbx import PPO
    import pickle
    from pathlib import Path
    import cv2

    path = Path(
        "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/defend/test_0"
    )
    with open(path / "best_model.zip", "rb") as f:
        model = PPO.load(f)
    #     with open(path / "ppo_defend_slow.zip", "rb") as f:
    #         model = PPO.load(f)
    with open(path / "vecnormalize.pkl", "rb") as f:
        normalizer = pickle.load(f)

    env = EnvDefend()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(10000)

    key, rng = jax.random.split(rng)

    state = jit_reset(key)
    norm_obs = normalizer.normalize_obs(state.obs)
    print(norm_obs)
    # exit()
    # agent.episode_start()

    done = False

    states = [state.pipeline_state]

    reward = 0.0

    episode_step = 0
    for i in range(1000):
        action, _ = model.predict(norm_obs, deterministic=True)
        print(action)
        exit()
        state = jit_step(state, action)
        # img = env.render(state.pipeline_state)
        # cv2.imshow("Air Hockey", img[:, :, ::-1])
        # cv2.waitKey(1)
        states.append(state.pipeline_state)
        norm_obs = normalizer.normalize_obs(state.obs)
        reward += state.reward

        episode_step += 1
        if state.done:
            print(f"Episode finished in step {episode_step}")
            print(f"Final reward: {reward}")

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
