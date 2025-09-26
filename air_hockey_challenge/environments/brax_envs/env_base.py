from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
import os
from brax.math import quat_to_3x3
import numpy as np

from scipy.spatial.transform import Rotation
from jax.scipy.spatial.transform import Rotation as R

from air_hockey_challenge.environments.data.mjx import __file__ as env_path
from mujoco import mjx

from air_hockey_challenge.utils.mjx.kinematics import (
    forward_kinematics,
)

from air_hockey_challenge.utils.kinematics import inverse_kinematics


class AirHockeyBase(PipelineEnv):
    def __init__(self, n_agents, timestep=1 / 1000, n_intermediate_steps=20):
        self.n_agents = n_agents
        self._timestep = timestep
        self._n_intermediate_steps = n_intermediate_steps

        xml_file = os.path.join(
            os.path.dirname(os.path.abspath(env_path)),
            "single.xml" if n_agents == 1 else "double.xml",
        )

        mj_model = mujoco.MjModel.from_xml_path(xml_file)
        mj_model.opt.timestep = timestep
        sys = mjcf.load_model(mj_model)

        super().__init__(sys, backend="mjx")

        self.env_info = dict()
        self.env_info["table"] = {"length": 1.948, "width": 1.038, "goal_width": 0.25}
        self.env_info["puck"] = {"radius": 0.03165}
        self.env_info["mallet"] = {"radius": 0.04815}
        self.env_info["n_agents"] = self.n_agents
        self.env_info["robot"] = {
            "n_joints": 7,
            "ee_desired_height": 0.1645,
            "joint_vel_limit": jnp.array(
                [
                    [-85, -85, -100, -75, -130, -135, -135],
                    [85, 85, 100, 75, 130, 135, 135],
                ]
            )
            / 180.0
            * jnp.pi,
            "joint_acc_limit": jnp.array(
                [
                    [-85, -85, -100, -75, -130, -135, -135],
                    [85, 85, 100, 75, 130, 135, 135],
                ]
            )
            / 180.0
            * jnp.pi
            * 10,
            "base_frame": [],
            "universal_height": 0.0645,
            "control_frequency": 50,
        }

        self.env_info["dt"] = self.dt
        self.env_info["robot"]["joint_pos_limit"] = np.array(
            [mj_model.joint(f"iiwa_1/joint_{i + 1}").range for i in range(7)]
        ).T
        self.env_info["puck_pos_ids"] = [0, 1, 2]
        self.env_info["puck_vel_ids"] = [3, 4, 5]
        self.env_info["joint_pos_ids"] = [6, 7, 8, 9, 10, 11, 12]
        self.env_info["joint_vel_ids"] = [13, 14, 15, 16, 17, 18, 19]
        if self.n_agents == 2:
            self.env_info["opponent_ee_ids"] = [20, 21, 22]
        else:
            self.env_info["opponent_ee_ids"] = []

        self.robot_model = mujoco.MjModel.from_xml_path(
            os.path.join(os.path.dirname(os.path.abspath(env_path)), "iiwa_only.xml")
        )
        self.robot_model.body("iiwa_1/base").pos = np.zeros(3)
        self.robot_data = mujoco.MjData(self.robot_model)

        # self.env_info["robot"]["robot_model"] = self.robot_model
        # self.env_info["robot"]["robot_data"] = self.robot_data

        robot_mjx_model = mjx.put_model(self.robot_model)
        robot_mjx_data = mjx.put_data(self.robot_model, self.robot_data)

        self.env_info["robot"]["robot_mjx_model"] = robot_mjx_model
        self.env_info["robot"]["robot_mjx_data"] = robot_mjx_data

        frame_T = jnp.eye(4)
        temp = quat_to_3x3(mj_model.body("iiwa_1/base").quat)
        frame_T = frame_T.at[:3, :3].set(temp)
        frame_T = frame_T.at[:3, 3].set(mj_model.body("iiwa_1/base").pos)
        self.env_info["robot"]["base_frame"].append(frame_T.copy())

        if self.n_agents == 2:
            temp = quat_to_3x3(mj_model.body("iiwa_2/base").quat)
            frame_T = frame_T.at[:3, :3].set(temp)
            frame_T = frame_T.at[:3, 3].set(mj_model.body("iiwa_2/base").pos)
            self.env_info["robot"]["base_frame"].append(frame_T.copy())

        self.puck_ids = jnp.array(
            [
                self.sys.mj_model.joint("puck_x").id,
                self.sys.mj_model.joint("puck_y").id,
                self.sys.mj_model.joint("puck_yaw").id,
            ]
        )

        self.joint_ids = jnp.array(
            [self.sys.mj_model.joint(f"iiwa_1/joint_{i + 1}").id for i in range(7)]
        )

        self.ee_pos_id = self.sys.mj_model.body(f"iiwa_1/striker_joint_link").id

        self.universal_joint_ids = jnp.array(
            [
                self.sys.mj_model.joint("iiwa_1/striker_joint_1").id,
                self.sys.mj_model.joint("iiwa_1/striker_joint_2").id,
            ]
        )

        self.universal_joint_ctrl_ids = jnp.array(
            [
                self.sys.mj_model.actuator("iiwa_1/striker_joint_1").id,
                self.sys.mj_model.actuator("iiwa_1/striker_joint_2").id,
            ]
        )

        self.action_ids = jnp.array(
            [self.sys.mj_model.actuator(f"iiwa_1/joint_{i + 1}").id for i in range(7)]
        )

        if self.n_agents == 2:
            self.opponent_joint_ids = jnp.array(
                [self.sys.mj_model.joint(f"iiwa_2/joint_{i + 1}").id for i in range(7)]
            )

            self.opponent_ee_pos_id = self.sys.mj_model.body(
                f"iiwa_2/striker_joint_link"
            ).id

            self.universal_joint_ids = jnp.concatenate(
                [
                    self.universal_joint_ids,
                    jnp.array(
                        [
                            self.sys.mj_model.joint("iiwa_2/striker_joint_1").id,
                            self.sys.mj_model.joint("iiwa_2/striker_joint_2").id,
                        ]
                    ),
                ]
            )

            self.universal_joint_ctrl_ids = jnp.concatenate(
                [
                    self.universal_joint_ctrl_ids,
                    jnp.array(
                        [
                            self.sys.mj_model.actuator("iiwa_2/striker_joint_1").id,
                            self.sys.mj_model.actuator("iiwa_2/striker_joint_2").id,
                        ]
                    ),
                ]
            )

            self.action_ids = jnp.concatenate(
                [
                    self.action_ids,
                    jnp.array(
                        [
                            self.sys.mj_model.actuator(f"iiwa_2/joint_{i + 1}").id
                            for i in range(7)
                        ]
                    ),
                ]
            )

        collision_spec = [
            ("puck", ["puck"]),
            (
                "rim",
                [
                    "rim_home_l",
                    "rim_home_r",
                    "rim_away_l",
                    "rim_away_r",
                    "rim_left",
                    "rim_right",
                ],
            ),
            (
                "rim_short_sides",
                ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"],
            ),
            ("robot_1/ee", ["iiwa_1/ee"]),
        ]

        if self.n_agents == 2:
            collision_spec.append(("robot_2/ee", ["iiwa_2/ee"]))

        self.collision_groups = {}
        for name, geom_names in collision_spec:
            col_group = list()
            for geom_name in geom_names:
                # mj_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                mj_id = self.sys.mj_model.geom(geom_name).id
                assert (
                    mj_id != -1
                ), f'geom "{geom_name}" not found! Can\'t be used for collision-checking.'
                col_group.append(mj_id)
            self.collision_groups[name] = jnp.array(col_group)

        self._compute_init_state()

        self.filter_ratio = 0.273

        self.actuator_joint_ids = jnp.array(
            [
                self.sys.mj_model.actuator(f"iiwa_1/joint_{i + 1}").trnid[0]
                for i in range(7)
            ]
        )

        if self.n_agents == 2:
            opponent_ids = jnp.array(
                [
                    self.sys.mj_model.actuator(f"iiwa_2/joint_{i + 1}").trnid[0]
                    for i in range(7)
                ]
            )
            self.actuator_joint_ids = jnp.concatenate(
                [self.actuator_joint_ids, opponent_ids]
            )

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps

    def reset(self, rng: jax.Array) -> State:
        qpos, qvel = self._setup(rng)

        # Determine start side based on puck x-position
        puck_x = qpos[self.puck_ids[0]]
        start_side = jnp.where(puck_x < 0, jnp.array(1), jnp.array(-1))

        data = self.pipeline_init(qpos, qvel)

        info = dict(
            u_joint_pos_prev=None,
            u_joint_vel_prev=jnp.zeros(2 * self.n_agents),
            u_joint_pos_des=jnp.zeros(2 * self.n_agents),
            tournament_start_side=start_side,
            tournament_prev_side=start_side,
            tournament_episode_faults=jnp.array([0, 0]),
            tournament_timer=jnp.array([0]),
            q_pos_prev=jnp.zeros(self.n_agents * self.env_info["robot"]["n_joints"]),
            q_vel_prev=jnp.zeros(self.n_agents * self.env_info["robot"]["n_joints"]),
        )

        obs = self._get_obs(data)
        reward, done = jnp.zeros(2)
        metrics = dict()

        state = State(data, obs, reward, done, metrics, info)
        state, _ = self._control_universal_joint(state)

        state.info.update(u_joint_vel_prev=jnp.zeros(2 * self.n_agents))

        qpos = state.pipeline_state.qpos.at[self.universal_joint_ids].set(
            state.info["u_joint_pos_des"]
        )
        new_data = self.pipeline_init(qpos, qvel)

        cur_obs = self._get_obs(new_data)
        cur_obs = self._create_observation(state, cur_obs)
        mod_obs = self._modify_observation(cur_obs)

        state.info.update(internal_obs=cur_obs, has_hit_robot1=False)

        return State(new_data, mod_obs, reward, done, metrics, state.info)

    def step(self, state: State, action: jax.Array) -> State:
        state, action = self._preprocess_action(state, action)
        state = self._step_init(state, action)

        prev_obs = state.info["internal_obs"]

        def f(carry, _):
            state, action = carry
            state, torque = self._compute_action(state, action)
            state = self._single_step(state, torque)

            return (state, action), None

        (state, action), _ = jax.lax.scan(
            f, (state, action), None, length=self._n_intermediate_steps
        )

        state = self._step_finalize(state)

        # cur_obs = self._get_obs(state.pipeline_state)
        # cur_obs = self._create_observation(state, cur_obs)

        cur_obs = state.info["internal_obs"]
        mod_obs = self._modify_observation(cur_obs)

        state = state.replace(obs=mod_obs)

        state, done = self._is_absorbing(state)
        state = self._reward(state)

        # state.info.update(internal_obs=cur_obs)

        return state.replace(done=done)

    def _single_step(self, state: State, action: jax.Array) -> State:
        """Performs a single step in the environment."""
        state = self._simulation_pre_step(state)

        state, u_torque = self._control_universal_joint(state)

        act = jnp.zeros(self.n_agents * 9)
        act = act.at[self.action_ids].set(action)
        act = act.at[self.universal_joint_ctrl_ids].set(u_torque)

        pipeline_state = self.pipeline_step(state.pipeline_state, act)
        obs = self._get_obs(pipeline_state)
        obs = self._create_observation(state, obs)
        # mod_obs = self._modify_observation(obs)

        state.info.update(internal_obs=obs)
        state = state.replace(pipeline_state=pipeline_state)

        return self._simulation_post_step(state)

    def _simulation_pre_step(self, state: State) -> State:
        return state

    def _simulation_post_step(self, state: State) -> State:
        has_hit_robot1 = jnp.logical_or(
            state.info["has_hit_robot1"],
            self._check_collision(state.pipeline_state, "puck", "robot_1/ee"),
        )

        state.info.update(has_hit_robot1=has_hit_robot1)
        return state

    def _step_finalize(self, state: State) -> State:
        return state

    def _get_obs(self, data: mjx.Data):
        puck_pos = data.qpos[self.puck_ids]
        puck_vel = data.qvel[self.puck_ids]
        joint_pos = data.qpos[self.joint_ids]
        joint_vel = data.qvel[self.joint_ids]

        obs = jnp.concatenate(
            [
                puck_pos,
                puck_vel,
                joint_pos,
                joint_vel,
            ]
        )

        if self.n_agents == 2:
            opponent_ee_pos = data.xpos[self.opponent_ee_pos_id]
            opponent_joint_pos = data.qpos[self.opponent_joint_ids]
            opponent_joint_vel = data.qvel[self.opponent_joint_ids]
            ee_pos = data.xpos[self.ee_pos_id]

            obs = jnp.concatenate(
                [
                    obs,
                    opponent_ee_pos,
                    puck_pos,
                    puck_vel,
                    opponent_joint_pos,
                    opponent_joint_vel,
                    ee_pos,
                ]
            )

        return obs

    def _create_observation(self, state: State, obs: jax.Array) -> jax.Array:
        return obs

    def _modify_observation(self, obs: jax.Array) -> jax.Array:
        return obs

    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        boundary = (
            jnp.array(
                [self.env_info["table"]["length"], self.env_info["table"]["width"]]
            )
            / 2
        )
        puck_pos, puck_vel = self.get_puck(state.info["internal_obs"])

        done = jnp.logical_or(
            jnp.any(jnp.abs(puck_pos[:2]) > boundary),
            jnp.linalg.norm(puck_vel) > 100,
        )
        return state, done * 1.0

    def _reward(self, state: State) -> State:
        # Placeholder for reward calculation, return 0
        return state.replace(reward=jnp.zeros(()))

    def get_puck(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        puck_pos = obs[jnp.array(self.env_info["puck_pos_ids"])]
        puck_vel = obs[jnp.array(self.env_info["puck_vel_ids"])]

        return puck_pos, puck_vel

    @staticmethod
    def _puck_pose_2d_in_robot_frame(
        puck_in: jax.Array, robot_frame: jax.Array
    ) -> jax.Array:
        """
        Convert puck position from world frame to robot frame.
        puck_in: Puck position in world frame (x, y, z).
        robot_frame: Robot base frame transformation matrix (4x4).
        Returns puck position in robot frame.
        """
        puck_w = jnp.eye(4)
        puck_w = puck_w.at[:2, 3].set(puck_in[:2])
        puck_w = puck_w.at[:3, :3].set(
            R.from_euler("xyz", [0, 0, puck_in[2]]).as_matrix()
        )

        puck_r = jnp.linalg.inv(robot_frame) @ puck_w
        puck_out = jnp.concatenate(
            [
                puck_r[:2, 3],
                R.from_matrix(puck_r[:3, :3]).as_euler("xyz")[2:3],
            ]
        )

        return puck_out

    @staticmethod
    def _puck_vel_2d_in_robot_frame(
        puck_in: jax.Array, robot_frame: jax.Array
    ) -> jax.Array:

        rot_mat = robot_frame[:3, :3]

        vel_lin = jnp.array([*puck_in[:2], 0.0])
        vel_ang = jnp.array([0.0, 0.0, puck_in[2]])

        vel_lin_r = rot_mat.T @ vel_lin
        vel_ang_r = rot_mat.T @ vel_ang

        puck_out = jnp.concatenate([vel_lin_r[:2], vel_ang_r[2:3]])

        return puck_out

    def _preprocess_action(
        self, state: State, action: jax.Array
    ) -> tuple[State, jax.Array]:
        return state, action

    def _compute_action(
        self, state: State, action: jax.Array
    ) -> tuple[State, jax.Array]:
        return state, action

    def _step_init(self, state: State, action: jax.Array) -> State:
        return state

    def _compute_init_state(self):
        init_state = np.array([0.0, -0.1961, 0.0, -1.8436, 0.0, 0.9704, 0.0])

        success, self.init_state = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            np.array([0.65, 0.0, 0.1645]),
            Rotation.from_euler("xyz", [0, 5 / 6 * np.pi, 0]).as_matrix(),
            initial_q=init_state,
        )

        assert jnp.all(success), "Inverse kinematics failed for initial state!"

    def _setup(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        qpos = self.sys.qpos0

        for i in range(7):
            iiwa1_jnt_id = self.sys.mj_model.joint(f"iiwa_1/joint_{i + 1}").id
            qpos = qpos.at[iiwa1_jnt_id].set(self.init_state[i])

        if self.n_agents == 2:
            for i in range(7):
                iiwa2_jnt_id = self.sys.mj_model.joint(f"iiwa_2/joint_{i + 1}").id
                qpos = qpos.at[iiwa2_jnt_id].set(self.init_state[i])

        qvel = jnp.zeros(self.sys.nv)

        return qpos, qvel

    def _control_universal_joint(self, state: State) -> tuple[State, jax.Array]:
        u_joint_pos_prev = state.info["u_joint_pos_prev"]
        u_joint_pos_des = jnp.zeros(2 * self.n_agents)

        all_joint_ids = (
            self.joint_ids
            if self.n_agents == 1
            else jnp.concatenate([self.joint_ids, self.opponent_joint_ids])
        )

        for i in range(self.n_agents):
            q = state.pipeline_state.qpos[all_joint_ids[i * 7 : (i + 1) * 7]]

            pos, rot_mat = forward_kinematics(
                self.env_info["robot"]["robot_mjx_model"],
                self.env_info["robot"]["robot_mjx_data"],
                q,
            )

            v_x = rot_mat[:, 0]
            v_y = rot_mat[:, 1]

            # The desired position of the x-axis is the cross product of the desired z (0, 0, 1).T
            # and the current y-axis. (0, 0, 1).T x v_y
            x_desired = jnp.array([-v_y[1], v_y[0], 0])

            # Find the signed angle from the current to the desired x-axis around the y-axis
            # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
            q1 = jnp.arctan2(self._cross_3d(v_x, x_desired) @ v_y, v_x @ x_desired)

            if u_joint_pos_prev is not None:
                # if q1 - u_joint_pos_prev[0] > jnp.pi:
                #     q1 -= jnp.pi * 2
                # elif q1 - u_joint_pos_prev[0] < -jnp.pi:
                #     q1 += jnp.pi * 2
                q1 = jnp.where(
                    jnp.abs(q1 - u_joint_pos_prev[0]) > jnp.pi,
                    q1 - jnp.sign(q1 - u_joint_pos_prev[0]) * jnp.pi * 2,
                    q1,
                )

            w = jnp.array(
                [[0, -v_y[2], v_y[1]], [v_y[2], 0, -v_y[0]], [-v_y[1], v_y[0], 0]]
            )

            r = jnp.eye(3) + w * jnp.sin(q1) + w**2 * (1 - jnp.cos(q1))
            v_x_rotated = r @ v_x

            # The desired position of the y-axis is the negative cross product of the desired z (0, 0, 1).T and the current
            # x-axis, which is already rotated around y. The negative is there because the x-axis is flipped.
            # -((0, 0, 1).T x v_x))
            y_desired = jnp.array([v_x_rotated[1], -v_x_rotated[0], 0])

            # Find the signed angle from the current to the desired y-axis around the new rotated x-axis
            q2 = jnp.arctan2(
                self._cross_3d(v_y, y_desired) @ v_x_rotated, v_y @ y_desired
            )

            if u_joint_pos_prev is not None:
                # if q2 - u_joint_pos_prev[1] > jnp.pi:
                #     q2 -= jnp.pi * 2
                # elif q2 - u_joint_pos_prev[1] < -jnp.pi:
                #     q2 += jnp.pi * 2
                q2 = jnp.where(
                    jnp.abs(q2 - u_joint_pos_prev[1]) > jnp.pi,
                    q2 - jnp.sign(q2 - u_joint_pos_prev[1]) * jnp.pi * 2,
                    q2,
                )

            alpha_y = jnp.minimum(
                jnp.maximum(q1, -jnp.pi / 2 * 0.95), jnp.pi / 2 * 0.95
            )
            alpha_x = jnp.minimum(
                jnp.maximum(q2, -jnp.pi / 2 * 0.95), jnp.pi / 2 * 0.95
            )

            u_joint_pos_des = u_joint_pos_des.at[i * 2 : i * 2 + 2].set(
                jnp.array([alpha_y, alpha_x])
            )

        u_joint_pos_prev = state.pipeline_state.qpos[self.universal_joint_ids]
        u_joint_vel_prev = (
            self.filter_ratio * state.pipeline_state.qvel[self.universal_joint_ids]
            + (1 - self.filter_ratio) * state.info["u_joint_vel_prev"]
        )

        Kp = 4
        Kd = 0.31
        torque = Kp * (u_joint_pos_des - u_joint_pos_prev) - Kd * u_joint_vel_prev

        # ctrl = state.pipeline_state.ctrl
        # ctrl = ctrl.at[self.universal_joint_ctrl_ids].set(torque)
        # pipeline_state = state.pipeline_state.replace(ctrl=ctrl)

        state.info.update(
            u_joint_pos_prev=u_joint_pos_prev,
            u_joint_vel_prev=u_joint_vel_prev,
            u_joint_pos_des=u_joint_pos_des,
        )

        return state, torque

    def get_ee(self):
        raise NotImplementedError

    def get_joints(self, obs):
        raise NotImplementedError

    def _cross_3d(self, a, b):
        return jnp.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )

    def _check_collision(self, data: mjx.Data, group1, group2):
        """
        Check if any of the geoms in group1 are colliding with any of the geoms in group2.
        Returns True if there is a collision, False otherwise.
        JIT-compatible.
        """
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        geom1 = data.contact.geom1[: data.ncon]
        geom2 = data.contact.geom2[: data.ncon]
        dist = data.contact.dist[: data.ncon]

        # Expand dims for broadcasting
        ids1 = ids1.reshape(-1, 1)
        ids2 = ids2.reshape(-1, 1)

        # Check for collisions: (geom1 in ids1 and geom2 in ids2) or (geom1 in ids2 and geom2 in ids1)
        mask1 = (geom1[None, :] == ids1).any(0) & (geom2[None, :] == ids2).any(0)
        mask2 = (geom1[None, :] == ids2).any(0) & (geom2[None, :] == ids1).any(0)
        mask = (mask1 | mask2) & (dist <= 0)

        return jnp.any(mask)
