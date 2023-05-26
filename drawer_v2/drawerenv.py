import numpy as np
from gym.spaces import Box as SamplingSpace
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene
# from metaworld.envs import reward_utils
from drawer_v2 import reward_utils


class DrawerEnv(GymEnvWrapper):
    def __init__(
            self,
            simulator: str,
            n_substeps: int = 10,
            max_steps_per_episode: int = 250,
            debug: bool = False,
            random_env: bool = False,
            render=False,
            rep_id=0
    ):
        sim_factory = SimRepository.get_factory(simulator)
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(render=render_mode, surrounding="kit_lab_surrounding_drawer", proc_id=rep_id)
        robot = sim_factory.create_robot(scene)
        controller = GymTorqueController(robot)
        robot.cartesianPosQuatTrackingController.neglect_dynamics = False
        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.random_env = random_env

        self.target_min_dist = 0.03

        self.observation_space = SamplingSpace(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64)

        self.action_space = self.controller.action_space()
        self.start()

        self.init_config = {
            'obj_init_pos': np.array([0.9, 0.0, 0.0], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']

        self.maxDist = 0.2

        self.tcp_center = self.robot.current_c_pos
        self.robot_init_c_pos = [5.50899712e-01, -1.03382391e-08, 6.99822168e-01]

    def get_observation(self) -> np.ndarray:
        drawer_handle_pos = self.scene.sim.data.get_site_xpos('handleStart')  # 3
        goal_pos = self._target_pos  # 3
        self.tcp_pos = self.robot.current_c_pos  # 3
        dist_tcp_goal, rel_goal_tcp_pos = obj_distance(goal_pos, self.tcp_pos)  # 1,3

        env_state = np.concatenate([drawer_handle_pos, goal_pos, [dist_tcp_goal], rel_goal_tcp_pos])  # 3+3+1+3=10
        robot_state = self.robot_state()  # 27
        return np.concatenate([robot_state, env_state])  # 27+10=37

    def get_reward(self):
        obs = self.get_observation()
        gripper = self.tcp_pos
        handle = obs[27:30]

        handle_error_dist = handle - self._target_pos
        handle_error = np.linalg.norm(handle_error_dist)

        reward_for_opening = reward_utils.tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=self.maxDist,
            sigmoid='long_tail'
        )

        handle_pos_init = self._target_pos + np.array([self.maxDist, .0, .0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        self.init_tcp = self.robot_init_c_pos
        handle_exact = handle + np.array([.02, .0, -.04])
        gripper_error_dist = (handle_exact - gripper) * scale
        gripper_error = np.linalg.norm(gripper_error_dist)
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            gripper_error,
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0
        return reward

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        goal_pos = self._target_pos
        drawer_handle_pos = self.scene.sim.data.get_site_xpos('handleStart')
        dist_tcp_goal, _ = obj_distance(goal_pos, drawer_handle_pos)

        if dist_tcp_goal <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True
        self.terminated = False
        return False


    # # condition2: don't early termination
    # def is_finished(self):
    #     """Checks if the robot either exceeded the maximum number of steps or is terminated according to another task
    #     dependent metric.
    #
    #     Returns:
    #         True if robot should terminate
    #     """
    #     if (
    #         self.env_step_counter >= self.max_steps_per_episode - 1
    #     ):
    #         return True
    #     return False


    def _reset_env(self):
        # Compute nightstand position
        self.obj_init_pos = self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        self.scene.sim.model.body_pos[self.scene.sim.model.body_name2id(
            'drawer'
        )] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array([-.16 - self.maxDist, .0, .08])

        self.scene.reset()
        observation = self.get_observation()

        return observation

    def step(self, action, gripper_width=None):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """

        if gripper_width is not None:
            self.robot.set_gripper_width = gripper_width

        self.controller.set_action(action)
        self.controller.execute_action(n_time_steps=self.n_substeps)

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_finished()

        #  condition1: check each step if success
        if self.terminated:
            success = 1
        else:
            success = 0

        info = {
            'is_success': success
        }



        self.env_step_counter += 1

        # #  condition2: check if last step success
        # if self.env_step_counter >= self.max_steps_per_episode - 1:
        #     goal_pos = self._target_pos
        #     drawer_handle_pos = self.scene.sim.data.get_site_xpos('handleStart')
        #
        #     if drawer_handle_pos[0] <= goal_pos[0]:
        #         success = 1
        #     else:
        #         success = 0
        # else:
        #     success = 0
        #
        # info = {
        #     'is_success': success
        # }


        return observation, reward, done, info
