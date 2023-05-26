import numpy as np
from gym.spaces import Box as SamplingSpace
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene
from lever_v2 import reward_utils


class LeverEnv(GymEnvWrapper):
    duration = 2
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
        scene = sim_factory.create_scene(render=render_mode, surrounding="kit_lab_surrounding_lever", proc_id=rep_id)
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

        # self.target_min_dist = 0.03
        self.target_min_dist = 0.15

        self.observation_space = SamplingSpace(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64)

        self.action_space = self.controller.action_space()
        self.start()

        self.init_config = {
            'obj_init_pos': np.array([0.88, -0.12, 0.25], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self._lever_pos_init = None


        self.LEVER_RADIUS = 0.2

        self.tcp_center = self.robot.current_c_pos
        self.robot_init_c_pos = [5.50899712e-01, -1.03382391e-08, 6.99822168e-01]

    def get_observation(self) -> np.ndarray:
        lever_pos = self.scene.sim.data.get_site_xpos('leverStart')  # 3  lever end live pos
        goal_pos = self._target_pos  # 3  lever end target pos
        self.tcp_pos = self.robot.current_c_pos  # 3
        dist_tcp_goal, rel_goal_tcp_pos = obj_distance(goal_pos, self.tcp_pos)  # 1,3

        env_state = np.concatenate([lever_pos, goal_pos, [dist_tcp_goal], rel_goal_tcp_pos])  # 3+3+1+3=10
        robot_state = self.robot_state()  # 27
        return np.concatenate([robot_state, env_state])  # 27+10=37

    def get_reward(self):
        obs = self.get_observation()
        gripper = self.tcp_pos
        lever = obs[27:30]

        # De-emphasize y error so that we get Sawyer's shoulder underneath the
        # lever prior to bumping on against
        scale = np.array([1., 4., 4.])
        # Offset so that we get the Sawyer's shoulder underneath the lever,
        # rather than its fingers
        offset = np.array([.0, -.12, .15])

        shoulder_to_lever = (gripper + offset - lever) * scale
        shoulder_to_lever_init = (
                                         self.robot_init_c_pos + offset - self._lever_pos_init
                                 ) * scale

        # This `ready_to_lift` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually lifting the lever
        ready_to_lift = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lever),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lever_init),
            sigmoid='long_tail',
        )

        # The skill of the agent should be measured by its ability to get the
        # lever to point straight upward. This means we'll be measuring the
        # current angle of the lever's joint, and comparing with 90deg.
        lever_angle = -self.scene.sim.data.get_joint_qpos('LeverAxis')
        lever_angle_desired = np.pi / 2.0

        lever_error = abs(lever_angle - lever_angle_desired)

        # We'll set the margin to 15deg from horizontal. Angles below that will
        # receive some reward to incentivize exploration, but we don't want to
        # reward accidents too much. Past 15deg is probably intentional movement
        lever_engagement = reward_utils.tolerance(
            lever_error,
            bounds=(0, np.pi / 48.0),
            margin=(np.pi / 2.0) - (np.pi / 12.0),
            sigmoid='long_tail'
        )

        target = self._target_pos
        obj_to_target = np.linalg.norm(lever - target)
        in_place_margin = (np.linalg.norm(self._lever_pos_init - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                          bounds=(0, 0.04),
                                          margin=in_place_margin,
                                          sigmoid='long_tail', )

        # reward = 2.0 * ready_to_lift + 8.0 * lever_engagement
        reward = 10.0 * reward_utils.hamacher_product(ready_to_lift, in_place)

        return reward
        # return [
        #     reward,
        #     np.linalg.norm(shoulder_to_lever),
        #     ready_to_lift,
        #     lever_error,
        #     lever_engagement
        # ]


    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        goal_pos = self._target_pos
        lever_pos = self.scene.sim.data.get_site_xpos('leverStart')
        dist_tcp_goal, _ = obj_distance(goal_pos, lever_pos)

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
        # lever link init pos
        self.obj_init_pos = self.init_config['obj_init_pos']
        # Set lever link to init position
        self.scene.sim.model.body_pos[self.scene.sim.model.body_name2id(
            'lever_link'
        )] = self.obj_init_pos

        # lever end init pos
        self._lever_pos_init = self.obj_init_pos + np.array(
            [-self.LEVER_RADIUS, 0, 0]
        )

        # lever end target pos
        self._target_pos = self.obj_init_pos + np.array(
            [0, .0, self.LEVER_RADIUS]
        )

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
