import numpy as np
from gym.spaces import Box as SamplingSpace
from alr_sim.gyms.gym_env_wrapper import GymEnvWrapper
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere
from alr_sim.gyms.gym_controllers import GymTorqueController
from alr_sim.sims.SimFactory import SimRepository
from alr_sim.core.Scene import Scene
import gym


class ReachEnv(GymEnvWrapper):
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
        scene = sim_factory.create_scene(render=render_mode, surrounding="kit_lab_surrounding", proc_id=rep_id)
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

        self.goal = Sphere(
            name="goal",
            size=[0.02],
            init_pos=[0.5, 0, 0.1],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
            visual_only=True
        )
        self.goal_space = SamplingSpace(
            low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5])
        )
        self.scene.add_object(self.goal)

        self.target_min_dist = 0.03

        self.observation_space = SamplingSpace(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float64)
        self.action_space = self.controller.action_space()
        self.start()


    def get_observation(self) -> np.ndarray:
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, rel_goal_tcp_pos = obj_distance(goal_pos, tcp_pos)

        env_state = np.concatenate([goal_pos, [dist_tcp_goal], rel_goal_tcp_pos])
        robot_state = self.robot_state()
        return np.concatenate([robot_state, env_state])

    def get_reward(self):
        tcp = self.robot.current_c_pos
        target = self.scene.get_obj_pos(self.goal)

        reward = -np.linalg.norm(tcp - target)

        return reward

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, _ = obj_distance(goal_pos, tcp_pos)

        if dist_tcp_goal <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True
        return False

    def _reset_env(self):
        if self.random_env:
            new_goal = [self.goal, self.goal_space.sample()]
            joint_noise = np.random.randn(7) * 0.1
            self.scene.reset([new_goal], joint_noise)
            observation = self.get_observation()
        else:
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


        if self.terminated:
            success = 1
        else:
            success = 0

        info = {
            'is_success': success
        }



        self.env_step_counter += 1


        return observation, reward, done, info


