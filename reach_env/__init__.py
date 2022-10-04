
from gym.envs.registration import register
from reach_env.reachenv import ReachEnv

register(
    id="ReachEnv-v0",
    entry_point="reach_env.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_env": False,
            "simulator": 'mujoco',
            "render": True}
)