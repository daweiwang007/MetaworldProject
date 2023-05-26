from gym.envs.registration import register
from reach_v2.reachenv import ReachEnv

register(
    id="ReachEnv-v0",
    entry_point="reach_v2.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_env": False,
            "simulator": 'mujoco',
            "render": True,
            "rep_id": 1}

)
register(
    id="ReachEnv-v1",
    entry_point="reach_v2.reachenv:ReachEnv",
    max_episode_steps=250,
    kwargs={"n_substeps": 10,
            "random_env": True,
            "simulator": 'mujoco',
            "render": False,
            "rep_id": 1}

)