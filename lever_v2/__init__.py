from gym.envs.registration import register
from lever_v2.leverenv import LeverEnv

register(
    id="LeverEnv-v0",
    entry_point="lever_v2.leverenv:LeverEnv",
    # max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "max_steps_per_episode": 625,
            "simulator": 'mujoco',
            "render": True,
            "rep_id": 0
            }
)