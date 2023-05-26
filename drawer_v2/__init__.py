from gym.envs.registration import register
from drawer_v2.drawerenv import DrawerEnv

register(
    id="DrawerEnv-v0",
    entry_point="drawer_v2.drawerenv:DrawerEnv",
    # max_episode_steps=625,
    kwargs={"n_substeps": 10,
            "max_steps_per_episode": 625,
            "simulator": 'mujoco',
            "render": True,
            "rep_id": 0
            }
)