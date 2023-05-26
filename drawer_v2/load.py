import gym
from drawer_v2.drawerenv import DrawerEnv
from stable_baselines3 import SAC

if __name__ == "__main__":

    env = gym.make("DrawerEnv-v0", render=True)

    model_dir = "models/SAC_Drawer_test"
    model_path = f"{model_dir}/2000000"
    model = SAC.load(model_path, env=env)
    # model = SAC.load("sac_drawer_3M")

    obs = env.reset()
    # print(env.robot.current_c_pos)
    while True:
        #action = env.action_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.scene.render()
        # print(reward)
        if done:
            # break
            obs = env.reset()
