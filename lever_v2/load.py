import gym
from lever_v2.leverenv import LeverEnv
from stable_baselines3 import SAC
from stable_baselines3 import TD3

if __name__ == "__main__":

    env = gym.make("LeverEnv-v0", render=True)

    # model_dir = "models/SAC_2M_Seed2"
    model_dir = "models/TD3_2M_Seed2"
    model_path = f"{model_dir}/2000000"
    model = TD3.load(model_path, env=env)
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
