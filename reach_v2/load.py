import gym
from reach_v2.reachenv import ReachEnv
from stable_baselines3 import SAC


if __name__ == "__main__":

    env = gym.make("ReachEnv-v0", random_env=True, render=True)

    model_dir = "models/SAC_Reach_1M_Seed0"
    model_path = f"{model_dir}/1000000"
    model = SAC.load(model_path, env=env)

    obs = env.reset()
    while True:
        #action = env.action_space.sample()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.scene.render()
        if done:
            obs = env.reset()