import gym
from lever_v2.leverenv import LeverEnv
from stable_baselines3 import SAC
import os

if __name__ == "__main__":

    model_dir = "models/SAC_2M_Seed9"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = gym.make("LeverEnv-v0", render=False)
    env.seed(10)
    env_test = gym.make("LeverEnv-v0", render=False)
    env_test.seed(11)


    model = SAC("MlpPolicy", env, verbose=1, create_eval_env=True, seed=9, tensorboard_log='./logs')

    model.learn(total_timesteps=2000000, log_interval=10, n_eval_episodes=1, eval_env=env_test, eval_freq=1000,
                tb_log_name='SAC_LEVER_2M_SEED0')
    model.save(f"{model_dir}/{2000000}")

    del model
