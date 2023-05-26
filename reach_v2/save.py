import gym
from reach_v2.reachenv import ReachEnv
from stable_baselines3 import SAC
import os


def save():
    model_dir = "models/SAC_Reach_2M"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = gym.make("ReachEnv-v0", random_env=True, render=False)
    env.seed(10)
    env_test = gym.make("ReachEnv-v0", random_env=True, render=False)
    env_test.seed(11)

    model = SAC("MlpPolicy", env, verbose=1, create_eval_env=True, seed=0, tensorboard_log='./logs')

    model.learn(total_timesteps=2000000, log_interval=10, eval_env=env_test, eval_freq=1000,
                tb_log_name='SAC_REACH_2M')
    model.save(f"{model_dir}/{2000000}")

    del model



if __name__ == "__main__":

    save()
