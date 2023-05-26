import gym
from drawer_v2.drawerenv import DrawerEnv
from stable_baselines3 import SAC
import os

if __name__ == "__main__":

    model_dir = "models/SAC_Drawer_2M_Seed0"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    env = gym.make("DrawerEnv-v0", render=False)
    env.seed(10)
    env_test = gym.make("DrawerEnv-v0", render=False)
    env_test.seed(11)


    # TIMESTEPS = 300000
    model = SAC("MlpPolicy", env, verbose=1, create_eval_env=True, seed=0, tensorboard_log='./logs')
    # for i in range(1, 20):
    #     model.learn(total_timesteps=TIMESTEPS, log_interval=10, eval_env=env, eval_freq=1000, tb_log_name='SAC_DRAWER_10M_3')#reset_num_timesteps=False
    #     model.save(f"{model_dir}/{TIMESTEPS*i}")

    model.learn(total_timesteps=2000000, log_interval=10, n_eval_episodes=1, eval_env=env_test, eval_freq=1000,
                tb_log_name='SAC_DRAWER_2M_SEED0')
    model.save(f"{model_dir}/{2000000}")

    del model
