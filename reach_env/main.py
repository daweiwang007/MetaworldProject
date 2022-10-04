import gym
from reach_env.reachenv import ReachEnv
from stable_baselines3 import SAC

if __name__ == "__main__":

    env = gym.make("ReachEnv-v0", render=True)
    env.seed(10)
    model = SAC("MlpPolicy", env, verbose=1, create_eval_env=True, tensorboard_log = './logs')
    model.learn(total_timesteps=3000000, log_interval=10, eval_env=env, eval_freq=1000, tb_log_name='SAC_3M')
    model.save('sac_3M')

    del model

    model = SAC.load("sac_3M")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.scene.render()
        if done:
            obs = env.reset()