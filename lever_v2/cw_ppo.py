import gym
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from lever_v2.leverenv import LeverEnv


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        params = config['params']
        env_id = params['env_id']

        # env = gym.make(env_id, random_env=True, render=False, rep_id=rep)
        env = make_vec_env(env_id, n_envs=8, seed=rep, env_kwargs={'rep_id': rep})
        env = VecNormalize(venv=env, norm_obs=True, norm_reward=True, training=True)


        # env_test = gym.make(env_id, random_env=True, render=False, rep_id=rep)
        eval_env = make_vec_env(env_id, n_envs=1, seed=rep+100, env_kwargs={'rep_id': rep})
        eval_env = VecNormalize(venv=eval_env, norm_obs=True, norm_reward=False, training=False)


        tensorboard_log = config['_rep_log_path'] + '/tensorboard/'

        model = PPO("MlpPolicy", env, verbose=1, create_eval_env=True, seed=rep+20, tensorboard_log=tensorboard_log)
        tot_timesteps = params['total_timesteps']
        log_interval = params['log_interval']
        eval_freq = params['eval_freq']
        tb_log_name = params['tb_log_name']

        model.learn(total_timesteps=tot_timesteps, log_interval=log_interval, eval_env=eval_env, eval_freq=eval_freq,
                    tb_log_name=tb_log_name)

        model_log_path = config['_rep_log_path'] + '/model/PPO_lever'
        model.save(model_log_path)
        del model

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    # cw.add_logger(...)

    # RUN!
    cw.run()
