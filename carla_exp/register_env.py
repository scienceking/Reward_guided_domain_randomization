from ray.tune.registry import register_env
from ACC_env import CustomEnv  #  custom_env.py


def register_custom_env():
    def env_creator(config):
        return CustomEnv(config)
    register_env("CustomEnv", env_creator)
