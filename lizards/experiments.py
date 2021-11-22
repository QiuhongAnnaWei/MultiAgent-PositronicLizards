from main_utils import *
# from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.pg as pg
from ray.tune.logger import pretty_print
from gym.spaces import Box, Discrete
import numpy as np
import time
from tensorflow import Tensor
import tensorflow as tf


env_directory = {'adversarial-pursuit': adversarial_pursuit_v3, 'tiger-deer': tiger_deer_v3, 'battle': battle_v3,
                 'battlefield': battlefield_v3}


def experiment_1():
    battlefield = convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0))
    model = train(battlefield, PPO, time_steps=10000, save_name='trained_policies/battlefield-10e4-V1')
    evaluate_model(battlefield, model)


def view_results():
    evaluate_model(convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0)),
                   PPO.load('trained_policies/battlefield-10e6-V1'))


def ray_experiment_1():
    auto_register_env_ray("battle", battle_v3)

    obs = Box(low=0.0, high=1.0, shape=(13, 13, 5), dtype=np.float32)
    act = Discrete(21)

    policy_dict, policy_fn = get_policy_config(battle_v3, action_space=act, obs_space=obs)

    trainer = ppo.PPOTrainer(env='battle', config={
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn": policy_fn
        },
        "model": {
            "conv_filters": [
                [21, 13, 1]
            ]
        },
        "env_config": {
            "map_size": 12
        },
        "num_gpus": 0.8
    })

    for i in range(10):
        print(trainer.train())


def ray_experiment_AP():
    auto_register_env_ray("adversarial-pursuit", adversarial_pursuit_v3)

    obs = Box(low=0.0, high=2.0, shape=(10, 10, 5), dtype=np.float32)
    act = Discrete(13)

    policy_dict, policy_fn = get_policy_config(adversarial_pursuit_v3, action_space=act, obs_space=obs,
                                               method='predator_prey')

    env_kwargs = {"map_size": 30}

    trainer = ppo.PPOTrainer(config={
        "env": 'adversarial-pursuit',
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn": policy_fn
        },
        "model": {
            "conv_filters": [
                [13, 10, 1]
            ]
        },
        "env_config": env_kwargs, # passed to the env creator as an EnvContext object
        ## For CPUs:
        # "num_gpus": 0,
        # "num_cpus_per_worker": 1,
        ## For GPUS:
        "num_gpus": 0.9,
    })
    checkpoint = None
    start = time.time()
    for i in range(1, 11):
        print(f"Starting training on batch {i}")
        start = time.time()
        result = trainer.train()
        print(pretty_print(result))
        print(f"batch {i}: took {time.time()-start} seconds")
        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)

    if checkpoint:
        render_from_checkpoint(checkpoint, trainer, adversarial_pursuit_v3, env_kwargs, policy_fn)


if __name__ == "__main__":
    ray_experiment_AP()
