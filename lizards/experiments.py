import os

from main_utils import *
# from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.pg as pg
from ray.tune.logger import pretty_print
from gym.spaces import Box, Discrete
import numpy as np
import time
import json
import pickle
from tensorflow import Tensor
import tensorflow as tf
import argparse

env_directory = {'adversarial-pursuit': adversarial_pursuit_v3, 'tiger-deer': tiger_deer_v3, 'battle': battle_v3,
                 'battlefield': battlefield_v3}

env_spaces = {'adversarial-pursuit':
                  {'action_space': Discrete(13), 'obs_space': Box(low=0.0, high=2.0, shape=(10, 10, 5), dtype=np.float32)},
              'battle':
                  {'action_space': Discrete(21), 'obs_space': Box(low=0.0, high=1.0, shape=(13, 13, 5), dtype=np.float32)}}


def experiment_1():
    # DEPRECATED
    battlefield = convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0))
    model = train(battlefield, PPO, time_steps=10000, save_name='trained_policies/battlefield-10e4-V1')
    evaluate_model(battlefield, model)


def view_results():
    # DEPRECATED
    evaluate_model(convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0)),
                   PPO.load('trained_policies/battlefield-10e6-V1'))


def ray_experiment_1(*args, gpu=True):
    policy_dict, policy_fn = get_policy_config(**env_spaces['battle'])

    env_kwargs = {"map_size": 12}

    trainer_config = {
        "env": "battle",
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn": policy_fn
        },
        "model": {
            "conv_filters": [
                [21, 13, 1]
            ]
        },
        "env_config": env_kwargs,
        "rollout_fragment_length": 100
    }

    trainer = ppo.PPOTrainer(config=trainer_config)

    if gpu:
        trainer_config["num_gpus"] = 1
        trainer_config["num_gpus_per_worker"] = 0.5
    else:  ## For CPUs:
        trainer_config["num_gpus"] = 0

    for i in range(10):
        print(trainer.train())


def ray_experiment_AP_training_shared(*args, gpu=True):
    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_1_name='predator', team_2_name='prey')

    env_kwargs = {"map_size": 30}

    trainer_config = {
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
        "env_config": env_kwargs,  # passed to the env creator as an EnvContext object
        "rollout_fragment_length": 100
    }

    if gpu:
        trainer_config["num_gpus"] = 1
        trainer_config["num_gpus_per_worker"] = 0.5
    else:  ## For CPUs:
        trainer_config["num_gpus"] = 0

    trainer = ppo.PPOTrainer(config=trainer_config)

    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Some_checkpoint_filename')

    checkpoint = train_ray_trainer(trainer, num_iters=1, log_intervals=1)

    if checkpoint:
        render_from_checkpoint(checkpoint, trainer, adversarial_pursuit_v3, env_kwargs, policy_fn)


def ray_experiment_AP_eval(*args, gpu=True):
    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_1_name='predator', team_2_name='prey')

    env_kwargs = {"map_size": 12}

    trainer_config = {
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
        "env_config": env_kwargs,  # passed to the env creator as an EnvContext object
        "rollout_fragment_length": 100
    }

    if gpu:
        trainer_config["num_gpus"] = 1
        trainer_config["num_gpus_per_worker"] = 0.5
    else:  ## For CPUs:
        trainer_config["num_gpus"] = 0

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/ttt')

    # checkpoint = train_ray_trainer(trainer, num_iters=1, log_intervals=1, log_dir='logs/ttt')
    checkpoint = log_dir + '/checkpoint_000001/checkpoint-1'

    if checkpoint:
        rewards = evaluate_policies(checkpoint, trainer, adversarial_pursuit_v3, env_kwargs, policy_fn, max_iter=100)
        print(rewards)


def ray_experiment_AP_training_share_split(*args, gpu=True):
    env_kwargs = {"map_size": 30}

    prey_count = get_num_agents(adversarial_pursuit_v3, env_kwargs)['prey']
    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_1_name='predator',
                                               team_2_name='prey', team_2_policy='split', team_2_count=prey_count)

    trainer_config = {
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
        "env_config": env_kwargs,  # passed to the env creator as an EnvContext object
        "rollout_fragment_length": 100
    }

    if gpu:
        trainer_config["num_gpus"] = 1
        trainer_config["num_gpus_per_worker"] = 0.5
    else:  ## For CPUs:
        trainer_config["num_gpus"] = 0

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/PPO_adversarial-pursuit_prey-split_100')

    checkpoint = train_ray_trainer(trainer, num_iters=100, log_intervals=20, log_dir=log_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)
    # ray_experiment_AP_training_split(args)
    ray_experiment_AP_training_share_split(gpu=args.gpu)
