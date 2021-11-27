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


class TeamPolicyConfig:
    def __init__(self, team_name, method='shared', count=None):
        """
        For specifying policy breakdowns for teams
        :param team_name: 'red', 'preditor', etc.
        :param method: 'shared': one policy shared for all agents starting with `team_name`, or 'split': one per agent
        :param count: (not required if method='shared') number of agents on team
        """
        self.team_name = team_name
        self.method = method
        self.count = count


def experiment_1():
    # DEPRECATED
    battlefield = convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0))
    model = train(battlefield, PPO, time_steps=10000, save_name='trained_policies/battlefield-10e4-V1')
    evaluate_model(battlefield, model)


def view_results():
    # DEPRECATED
    evaluate_model(convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0)),
                   PPO.load('trained_policies/battlefield-10e6-V1'))


def ray_experiment_AP_training_shared(*args, gpu=True):
    team_data = [TeamPolicyConfig('predator'), TeamPolicyConfig('prey')]
    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_data=team_data)

    env_config = {"map_size": 30}

    trainer_config = get_trainer_config('adversarial-pursuit', policy_dict, policy_fn, env_config, gpu=gpu)

    trainer = ppo.PPOTrainer(config=trainer_config)

    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Some_checkpoint_filename')

    checkpoint = train_ray_trainer(trainer, num_iters=1, log_intervals=1)

    if checkpoint:
        render_from_checkpoint(checkpoint, trainer, adversarial_pursuit_v3, env_config, policy_fn)


def ray_experiment_AP_eval(*args, gpu=True):
    team_data = [TeamPolicyConfig('predator'), TeamPolicyConfig('prey')]
    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_data=team_data)

    env_config = {"map_size": 12}

    trainer_config = get_trainer_config('adversarial-pursuit', policy_dict, policy_fn, env_config, gpu=gpu)

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/ttt')

    # checkpoint = train_ray_trainer(trainer, num_iters=1, log_intervals=1, log_dir='logs/ttt')
    checkpoint = log_dir + '/checkpoint_000001/checkpoint-1'

    if checkpoint:
        rewards = evaluate_policies(checkpoint, trainer, adversarial_pursuit_v3, env_config, policy_fn, max_iter=100)
        print(rewards)


def ray_experiment_BA_visualize(*args, gpu=True):
    env_config = {"map_size": 19}
    red_count = get_num_agents(battle_v3, env_config)['red']
    team_data = [TeamPolicyConfig('red', method='split', count=red_count), TeamPolicyConfig('blue')]

    policy_dict, policy_fn = get_policy_config(**env_spaces['battle'], team_data=team_data)

    trainer_config = get_trainer_config('battle', policy_dict, policy_fn, env_config, gpu=gpu)

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/PPO_battle_red-split_100')

    # checkpoint = train_ray_trainer(trainer, num_iters=1, log_intervals=1, log_dir='logs/ttt')
    checkpoint = log_dir + '/checkpoint_000081/checkpoint-81'

    if checkpoint:
        # rewards = evaluate_policies(checkpoint, trainer, battle_v3, env_kwargs, policy_fn, max_iter=1000)
        # print(rewards)
        render_from_checkpoint(checkpoint, trainer, battle_v3, env_config, policy_fn)


def ray_experiment_AP_training_share_split(*args, gpu=True):
    env_config = {"map_size": 30}
    predator_count = get_num_agents(adversarial_pursuit_v3, env_config)['predator']
    team_data = [TeamPolicyConfig('predator', method='split', count=predator_count), TeamPolicyConfig('prey')]

    policy_dict, policy_fn = get_policy_config(**env_spaces['adversarial-pursuit'], team_data=team_data)

    trainer_config = get_trainer_config('adversarial-pursuit', policy_dict, policy_fn, env_config, gpu=gpu)

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/PPO_adversarial-pursuit_predator-split_100')

    checkpoint = train_ray_trainer(trainer, num_iters=100, log_intervals=20, log_dir=log_dir)


def ray_experiment_BA_training_share_split(*args, gpu=True):
    env_config = {"map_size": 19}
    red_count = get_num_agents(battle_v3, env_config)['red']
    team_data = [TeamPolicyConfig('red', method='split', count=red_count), TeamPolicyConfig('blue')]

    policy_dict, policy_fn = get_policy_config(**env_spaces['battle'], team_data=team_data)

    trainer_config = get_trainer_config('battle', policy_dict, policy_fn, env_config, gpu=gpu)

    trainer = ppo.PPOTrainer(config=trainer_config)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/PPO_battle_red-split_100')

    checkpoint = train_ray_trainer(trainer, num_iters=100, log_intervals=20, log_dir=log_dir)


def parse_args():
    env_abreviation_dict = {'BA': 'battle',
                            'AP': 'adversarial-pursuit',
                            'BF': 'battlefield',
                            'TD': 'tiger-deer'}
    env_team_names = {'BA': ['red', 'blue'],
                      'AP': ['predator', 'prey'],
                      'BF': ['red', 'blue'],
                      'TD': ['tiger', 'deer']}

    parser = argparse.ArgumentParser()
    # parser.add_argument('experiment', help="peek")
    parser.add_argument('env', choices=['BA', 'AP', 'BF', 'TD'],
                        help=f"choice of environment for training\n{str(env_abreviation_dict)}")
    parser.add_argument('--no-gpu', dest='gpu', default=True, action='store_false',
                        help="disables gpu usage")

    args = parser.parse_args()
    args.env_name = env_abreviation_dict[args.env]
    args.team_names = env_team_names[args.env]

    return args


def main():
    args = parse_args()
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)
    # ray_experiment_AP_training_split(args)
    # ray_experiment_AP_training_share_split(gpu=args.gpu)
    # ray_experiment_BA_training_share_split(args)
    ray_experiment_BA_visualize(args)
    # x = get_num_agents(battle_v3, {'map_size': 19})
    # print(x)


if __name__ == "__main__":
    main()
    # args = parse_args()
    # print(args)
