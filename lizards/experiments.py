import os

from main_utils import *
# from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3, combined_arms_v5
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
import uuid
from itertools import product

env_directory = {'adversarial-pursuit': adversarial_pursuit_v3, 'tiger-deer': tiger_deer_v3, 'battle': battle_v3,
                 'battlefield': battlefield_v3, "combined-arms": combined_arms_v5}

env_spaces = {'adversarial-pursuit':
                  {'action_space': Discrete(13),
                   'obs_space': Box(low=0.0, high=2.0, shape=(10, 10, 5), dtype=np.float32)},
              'battle':
                  {'action_space': Discrete(21),
                   'obs_space': Box(low=0.0, high=1.0, shape=(13, 13, 5), dtype=np.float32)},
              'battlefield':
                  {'action_space': Discrete(21),
                   'obs_space': Box(low=0.0, high=1.0, shape=(13, 13, 5), dtype=np.float32)},
              'tiger-deer':
                  {'action_space': Discrete(9),
                   'obs_space': Box(low=0.0, high=1.0, shape=(9, 9, 5), dtype=np.float32)},
              'combined-arms':
                  {'action_space': Discrete(25),
                   'obs_space': Box(low=0.0, high=1.0, shape=(13, 13, 9), dtype=np.float32)}
              }


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

    def for_filename(self):
        if self.method == 'split':
            return f"_{self.team_name}-split"
        return ""

    def __str__(self):
        return f"TeamPolicyConfig: {self.team_name}, {self.method}, {self.count}"


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


def ray_train_generic(*args, end_render=True, **kwargs):
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'], gpu=kwargs['gpu'])
    trainer = ppo.PPOTrainer(config=trainer_config)

    policy_log_str = "".join([p.for_filename() for p in kwargs['team_data']])
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f"logs/PPO_{kwargs['env_name']}{policy_log_str}_{kwargs['train_iters']}-iters__{uuid.uuid4().hex[:5]}")
    print(f"(from ray_train_generic) `log_dir` has been set to {log_dir}")

    free_resources_after_train = kwargs.get("free_resources_after_train", False)
    checkpoint = train_ray_trainer(trainer, num_iters=kwargs['train_iters'], log_intervals=kwargs['log_intervals'], log_dir=log_dir, free_resources_after_train=free_resources_after_train)

    if kwargs['end_render']:
        render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'], kwargs['policy_fn'], max_iter=10000)
    return checkpoint, trainer


def ray_viz_generic(checkpoint, **kwargs):
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'],
                                        gpu=kwargs['gpu'])
    trainer = ppo.PPOTrainer(config=trainer_config)

    render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'],
                           kwargs['policy_fn'], max_iter=10000)


def ray_BF_training_share_split_retooled():
    env_config = {'map_size': 55, 'dead_penalty': -6}
    red_count = get_num_agents(battlefield_v3, env_config)['red']
    team_data = [TeamPolicyConfig('red', method='split', count=red_count), TeamPolicyConfig('blue')]
    kwargs = {
        'env_name': 'battlefield',
        'team_data': team_data,
        'env_config': env_config,
        'train_iters': 100,
        'log_intervals': 10,
    }

    ray_train_generic(**kwargs, end_render=True)


def ray_TD_training_share_split_retooled():
    env_config = {'map_size': 30, 'max_cycles': 10000}
    tiger_count = get_num_agents(tiger_deer_v3, env_config)['tiger']
    team_data = [TeamPolicyConfig('tiger', method='split', count=tiger_count), TeamPolicyConfig('deer')]
    policy_dict, policy_fn = get_policy_config(**env_spaces['tiger-deer'], team_data=team_data)
    kwargs = {
        'env_name': 'tiger-deer',
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 20,
        'log_intervals': 10,
        'gpu': True
    }

    # ray_train_generic(**kwargs)
    ray_viz_generic(
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_tiger-deer_tiger-split_100-iters__f1282/checkpoint_000100/checkpoint-100',
        **kwargs)


def ray_CA_red_split_blue_shared_TEST(map_sz=16, train_iters=8, log_intervals=2):
    """
    Testing using a specific split-share combination before generalizing
    """
    env_nm = "combined-arms"
    ca_fn = env_directory[env_nm]
    env_config = {'map_size': map_sz} # min map sz is 16

    teams = ("redmelee", "redranged", "bluemele", "blueranged")
    counts = {t: get_num_agents(ca_fn, env_config)[t] for t in teams}

    team_data = [TeamPolicyConfig(teams[0], method='split', count=counts[teams[0]]),
                 TeamPolicyConfig(teams[1], method='split', count=counts[teams[1]]),
                 TeamPolicyConfig(teams[2], method='shared', count=counts[teams[2]]), 
                 TeamPolicyConfig(teams[3], method='shared', count=counts[teams[3]])] 

    kwargs = {
        'env_name': env_nm,
        'team_data': team_data,
        'env_config': env_config,
        'train_iters': train_iters,
        'log_intervals': log_intervals,
        "free_resources_after_train": True,
        'gpu': False,
        'end_render': True,
    }

    ray_train_generic(**kwargs)


def ray_CA_generalized(map_sz=16):
    """ Generalizing to all split-share combinations, but with fixed map_siz = 16 """

    env_nm = "combined-arms"
    ca_fn = env_directory[env_nm]
    env_config = {'map_size': 16} # min map sz is 16


    teams = ("redmelee", "redranged", "bluemele", "blueranged")
    counts = {t: get_num_agents(ca_fn, env_config)[t] for t in teams}

    def td_given_mcomb(method_comb):
        return [TeamPolicyConfig(t_i, method=m_i, count=(counts[t_i] if m_i=="split" else None)) for t_i, m_i in zip(teams, method_comb)]

    train_data_combs = [td_given_mcomb(m_comb) for m_comb in product(("split", "shared"), repeat=4)]
    assert len((train_data_combs)) == 16

    # To see that this gives us the training combinations we want, print the following
    for i, comb in enumerate(train_data_combs):
        print(f"\nTrain data idx {i}") 
        for tpc in comb: 
            print(tpc.__str__()) 

    parametrized_kwargs = lambda t_data: {'env_name': env_nm,
                                          'team_data': t_data,
                                          'env_config': env_config,
                                          'train_iters': 100,
                                          'log_intervals': 20,
                                          'gpu': True, 
                                          'end_render': True,
                                          "free_resources_after_train": True, 
                                          # might need this if training back to back, but not sure. feel free to toggle it
                                          }

    train_comb_kwargs = [parametrized_kwargs(td) for td in train_data_combs] 

    for i, kws in enumerate(train_comb_kwargs):
        print(f"\nStarting on training combination idx {i}")
        ray_train_generic(**kws)




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
    # parser.add_argument('experiment', help="train, peek")
    parser.add_argument('env', choices=['BA', 'AP', 'BF', 'TD'],
                        help=f"choice of environment for training\n{str(env_abreviation_dict)}")
    parser.add_argument('--no-gpu', dest='gpu', default=True, action='store_false',
                        help="disables gpu usage")
    parser.add_argument('--render_end', dest='end_render', default=False, action='store_true',
                        help="renders the last checkpoint after training")
    parser.add_argument('-i', '--train-iters', dest='train_iters', default=100,
                        help="number of training iterations")
    parser.add_argument('-li', '--log-intervals', dest='log_intervals', default=20,
                        help="logging interval")

    args = parser.parse_args()
    args.env_name = env_abreviation_dict[args.env]
    args.team_names = env_team_names[args.env]

    return args


def main():
    # kwargs = parse_args()
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    # print(kwargs)
    # pettingzoo_peek(tiger_deer_v3, {'map_size': 30})
    # ray_TD_training_share_split_retooled()
    ray_CA_generalized()


if __name__ == "__main__":
    main()
