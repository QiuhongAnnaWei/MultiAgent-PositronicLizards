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
                   'obs_space': Box(low=0.0, high=2.0, shape=(13, 13, 5), dtype=np.float32)},
              'battlefield':
                  {'action_space': Discrete(21),
                   'obs_space': Box(low=0.0, high=2.0, shape=(13, 13, 5), dtype=np.float32)},
              'tiger-deer':
                  {'action_space': Discrete(9),
                   'obs_space': Box(low=0.0, high=2.0, shape=(9, 9, 5), dtype=np.float32)},
              'combined-arms':
                  {'action_space': Discrete(25),
                   'obs_space': Box(low=0.0, high=2.0, shape=(13, 13, 9), dtype=np.float32)}
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


def ray_train_generic(*args, end_render=True, savefile=False, **kwargs):
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'], gpu=kwargs['gpu'])
    trainer = ppo.PPOTrainer(config=trainer_config)

    policy_log_str = "".join([p.for_filename() for p in kwargs['team_data']])
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f"logs/PPO_{kwargs['env_name']}{policy_log_str}_{kwargs['train_iters']}-iters__{uuid.uuid4().hex[:5]}")
    print(f"\n(from ray_train_generic) `log_dir` has been set to {log_dir}")

    checkpoint = train_ray_trainer(trainer, num_iters=kwargs['train_iters'], log_intervals=kwargs['log_intervals'], log_dir=log_dir)

    if end_render:
        render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'], kwargs['policy_fn'], max_iter=10000, savefile=savefile)
    return checkpoint, trainer


def ray_viz_generic(checkpoint, max_iter=10000, savefile=False, **kwargs):
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'],
                                        gpu=kwargs['gpu'])
    trainer = ppo.PPOTrainer(config=trainer_config)

    # rewards = evaluate_policies(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'], kwargs['policy_fn'])
    # print(rewards)

    render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'],
                           kwargs['policy_fn'], max_iter=max_iter, savefile=savefile)


def ray_BF_training_share_split_retooled():
    env_config = {'map_size': 55}
    red_count = get_num_agents(battlefield_v3, env_config)['red']
    team_data = [TeamPolicyConfig('red', method='split', count=red_count), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces['battlefield'], team_data=team_data)
    kwargs = {
        'env_name': 'battlefield',
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 200,
        'log_intervals': 50,
        'gpu': True
    }

    # ray_train_generic(**kwargs, end_render=True)
    ray_viz_generic(
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_battlefield_red-split_100'
                   '-iters__3a4d3/checkpoint_000100/checkpoint-100',
        **kwargs)


def ray_BA_training_share_pretrained(checkpoint='/logs/pretrained/PPO_battle_100-iters__cad08/checkpoint_000100/checkpoint-100',
                                    end_render=True, pre_trained_policy="red_shared"):
    env_name = 'battle'
    env_config = {'map_size': 19}
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 2,
        'log_intervals': 1,
        'gpu': False
    }
    ## Transfer partial weights
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'], gpu=kwargs['gpu'])
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(checkpoint)
    weights = temp_trainer.get_policy(pre_trained_policy).get_weights() # get the learnt weights
    temp_trainer.stop()
    trainer = ppo.PPOTrainer(config=trainer_config)
    trainer.get_policy(pre_trained_policy).set_weights(weights) # transfer the weigths (blue has untrained weigths)
    
    ## Train
    policy_log_str = "".join([p.for_filename() for p in kwargs['team_data']])
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f"logs/PPO_{kwargs['env_name']}{policy_log_str}_pretrained-{pre_trained_policy}_{kwargs['train_iters']}-iters__{uuid.uuid4().hex[:5]}")
    print(f"(from ray_BA_training_share_pretrained) `log_dir` has been set to {log_dir}")
    checkpoint = train_ray_trainer(trainer, num_iters=kwargs['train_iters'], log_intervals=kwargs['log_intervals'], log_dir=log_dir)

    ## Render/Evaluate
    if end_render:
        render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'], kwargs['policy_fn'], max_iter=10000, savefile=True)
    rewards = evaluate_policies(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=1000)
    print("\n ### POLICY EVALUATION: REWARDS ###")
    for key in rewards:
        print(f"{key}: {rewards[key]}")


def ray_BA_training_share_split_retooled():
    env_name = 'battle'
    env_config = {'map_size': 19}
    counts = get_num_agents(env_directory[env_name], env_config)
    team_data = [TeamPolicyConfig('red', method='split', count=counts['red']),
                 TeamPolicyConfig('blue')]
    # team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 200,
        'log_intervals': 40,
        'gpu': True
    }

    # ray_train_generic(**kwargs, end_render=True)
    ray_viz_generic(
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_battle_red-split_100'
                   '/checkpoint_000081/checkpoint-81',
        **kwargs)


def ray_TD_training_share_split_retooled():
    env_config = {'map_size': 30}
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
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_tiger-deer_tiger-split_100'
                   '-iters__f1282/checkpoint_000100/checkpoint-100',
        **kwargs)


def ray_AP_training_share_split_retooled():
    env_name = 'adversarial-pursuit'
    env_config = {'map_size': 40}
    predator_count = get_num_agents(env_directory[env_name], env_config)['predator']
    team_data = [TeamPolicyConfig('predator', method='split', count=predator_count), TeamPolicyConfig('prey')]
    # team_data = [TeamPolicyConfig('predator'), TeamPolicyConfig('prey')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 120,
        'log_intervals': 20,
        'gpu': True
    }

    # ray_train_generic(**kwargs, end_render=True)
    ray_viz_generic(
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_adversarial-pursuit_predator-split_120-iters__6cda8/checkpoint_000120/checkpoint-120',
        **kwargs)


def ray_CA_red_split_blue_shared_TEST(map_size=16, train_iters=8, log_intervals=2):
    """
    Testing using a specific split-share combination before generalizing
    """
    env_name = "combined-arms"
    ca_fn = env_directory[env_name]
    env_config = {'map_size': map_size} # min map sz is 16

    teams = ("redmelee", "redranged", "bluemele", "blueranged")
    counts = {t: get_num_agents(ca_fn, env_config)[t] for t in teams}

    team_data = [TeamPolicyConfig(teams[0], method='split', count=counts[teams[0]]),
                 TeamPolicyConfig(teams[1], method='split', count=counts[teams[1]]),
                 TeamPolicyConfig(teams[2], method='shared', count=counts[teams[2]]), 
                 TeamPolicyConfig(teams[3], method='shared', count=counts[teams[3]])] 
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)

    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': train_iters,
        'log_intervals': log_intervals,
        'gpu': False,
    }

    ray_train_generic(**kwargs, end_render=True)


def ray_CA_generalized(map_size=16):
    """ Generalizing to more than one split-share combination, but with fixed map_size"""

    env_name = "combined-arms"
    ca_fn = env_directory[env_name]
    env_config = {'map_size': map_size} # min map sz is 16

    teams = ("redmelee", "redranged", "bluemele", "blueranged")
    counts = {team: get_num_agents(ca_fn, env_config)[team] for team in teams}

    # First make the method combinations
    method_combos_red = product(("split", "shared"), repeat=2)  # cartesian product coz, conceptually, redmelee diff from redranged
    method_combos_all = map(lambda red_combo: red_combo + ("shared", "shared"), method_combos_red)

    # Then map over the method combinations to get the team data objects
    def train_data_given_m_comb(method_comb):
        return [TeamPolicyConfig(t_i, method=m_i, count=(counts[t_i] if m_i=="split" else None)) for t_i, m_i in zip(teams, method_comb)]

    train_data_combs = [train_data_given_m_comb(m_comb) for m_comb in method_combos_all]
    assert len(train_data_combs) == 4

    # To see that this gives us the training combinations we want, print the following
    for i, comb in enumerate(train_data_combs):
        print(f"\nTrain data idx {i}") 
        for team_policy_config in comb: 
            print(team_policy_config) 

    # Finally, map over team data combinations to get kwargs for each of them
    def parametrized_kwargs(team_data): 
        policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)

        return {'team_data': team_data,
                'policy_dict': policy_dict,
                'policy_fn': policy_fn,
                'env_name': env_name,
                'env_config': env_config,
                'train_iters': 100,
                'log_intervals': 20,
                'gpu': True
                }

    train_comb_kwargs = [parametrized_kwargs(td) for td in train_data_combs] 

    for i, kwargs in enumerate(train_comb_kwargs):
        print(f"\nStarting on training with team data combination idx {i}")
        ray_train_generic(**kwargs, end_render=True)


def ray_experiment_BF_training_arch(*args):
    env_name = 'battlefield'
    env_config = {'map_size': 55}
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    train_iters = 50
    log_intervals = 10
    gpu = False
    # [out_channels, kernel, stride] 
    new_arch = [[7, [3, 3], 1], [21, [3, 3], 2], [21, [7,7], 1]] # 7(13x13)-3(7x7)-1(1x1) filters
    old_arch = [[21, 13, 1]]
    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),  f"logs/PPO_battlefield_10-iters-test")
    # print(f"\n(from ray_train_generic) `log_dir` has been set to {log_dir}")
    if False:
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        trainer_config["model"]["conv_filters"] = new_arch
        trainer = ppo.PPOTrainer(config=trainer_config)
        checkpoint = train_ray_trainer(trainer, num_iters=train_iters, log_intervals=log_intervals, log_dir=log_dir)
    else:        
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        trainer_config["model"]["conv_filters"] = new_arch
        temp_trainer = ppo.PPOTrainer(config=trainer_config)
        temp_trainer.restore('logs/ccv/PPO_battlefield_10-iters__85fbe/checkpoint_000090/checkpoint-90')
        red_new_weights = temp_trainer.get_policy("red_shared").get_weights()
        temp_trainer.stop()

        trainer_config["model"]["conv_filters"] = old_arch
        temp_trainer = ppo.PPOTrainer(config=trainer_config)
        temp_trainer.restore('logs/ccv/PPO_battlefield_10-iters__fa60e/checkpoint_000060/checkpoint-60')
        blue_old_weights = temp_trainer.get_policy("blue_shared").get_weights()
        temp_trainer.stop()

        policy_dict["red_shared"] = (policy_dict["red_shared"][0], policy_dict["red_shared"][1], policy_dict["red_shared"][2], 
                { "model": {  "conv_filters": new_arch, "conv_activation": "relu" }})
        policy_dict["blue_shared"] = (policy_dict["blue_shared"][0], policy_dict["blue_shared"][1], policy_dict["blue_shared"][2], 
                { "model": { "conv_filters": old_arch, "conv_activation": "relu" }})
        env_config = {'map_size': 100}
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        del trainer_config["model"]
        trainer = ppo.PPOTrainer(config=trainer_config)
        trainer.get_policy("red_shared").set_weights(red_new_weights) # transfer the weights
        trainer.get_policy("blue_shared").set_weights(blue_old_weights)
        checkpoint = None
    
    render_from_checkpoint(checkpoint, trainer, env_directory[env_name], env_config, policy_fn, max_iter=10000, savefile=True) 
    rewards = evaluate_policies(checkpoint, trainer, battlefield_v3, env_config, policy_fn, max_iter=10000)
    print("\n ### (ray_experiment_BF_training_arch) POLICY EVALUATION: REWARDS ###")
    for key in rewards:
        print(f"{key}: {rewards[key]}")


def ray_experiment_BA_training_arch(*args):
    env_name = 'battle'
    env_config = {'map_size': 30}
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    train_iters = 2
    log_intervals = 2
    gpu = False
    new_arch = [[7, [5, 5], 2], [21, [3, 3], 2], [21, [4,4], 1]] # (13,13,5) -> (7,5,5) -> (21,3,3) -> (21,1,1)
    old_arch = [[21, 13, 1]] 
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),  f"logs/PPO_battle_newarch_{uuid.uuid4().hex[:5]}")
    print(f"\n(from ray_train_generic) `log_dir` has been set to {log_dir}")
    if True:
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        trainer_config["model"]["conv_filters"] = new_arch
        trainer = ppo.PPOTrainer(config=trainer_config)
        checkpoint = train_ray_trainer(trainer, num_iters=train_iters, log_intervals=log_intervals, log_dir=log_dir,
            render=True, env=battle_v3, env_config=env_config, policy_fn=policy_fn, max_iter=10000)
    else:        
        pass
    
    # render_from_checkpoint(checkpoint, trainer, env_directory[env_name], env_config, policy_fn, max_iter=10000, savefile=True) 
    rewards = evaluate_policies(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000)
    print("\n ### (ray_experiment_BA_training_arch) POLICY EVALUATION: REWARDS ###")
    for key in rewards:
        print(f"{key}: {rewards[key]}")


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
    # pettingzoo_peek(battle_v3, {'map_size': 30})
    # ray_TD_training_share_split_retooled()
    # ray_CA_generalized()
    # ray_experiment_BF_training_arch()
    ray_experiment_BA_training_arch()
    # ray_BF_training_share_split_retooled()
    # ray_AP_training_share_split_retooled()  # Run this after Local (2) finishes.
    # ray_BF_training_share_split_retooled()
    # ray_BA_training_share_pretrained(checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_battle_100-iters__cad08/checkpoint_000100/checkpoint-100')
    # ray_BA_training_share_split_retooled()
    # ray_AP_training_share_split_retooled()
    print("\nDONE")


if __name__ == "__main__":
    main()
