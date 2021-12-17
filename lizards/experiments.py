import os

from main_utils import *
from quantitative_analysis.stats import *
# from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3, combined_arms_v5
import ray.rllib.agents.ppo as ppo
# import ray
# import ray.rllib.agents.pg as pg
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
from datetime import datetime
from pathlib import Path

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


# (Filestructure constants for checkpoint analysis)

EXP2_BASE_PATH = Path("/Users/eli/Downloads/MultiAgent-PositronicLizards/lizards/downloaded_checkpoints/")
EXP2_CHECKPOINT_SUFFIX_TO_DIR= {"baseline": "PPO_battle_oldarch__ms19_cad08", # (this mirrors Google drive filenenames, compared to below)
                            "self-play": "PPO_battleself-play-ms19_120-iters_ms19_c6f6a", # YM: changed this to the right checkpoint. Sorry about using the wrong one before.
                            "pretrained": "PRETRAINED-cad08_120-iters__cd1c3",
                            "random":  "VS_RANDOM_train_time_2h/PPO_battle_120-iters_ms19_bcdcc"}

EXP2_CHECKPOINT_FILE = "checkpoint_000120/checkpoint-120"
EXP2_FULL_SUFFIX_CHECKPOINT_PATHS = {suffix: EXP2_BASE_PATH / pth / EXP2_CHECKPOINT_FILE for suffix, pth in EXP2_CHECKPOINT_SUFFIX_TO_DIR.items()}

def get_all_YM_chkpt_paths():
    EXP2_BASE_PATH = Path("/gpfs/scratch/yh31/projects/MultiAgent-PositronicLizards/lizards/saved_checkpoints/for_evals/")
    EXP2_CHECKPOINT_SUFFIX_TO_DIR= {"baseline": "BASELINE_oldarch__ms19_cad08",
                                "self-play": "SELF-PLAY-ms19_120-iters_ms19_c6f6a",
                                "pretrained": "PRETRAINED-cad08_120-iters__cd1c3",
                                "random":  "VS_RANDOM_train_time_2h/PPO_battle_120-iters_ms19_bcdcc"}
    EXP2_CHECKPOINT_FILE = "checkpoint_000120/checkpoint-120"
    EXP2_FULL_SUFFIX_CHECKPOINT_PATHS = {suffix: EXP2_BASE_PATH / pth / EXP2_CHECKPOINT_FILE for suffix, pth in EXP2_CHECKPOINT_SUFFIX_TO_DIR.items()}

    return EXP2_FULL_SUFFIX_CHECKPOINT_PATHS

def get_YM_chkpt_paths(non_baseline):
    EXP2_FULL_SUFFIX_CHECKPOINT_PATHS = get_all_YM_chkpt_paths()
    baselinepath, non_baseline_path = EXP2_FULL_SUFFIX_CHECKPOINT_PATHS["baseline"], EXP2_FULL_SUFFIX_CHECKPOINT_PATHS[non_baseline] 
    return baselinepath, non_baseline_path





class TeamPolicyConfig:
    def __init__(self, team_name, method='shared', count=None, random_action_team=False):
        """
        For specifying policy breakdowns for teams
        :param team_name: 'red', 'preditor', etc.
        :param method: 'shared': one policy shared for all agents starting with `team_name`, or 'split': one per agent
        :param count: (not required if method='shared') number of agents on team
        :param random_action_team: a special boolean flag that makes this instance a dummy-team that acts randomly, and will never learn.
        """
        self.team_name = team_name
        self.method = method
        self.count = count
        self.random_action_team = random_action_team

    def for_filename(self):
        if self.method == 'split':
            return f"_{self.team_name}-split"
        return ""

    def __str__(self):
        return f"TeamPolicyConfig: {self.team_name}, {self.method}, {self.count}"


# def experiment_1():
#     # DEPRECATED
#     battlefield = convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0))
#     model = train(battlefield, PPO, time_steps=10000, save_name='trained_policies/battlefield-10e4-V1')
#     evaluate_model(battlefield, model)
#
#
# def view_results():
#     # DEPRECATED
#     evaluate_model(convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0)),
#                    PPO.load('trained_policies/battlefield-10e6-V1'))


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


def iterate_BA_stats_eval_trials(run_path, checkpoint_path, num_trials = 100, gpu = False, save_viz=True):
    """Gets the stats for Battle from a checkpoint including the number of attacks."""

    # Make sure valid log directory exists:
    # if log_dir is None:
    #     log_dir = Path(checkpoint_path).parents[0]
    # if not log_dir.is_dir(): log_dir.mkdir()

    # Initial environment settings:
    env_config = {"map_size": 19}
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')] #
    policy_dict, policy_fn = get_policy_config(**env_spaces['battle'], team_data=team_data) #

    # Create representative trainer from settings/checkpoint:
    trainer_config = get_trainer_config('battle', policy_dict, policy_fn, env_config, gpu=gpu)
    representative_trainer = ppo.PPOTrainer(config=trainer_config)
    representative_trainer.restore(checkpoint_path)

    for trial_i in range(num_trials):
        trial_path = run_path / ("trial_" + str(trial_i))
        losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df = collect_stats_from_eval(representative_trainer, battle_v3, env_config, policy_fn, trial_path, save_viz=save_viz)
        yield losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df


# ------------------------------------------------------------
# We are always comparing baseline against some other policy.
# ------------------------------------------------------------
def run_selfplay_against_baseline(n_trials, log_dir="logs/evals", gpu=False, env_name="battle"):
    baseline_chkpt_path, NON_baseline_chkpt_path = get_YM_chkpt_paths("self-play")

    if not Path(NON_baseline_chkpt_path).exists() or not Path(baseline_chkpt_path).exists(): 
        raise Exception("checkpoint does not exist at path!")

    run_name = "selfplay_vs_baseline" # timestamp will be added in stats fn

    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    
    # eval_config = {
    #     "red_ckpt": str(baseline_chkpt_path),
    #     "red_load": "red_shared",

    #     "blue_ckpt": str(NON_baseline_chkpt_path),
    #     "blue_load": policy_to_load,
    # }
    # log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])

    # Get weights from self play checkpoint
    def get_selfplay_weights():
        policy_dict = {'all': (None, env_spaces[env_name]['obs_space'], env_spaces[env_name]['action_space'], dict())}
        policy_fn = lambda *args, **kwargs: 'all'
        self_play_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)

        # get weights from the policy named 'all' 
        temp_trainer = ppo.PPOTrainer(config=self_play_trainer_config)
        temp_trainer.restore(str(NON_baseline_chkpt_path))
        weights = temp_trainer.get_policy("all").get_weights()
        temp_trainer.stop()

        return weights

    self_play_weights = get_selfplay_weights()

    # Load baseline checkpoint, use its red policy as baseline. 
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    eval_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    eval_trainer_config["env_config"] = eval_env_config
    eval_trainer = ppo.PPOTrainer(config=eval_trainer_config)
    eval_trainer.restore(str(baseline_chkpt_path))

    # Transfer self play policy weights to the blue of eval_trainer
    eval_trainer.get_policy("blue_shared").set_weights(self_play_weights)

    # and evaluate
    write_BA_stats_CSVs(n_trials, eval_trainer, log_dir, env_config, policy_fn, save_viz=True, gpu=False, run_name_from_user=run_name)


def run_pretrained_against_baseline(n_trials, log_dir="logs/evals", gpu=False, env_name="battle"):
    baseline_chkpt_path, NON_baseline_chkpt_path = get_YM_chkpt_paths("pretrained")

    if not Path(NON_baseline_chkpt_path).exists() or not Path(baseline_chkpt_path).exists(): 
        raise Exception("checkpoint does not exist at path!")

    run_name = "pretrained_vs_baseline" 

    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    
    # eval_config = {
    #     "red_ckpt": str(baseline_chkpt_path),
    #     "red_load": "red_shared",

    #     "blue_ckpt": str(NON_baseline_chkpt_path),
    #     "blue_load": policy_to_load,
    # }
    # log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])

    # Get weights for the red ('from-scratch') policy from 'pretrained' checkpoint
    # (The pretrained exp was: red was trained from scratch against a blue that had alrdy been trained for 40 iters
    def get_red_weights_from_pretrained():
        team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
        policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
        pt_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)

        # get weights from the policy named 'red_shared' 
        temp_trainer = ppo.PPOTrainer(config=pt_trainer_config)
        temp_trainer.restore(str(NON_baseline_chkpt_path))
        weights = temp_trainer.get_policy("red_shared").get_weights()
        temp_trainer.stop()

        return weights

    red_from_pt_weights = get_red_weights_from_pretrained()


    # Load baseline checkpoint, use its red policy as baseline. 
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    eval_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    eval_trainer_config["env_config"] = eval_env_config
    eval_trainer = ppo.PPOTrainer(config=eval_trainer_config)
    eval_trainer.restore(str(baseline_chkpt_path))

    # Transfer the other policy weights to the blue of eval_trainer
    eval_trainer.get_policy("blue_shared").set_weights(red_from_pt_weights)

    # and evaluate
    write_BA_stats_CSVs(n_trials, eval_trainer, log_dir, env_config, policy_fn, save_viz=True, gpu=False, run_name_from_user=run_name)


def run_rand_trained_pol_against_baseline(n_trials, log_dir="logs/evals", gpu=False, env_name="battle"):
    
    cp_path_id="random"
    run_name = "randtrained_vs_baseline" 
    
    baseline_chkpt_path, NON_baseline_chkpt_path = get_YM_chkpt_paths(cp_path_id)

    if not Path(NON_baseline_chkpt_path).exists() or not Path(baseline_chkpt_path).exists(): 
        raise Exception("checkpoint does not exist at path!")

    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    
    # eval_config = {
    #     "red_ckpt": str(baseline_chkpt_path),
    #     "red_load": "red_shared",

    #     "blue_ckpt": str(NON_baseline_chkpt_path),
    #     "blue_load": policy_to_load,
    # }
    # log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])

    # Get weights for policy that was trained against random ('blue' was the non-random one)
    def get_other_weights():
        team_data_for_random = [TeamPolicyConfig('red', random_action_team=True),
                                TeamPolicyConfig('blue')]
        policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data_for_random)
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)

        # get weights from the policy named 'blue_shared' 
        temp_trainer = ppo.PPOTrainer(config=trainer_config)
        temp_trainer.restore(str(NON_baseline_chkpt_path))
        weights = temp_trainer.get_policy("blue_shared").get_weights()
        temp_trainer.stop()

        return weights

    weights_of_policy_tt_was_trained_against_random = get_other_weights()


    # Load baseline checkpoint, use its red policy as baseline. 
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    eval_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    eval_trainer_config["env_config"] = eval_env_config
    eval_trainer = ppo.PPOTrainer(config=eval_trainer_config)
    eval_trainer.restore(str(baseline_chkpt_path))

    # Transfer the other policy weights to the blue of eval_trainer
    eval_trainer.get_policy("blue_shared").set_weights(weights_of_policy_tt_was_trained_against_random)

    # and evaluate
    write_BA_stats_CSVs(n_trials, eval_trainer, log_dir, env_config, policy_fn, save_viz=True, gpu=False, run_name_from_user=run_name)


def run_self_play_against_pretrained(n_trials, log_dir="logs/evals", gpu=False):
    run_name = "selfplay_vs_pretrained" 

    # get checkpoint paths
    checkpoint_path_dict = get_all_YM_chkpt_paths
    selfplay_cp_path, pretrained_cp_path = checkpoint_path_dict["self-play"], checkpoint_path_dict["pretrained"]
    if not Path(selfplay_cp_path).exists() or not Path(pretrained_cp_path).exists(): 
            raise Exception("checkpoint does not exist at path!")

    # config set up
    env_name="battle"
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}

    # Get weights from self play checkpoint
    def get_selfplay_weights():
        policy_dict = {'all': (None, env_spaces[env_name]['obs_space'], env_spaces[env_name]['action_space'], dict())}
        policy_fn = lambda *args, **kwargs: 'all'
        self_play_trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)

        # get weights from the policy named 'all' 
        temp_trainer = ppo.PPOTrainer(config=self_play_trainer_config)
        temp_trainer.restore(str(selfplay_cp_path))
        weights = temp_trainer.get_policy("all").get_weights()
        temp_trainer.stop()

        return weights

    self_play_weights = get_selfplay_weights()


    # Set up evaluator trainer; note that 'red_shared' is the policy tt was trained from scratch
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)

    eval_trainer = ppo.PPOTrainer(config=trainer_config)
    eval_trainer.restore(str(pretrained_cp_path))

    # Set the blue policy of eval_trainer to the self play policy, so tt we have 'from scratch' playing against 'self play'
    eval_trainer.get_policy("blue_shared").set_weights(self_play_weights)

    # and evaluate
    write_BA_stats_CSVs(n_trials, eval_trainer, log_dir, env_config, policy_fn, save_viz=True, gpu=False, run_name_from_user=run_name)





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


def ray_train_generic(*args, end_render=True, savefile=False, policy_log_str=None, test_mode=False, **kwargs):
    trainer_config = get_trainer_config(kwargs['env_name'], kwargs['policy_dict'], kwargs['policy_fn'],
                                        kwargs['env_config'], gpu=kwargs['gpu'], num_workers=kwargs.get('num_workers', 1))
    if test_mode: 
        trainer_config["train_batch_size"] = 1000
    print(f"trainer_config is {trainer_config}")

    trainer = ppo.PPOTrainer(config=trainer_config)

    if policy_log_str is None:
        policy_log_str = "".join([p.for_filename() for p in kwargs['team_data']])
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           f"logs/PPO_{kwargs['env_name']}{policy_log_str}_{kwargs['train_iters']}-iters_ms{kwargs['env_config']['map_size']}_{uuid.uuid4().hex[:5]}")
    print(f"\n(from ray_train_generic) `log_dir` has been set to {log_dir}")

    checkpoint = train_ray_trainer(trainer, num_iters=kwargs['train_iters'], log_intervals=kwargs['log_intervals'], log_dir=log_dir)

    is_battle = True if kwargs['env_name'] == "battle" else False
    if end_render:
        render_from_checkpoint(checkpoint, trainer, env_directory[kwargs['env_name']], kwargs['env_config'], kwargs['policy_fn'], max_iter=10000, savefile=savefile, is_battle=is_battle)
    return checkpoint, trainer


def ray_viz_generic(checkpoint, max_iter=20000, savefile=False, **kwargs):
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
    env_config = {'map_size': 30}
    # policy_dict = {'all': (None, env_spaces['battle']['obs_space'], env_spaces['battle']['action_space'], dict())}
    # policy_fn = lambda *args, **kwargs: 'all'
    counts = get_num_agents(env_directory[env_name], env_config)
    team_data = [TeamPolicyConfig('red', method='split', count=counts['red']),
                 TeamPolicyConfig('blue')]
    # team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'env_config': env_config,
        'team_data': team_data,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 120,
        'log_intervals': 20,
        'gpu': True
    }

    # ray_train_generic(**kwargs, end_render=True)
    ray_viz_generic(savefile=True,
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_battle_red-split_120-iters__3914d/checkpoint_000120/checkpoint-120',
        **kwargs)

def ray_BA_training_share_randomized_retooled(test_mode=False):
    env_name = 'battle'
    env_config = {'map_size': 19}
    team_data = [TeamPolicyConfig('red', random_action_team=True),
                 TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 120,
        'log_intervals': 30,
        'gpu': False,
    }

    if test_mode: 
        kwargs['train_iters']=1

    ray_train_generic(**kwargs, test_mode=test_mode, savefile=True, end_render=True)

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
    # ray_viz_generic(
    #     checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_tiger-deer_tiger-split_100'
    #                '-iters__f1282/checkpoint_000100/checkpoint-100',
    #     **kwargs)


def ray_AP_training_share_split_retooled():
    env_name = 'adversarial-pursuit'
    env_config = {'map_size': 19}
    # predator_count = get_num_agents(env_directory[env_name], env_config)['predator']
    # team_data = [TeamPolicyConfig('predator', method='split', count=predator_count), TeamPolicyConfig('prey')]
    team_data = [TeamPolicyConfig('predator'), TeamPolicyConfig('prey')]
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
    ray_viz_generic(savefile=True,
        checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_adversarial-pursuit_120-iters_ms19_33090/checkpoint_000120/checkpoint-120',
        **kwargs)


def ray_AP_training_share_randomized_retooled(test_mode=False):
    env_name = 'adversarial-pursuit'
    env_config = {'map_size': 19} # making this the same as in `ray_AP_training_share_split_retooled`
    team_data = [TeamPolicyConfig('predator', random_action_team=True), TeamPolicyConfig('prey')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    kwargs = {
        'env_name': env_name,
        'team_data': team_data,
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 120,
        'log_intervals': 30,
        'gpu': True
    }

    ray_train_generic(**kwargs, test_mode=test_mode, savefile=True, end_render=True)


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
                'train_iters': 120,
                'log_intervals': 20,
                'gpu': True
                }

    train_comb_kwargs = [parametrized_kwargs(td) for td in train_data_combs] 

    for i, kwargs in enumerate(train_comb_kwargs):
        print(f"\nStarting on training with team data combination idx {i}")
        ray_train_generic(**kwargs, end_render=True)


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


def all_experiment_1():
    # Self-play on Battle
    env_config = {'map_size': 19}
    policy_dict = {'all': (None, env_spaces['battle']['obs_space'], env_spaces['battle']['action_space'], dict())}
    policy_fn = lambda *args, **kwargs: 'all'
    kwargs = {
        'env_name': 'battle',
        'env_config': env_config,
        'policy_dict': policy_dict,
        'policy_fn': policy_fn,
        'train_iters': 120,
        'log_intervals': 20,
        'gpu': True
    }
    ray_train_generic(policy_log_str="self-play-ms19", **kwargs)

    # # Shared-split Symmetric battle
    # env_name = 'battle'
    # env_config = {'map_size': 30}
    # counts = get_num_agents(env_directory[env_name], env_config)
    # team_data = [TeamPolicyConfig('red', method='split', count=counts['red']),
    #              TeamPolicyConfig('blue')]
    # policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    # kwargs = {
    #     'env_name': env_name,
    #     'team_data': team_data,
    #     'env_config': env_config,
    #     'policy_dict': policy_dict,
    #     'policy_fn': policy_fn,
    #     'train_iters': 120,
    #     'log_intervals': 20,
    #     'gpu': True
    # }
    # ray_train_generic(**kwargs, end_render=True)
    #
    # # Shared-shared Symmetric battle
    # env_name = 'battle'
    # env_config = {'map_size': 30}
    # team_data = [TeamPolicyConfig('red'),
    #              TeamPolicyConfig('blue')]
    # policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    # kwargs = {
    #     'env_name': env_name,
    #     'team_data': team_data,
    #     'env_config': env_config,
    #     'policy_dict': policy_dict,
    #     'policy_fn': policy_fn,
    #     'train_iters': 120,
    #     'log_intervals': 20,
    #     'gpu': True
    # }
    # ray_train_generic(**kwargs, end_render=True)

    # Shared-shared Asymmetric AP
    env_name = 'adversarial-pursuit'
    env_config = {'map_size': 19}
    team_data = [TeamPolicyConfig('predator'), TeamPolicyConfig('prey')]
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
    ray_train_generic(**kwargs, end_render=True)

    # Shared-split Asymmetric AP
    env_name = 'adversarial-pursuit'
    env_config = {'map_size': 19}
    predator_count = get_num_agents(env_directory[env_name], env_config)['predator']


def main():
    # kwargs = parse_args()
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    # print(os.environ.get("RLLIB_NUM_GPUS", "0"))
    # ray.init(num_gpus=1, local_mode=True)
    # print(ray.get_gpu_ids())
    # print(kwargs)
    # pettingzoo_peek(adversarial_pursuit_v3, {'map_size': 30})
    # pettingzoo_peek(tiger_deer_v3, {'map_size': 30})
    # pettingzoo_peek(battle_v3, {'map_size': 30})
    # all_experiment_1()
    # ray_TD_training_share_split_retooled()
    # ray_CA_generalized()
    # ray_BF_training_share_split_retooled()
    # ray_AP_training_share_split_retooled()  # Run this after Local (2) finishes.
    # ray_BF_training_share_split_retooled()
    # ray_BA_training_share_pretrained(checkpoint='/home/ben/Code/MultiAgent-PositronicLizards/lizards/logs/PPO_battle_100-iters__cad08/checkpoint_000100/checkpoint-100')
    # ray_BA_training_share_split_retooled()
    # ray_AP_training_share_split_retooled()
    # ray_BA_training_share_split_retooled()
    # ray_AP_training_share_split_retooled()

    # Randomized experiments
    #ray_BA_training_share_randomized_retooled(test_mode=False)
    #print("Done with BA exp!")
    # ray_AP_training_share_randomized_retooled()
    # print("\nDONE")

    # run_selfplay_against_baseline(n_trials=800) 
    #(Eli): Yongming, can you actually run this many trials? [Eli was referring to 3_000]
    # YM: I ran it over night. I think it took like 5 hours? I had set a high figure on a whim.
    
    run_self_play_against_pretrained(n_trials=500) 

    # ray_BA_training_share_randomized_retooled(test_mode=False)
    # print("Done with BA exp!")
    # ray_AP_training_share_randomized_retooled()
    # print("\nDONE")


if __name__ == "__main__":
    main()