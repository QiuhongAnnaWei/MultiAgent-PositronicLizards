from experiments import *
from main_utils import *
from policy_freezing_helper import *

import argparse
from copy import deepcopy
from pathlib import Path
import pandas as pd
from functools import reduce, partial
from itertools import starmap
import time
import pytz
from datetime import datetime
from typing import Sequence, Dict, Optional, Tuple, List
import numpy.typing as npt

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.examples.policy.random_policy import RandomPolicy
# from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env

convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}



const_exp_info = {"help": "Non-Littman Battle experiment setup; *not* experimenting with split-share distinction here",
                  "env_name": "battle",
                  "env_fn": env_directory["battle"],
                  "map_size": 19,
                  "policies": ("red", "blue")}


gen_dynamic_info = {"timestamp": None,
                    "r_num": None,
                    "b_num": None,
                    "policyset_to_start_with": None,
                    "log_dir": Path("./logs/pol_freezing"),
                    "test_mode": True,
                    "num_iters": None,
                    "no_alt_pfreeze": False}


def chk_time():
    timestamp = gen_dynamic_info["timestamp"] if gen_dynamic_info["timestamp"] is not None else get_timestamp()
    gen_dynamic_info["timestamp"] = timestamp
    return timestamp


# testing_info = {}



""" 
Assumptions:
* Code below was designed for 2-team environments
* names of policies == prefixes of teams (i.e., either "predator" or "prey")
"""

class APTCallback_BA_to_wrap(DefaultCallbacks):

    def __init__(self, team_turn_len_tuples: list, burn_in_iters = 0):
        super().__init__()

        # Sum all the turn lengths -> Can normalize trainer.iteration w/ this value.
        self.turn_lengths_sum = sum([x[1] for x in team_turn_len_tuples])

        # Then can map certain modulus values to next team (e.g. 0->Red, 5-> Blue, when team_turn_len_tuples=[(Red, 5), (Blue, 1)])
        self.turn_modulus_to_next_team = {}
        curr_turns = 0
        for team_name, turn_len in team_turn_len_tuples:
            self.turn_modulus_to_next_team[curr_turns] = team_name
            curr_turns += turn_len

        # How many iterations to train normally for, before starting alternating policy training/freezing regimen
        self.burn_in_iters = burn_in_iters

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
        (Iteration 0 is the state when *no* training has been done.)"""

        curr_iter = trainer.iteration - 1 # want this to start at zero.
        print(f"Just finished train iter {curr_iter}")

        if curr_iter >= self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.turn_lengths_sum) in self.turn_modulus_to_next_team:
            new_team = self.turn_modulus_to_next_team[((curr_iter - self.burn_in_iters) % self.turn_lengths_sum)]
            print(f"On iter {curr_iter + 1}, switching to train {new_team}")

            def _set(worker):
                print(f"_set has been called; callingset_policies_to_train with {new_team}")
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train({new_team})
            trainer.workers.foreach_worker(_set)


def BA_pol_mapping_fn(agent_id, episode, worker, **kwargs):
    return "red" if agent_id.startswith("red") else "blue"

def BA_apt_1_30_PROTOTYPE(*args):


    if gen_dynamic_info["test_mode"]: 
        gen_dynamic_info["num_iters"] = 12
        gen_dynamic_info["log_intervals"] = None

    timestamp = gen_dynamic_info["timestamp"] if gen_dynamic_info["timestamp"] is not None else get_timestamp()

    env_name = const_exp_info["env_name"]
    env_config = {'map_size': const_exp_info["map_size"]} 
    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]


    class APTCallback_BA(APTCallback_BA_to_wrap):
        def __init__(self):
            super().__init__([("red", gen_dynamic_info["r_num"]), ("blue", gen_dynamic_info["b_num"])])


    ray_trainer_config = {

        "multiagent": {
            "policies": {"red": (None, obs_space, action_space, dict()),
                         "blue": (None, obs_space, action_space, dict())},
            "policy_mapping_fn": BA_pol_mapping_fn,
            # Do NOT use the `policies_to_train` setting to try to train only one policy for the first iter
            # This is currently handled by the training loop
        },
        # TO DO: try to come up with a better way to make clear which team will be trained on first iter in callback and here


        "env": env_name,
        "model": {
            "conv_filters": convs[env_name]
        },
        "env_config": env_config,
        "create_env_on_driver": True, # potentially disable this?
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    if not gen_dynamic_info.get("no_alt_pfreeze", False):
        ray_trainer_config["callbacks"] = APTCallback_BA

    if gen_dynamic_info["test_mode"]: 
        ray_trainer_config["train_batch_size"] = 1000

    trainer = ppo.PPOTrainer(config=ray_trainer_config)
    results_dicts, policy_weights_for_iters = train_for_pol_wt_freezing(trainer, const_exp_info, gen_dynamic_info)

    return results_dicts, policy_weights_for_iters # for interactive testing


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('experiment', help="train, peek")
    parser.add_argument('--test', dest='test_mode', default=False, action='store_true')
    parser.add_argument('--no-alt-pfreeze', dest='no_alt_pfreeze', default=False, action='store_true')
    
    parser.add_argument('-i', '--num-iters', type=int, dest='num_iters', default=120,
                        help="number of training iterations")
    parser.add_argument('-li', '--log-intervals', dest='log_intervals', default=20,
                        help="logging interval")

    parser.add_argument('-r', dest='r_num', type=int, default=5)
    parser.add_argument('-b', dest='b_num', type=int, default=1)

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    gen_dynamic_info["timestamp"] = get_timestamp()

    args = parse_args()
    gen_dynamic_info["test_mode"], gen_dynamic_info["log_intervals"], gen_dynamic_info["num_iters"] = args.test_mode, args.log_intervals, args.num_iters
    gen_dynamic_info["no_alt_pfreeze"] = bool(args.no_alt_pfreeze)

    if gen_dynamic_info["no_alt_pfreeze"]:
        gen_dynamic_info["r_num"], gen_dynamic_info["b_num"] = None, None
    else:
        gen_dynamic_info["r_num"], gen_dynamic_info["b_num"] = int(args.r_num), int(args.b_num)

    if int(args.b_num) == 1:
        gen_dynamic_info["policyset_to_start_with"] = {"blue"}
    else:
        gen_dynamic_info["policyset_to_start_with"] = None


    print(f"gen_dynamic_info is {gen_dynamic_info}")


    BA_apt_1_30_PROTOTYPE()
