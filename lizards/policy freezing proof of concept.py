from experiments import *
from main_utils import *
import pathlib

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.examples.policy.random_policy import RandomPolicy
# from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env
import pytz
import datetime
from copy import deepcopy
import pandas as pd
from typing import List, Dict
import numpy.typing as npt
import numpy as np
from functools import partial
from pathlib import Path

# policy freezing and unfreezing stuff
# ====================================
# For info on Rllib callbacks: https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_train_result

# Ray trainer code: https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/agents/trainer.py

# https://github.com/ray-project/ray/issues/6669
# https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/examples/self_play_league_based_with_open_spiel.py
# https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/examples/self_play_with_open_spiel.py

""" TO DO WED Dec 1
0. Write a callback for recording / logging the policy weihgts, so tt we can use tune.run instead of our manual training loop
1. Test training loop with APTCallback_BA_simplest_for_quicktest to check if policy freezing stuff has been recorded properly
2. Run each exp for couple of iters and use new policy freezing logging to check if freezing happening as desired.
3. Add checkpoint saving and evaluation (see experiments.py for how)
4. chk with Qiuhong / Anna re training architecture
5. Run the experiments for full iter length.
"""



# TO DO NOTE: We should try to switch over to the new training conv stuff that Qiuhong / Anna has made before actually training the experiments
convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}


def list_map(*args): return list(map(*args))
def tup_map(*args): return tuple(map(*args))
def np_itermap(*args, dtype=bool): return np.fromiter(map(*args), dtype=dtype)


def get_timestamp():
    tz = pytz.timezone('US/Eastern')
    short_timestamp = datetime.now(tz).strftime("%d.%m_%H.%M")
    return short_timestamp


def pairwise_eq_chk(team_name, wt_dict1, wt_dict2):
    """
    Returns True iff wts for team_name in both dicts are the same  
    Assumes keys are the same for both dicts; see examples below 
    """
    d1 = wt_dict1[team_name]
    d2 = wt_dict2[team_name]

    return np.array([True if np.array_equal(d1[key], d2[key]) else False for key in d1]).all()

test_dict_blue_123 = {"blue": {'blue/conv_value_1/bias': np.array([1, 2, 3])}}
test_dict_blue_123_copy = deepcopy(test_dict_blue_123)
test_dict_blue_321 = {"blue": {'blue/conv_value_1/bias': np.array([3, 2, 1])}}
test_dict_blue_321_copy = deepcopy(test_dict_blue_321)

assert pairwise_eq_chk("blue", test_dict_blue_123, test_dict_blue_123_copy) == True
assert pairwise_eq_chk("blue", test_dict_blue_123, test_dict_blue_321) == False


def check_eq_policy_wts_across_iters(pol_wts_across_iters: List[Dict[str, Dict[str, npt.ArrayLike]]], team_names: List[str]):
    """ 
    Given a list of policy weights across iterations, checks if the pol wts at each iteration is equal to the previous one
    pol_wts_across_iters's first value must be the initial random wts for each team; i.e., the wts at iteration 0 
    So at idx i of pol_wts_across_iters, we'll have the pol weights __at the end of__ the i-th iteration (where the idxing is 0-based)
    """
    return {team: tup_map(partial(pairwise_eq_chk, team), pol_wts_across_iters, pol_wts_across_iters[1:]) for team in team_names}


test_dict_br_123 = {"blue": {'blue/conv_value_1/bias': np.array([1, 2, 3])},
                    "red": {'red/conv_value_1/bias': np.array([4, 9])}}
test_dict_br_123_copy = deepcopy(test_dict_br_123)
test_dict_br_321 = {"blue": {'blue/conv_value_1/bias': np.array([3, 2, 1])},
                    "red": {'red/conv_value_1/bias': np.array([4, 9])}}
test_dict_br_321_copy = deepcopy(test_dict_br_321)

test_pw_across_iters = [test_dict_br_123, test_dict_br_123_copy, test_dict_br_321, test_dict_br_321_copy]
test_pw_eq_chk_dict = {'blue': (True, False, True), 'red': (True, True, True)}
assert check_eq_policy_wts_across_iters(test_pw_across_iters, ["blue", "red"]) == test_pw_eq_chk_dict


def get_changepoints(eq_chk_dict: dict):
    # False here means: the pol wts at that iteration not equal to those at previous iter
    return {team: np.nonzero(np.array(eq_chk_dict[team])==False) for team in eq_chk_dict}

# get_changepoints(test_pw_eq_chk_dict) 
# {'blue': (array([1]),), 'red': (array([], dtype=int64),)}



def save_results_dicts_pol_wts(results_dicts, policy_weights_for_iters, log_dir=Path("logs/pol_freezing")):
    #results_save_path = log_dir.joinpath(f"{timestamp}_results_stats.csv") TODO: timestamp is not defined
    results_save_path = log_dir.joinpath(f"{get_timestamp()}_results_stats.csv")
    pd.DataFrame(results_dicts).to_csv(results_save_path)

    print(f"results_dicts saved to {results_save_path}")

    # TO DO: 
    # 1. Save raw pol wts
    # 2. Save and print changepoints
    changepoints = get_changepoints(check_eq_policy_wts_across_iters(policy_weights_for_iters))

    for team in changepoints:
        print(f"changepoints for team {team} are:\n {changepoints}")



# TO DO: Incorporate this into tune.run 
def train_for_pol_wt_freezing(trainer, timestamp=get_timestamp(), num_iters=20, log_intervals=10, log_dir=Path("logs/pol_freezing")):

    true_start = time.time()

    results_dicts = []
    policy_weights_for_iters = []

    def get_and_log_wts(trainer):
        copied_policy_wts_from_local_worker = deepcopy(trainer.get_weights())
        # there was an issue on Rllib github tt made me think they might not be careful enough about managing state and refs when it comes to policy wts
        policy_weights_for_iters.append(copied_policy_wts_from_local_worker)


    for i in range(num_iters):
        print(f"Starting training on iter {i + 1}...")
        start = time.time()
        
        result = trainer.train()
        results_dicts.append(result)

        print(f"batch {i + 1}: took {time.time() - start} seconds")

        get_and_log_wts(trainer) 
        # could use a callback instead for this tho, e.g. https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        # but for now I like the idea of using a causal pathway distinct from tt which we are manipulating

        # TO DO: Fix log interval code
        # if (i + 1) % log_intervals == 0:
        #     checkpoint = trainer.save(log_dir)
        #     print("checkpoint saved at", checkpoint)

    print(f"Full training took {(time.time() - true_start) / 60.0} min")
    
    save_results_dicts_pol_wts(results_dicts, policy_weights_for_iters, log_dir)
    
    return results_dicts, policy_weights_for_iters


""" 
Assumptions:
* Code below was designed for 2-team environments
* names of policies == prefixes of teams (i.e., either "predator" or "prey")
"""

class APTCallback_BA_simplest_for_quicktest(DefaultCallbacks): 
    def __init__(self):
        super().__init__()

        self.teams = ("red", "blue") 
        self.curr_trainable_policies = {"red"} 

        self.interval_len = 2

    def get_other_team(self): 
        # Returns str
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1"""
        
        curr_iter = trainer.iteration
        print(f"Just finished train iter {curr_iter}")

        if curr_iter == 1 or curr_iter % self.interval_len == 0:
            team_to_freeze = self.curr_trainable_policies
            team_to_train = self.get_other_team()  # this will be str
            self.curr_trainable_policies = {team_to_train}

            print(f"For iter {curr_iter + 1}: Freezing {team_to_freeze} and training {team_to_train}")

            def _set(worker):
                print(f"_set has been called; self.curr_trainable_policies are {self.curr_trainable_policies}")
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)
            
            trainer.workers.foreach_worker(_set)


# APTCallback stands for: Alternating Policy Training Callback
class APTCallback_BA_to_wrap(DefaultCallbacks):
    # We'll make this cleaner by using the mixin once we're sure that this works

    def __init__(self, team_to_turn_length: dict):
        super().__init__()

        self.team_to_turn_length = team_to_turn_length
        self.turn_lengths_sum = sum(self.team_to_turn_length.values())
        self.teams = frozenset(self.team_to_turn_length.keys())
        self.curr_trainable_policies = {list(self.teams)[0]}

        self.burn_in_iters = 0
        # How many iterations to train normally for, before starting alternating policy training/freezing regimen

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
        (Iteration 0 is the state when *no* training has been done.)"""

        curr_iter = trainer.iteration
        print(f"Just finished train iter {curr_iter}")
        if curr_iter > self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.turn_lengths_sum) in self.team_to_turn_length.values():
            team_to_freeze = self.curr_trainable_policies # for debug

            self.curr_trainable_policies = self.teams - self.curr_trainable_policies # swaps teams

            team_to_train = self.curr_trainable_policies # for debug

            print(f"On iter {curr_iter + 1}, {team_to_freeze} will be frozen, {team_to_train} will be trained")

            def _set(worker):
                print(f"_set has been called; self.curr_trainable_policies are {self.curr_trainable_policies}")
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)

            trainer.workers.foreach_worker(_set)


# Demonstration of how to make code more reusable with mixins:
# I'm not 100% sure the mixin will work well with Ray though, so caveat emptor 
# Tho the mixin does at least seem to add the relevant methods
# This is definitely the way to make the code more reusable though --- just needs more testing first

# class TwoTeamOnTrainResultMixin:   
#     def get_other_team(self):
#         # this will need to be made more robust later
#         other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
#         return other_team

#     def on_train_result(self, *, trainer, result, **kwargs):
#         """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
#         (Iteration 0 is the state when *no* training has been done.)"""
        
#         curr_iter = trainer.iteration
#         if curr_iter > self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.interval_len == 0):
#             team_to_freeze = self.curr_trainable_policies
#             team_to_train = self.get_other_team()
#             print(team_to_train)
#             self.curr_trainable_policies = {team_to_train}

#             print(f"On iter {curr_iter + 1}, {team_to_freeze} will be frozen, {team_to_train} will be trained")

#             def _set(worker):
#                 # Note: `_set` must be enclosed in `on_train_result`!
#                 worker.set_policies_to_train(self.curr_trainable_policies)
            
#             trainer.workers.foreach_worker(_set)

# class APTCallback_AP(DefaultCallbacks, TwoTeamOnTrainResultMixin):

#     def __init__(self):
#         super().__init__()

#         self.teams = ("predator", "prey") 
#         self.curr_trainable_policies = {"predator"} 
#         # Start with predator being the one whose policy's being trained

#         self.burn_in_iters = 0 
#         # How many iterations to train normally for, before starting alternating policy training/freezing regimen

#         self.interval_len = 3
#         # The point of this, recall, is to train one team while keeping the other frozen for regular intervals
#         # interval_len := number of iterations that each interval consists in



# env-specific things
# ===================

def BA_pol_mapping_fn(agent_id, episode, worker, **kwargs):
    return "red" if agent_id.startswith("red") else "blue"

def BA_apt_1_30_PROTOTYPE(*args, map_size=19, timestamp=get_timestamp()):
    # started Wed Dec 1
    env_name = "battle"

    training_setup = {"num_iters": 16,
                     # "log_intervals": 4, # TO DO: might need to set this properly
                     "log_dir": 'logs/BA_testing'}

    env_fn = env_directory[env_name]
    env_config = {'map_size': map_size} 
    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    class APTCallback_BA_test_1_30(APTCallback_BA_to_wrap):
        def __init__(self):
            super().__init__({"red": 1, "blue": 30})

    ray_trainer_config = {

        "callbacks": APTCallback_BA_test_1_30, # IMPT 

        "multiagent": {
            "policies": {"red": (None, obs_space, action_space, dict()),
                         "blue": (None, obs_space, action_space, dict())},
            "policy_mapping_fn": BA_pol_mapping_fn
        },
        "policies_to_train": ["red"], # Red, and __only__ red, will be trained on first iter
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

    trainer = ppo.PPOTrainer(config=ray_trainer_config)
    train_for_pol_wt_freezing(trainer, timestamp=timestamp)

    # TO DO: Add eval stuff!


if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    BF_alternating_pol_training_PROTOTYPE()

    # AP_alternating_pol_train_PROTOTYPE()



