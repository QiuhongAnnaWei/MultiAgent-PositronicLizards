from experiments import *
from main_utils import *

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


# Stepping thru to get the trainer obj 


def list_map(*args): return list(map(*args))
def tup_map(*args): return tuple(map(*args))
def np_itermap(*args, dtype=bool): return np.fromiter(map(*args), dtype=dtype)

# This is for env BATTLE, not BF
class APTCallback_BA_simplest(DefaultCallbacks):
    # We'll make this cleaner by using the mixin once we're sure that this works

    def __init__(self):
        super().__init__()

        self.teams = ("red", "blue") 
        self.curr_trainable_policies = {"red"} 


        self.interval_len = 2
        # The point of this, recall, is to train one team while keeping the other frozen for regular intervals
        # interval_len := number of iterations that each interval consists in


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



# Temp setup code to just play with ray trainer. TO DO: clean up later

map_size = 14 # min map size is 12
env_name = "battle"
env_fn = env_directory[env_name]
env_config = {'map_size': map_size} 

training_setup = {"num_iters": 16,
                 "log_intervals": 4,
                 "log_dir": 'logs/BA_testing'}


action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

def pol_mapping_fn(agent_id, episode, worker, **kwargs):
    return "red" if agent_id.startswith("red") else "blue"


ray_trainer_config = {

    "callbacks": APTCallback_BA_simplest, # IMPT 
    # "callbacks": APTCallback_BF_test_1,

    "multiagent": {
        "policies": {"red": (None, obs_space, action_space, dict()),
                     "blue": (None, obs_space, action_space, dict())},
        "policy_mapping_fn": pol_mapping_fn
    },
    
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



# Get list of weights of each worker, including remote replicas
# trainer.workers.foreach_worker(lambda ev: ev.get_policy().get_weights())
# this errored when run before it's done any training


# ============================================================
# Stepping thru to understand how it works before generalizing
# ============================================================

trainer.get_weights() 
# this actually works, tho i thihkn it works ojnly for local worker

# In [45]: trainer.iteration
# Out[45]: 0


result1 = trainer.train()


# In [43]: type(trainer.get_weights()["red"])
# Out[43]: collections.OrderedDict

# In [44]: trainer.get_weights()["red"].keys()
# Out[44]: odict_keys(['red/conv_value_1/kernel', 'red/conv_value_1/bias', 'red/conv1/kernel', 'red/conv1/bias', 'red/conv_value_out/kernel', 'red/conv_value_out/bias', 'red/conv_out/kernel', 'red/conv_out/bias'])
# In [47]: type(trainer.get_weights()["red"]['red/conv_value_1/kernel'])
# Out[47]: numpy.ndarray
# In [48]: type(trainer.get_weights()["red"]['red/conv_value_out/bias'])
# Out[48]: numpy.ndarray


# trainer.config["num_workers"]


def get_timestamp():
    tz = pytz.timezone('US/Eastern')
    short_timestamp = datetime.now(tz).strftime("%d.%m_%H.%M")
    return short_timestamp



def train_for_testing_pol_wt_freezing(trainer, num_iters=20, log_intervals=10, log_dir=Path("logs/pol_freezing")):

    def get_and_log_wts(trainer):
        copied_policy_wts_from_local_worker = deepcopy(trainer.get_weights())
        # there was an issue on Rllib github tt made me think they might not be careful enough about managing state and refs when it comes to policy wts
        policy_weights.append(copied_policy_wts_from_local_worker)

    true_start = time.time()

    results_dicts = []
    policy_weights_for_iters = []

    for i in range(num_iters):
        print(f"Starting training on iter {i + 1}...")
        start = time.time()
        
        result = trainer.train()
        results_dicts.append(result)

        print(f"batch {i + 1}: took {time.time() - start} seconds")

        get_and_log_wts(trainer)

        # if (i + 1) % log_intervals == 0:
        #     checkpoint = trainer.save(log_dir)
        #     print("checkpoint saved at", checkpoint)

    timestamp = get_timestamp()

    results_save_path = log_dir.joinpath(f"{timestamp}_results_stats.csv")
    pd.DataFrame(results_dicts).to_csv(results_save_path)

    print(f"results_dicts saved to {results_save_path}")

    print(f"Full training took {(time.time() - true_start) / 60.0} min")

    return results_dicts, policy_weights_for_iters


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


test_dict_br_123 = {"blue": {'blue/conv_value_1/bias': np.array([1, 2, 3])},
                    "red": {'red/conv_value_1/bias': np.array([4, 9])}}
test_dict_br_123_copy = deepcopy(test_dict_br_123)
test_dict_br_321 = {"blue": {'blue/conv_value_1/bias': np.array([3, 2, 1])},
                    "red": {'red/conv_value_1/bias': np.array([4, 9])}}
test_dict_br_321_copy = deepcopy(test_dict_br_321)

def check_eq_policy_wts_across_iters(pol_wts_across_iters: List[Dict[str, Dict[str, npt.ArrayLike]]], team_names: List[str]):
    """ 
    pol_wts_across_iters's first value must be the initial random wts for each team; i.e., the wts at iteration 0 
    So at idx i of pol_wts_across_iters, we'll have the pol weights __at the end of__ the i-th iteration (where the idxing is 0-based)
    """
    return {team: tup_map(partial(pairwise_eq_chk, team), pol_wts_across_iters, pol_wts_across_iters[1:]) for team in team_names}

test_pw_across_iters = [test_dict_br_123, test_dict_br_123_copy, test_dict_br_321, test_dict_br_321_copy]
test_pw_eq_chk_dict = {'blue': (True, False, True), 'red': (True, True, True)}
assert check_eq_policy_wts_across_iters(test_pw_across_iters, ["blue", "red"]) == test_pw_eq_chk_dict

def get_changepoints(eq_chk_dict: dict):
    # False here means: the pol wts at that iteration not equal to those at previous iter
    return {team: np.nonzero(np.array(eq_chk_dict[team])==False) for team in eq_chk_dict}

get_changepoints(test_pw_eq_chk_dict) 
# {'blue': (array([1]),), 'red': (array([], dtype=int64),)}

if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

