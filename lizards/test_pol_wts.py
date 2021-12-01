from experiments import *
from main_utils import *

from copy import deepcopy
import pathlib
import pandas as pd

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.examples.policy.random_policy import RandomPolicy
# from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env

import time




# Stepping thru to get the trainer obj 

# See if I can get the num_worker settings too



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



def train_for_testing_pol_wt_freezing(trainer, num_iters=20, log_intervals=10, log_dir=None):
    true_start = time.time()

    results_dicts = []
    policy_weights_for_iters = []


    for i in range(num_iters):
        print(f"Starting training on iter {i + 1}...")
        start = time.time()
        
        result = trainer.train()
        results_dicts.append(result)

        copied_policy_wts_from_local_worker = deepcopy(trainer.get_weights())
        policy_weights.append(copied_policy_wts_from_local_worker)


        print(f"batch {i + 1}: took {time.time() - start} seconds")

        # if (i + 1) % log_intervals == 0:
        #     checkpoint = trainer.save(log_dir)
        #     print("checkpoint saved at", checkpoint)

    # TO ADD: save results_dicts, e.g. as csv

    print(f"Full training took {(time.time() - true_start) / 60.0} min")

    return results_dicts, policy_weights_for_iters


def pairwise_chk(wt_dict1, wt_dict2, team_name):
    """ assumes keys are the same for both dicts """
    d1 = wt_dict1[team_name]
    d2 = wt_dict2[team_name]

    return np.array([True if np.array_equal(d1[key], d2[key]) else False for key in d1]).all()

assert pairwise_chk(dt1, dt2, "blue") == False
assert pairwise_chk(dt1, dt2, "red") == True

assert pairwise_chk(dt2, dt3, "red") == False
assert pairwise_chk(dt2, dt3, "blue") == True

dt3 = wts_from_trainer_getwts[3]



if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

