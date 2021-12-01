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


# policy freezing and unfreezing stuff
# ====================================
# For info on Rllib callbacks: https://docs.ray.io/en/latest/_modules/ray/rllib/agents/callbacks.html#DefaultCallbacks.on_train_result

# Ray trainer code: https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/agents/trainer.py

# https://github.com/ray-project/ray/issues/6669
# https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/examples/self_play_league_based_with_open_spiel.py
# https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/examples/self_play_with_open_spiel.py


# TO DO NOTE: We should try to switch over to the new training conv stuff that Qiuhong / Anna has made before actually training the experiments
convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}


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


""" 
Assumptions:
* Code below was designed for 2-team environments
* names of policies == prefixes of teams (i.e., either "predator" or "prey")
"""

class TwoTeamOnTrainResultMixin:   
    def get_other_team(self):
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
        (Iteration 0 is the state when *no* training has been done.)"""
        
        curr_iter = trainer.iteration
        if curr_iter > self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.interval_len == 0):
            team_to_freeze = self.curr_trainable_policies
            team_to_train = self.get_other_team()
            print(team_to_train)
            self.curr_trainable_policies = {team_to_train}

            print(f"On iter {curr_iter + 1}, {team_to_freeze} will be frozen, {team_to_train} will be trained")

            def _set(worker):
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)
            
            trainer.workers.foreach_worker(_set)


# APTCallback stands for: Alternating Policy Training Callback
class APTCallback_BF(DefaultCallbacks):
    # We'll make this cleaner by using the mixin once we're sure that this works

    def __init__(self):
        super().__init__()

        self.teams = ("red", "blue") 
        self.curr_trainable_policies = {"red"} 
        # Start with red being the one whose policy's being trained

        self.burn_in_iters = 4 
        # How many iterations to train normally for, before starting alternating policy training/freezing regimen

        self.interval_len = 4
        # The point of this, recall, is to train one team while keeping the other frozen for regular intervals
        # interval_len := number of iterations that each interval consists in


    def get_other_team(self): 
        # Returns str
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team

    def on_train_result(self, *, trainer, result, **kwargs):
                """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
        (Iteration 0 is the state when *no* training has been done.)"""
        
        curr_iter = trainer.iteration
        print(f"Just finished train iter {curr_iter}")
        if curr_iter > self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.interval_len == 0):
            team_to_freeze = self.curr_trainable_policies
            team_to_train = self.get_other_team()  # this will be str
            self.curr_trainable_policies = {team_to_train}

            print(f"On iter {curr_iter + 1}, {team_to_freeze} will be frozen, {team_to_train} will be trained")

            def _set(worker):
                print(f"_set has been called; self.curr_trainable_policies are {self.curr_trainable_policies}")
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)
            
            trainer.workers.foreach_worker(_set)

# APTCallback stands for: Alternating Policy Training Callback
class APTCallback_BF_to_wrap(DefaultCallbacks):
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

class APTCallback_AP(DefaultCallbacks):

    def __init__(self):
        super().__init__()

        self.teams = ("predator", "prey") 
        self.curr_trainable_policies = {"predator"} 
        # Start with predator being the one whose policy's being trained

        self.burn_in_iters = 0 
        # How many iterations to train normally for, before starting alternating policy training/freezing regimen

        self.interval_len = 3
        # The point of this, recall, is to train one team while keeping the other frozen for regular intervals
        # interval_len := number of iterations that each interval consists in

    
    def get_other_team(self): 
        # Returns str
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1. 
        (Iteration 0 is the state when *no* training has been done.)"""
        
        curr_iter = trainer.iteration
        if curr_iter > self.burn_in_iters and ((curr_iter - self.burn_in_iters) % self.interval_len == 0):
            team_to_freeze = self.curr_trainable_policies
            team_to_train = self.get_other_team()  # this will be str
            self.curr_trainable_policies = {team_to_train}

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
def AP_alternating_pol_train_PROTOTYPE(map_size=25, *args):
    """ This uses policy sharing for each of the teams to keep things simple """

    # training_setup = {"num_iters": 4,
    #                  # "log_intervals": 1,
    #                  "log_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/AP_alternating_policy_freezing_proof_concept')}

    conv_filters = None

    env_name = "adversarial-pursuit"
    env_fn = env_directory[env_name]
    env_config = {'map_size': map_size} # min map sz is 7 for AP

    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    def pol_mapping_fn(agent_id, episode, worker, **kwargs):
        return "predator" if agent_id.startswith("predator") else "prey"
    
    ray_trainer_config = {

        "callbacks": APTCallback_AP, # IMPT

        "env": env_name,
        "multiagent": {
            "policies": {"predator": (None, obs_space, action_space, dict()),
                         "prey": (None, obs_space, action_space, dict())},
            "policy_mapping_fn": pol_mapping_fn
        },
        "model": {
            "conv_filters": convs[env_name] if conv_filters is None else conv_filters
        },
        "env_config": env_config,
        "create_env_on_driver": True, # potentially disable this?
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    trainer = ppo.PPOTrainer(config=ray_trainer_config)

    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Some_checkpoint_filename')
    #checkpoint = train_ray_trainer(trainer, num_iters=training_setup["train_iters"], 
                                           # log_intervals=training_setup['log_intervals'], 
                                            #log_dir=training_setup["log_dir"])
    checkpoint = train_ray_trainer(trainer)


    # if checkpoint:
        # render_from_checkpoint(checkpoint, trainer, env_fn, env_config, policy_fn)



def BF_alternating_pol_training_PROTOTYPE(map_size=50, *args):
    # min map sz for BF is 46

    training_setup = {"num_iters": 16,
                     "log_intervals": 4,
                     "log_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/BF_apt_PROTOTYPE')}

    env_name = "battlefield"
    env_fn = env_directory[env_name]
    env_config = {'map_size': map_size} 

    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    def pol_mapping_fn(agent_id, episode, worker, **kwargs):
        return "red" if agent_id.startswith("red") else "blue"
    
    class APTCallback_BF_test_1(APTCallback_BF_to_wrap):
        def __init__(self):
            super().__init__({"red": 1, "blue": 30})

    ray_trainer_config = {

        # "callbacks": APTCallback_BF, # IMPT # Testing new callback below (Eli):
        "callbacks": APTCallback_BF_test_1,

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

    train_ray_trainer(trainer, **training_setup)



if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    BF_alternating_pol_training_PROTOTYPE()

    # AP_alternating_pol_train_PROTOTYPE()



