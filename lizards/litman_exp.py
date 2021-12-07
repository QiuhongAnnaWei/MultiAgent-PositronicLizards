from ray.tune.trainable import Trainable
from experiments import *
from main_utils import *
from early_stopping import ExperimentPlateauOrMaxStepsStopper
from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np
from ray import tune
import json
import random
from pathlib import Path
from copy import deepcopy
import pandas as pd

convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}

gen_dyn_info_1inf = {"timestamp": None,
                    "log_dir": Path("./logs/pol_freezing/1_inf_Dec7"),
                    "wts_log_path": Path("./logs/pol_freezing/1_inf_Dec7").joinpath("log_1_inf_pol_wts.csv")}


exp_stopper = ExperimentPlateauOrMaxStepsStopper("episode_reward_mean")

policy_weights_for_iters = []

class BattleTrainerCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()

        self.teams = ("red", "blue") 
        self.curr_trainable_policies = {"blue"} # impt tt this be set to blue

        self.total_iterations = 0
        self.top = 4 # Minimum number of iterations that the standard deviation must be within a threshold.
        self.std_cutoff = 0.1
        self.max_iterations = 400
        self.min_iters_until_stop = 30
        self.policy_rewards = []

    def get_other_team(self): 
        # Returns str
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team

    def switch_teams(self, trainer):
        team_to_freeze = self.curr_trainable_policies
        team_to_train = self.get_other_team()  # this will be str
        self.curr_trainable_policies = {team_to_train}
        def _set(worker):
                print(f"_set has been called; self.curr_trainable_policies are {self.curr_trainable_policies}")
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)

        trainer.workers.foreach_worker(_set)
        

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train(), so that the first time this is called, trainer.iteration will == 1"""
        copied_policy_wts_from_local_worker = deepcopy(trainer.get_weights())
        policy_weights_for_iters.append(copied_policy_wts_from_local_worker)
        
        # with open(gen_dyn_info_1inf["wts_log_path"], "w") as f:
        # with open("/users/yh31/scratch/projects/MultiAgent-PositronicLizards/lizards/logs/pol_freezing", "w") as f:
        #     f.write(json.dumps(deepcopy(trainer.get_weights())))

        team = list(self.curr_trainable_policies)[0]
        if team == "red":
            red_policy_mean = result["policy_reward_mean"]['red']
            self.total_iterations += 1 
            self.policy_rewards.append(red_policy_mean)
            if (self.total_iterations > self.top and np.std(self.policy_rewards[-self.top:]) < self.std_cutoff) or self.min_iters_until_stop <= self.total_iterations:
                # 1. If the standard deviation within the last self.top iterations is smaller than the cutoff
                # 2. Reach max number of timesteps
                self.total_iterations = 0
                self.policy_rewards = []
                self.switch_teams(trainer)
        else:
            self.switch_teams(trainer) 

def BattleTrainerInfinite(*args, map_size=19, num_iters = 30, test = False):
    """
    Begins the Training on the Battle Trainer
    """
    env_name = "battle"

    training_setup = {"num_iters": num_iters, 
                     "log_intervals": 4,
                     "log_dir": 'logs/BA_testing'}

    env_fn = env_directory[env_name]
    env_config = {'map_size': map_size} 
    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    def BA_pol_mapping_fn(agent_id, episode, worker, **kwargs):
        return "red" if agent_id.startswith("red") else "blue"

    ray_trainer_config = {
        "callbacks": BattleTrainerCallback,
        "multiagent": {
            "policies": {"red": (None, obs_space, action_space, dict()),
                         "blue": (None, obs_space, action_space, dict())},
            "policy_mapping_fn": BA_pol_mapping_fn,
            #"policies_to_train": None, # Red, and __only__ red, will be trained on first iter
        },
        # TO DO: try to come up with a better way to make clear which team will be trained on first iter in callback and here

        "env": env_name,
        "model": {
            "conv_filters": convs[env_name]
        },
        "env_config": env_config,
        "create_env_on_driver": True, # potentially disable this?
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }
    
    if test:
        ray_trainer_config["train_batch_size"] = 1000

    #tune.run(ppo.PPOTrainer, config = ray_trainer_config, stop= exp_stopper)
    if test:
        tune.run(ppo.PPOTrainer, config = ray_trainer_config, keep_checkpoints_num=1, checkpoint_score_attr="accuracy", stop={"training_iteration": 1})
    else:
        tune.run(ppo.PPOTrainer, config = ray_trainer_config, keep_checkpoints_num=1, checkpoint_score_attr="accuracy")

    # save logged pol wts
    pd.DataFrame(policy_weights_for_iters).to_csv(gen_dyn_info_1inf["wts_log_path"])



    #trainer = ppo.PPOTrainer(config=ray_trainer_config)
    #train_for_pol_wt_freezing(trainer, timestamp=timestamp)

    # TO DO: Add eval stuff!

if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    log_dir = gen_dyn_info_1inf["log_dir"]
    if not log_dir.is_dir(): log_dir.mkdir()

    gen_dyn_info_1inf["wts_log_path"].touch(exist_ok=True)

    BattleTrainerInfinite()
