from ray.tune.trainable import Trainable
from experiments import *
from main_utils import *
from early_stopping import ExperimentPlateauOrMaxStepsStopper
from ray.rllib.agents.callbacks import DefaultCallbacks
import numpy as np
from ray import tune

from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec

convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}

def BattleTrainerRandom(*args, map_size=19):
    """
    Begins the Training on the Battle Trainer
    """
    env_name = "battle"

    # training_setup = {"num_iters": num_iters, 
    #                  "log_intervals": 20,
    #                  "log_dir": 'logs/BA_randomized'}

    env_config = {'map_size': map_size} 
    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    def BA_pol_mapping_fn(agent_id, episode, worker, **kwargs):
        return "red" if agent_id.startswith("red") else "blue"

    ray_trainer_config = {
        "multiagent": {
            "policies": {"red": (None, obs_space, action_space, dict()),
                         "blue": PolicySpec(policy_class=RandomPolicy)},
            "policy_mapping_fn": BA_pol_mapping_fn,
            "policies_to_train": ["red"],
        },

        "env": env_name,
        "model": {
            "conv_filters": convs[env_name]
        },
        "env_config": env_config,
        "num_workers": 4,
        "create_env_on_driver": True, # potentially disable this?
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    # ray_trainer_config["train_batch_size"] = 2000
    tune.run(ppo.PPOTrainer, name="red_train_against_blue_rand_120iters_mapsz19_battle", keep_checkpoints_num=3, config = ray_trainer_config, stop={"training_iteration": 120})


if __name__ == "__main__":
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    BattleTrainerRandom()