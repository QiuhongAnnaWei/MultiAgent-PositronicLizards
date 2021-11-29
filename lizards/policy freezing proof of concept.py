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


class AlternatingPolicyTrainCallback(DefaultCallbacks):
    """ 
    Assumptions being made:
    * This proof of concept prototype is specialized to the Adversarial Pursuit environment
    * names of policies == prefixes of teams (i.e., either "predator" or "prey")
    """

    def __init__(self):
        super().__init__()

        self.teams = ("predator", "prey") # to abstract later
        self.curr_trainable_policies = {"predator"} 
        # Start with predator being the one whose policy's being trained


        self.interval_len = 5  
        # The point of this, recall, is to train one team while keeping the other frozen for regular intervals
        # interval_len := number of iterations that each interval consists in

    def get_other_team(self):
        # this will need to be made more robust later
        other_team = self.teams[0] if self.teams[0] not in self.curr_trainable_policies else self.teams[1]
        return other_team
        

    def on_train_result(self, *, trainer, result, **kwargs):
        """ will be called at the end of Trainable.train()"""
        
        curr_iter = trainer.iteration
        if curr_iter > 0 and curr_iter % self.interval_len == 0:
            team_to_freeze = self.curr_team_being_trained
            team_to_train = self.get_other_team()

            print(f"Iter {curr_iter}: Freezing {team_to_freeze} and training {team_to_train}")

            def _set(worker):
                # Note: `_set` must be enclosed in `on_train_result`!
                worker.set_policies_to_train(self.curr_trainable_policies)
            
            trainer.workers.foreach_worker(_set)




# env-specific things
# ===================
def ray_AP_alternating_pol_freezing_PROTOTYPE(map_size=7, *args, gpu=True):
    """ This uses policy sharing for each of the teams to keep things simple """

    training_setup = {"train_iters": 4,
                      "log_intervals": 1,
                      "log_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/AP_alternating_policy_freezing_proof_concept')}

    env_name = "adversarial-pursuit"
    env_fn = env_directory[env_name]
    env_config = {'map_size': map_size, 'max_cycles': 5000} # min map sz is 7 for AP

    action_space, obs_space = env_spaces[env_name]["action_space"], env_spaces[env_name]["obs_space"]

    pol_mapping_fn = lambda agent_id: "predator" if agent_id.startswith("predator") else "prey"
    
    ray_trainer_config = {

        "callbacks": AlternatingPolicyTrainCallback, # IMPT

        "env": env_name,
        "multiagent": {
            "policies": {"predator": (None, obs_space, action_space, dict()),
                         "prey": (None, obs_space, action_space, dict())}
            "policy_mapping_fn": pol_mapping_fn
        },
        "model": {
            "conv_filters": convs[env_name] if conv_filters is None else conv_filters
        },
        "env_config": env_config,
        "rollout_fragment_length": 100,
        "create_env_on_driver": True, # potentially disable this?
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    trainer = ppo.PPOTrainer(config=ray_trainer_config)

    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Some_checkpoint_filename')
    checkpoint = train_ray_trainer(trainer, num_iters=training_setup["train_iters"], 
                                            log_intervals=training_setup['log_intervals'], 
                                            log_dir=training_setup["log_dir"])


    # if checkpoint:
        # render_from_checkpoint(checkpoint, trainer, env_fn, env_config, policy_fn)



if __name__ == "__main__":
    ray_AP_alternating_pol_freezing_PROTOTYPE()










