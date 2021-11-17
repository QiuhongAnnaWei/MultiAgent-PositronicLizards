from main_utils import *
from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.pg as pg


env_directory = {'adversarial-pursuit': adversarial_pursuit_v3, 'tiger-deer': tiger_deer_v3, 'battle': battle_v3,
                 'battlefield': battlefield_v3}


def experiment_1():
    battlefield = convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0))
    model = train(battlefield, PPO, time_steps=10000, save_name='trained_policies/battlefield-10e4-V1')
    evaluate_model(battlefield, model)


def view_results():
    evaluate_model(convert_to_sb3_env(battlefield_v3.env(dead_penalty=-10.0)),
                   PPO.load('trained_policies/battlefield-10e6-V1'))


def ray_experiment_1():
    auto_register_env_ray("battle", battle_v3)
    policy_dict, policy_fn, obs_shape = get_policy_config(battle_v3)

    trainer = ppo.PPOTrainer(env='battle', config={
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn": policy_fn
        },
        "model": {
            "conv_filters": [
                [21, 13, 1]
            ]
        },
        "env_config": {
            "map_size": 12
        },
        "num_gpus": 1
    })

    for i in range(10):
        print(trainer.train())


if __name__ == "__main__":
    ray_experiment_1()
