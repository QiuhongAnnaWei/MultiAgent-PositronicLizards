from main_utils import *
from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3


env_directory = {'adversarial-pursuit': adversarial_pursuit_v3, 'tiger-deer': tiger_deer_v3, 'battle': battle_v3,
                 'battlefield': battlefield_v3}


def experiment_1():
    battlefield = convert_to_sb3_env(battlefield_v3.env())
    model = train(battlefield, PPO, time_steps=100000, save_name='trained_policies/battlefield-10e5-V1')
    evaluate_model(battlefield, model, render=True)


if __name__ == "__main__":
    experiment_1()
