import supersuit as ss
from pettingzoo.utils.conversions import to_parallel
import multiprocessing
import time
from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from gym.spaces import Box
import numpy as np


multiprocessing.set_start_method("fork")
# From supersuit docs: On MacOS with python>=3.8, need to use fork multiprocessing instead of spawn multiprocessing


def convert_to_sb3_env(env, num_cpus=1):
    """
    Converts a pettingzoo environment to a stable_baselines3 environment.

    Note: This function currently can only accept pettingzoo environments with the following conditions:
        1. No agent death or generation
        2. Each agent must have the same observation and action space

    :param env: a pettingzoo environment
    :return: a stable_baselines3 environment
    """
    sb3_env = ss.pettingzoo_env_to_vec_env_v1(to_parallel(env))
    sb3_env = ss.concat_vec_envs_v1(sb3_env, 4, num_cpus=num_cpus, base_class='stable_baselines3')
    return sb3_env


def train(sb3_env, model_class, policy_type="MlpPolicy", time_steps=1000, save_name=None):
    """
    Trains an agent of type model_class on the environment, sb3_env

    :param sb3_env: stable_baselines3 environment (I think it's a parallel one)
    :param model_class: PPO, DQN, HER, etc. See stable_baselines3
    :param policy_type: "MlpPolicy" or "CnnPolicy", see sb3 documentation
    :param time_steps: training steps
    :param save_name: try to prefix this with 'trained_policies/' to put them in that directory
    :return: a trained model
    """
    model = model_class(policy_type, sb3_env)   # Instantiates an RL algorithm (model_class) with an sb3 environment
    model.learn(total_timesteps=time_steps)
    if save_name:
        model.save(save_name)   # Saves the model as a zip
    return model


def evaluate_model(sb3_env, model, render=True, time_steps=1000):
    """

    :param sb3_env: stable_baselines3 environment
    :param model: a pre-trained model
    :param render: boolean, default True
    :param time_steps:
    :return: None
    """
    obs = sb3_env.reset()
    for _ in range(time_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = sb3_env.step(action)
        if render:
            sb3_env.render()
        # time.sleep(0.1)
        # print(rewards)
    sb3_env.close()


def auto_register_env_ray(env_name, env):
    """
    Registers a pettingzoo environment with ray
    :param env_name: desired name of environment in registry
    :param env: pettingzoo environment
    :return: None
    """
    def env_creator(config):
        penv = env.parallel_env(**config)
        penv = ss.pad_observations_v0(penv)
        penv = ss.pad_action_space_v0(penv)
        # penv = ss.normalize_obs_v0(penv, env_min=0, env_max=1)
        return penv

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))


def get_policy_config(action_space, obs_space, method='red_blue'):
    """
    Gets some objects needed for instantiating a ray Trainer
    :param env: pettingzoo environment
    :param action_space: a gym Space (largest of all agents)
    :param obs_space: a gym Space (largest of all agents)
    :param method: [optional] split policies by color, species
    :return: policy_dict, policy_fn
    """
    team_1 = None
    team_2 = None

    if method == 'red_blue':
        team_1 = "red"
        team_2 = "blue"

    elif method == 'predator_prey':
        team_1 = "predator"
        team_2 = "prey"

    # policies: {policy ids: (policy_cls, obs_space, act_space, config)}
    policy_dict = {team_1: (None, obs_space, action_space, dict()),
                   team_2: (None, obs_space, action_space, dict())}
    # policy_mapping_fn: map agent ids -> policy ids
    policy_fn = lambda agent_id, episode, **kwargs: team_1 if agent_id.startswith(team_1) else team_2

    return policy_dict, policy_fn


def train_ray_trainer(trainer, num_iters=100, log_intervals=10, log_dir=None):
    """
    Trains a Ray Trainer and saves checkpoints
    :param trainer: a Ray Trainer
    :param num_iters: (optional) number of training iterations
    :param log_intervals: (optional) saves a checkpoint for every 'log_intervals' training iterations
    :param log_dir: (optional) file path to save checkpoints
    :return: file path of the final checkpoint
    """
    checkpoint = None
    true_start = time.time()
    for i in range(num_iters):
        print(f"Starting training on batch {i}...")
        start = time.time()
        result = trainer.train()
        print(pretty_print(result))
        print(f"batch {i}: took {time.time() - start} seconds")
        if i % log_intervals == 0:
            checkpoint = trainer.save(log_dir)
            print("checkpoint saved at", checkpoint)
    print(f"Full training took {(time.time() - true_start) / 60.0} minutes")

    return checkpoint


def render_from_checkpoint(checkpoint, trainer, env, config, policy_fn):
    """
    Visualize from given checkpoint. 
    Reference: https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/render_rllib_leduc_holdem.py
    :param checkpoint: a file path to a checkpoint to load to generate visualizations
    :param trainer: trainer associated with the checkpoint
    :param env: pettingzoo env to use (e.g., adversarial_pursuit_v3)
    :param config: config dictionary for the environment (e.g. {"map_size":30})
    :param policy_fn: policy_fn returned from get_policy_config()
    :return: None
    """
    trainer.restore(checkpoint)
    env = env.env(**config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        if done:
            action = None
        else:
            agentpolicy = policy_fn(agent, None) # map agent id to policy id
            # print("get_policy: ", trainer.get_policy(agent))
            policy = trainer.get_policy(agentpolicy)
            batch_obs = {
                'obs': np.expand_dims(observation, 0) # (10,10,5) -> (1,10,10,5)
                # { 
                #     'observation': np.expand_dims(observation, 0),
                #     'action_mask': np.expand_dims(observation['action_mask'], 0)
                # }
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action
            # print(f"action={action}")
        env.step(action)
        env.render(mode='human')
    env.close()
