import supersuit as ss
from pettingzoo.utils.conversions import to_parallel
import multiprocessing
import time
from datetime import datetime
# from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy
from gym.spaces import Box
import numpy as np
# import pygame
# from pygame.locals import*
import PIL
from PIL import ImageDraw
import os
import cv2

multiprocessing.set_start_method("fork")


# From supersuit docs: On MacOS with python>=3.8, need to use fork multiprocessing instead of spawn multiprocessing


def convert_to_sb3_env(env, num_cpus=1):
    """
    DEPRECATED
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
    DEPRECATED
    Trains an agent of type model_class on the environment, sb3_env

    :param sb3_env: stable_baselines3 environment (I think it's a parallel one)
    :param model_class: PPO, DQN, HER, etc. See stable_baselines3
    :param policy_type: "MlpPolicy" or "CnnPolicy", see sb3 documentation
    :param time_steps: training steps
    :param save_name: try to prefix this with 'trained_policies/' to put them in that directory
    :return: a trained model
    """
    model = model_class(policy_type, sb3_env)  # Instantiates an RL algorithm (model_class) with an sb3 environment
    model.learn(total_timesteps=time_steps)
    if save_name:
        model.save(save_name)  # Saves the model as a zip
    return model


def evaluate_model(sb3_env, model, render=True, time_steps=1000):
    """
    DEPRECATED
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


def get_num_agents(env, env_config):
    """
    Gets the number of agents in an environment with a given config
    :param env: a pettingzoo environment
    :param env_config: a dictionary of arguments for the environment (e.g. map_size=30)
    :return: a dictionary in the form {"team_name": count}
    """
    e = env.env(**env_config)
    team_names = list(set([a.split('_')[0] for a in e.possible_agents]))
    agent_counts = dict()
    for team_name in team_names:
        agent_counts[team_name] = max([int(a.split('_')[-1]) for a in e.possible_agents if a.startswith(team_name)]) + 1
    return agent_counts


def get_policy_config(action_space, obs_space, team_data):
    """
    Gets some objects needed for instantiating a ray Trainer
    :param action_space: a gym Space (largest of all agents)
    :param obs_space: a gym Space (largest of all agents)
    :param team_data: a list of TeamPolicyConfig objects (usually one per 'team' e.g. 'red')
    :return: policy_dict, policy_fn
    """

    def policy_fn(agent_id, episode, **kwargs):
        subdict = policy_fn_dict[agent_id.split('_')[0]]
        if isinstance(subdict, str):
            return subdict
        else:
            return agent_id

    policy_dict = dict()
    policy_fn_dict = dict()

    for team in team_data:
        name = team.team_name
        method = team.method
        count = team.count

        policy_dict_updates = dict()
        if method == 'shared':
            policy_dict_updates[name + "_shared"] = (None, obs_space, action_space, dict())
            policy_fn_dict[name] = name + "_shared"
        elif method == 'split':
            policy_fn_dict[name] = None
            for i in range(count):
                policy_dict_updates[f"{name}_{i}"] = (None, obs_space, action_space, dict())
        if team.random_action_team:
            # Override all team-policies with random, untrainable ones.
            for k in policy_dict_updates:
                policy_dict_updates[k] = PolicySpec(policy_class=RandomPolicy)
        policy_dict.update(policy_dict_updates)

    return policy_dict, policy_fn


def get_trainer_config(env_name, policy_dict, policy_fn, env_config, conv_filters=None, gpu=True, create_env_on_driver=False, **kwargs):
    """
    Gets a config dictionary for a Ray Trainer
    :param env_name: the Ray-registered environment name (e.g. 'adversarial-pursuit')
    :param policy_dict: policy_dict from get_policy_config ({policy_id: (None, obs_space, action_space, dict())})
    :param policy_fn: policy_fn from get_policy_config (agent_id maps to policy_id)
    :param env_config: a dictionary of arguments for the environment (e.g. map_size=30)
    :param conv_filters: [optional] a list of convolutional filters (out_channels, kernel, stride)
    :param kwargs: any other keyword arguments you want to put into the trainer config dict
    :return: a config dict for a Ray Trainer
    """
    convs = {"adversarial-pursuit": [[13, 10, 1]],
             "battle": [[21, 13, 1]],
             "battlefield": [[21, 13, 1]],
             "tiger-deer": [[9, 9, 1]],
             "combined-arms": [[25, 13, 1]]}
    conv_activation = "relu" #  "tanh", "relu", "swish" (or "silu")

    trainer_config = {
        "env": env_name,
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn": policy_fn
        },
        "model": {
            "conv_filters": convs[env_name] if conv_filters is None else conv_filters,
            "conv_activation": conv_activation
        },
        "env_config": env_config,
        "rollout_fragment_length": 500
    }

    # Train policies that aren't RandomPolicy instances:
    policies_to_train = []
    for policy_name, p in policy_dict.items():
        if isinstance(p, PolicySpec) and p[0] == RandomPolicy:
           continue
        policies_to_train.append(policy_name)
    trainer_config["multiagent"]["policies_to_train"] = policies_to_train

    if create_env_on_driver:
        trainer_config["create_env_on_driver"] = True
        # according to Ray Rllib video tutorial, setting this to true 
        # facilitates evaluation after training (at least with their Trainer.evaluate() method)
        # haven't tried this yet though. keeping it set to F by default since tt's the current default

    if gpu:
        trainer_config["num_gpus"] = 1
        trainer_config["num_gpus_per_worker"] = 0.5
    else:  # For CPU training only:
        trainer_config["num_gpus"] = 0
        # trainer_config["num_workers"] = 2
        # trainer_config["num_cpus_per_worker"] = 16

    trainer_config.update(kwargs)
    return trainer_config


def train_ray_trainer(trainer, num_iters=100, log_intervals=10, log_dir=None, 
        render=False, env=None, env_config=None, policy_fn=None, max_render_iter=10000, is_battle=False):
    """
    Trains a Ray Trainer and saves checkpoints
    :param trainer: a Ray Trainer
    :param num_iters: (optional) number of training iterations
    :param log_intervals: (optional) saves a checkpoint for every 'log_intervals' training iterations
    :param log_dir: (optional) file path to save checkpoints
    :param render: (optional) for rendering after saving checkpoint. If True, env, env_config, policy_fn must be set.
    :return: file path of the final checkpoint
    """
    checkpoint = None
    true_start = time.time()
    for i in range(num_iters):
        print(f"Starting training on batch {i + 1}...")
        start = time.time()
        result = trainer.train()
        print(pretty_print(result))
        print(f"batch {i + 1}: took {time.time() - start} seconds")
        if (i + 1) % log_intervals == 0:
            checkpoint = trainer.save(log_dir)
            print("checkpoint saved at", checkpoint)
            if render and (env is not None) and (env_config is not None) and (policy_fn is not None):
                render_from_checkpoint(checkpoint, trainer, env, env_config, policy_fn, max_iter=max_render_iter, savefile=True, is_battle=is_battle)
    print(f"Full training took {(time.time() - true_start) / 60.0} minutes")

    trainer.stop()
    return checkpoint


def render_from_checkpoint(checkpoint, trainer, env, env_config, policy_fn, max_iter=2 ** 8, savefile=False, is_battle=False, logname = None):
    """
    Visualize from given checkpoint.
    Reference: https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/render_rllib_leduc_holdem.py
    :param checkpoint: a file path to a checkpoint to load to generate visualizations; if None, expect trainer to already been restored
    :param trainer: trainer associated with the checkpoint
    :param env: pettingzoo env to use (e.g., adversarial_pursuit_v3)
    :param env_config: config dictionary for the environment (e.g. {"map_size":30})
    :param policy_fn: policy_fn returned from get_policy_config()
    :param is_battle: if set to true, will add captions of team hp (3rd and 5th channels of state)
    :param logname: filepath to log to (without extension); e.g. 'logs/battle/evaluation/12345'
    :return: None
    """
    if checkpoint:
        trainer.restore(checkpoint)
    # else: # for path
    #     checkpoint = f"logs/battle/evaluation/{logname}"
    #     if not os.path.exists(os.path.split(checkpoint)[0]): os.makedirs(os.path.split(checkpoint)[0])
    if logname is None:
        logname = checkpoint
    else:
        if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env.reset()
    if savefile:
        diff_frame_list = []
        width,height,img = None, None, None
    i = 0
    for agent in env.agent_iter(max_iter=max_iter):
        if i % 1000 == 0:
            print(f"rendering: iteration {i} ...")
        observation, reward, done, info = env.last() # (observation[:,:,3/4]==0).sum()
        if done:
            action = None
        else:
            agentpolicy = policy_fn(agent, None)  # map agent id to policy id
            policy = trainer.get_policy(agentpolicy)
            batch_obs = { 'obs': np.expand_dims(observation, 0)} # (10,10,5) -> (1,10,10,5)
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action
        try:
            s = env.state() # (map_size, map_size, 5)
        except:
            print(f"At {i}: one team eliminated - env.agents = {env.agents}") 
            break
        # out = False
        env.step(action)
        if savefile:
            img2 = PIL.Image.fromarray(env.render(mode='rgb_array'))
            if img is None:
                width, height = img2.width, img2.height
                img = np.zeros((width, height))
            if np.array_equal(np.array(img), np.array(img2)) == False:
                img = img2.copy()
                ImageDraw.Draw(img2).text( (2, height-10), f"iter={i}",  (0, 0, 0))
                if is_battle:
                    ImageDraw.Draw(img2).text( (2, 13), f"HP={str(round((s[:,:,2]).sum(), 2))}",  (0, 0, 0)) 
                    ImageDraw.Draw(img2).text( (width-55, 13), f"HP={str(round((s[:,:,4]).sum(), 2))}",  (0, 0, 0)) # "{:.4f}".format()
                diff_frame_list.append(img2)
        else:
            pass
            # env.render(mode='human')
            # ANNA: This code fixes my visualization
            # for event in pygame.event.get():
            #     time.sleep(0.1)
            #     if event.type == pygame.QUIT:
            #         out = True
        # if out:  break
        i += 1
    env.close()
    if savefile:
        save_path = f"{logname}.gif"
        log(f"{logname}.txt", f"\n# Saving gif to: {save_path}")
        diff_frame_list[0].save(save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)
        save_path = f"{logname}.mp4"
        log(f"{logname}.txt", f"\n# Saving video to: {save_path}\n")
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
        for i, image in enumerate(diff_frame_list):
            video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        for i in [0, len(diff_frame_list)-1]:
            diff_frame_list[i].save(f"{logname}_{i}.jpg")


def evaluate_policies(checkpoint, trainer, env, env_config, policy_fn, gamma=0.99, max_iter=100):
    """
    Evaluates a set of policies on an environment
    :param checkpoint: a file path to a checkpoint to load to generate visualizations; if None, expect trainer to already been restored
    :param trainer: trainer associated with the checkpoint
    :param env: pettingzoo env to use (e.g., adversarial_pursuit_v3)
    :param env_config: config dictionary for the environment (e.g. {"map_size":30})
    :param policy_fn: policy_fn returned from get_policy_config()
    :param gamma: gamma
    :param max_iter: number of iterations to evaluate policies
    :return: dictionary of cumulative discounted rewards per each policy in the trainer
    """
    if checkpoint:
        trainer.restore(checkpoint)
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    rewards = dict()
    rewards_log = dict()
    gamma_mul = 1.0 * gamma
    first_agent = None

    env.reset()
    for agent in env.agent_iter(max_iter=max_iter):
        if first_agent and first_agent == agent:
            gamma_mul *= gamma
        elif not first_agent:
            first_agent = agent

        observation, reward, done, info = env.last()
        if done:
            action = None
        else:
            agent_policy = policy_fn(agent, None)  # map agent id to policy id
            policy = trainer.get_policy(agent_policy)
            batch_obs = {
                'obs': np.expand_dims(observation, 0)  # (10,10,5) -> (1,10,10,5)
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action

            if agent_policy in rewards:
                rewards[agent_policy] += reward * gamma_mul
                rewards_log[agent_policy].append(rewards[agent_policy])
            else:
                rewards[agent_policy] = reward * gamma_mul
                rewards_log[agent_policy] = [reward * gamma_mul]
        env.step(action)

    env.close()
    return rewards, rewards_log


def pettingzoo_peek(env, env_config):
    """
    For taking a peek at a pettingzoo environment
    :param env: pettingzoo env
    :param env_config: config dictionary for the environment (e.g. {"map_size":30})
    :return: None
    """
    e = env.env(**env_config)
    e.reset()
    e.render()
    input("Press Enter to close window...")


def log(fp, msg, toprint=True):
    """
    For printing and appending to log file
    :param msg: a singular message or an iterable of message
    """
    if toprint:
        if isinstance(msg, list):
            for m in msg:
                print(m)
        else:
            print(msg)
    with open(fp, 'a') as f:
        if isinstance(msg, list):
            f.writelines(msg)
        else:
            f.write(msg)
