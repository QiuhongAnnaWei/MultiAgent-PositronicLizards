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
import pygame
# from pygame.locals import*
import PIL
import os

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
    model = model_class(policy_type, sb3_env)   # Instantiates an RL algorithm (model_class) with an sb3 environment
    model.learn(total_timesteps=time_steps)
    if save_name:
        model.save(save_name)   # Saves the model as a zip
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


def get_policy_config(action_space, obs_space, team_1_name='red', team_2_name='blue',
                      team_1_policy='shared', team_2_policy='shared', team_1_count=None, team_2_count=None):
    """
    Gets some objects needed for instantiating a ray Trainer
    :param action_space: a gym Space (largest of all agents)
    :param obs_space: a gym Space (largest of all agents)
    :param team_1_name: 'prey', 'red', etc.
    :param team_2_name: 'predator', 'blue', etc.
    :param team_1_policy: 'shared' (one per team) or 'split' (one per agent)
    :param team_2_policy: 'shared' (one per team) or 'split' (one per agent)
    :param team_1_count: [optional] number of policies (necessary for 'split' param only)
    :param team_2_count: [optional] number of policies (necessary for 'split' param only)
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

    for team_name, team_policy, team_count in [(team_1_name, team_1_policy, team_1_count), (team_2_name, team_2_policy, team_2_count)]:
        if team_policy == 'shared':
            policy_dict[team_name+"_shared"] = (None, obs_space, action_space, dict())
            policy_fn_dict[team_name] = team_name+"_shared"
        elif team_policy == 'split':
            policy_fn_dict[team_name] = None
            for i in range(team_count):
                policy_dict[f"{team_name}_{i}"] = (None, obs_space, action_space, dict())

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


def render_from_checkpoint(checkpoint, trainer, env, env_config, policy_fn, max_iter=2**8, savefile=False):
    """
    Visualize from given checkpoint.
    Reference: https://github.com/Farama-Foundation/PettingZoo/blob/master/tutorials/render_rllib_leduc_holdem.py
    :param checkpoint: a file path to a checkpoint to load to generate visualizations
    :param trainer: trainer associated with the checkpoint
    :param env: pettingzoo env to use (e.g., adversarial_pursuit_v3)
    :param env_config: config dictionary for the environment (e.g. {"map_size":30})
    :param policy_fn: policy_fn returned from get_policy_config()
    :return: None
    """
    trainer.restore(checkpoint)
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    i = 0
    env.reset()

    if savefile:
        width, height = 240, 255
        img = np.zeros((width, height))
        diff_frame_list = []
        import cv2
        save_path = os.path.join(os.path.split(checkpoint)[0], f'{os.path.split(checkpoint)[1]}.mp4')
        print("\n# Saving video to:", save_path)
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    for agent in env.agent_iter(max_iter=max_iter):
        observation, reward, done, info = env.last()
        if done:
            action = None
        else:
            agentpolicy = policy_fn(agent, None) # map agent id to policy id
            policy = trainer.get_policy(agentpolicy)
            batch_obs = {
                'obs': np.expand_dims(observation, 0) # (10,10,5) -> (1,10,10,5)
            }
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action
        env.step(action)

        out = False
        if savefile:
            img2 = PIL.Image.fromarray(env.render(mode='rgb_array'))
            if np.array_equal(np.array(img),np.array(img2)) == False:
                diff_frame_list.append(img2)
            img = img2
            # video.write(cv2.cvtColor(np.array( PIL.Image.fromarray(env.render(mode='rgb_array')) ), cv2.COLOR_RGB2BGR))
            # if (i-1) % (env.num_agents) == 0: #33 (0, 34, 67, 100, 133, 166, 199, 232 )
            #     frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
        else:
            env.render(mode='human')
            for event in pygame.event.get():
                time.sleep(0.1)
                if event.type == pygame.QUIT:
                    out = True
        if out: break
        i += 1
    env.close()
    if savefile:
        save_path = os.path.join(os.path.split(checkpoint)[0], f'{os.path.split(checkpoint)[1]}.gif')
        print("\n# Saving gif to:", save_path)
        diff_frame_list[0].save(save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)
        # for i, image in enumerate(diff_frame_list):
        #     video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        #     diff_frame_list[i].save(os.path.join(os.path.split(checkpoint)[0], f'{os.path.split(checkpoint)[1]}_{i}.jpg'))


def evaluate_policies(checkpoint, trainer, env, env_config, policy_fn, gamma=0.99, max_iter=500):
    """
    Evaluates a set of policies on an environment
    :param checkpoint: a file path to a checkpoint to load to generate visualizations
    :param trainer: trainer associated with the checkpoint
    :param env: pettingzoo env to use (e.g., adversarial_pursuit_v3)
    :param env_config: config dictionary for the environment (e.g. {"map_size":30})
    :param policy_fn: policy_fn returned from get_policy_config()
    :param gamma: gamma
    :param max_iter: number of iterations to evaluate policies
    :return: dictionary of cumulative discounted rewards per each policy in the trainer
    """
    trainer.restore(checkpoint)
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    rewards = dict()
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
            else:
                rewards[agent_policy] = reward * gamma_mul
        env.step(action)

    env.close()
    return rewards
