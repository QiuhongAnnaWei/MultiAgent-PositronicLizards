import supersuit as ss
from pettingzoo.utils.conversions import to_parallel
import multiprocessing
import time
from stable_baselines3 import PPO
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from gym.spaces import Box


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


def get_policy_config(env, action_space, obs_space, method='red_blue'):
    """
    Gets some objects needed for instantiating a ray Trainer
    :param env: pettingzoo environment
    :param action_space: a gym Space (largest of all agents)
    :param obs_space: a gym Space (largest of all agents)
    :param method: [optional] split policies by color, species
    :return: policy_dict, policy_fn, observation_space.shape
    """
    team_1 = None
    team_2 = None

    if method == 'red_blue':
        team_1 = "red"
        team_2 = "blue"

    elif method == 'predator_prey':
        team_1 = "predator"
        team_2 = "prey"

    policy_dict = {team_1: (None, obs_space, action_space, dict()),
                   team_2: (None, obs_space, action_space, dict())}
    policy_fn = lambda agent_id, episode, **kwargs: team_1 if agent_id.startswith(team_1) else team_2

    return policy_dict, policy_fn
