import supersuit as ss
from pettingzoo.utils.conversions import to_parallel
import multiprocessing
import time

multiprocessing.set_start_method("fork")


def convert_to_sb3_env(env):
    sb3_env = ss.pettingzoo_env_to_vec_env_v1(to_parallel(env))
    sb3_env = ss.concat_vec_envs_v1(sb3_env, 4, num_cpus=1, base_class='stable_baselines3')
    return sb3_env


def train(env, model_class, policy_type="MlpPolicy", time_steps=1000, save_name=None):
    model = model_class(policy_type, env)
    model.learn(total_timesteps=time_steps)
    if save_name:
        model.save(save_name)
    return model


def evaluate_model(env, model, render=False, time_steps=100):
    obs = env.reset()
    for _ in range(time_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if render:
            env.render()
        # time.sleep(0.1)
        # print(rewards)
    env.close()
