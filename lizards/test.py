import numpy as np

from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3
from simple_rl.agents import QLearningAgent
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
import supersuit as ss
from pettingzoo.utils.conversions import to_parallel


def random_AP():
    env = adversarial_pursuit_v3.env(map_size=15)
    env.reset()
    for agent in env.agent_iter(max_iter=5000):
        observation, reward, done, info = env.last()
        if done:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        env.render(mode='human')
    env.close()


def random_TD():
    env = tiger_deer_v3.env()
    env.reset()
    for agent in env.agent_iter(max_iter=500):
        observation, reward, done, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)
        env.render(mode='human')
    env.close()


def simple_agent_test_AP():
    env = adversarial_pursuit_v3.env(map_size=15)
    ag = QLearningAgent(list(range(13)))
    agents = {'predator_0': ag}

    env.reset()
    for agent in env.agent_iter(max_iter=5000):
        observation, reward, done, info = env.last()
        if done:
            action = None
        else:
            if agent in agents:
                action = agents[agent].act(observation, reward)
            else:
                action = env.action_space(agent).sample()
        env.step(action)
        env.render(mode='human')
    env.close()


def test():
    env = adversarial_pursuit_v3.env(map_size=15)
    env.reset()
    for agent in env.agent_iter(max_iter=10):
        observation, reward, done, info = env.last()
        # print(observation)
        # print(len(observation))
        # print(np.shape(observation))
        # print(info)
        # print(agent)
        action = env.action_space(agent).sample()
        env.step(action)
        print(agent)

    env.close()


def test2():
    env = adversarial_pursuit_v3.env(map_size=15)
    env = to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v0(env, 8, num_cpus=1, base_class='stable_baselines3')
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=2000)
    model.save('policy')


if __name__ == "__main__":
    test2()
