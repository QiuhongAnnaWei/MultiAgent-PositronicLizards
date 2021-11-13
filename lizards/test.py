import supersuit as ss
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battlefield_v3
from pettingzoo.utils.conversions import to_parallel
from stable_baselines3 import PPO
# from simple_rl.agents import QLearningAgent
import multiprocessing
import time

multiprocessing.set_start_method("fork")



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
    env = tiger_deer_v3.env(map_size=15)
    env.reset()
    for agent in env.agent_iter(max_iter=500):
        observation, reward, done, info = env.last()
        action = env.action_space(agent).sample()
        env.step(action)
        env.render(mode='human')
    env.close()


# def simple_agent_test_AP():
#     env = adversarial_pursuit_v3.env(map_size=15)
#     ag = QLearningAgent(list(range(13)))
#     agents = {'predator_0': ag}
#
#     env.reset()
#     for agent in env.agent_iter(max_iter=5000):
#         observation, reward, done, info = env.last()
#         if done:
#             action = None
#         else:
#             if agent in agents:
#                 action = agents[agent].act(observation, reward)
#             else:
#                 action = env.action_space(agent).sample()
#         env.step(action)
#         env.render(mode='human')
#     env.close()


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
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10)
    model.save('policy')


def train_battle_policy():
    env = battle_v3.env(map_size=12)
    env = ss.pettingzoo_env_to_vec_env_v1(to_parallel(env))
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class='stable_baselines3')
    model = PPO("MlpPolicy", env, verbose=3)
    model.learn(total_timesteps=1000, log_interval=4)
    model.save("bat_policy")


def run_saved_battle_policy():
    env = battle_v3.env(map_size=12)
    env = ss.pettingzoo_env_to_vec_env_v1(to_parallel(env))
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=1, base_class='stable_baselines3')
    model = PPO.load("bat_policy")

    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        # time.sleep(0.1)
        print(rewards)
    env.close()


if __name__ == "__main__":
    run_saved_battle_policy()
