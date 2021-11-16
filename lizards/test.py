import supersuit as ss
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battlefield_v3, battle_v3
from pettingzoo.utils.conversions import to_parallel
from stable_baselines3 import PPO
# from simple_rl.agents import QLearningAgent
import multiprocessing
import time
import ray.rllib.agents.pg as pg
from ray.tune.registry import register_env

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


def ray_experiment1():
    env_proto = battle_v3.parallel_env(map_size=12)

    def env_creator():
        return battle_v3.parallel_env

    env = env_creator()
    register_env('battle', lambda config: env(map_size=12))

    # "car1": (None, car_obs_space, car_act_space, {"gamma": 0.85}),

    # policy_dict = dict()
    # for agent in env.agents:
    #     policy_dict[agent] = (None, env.observation_space(agent), env.action_space(agent), {"gamma": 0.95})

    policy_dict = dict()
    unique_agents = set([agent.split('_')[0] for agent in env_proto.agents])
    for agent in unique_agents:
        policy_dict[agent] = (None, env_proto.observation_space(f"{agent}_0"), env_proto.action_space(f"{agent}_0"), {"gamma": 0.95})

    trainer = pg.PGTrainer(env='battle', config={
        "multiagent": {
            "policies": policy_dict,
            "policy_mapping_fn":
                lambda agent_name: "red" if agent_name.startswith("red") else "blue"
        }
    })

    while True:
        print(trainer.train())


if __name__ == "__main__":
    ray_experiment1()
