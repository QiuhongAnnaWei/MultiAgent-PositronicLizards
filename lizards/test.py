from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3
from simple_rl.agents import QLearningAgent


def random_AP():
    env = adversarial_pursuit_v3.env(map_size=15)
    env.reset()
    for agent in env.agent_iter(max_iter=50000):
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
    agent = QLearningAgent(list(range(13)))

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


if __name__ == "__main__":
    random_AP()
