from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3


def random_AP():
    env = adversarial_pursuit_v3.env()
    env.reset()
    for agent in env.agent_iter(max_iter=500):
        observation, reward, done, info = env.last()
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


if __name__ == "__main__":
    random_TD()
