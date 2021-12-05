import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme()


def get_simple_data(fname, iteration_steps=4000):
    with open(fname, "r") as fp:
        data = json.load(fp)
        if iteration_steps is not None:
            x_data = [d[1]/iteration_steps for d in data]
            y_data = [d[2] for d in data]
        else:
            x_data = [d[1] for d in data]
            y_data = [d[2] for d in data]
        return x_data, y_data


def get_result_data(fname, avg_prefixes=True, distribution_type='mean'):
    """

    :param fname:
    :param avg_prefixes:
    :param distribution_type: can be 'mean', 'min', or 'max'
    :return:
    """
    with open(fname, "r") as fp:
        data = [json.loads(line) for line in fp]
        i = 0
        while True:
            if data[i]['policy_reward_mean'] == {}:
                i += 1
            else:
                break

        reward_history = dict()

        if avg_prefixes:
            teams = set(map(lambda x: x.split("_")[0], list(data[i][f'policy_reward_{distribution_type}'].keys())))
            for t in teams:
                reward_history[t] = []
        else:
            for k, v in data[i][f'policy_reward_{distribution_type}'].items():
                reward_history[k] = []

        for i in data:
            rewards = i[f'policy_reward_{distribution_type}']

            if avg_prefixes:
                team_rewards = {t: 0 for t in teams}
                team_counts = {t: 0 for t in teams}
                for k, v in rewards.items():
                    team = k.split('_')[0]
                    team_rewards[team] += v
                    team_counts[team] += 1

                can_collect = True
                for t in teams:
                    if team_counts[t] == 0:
                        can_collect = False
                        break
                if can_collect:
                    for t in teams:
                        reward_history[t].append(team_rewards[t]/team_counts[t])
            else:
                for k, v in rewards.items():
                    reward_history[k].append(v)

        return reward_history


def simple_reward_viz(x_data, y_data, plot_title="None", xlabel='Episode', ylabel='Reward'):
    plt.plot(x_data, y_data, linewidth=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()


def comparative_reward_viz(x_data, y_data, plot_title="None", xlabel='Red Reward', ylabel='Blue Reward', style='full'):
    pairs = np.array(list(zip(x_data, y_data))).reshape(-1, 1, 2)
    segments = np.concatenate((pairs[:-1], pairs[1:]), axis=1)
    lc = mpl.collections.LineCollection(segments, linewidths=3)
    lc.set_array(list(range(len(x_data))))
    lc.set_cmap('winter')
    f, ax = plt.subplots()
    ax.add_collection(lc)

    if style == 'full':
        ax.set_aspect('equal', 'box')
        t = np.arange(min([*x_data, *y_data]), max([*x_data, *y_data]), 0.01)
        ax.plot(t, t, linestyle='--', linewidth=2)
        ax.scatter(x_data[0], y_data[0], label='Episode 0')
        ax.scatter(x_data[-1], y_data[-1], label=f"Episode {len(x_data)}")
        plt.legend()

    elif style == 'tight':
        ax.autoscale_view()
        t = np.arange(min(x_data), max(y_data), 0.01)
        ax.plot(t, t, linestyle='--', linewidth=2)
        ax.text(x_data[0], y_data[0], f"Episode 0")
        ax.text(x_data[-1], y_data[-1], f"Episode {len(x_data)}")
        ax.scatter([x_data[0], x_data[-1]], [y_data[0], y_data[-1]])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()


def main():
    # Battle shared shared plots
    battle_shared_shared = get_result_data("result_data/battle_shared_shared/result.json")
    simple_reward_viz(list(range(len(battle_shared_shared['red']))), battle_shared_shared['red'])
    simple_reward_viz(list(range(len(battle_shared_shared['blue']))), battle_shared_shared['blue'])
    comparative_reward_viz(battle_shared_shared['red'], battle_shared_shared['blue'], style='tight')

    # Battle shared split plots
    battle_shared_split = get_result_data("result_data/battle_shared_split/result.json")
    simple_reward_viz(list(range(len(battle_shared_split['red']))), battle_shared_split['red'])
    simple_reward_viz(list(range(len(battle_shared_split['blue']))), battle_shared_split['blue'])
    comparative_reward_viz(battle_shared_split['red'], battle_shared_split['blue'], style='tight')

    # Battle selfplay plots
    battle_selfplay = get_result_data("result_data/battle_selfplay/result.json")
    simple_reward_viz(list(range(len(battle_selfplay['all']))), battle_selfplay['all'])

    # AP shared shared plots
    ap_shared_shared = get_result_data("result_data/ap_shared_shared/result.json")
    simple_reward_viz(list(range(len(ap_shared_shared['prey']))), ap_shared_shared['prey'])
    simple_reward_viz(list(range(len(ap_shared_shared['predator']))), ap_shared_shared['predator'])
    comparative_reward_viz(ap_shared_shared['prey'], ap_shared_shared['predator'], xlabel='prey',
                           ylabel='predator', style='full')

    # AP shared split plots
    ap_shared_split = get_result_data("result_data/ap_shared_split/result.json")
    simple_reward_viz(list(range(len(ap_shared_split['prey']))), ap_shared_split['prey'], plot_title='Avg Prey Reward')
    simple_reward_viz(list(range(len(ap_shared_split['predator']))), ap_shared_split['predator'], plot_title='Avg Predator Reward')
    comparative_reward_viz(ap_shared_split['prey'], ap_shared_split['predator'], xlabel='prey',
                           ylabel='predator', style='full')


if __name__ == '__main__':
    main()
