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


def simple_reward_viz(x_data, y_data, plot_title="None", xlabel='Episode', ylabel='Reward', d_label='Shared Policy'):
    fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)
    ax.plot(x_data, y_data, linewidth=5, alpha=0.7, label=d_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(plot_title)
    ax.legend()
    plt.show()


def sidebyside_reward_viz(ax, x_data1, y_data1, x_data2, y_data2, plot_title=None, d1_label='Red Policy', d2_label='Blue Policy', d1_color='red', d2_color='blue'):
    ax.plot(x_data1, y_data1, linewidth=3.5, c=d1_color, alpha=0.65, label=d1_label)
    ax.plot(x_data2, y_data2, linewidth=3.5, c=d2_color, alpha=0.65, label=d2_label)
    ax.set_box_aspect()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    if plot_title:
        ax.set_title(plot_title)
    ax.legend()


def comparative_reward_viz(fig, ax, x_data, y_data, plot_title=None, xlabel='Red Reward', ylabel='Blue Reward', style='full', start_offsets=(0, 0), end_offsets=(0, 0), text_kwargs=None):
    if text_kwargs is None:
        text_kwargs = {'start': {}, 'end': {}}
    if 'start' not in text_kwargs:
        text_kwargs['start'] = {}
    if 'end' not in text_kwargs:
        text_kwargs['end'] = {}

    pairs = np.array(list(zip(x_data[::1], y_data[::1]))).reshape(-1, 1, 2)
    segments = np.concatenate((pairs[:-1], pairs[1:]), axis=1)
    lc = mpl.collections.LineCollection(segments, linewidths=5)
    lc.set_array(list(range(len(x_data))))
    lc.set_cmap('winter')
    ax.add_collection(lc)

    if style == 'full':
        ax.set_aspect('equal', 'box')
        t = np.arange(min([*x_data, *y_data]), max([*x_data, *y_data]), 0.01)
    elif style == 'tight':
        ax.autoscale_view()
        t = np.arange(min(x_data), max(x_data), 0.01)

    ax.plot(t, t, linestyle='--', linewidth=3, zorder=1)
    ax.text(x_data[0]+start_offsets[0], y_data[0]+start_offsets[1], "Episode 0", **text_kwargs['start'])
    ax.text(x_data[-1]+end_offsets[0], y_data[-1]+end_offsets[1], f"Episode {len(x_data)}", **text_kwargs['end'])
    ax.scatter([x_data[0], x_data[-1]], [y_data[0], y_data[-1]], zorder=2, color='red', s=80, alpha=0.75, marker='*')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(lc)
    cbar.set_label("Episode", loc='center')
    if plot_title:
        ax.set_title(plot_title)
    # plt.legend()


def main_plotting_fn(team_1, team_2, suptitle='None', start_offsets=(0, 0), end_offsets=(0, 0), team_1_name='Red',
                     team_2_name='Blue'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.65, 4), gridspec_kw={'width_ratios': [1, 1.5]}, tight_layout=True)
    sidebyside_reward_viz(ax1, list(range(len(team_1))), team_1,
                          list(range(len(team_2))), team_2, d1_label=f"{team_1_name} Policy",
                          d2_label=f"{team_2_name} Policy")
    comparative_reward_viz(fig, ax2, team_1, team_2,
                           style='full', start_offsets=start_offsets, end_offsets=end_offsets,
                           xlabel=f"{team_1_name} Reward", ylabel=f"{team_2_name} Reward")
    fig.suptitle(suptitle)
    plt.show()


def main():
    # Battle shared shared plots
    battle_shared_shared = get_result_data("result_data/battle_shared_shared/result.json")
    main_plotting_fn(battle_shared_shared['red'], battle_shared_shared['blue'],
                     suptitle='Battle Env - Red:Shared, Blue:Shared', start_offsets=(2, -1), end_offsets=(-17, -1))

    # # Battle shared split plots
    # battle_shared_split = get_result_data("result_data/battle_shared_split/result.json")
    # main_plotting_fn(battle_shared_split['red'], battle_shared_split['blue'],
    #                  suptitle='Battle Env - Red:Split(mean), Blue:Shared', start_offsets=(2, -1), end_offsets=(-17, -1))
    #
    # battle_shared_split = get_result_data("result_data/battle_shared_split/result.json", distribution_type='max')
    # main_plotting_fn(battle_shared_split['red'], battle_shared_split['blue'],
    #                  suptitle='Battle Env - Red:Split(max), Blue:Shared', start_offsets=(3, -2), end_offsets=(3, -2))
    #
    # battle_shared_split = get_result_data("result_data/battle_shared_split/result.json", distribution_type='min')
    # main_plotting_fn(battle_shared_split['red'], battle_shared_split['blue'],
    #                  suptitle='Battle Env - Red:Split(min), Blue:Shared', start_offsets=(-18, 2), end_offsets=(-22, -5))

    # # Battle selfplay plots
    # battle_selfplay = get_result_data("result_data/battle_selfplay/result.json")
    # simple_reward_viz(list(range(len(battle_selfplay['all']))), battle_selfplay['all'],
    #                   plot_title='Battle Env - Self-Play')
    #
    # # AP shared shared plots
    # ap_shared_shared = get_result_data("result_data/ap_shared_shared/result.json")
    # main_plotting_fn(ap_shared_shared['predator'], ap_shared_shared['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Shared, Prey:Shared', start_offsets=(2, 2),
    #                  end_offsets=(-40, -5), team_1_name='Predator', team_2_name='Prey')

    # AP shared split plots
    # ap_shared_split = get_result_data("result_data/ap_shared_split/result.json")
    # main_plotting_fn(ap_shared_split['predator'], ap_shared_split['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Split(mean), Prey:Shared', start_offsets=(-50, -160),
    #                  end_offsets=(-300, -170), team_1_name='Predator', team_2_name='Prey')

    # ap_shared_split = get_result_data("result_data/ap_shared_split/result.json", distribution_type='max')
    # main_plotting_fn(ap_shared_split['predator'], ap_shared_split['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Split(max), Prey:Shared', start_offsets=(0, 50),
    #                  end_offsets=(-500, 50), team_1_name='Predator', team_2_name='Prey')

    # ap_shared_split = get_result_data("result_data/ap_shared_split/result.json", distribution_type='min')
    # main_plotting_fn(ap_shared_split['predator'], ap_shared_split['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Split(min), Prey:Shared', start_offsets=(-50, -160),
    #                  end_offsets=(-300, -170), team_1_name='Predator', team_2_name='Prey')


if __name__ == '__main__':
    main()
