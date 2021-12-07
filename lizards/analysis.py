import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def get_all_result_data(fname, just_mean=False):
    all_results = {}
    if just_mean:
        dts = ['mean']
    else:
        dts = ('mean', 'min', 'max')
    for dist_type in dts:
        results = get_result_data(fname, distribution_type=dist_type)
        for k, v in results.items():
            if k not in all_results:
                all_results[k] = {}
            all_results[k][dist_type] = v

    return all_results


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


def multi_sidebyside_reward_viz(ax, team_1, team_2, dts=None, plot_title=None, d1_label='Red Policy', d2_label='Blue Policy', d1_color='red', d2_color='blue'):
    if dts is None:
        dts = ('mean', 'min', 'max')

    for dist_type in dts:
        if dist_type == 'mean':
            linestyle = 'solid'
        elif dist_type == 'min':
            linestyle = (0, (5, 1))
        else:
            linestyle = (0, (3, 1, 1, 1, 1, 1))

        ax.plot(list(range(len(team_1[dist_type]))), team_1[dist_type], linewidth=3, c=d1_color, alpha=0.5, linestyle=linestyle, label=f"{d1_label} ({dist_type})")
        ax.plot(list(range(len(team_2[dist_type]))), team_2[dist_type], linewidth=3, c=d2_color, alpha=0.5, linestyle=linestyle, label=f"{d2_label} ({dist_type})")

    ax.set_box_aspect()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_yscale('symlog')
    if plot_title:
        ax.set_title(plot_title)

    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='blue', lw=2)]
    ax.legend(custom_lines, [d1_label, d2_label])


def comparative_reward_viz(fig, ax, team_1, team_2, dts=None, plot_title=None, xlabel='Red Reward',
                           ylabel='Blue Reward', start_offsets=(0, 0), end_offsets=(0, 0)):
    if dts is None:
        # dts = ['mean']
        dts = ('mean', 'min', 'max')

    xmin = None
    xmax = None

    for dist_type in dts:
        if dist_type == 'mean' or True:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'

        x_data = team_1[dist_type]
        y_data = team_2[dist_type]
        pairs = np.array(list(zip(x_data[::1], y_data[::1]))).reshape(-1, 1, 2)
        segments = np.concatenate((pairs[:-1], pairs[1:]), axis=1)
        lc = mpl.collections.LineCollection(segments, linewidths=4, linestyle=linestyle)
        lc.set_array(list(range(len(x_data))))
        lc.set_cmap('winter')
        ax.add_collection(lc)

        if xmin is None:
            xmin = min([*x_data, *y_data])
        else:
            txmin = min([*x_data, *y_data])
            xmin = min(xmin, txmin)
        if xmax is None:
            xmax = max([*x_data, *y_data])
        else:
            txmax = max([*x_data, *y_data])
            xmax = max(xmax, txmax)

        # ax.text(x_data[0]+start_offsets[0], y_data[0]+start_offsets[1], "Episode 0")
        # ax.text(x_data[-1]+end_offsets[0], y_data[-1]+end_offsets[1], f"Episode {len(x_data)}")
        # ax.scatter([x_data[0], x_data[-1]], [y_data[0], y_data[-1]], zorder=2, color='red', s=80, alpha=0.75, marker='*')

    ax.set_aspect('equal', 'box')
    t = np.arange(xmin, xmax, 0.01)
    ax.plot(t, t, linestyle='dotted', linewidth=2, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('symlog')
    ax.set_xscale('symlog')
    cbar = fig.colorbar(lc)
    cbar.set_label("Episode", loc='center')
    if plot_title:
        ax.set_title(plot_title)
    # plt.legend()


def main_plotting_fn(team_1, team_2, dtsl=None, dtsr=None, suptitle='None', start_offsets=(0, 0), end_offsets=(0, 0), team_1_name='Red',
                     team_2_name='Blue', savename=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.65, 4), gridspec_kw={'width_ratios': [1, 1.5]}, tight_layout=True)

    multi_sidebyside_reward_viz(ax1, team_1, team_2, dts=dtsl, d1_label=team_1_name, d2_label=team_2_name)

    comparative_reward_viz(fig, ax2, team_1, team_2, dts=dtsr, start_offsets=start_offsets, end_offsets=end_offsets,
                           xlabel=f"{team_1_name} Reward", ylabel=f"{team_2_name} Reward")

    fig.suptitle(suptitle)
    if savename:
        plt.savefig(savename)
    plt.show()


def main():
    # Battle shared shared plots
    battle_shared_shared = get_all_result_data("result_data/battle_shared_shared/result.json")
    main_plotting_fn(battle_shared_shared['red'], battle_shared_shared['blue'],
                     suptitle='Battle Env - Red:Shared, Blue:Shared', dtsr=['mean'], start_offsets=(2, -1), end_offsets=(-17, -1),
                     savename='plots/Battle-ShSh.png')

    # Battle shared split plots
    battle_shared_split = get_all_result_data("result_data/battle_shared_split/result.json")
    main_plotting_fn(battle_shared_split['red'], battle_shared_split['blue'],
                     suptitle='Battle Env - Red:Split, Blue:Shared', dtsr=['mean'], start_offsets=(2, -1), end_offsets=(-17, -1),
                     savename='plots/Battle-ShSp.png')

    # Battle selfplay plots
    battle_selfplay = get_result_data("result_data/battle_selfplay/result.json")
    simple_reward_viz(list(range(len(battle_selfplay['all']))), battle_selfplay['all'],
                      plot_title='Battle Env - Self-Play')

    # AP shared shared plots
    ap_shared_shared = get_all_result_data("result_data/ap_shared_shared/result.json")
    main_plotting_fn(ap_shared_shared['predator'], ap_shared_shared['prey'],
                     suptitle='Adversarial Pursuit - Predator:Shared, Prey:Shared', start_offsets=(2, 2),
                     end_offsets=(-40, -5), team_1_name='Predator', team_2_name='Prey', dtsr=['min', 'mean', 'max'],
                     savename='plots/AP-ShSh.png')

    # AP shared split plots
    ap_shared_split = get_all_result_data("result_data/ap_shared_split/result.json")
    main_plotting_fn(ap_shared_split['predator'], ap_shared_split['prey'],
                     suptitle='Adversarial Pursuit - Predator:Split, Prey:Shared', start_offsets=(-50, -160),
                     end_offsets=(-300, -170), team_1_name='Predator', team_2_name='Prey',
                     savename='plots/AP-ShSp.png')


if __name__ == '__main__':
    main()
