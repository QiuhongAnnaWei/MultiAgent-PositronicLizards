import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import numpy
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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


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
        dts = ('mean', 'max')

    for dist_type in dts:
        if dist_type == 'mean':
            linestyle = 'solid'
            dist_type_str = 'Avg'
        elif dist_type == 'min':
            linestyle = (0, (5, 1))
        else:
            linestyle = (0, (3, 1, 1, 1, 1, 1))
            dist_type_str = 'Max'

        ax.plot(list(range(len(team_1[dist_type]))), team_1[dist_type], linewidth=3, c=d1_color, alpha=0.75, linestyle=linestyle, label=f"{d1_label} ({dist_type_str})")
        ax.plot(list(range(len(team_2[dist_type]))), team_2[dist_type], linewidth=3, c=d2_color, alpha=0.75, linestyle=linestyle, label=f"{d2_label} ({dist_type_str})")

    ax.set_box_aspect()
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Reward')
    # ax.set_yscale('symlog')
    if plot_title:
        ax.set_title(plot_title)

    # custom_lines = [Line2D([0], [0], color='red', lw=2),
    #                 Line2D([0], [0], color='blue', lw=2)]
    # ax.legend(custom_lines, [d1_label, d2_label])
    ax.legend()


def multi_sidebyside_reward_viz_2config(ax, team_group, plot_title=None, label='Red Policy', color='red', all_shared=False):
    dts = ('mean', 'max')
    training_config = ('shared-shared', 'shared-split')

    for dist_type in dts:
        if dist_type == 'mean':
            linestyle = 'solid'
            dist_type_str = 'Avg'
        elif dist_type == 'min':
            linestyle = (0, (5, 1))
        else:
            linestyle = (0, (3, 1, 1, 1, 1, 1))
            dist_type_str = 'Max'

        for tc in training_config:
            shsp = 'shared'
            if tc == 'shared-split' and not all_shared:
                shsp = 'split'
            team_1 = team_group[tc]
            ax.plot(list(range(len(team_1[dist_type]))), team_1[dist_type], linewidth=3, c=color, alpha=0.65, linestyle=linestyle, label=f"{dist_type_str} {label} ({shsp})")

    ax.set_box_aspect()
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Reward')
    # ax.set_yscale('symlog')
    if plot_title:
        ax.set_title(plot_title)

    # custom_lines = [Line2D([0], [0], color='red', lw=2),
    #                 Line2D([0], [0], color='blue', lw=2)]
    # ax.legend(custom_lines, [d1_label, d2_label])
    ax.legend()


def comparative_reward_viz(fig, ax, team_1, team_2, dts=None, plot_title=None, xlabel='Red Reward',
                           ylabel='Blue Reward', show_cbar=True, symmetrical=True):
    if dts is None:
        # dts = ['mean']
        dts = ('mean', 'max')

    xmin = None
    xmax = 0

    for dist_type in dts:
        x_data = team_1[dist_type]
        y_data = team_2[dist_type]
        pairs = np.array(list(zip(x_data[::1], y_data[::1]))).reshape(-1, 1, 2)
        segments = np.concatenate((pairs[:-1], pairs[1:]), axis=1)
        lc = mpl.collections.LineCollection(segments, linewidths=4, alpha=0.8)
        lc.set_array(list(range(len(x_data))))

        cmap = plt.get_cmap('summer')
        cmap = truncate_colormap(cmap, 0.0, 0.7)
        lc.set_cmap(cmap)
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

    t = np.arange(xmin, xmax, 0.01)
    ax.plot(t, numpy.zeros_like(t), linewidth=1, zorder=1, color='k')
    ax.plot(numpy.zeros_like(t), t, linewidth=1, zorder=1, color='k')

    ax.set_aspect('equal', 'box')

    if symmetrical:
        ax.plot(t, t, linestyle='--', linewidth=1, zorder=1, color='k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_yscale('symlog')
    # ax.set_xscale('symlog')

    if show_cbar:
        cbar = fig.colorbar(lc)
        cbar.set_label("Training Iteration", loc='center')
    if plot_title:
        ax.set_title(plot_title)
    # plt.legend()


def comparative_reward_viz_axflip(fig, ax, team_1_group, team_2_group, dts=None, plot_title=None, xlabel='Red Reward',
                                  ylabel='Blue Reward', show_cbar=True, symmetrical=True, loc='lower right'):
    if dts is None:
        dts = ['mean']

    training_config = ('shared-shared', 'shared-split')

    legend_colors = []

    xmin = None
    ymin = None
    xmax = 0
    ymax = 0

    for tc in training_config:
        team_1 = team_1_group[tc]
        team_2 = team_2_group[tc]
        for dist_type in dts:
            x_data = team_1[dist_type]
            y_data = team_2[dist_type]
            pairs = np.array(list(zip(x_data[::1], y_data[::1]))).reshape(-1, 1, 2)
            segments = np.concatenate((pairs[:-1], pairs[1:]), axis=1)
            lc = mpl.collections.LineCollection(segments, linewidths=4, alpha=0.9)
            lc.set_array(list(range(len(x_data))))

            if tc == 'shared-shared':
                cmap = plt.get_cmap('YlGn')
                cmap = truncate_colormap(cmap, 0.6, 1.0)
                lc.set_cmap(cmap)
                legend_colors.append(cmap(100))
            else:
                cmap = plt.get_cmap('Purples')
                cmap = truncate_colormap(cmap, 0.6, 1.0)
                lc.set_cmap(cmap)
                legend_colors.append(cmap(100))

            ax.add_collection(lc)

            if xmin is None:
                xmin = min([*x_data, *y_data])
                ymin = min(y_data)
            else:
                txmin = min([*x_data, *y_data])
                tymin = min(y_data)
                xmin = min(xmin, txmin)
                ymin = min(ymin, tymin)
            if xmax is None:
                xmax = max([*x_data, *y_data])
                ymax = max(y_data)
            else:
                txmax = max([*x_data, *y_data])
                tymax = max(y_data)
                xmax = max(xmax, txmax)
                ymax = max(ymax, tymax)

    if symmetrical:
        ax.set_aspect('equal', 'box')
        t = np.arange(xmin, xmax, 0.01)
        ax.plot(t, numpy.zeros_like(t), linewidth=1, zorder=1, color='k')
        ax.plot(numpy.zeros_like(t), t, linewidth=1, zorder=1, color='k')
        ax.plot(t, t, linestyle='--', linewidth=1, zorder=1, color='k')
    else:
        t = np.arange(xmin, xmax, 0.01)
        t2 = np.arange(ymin, ymax, 0.01)
        ax.plot(t, numpy.zeros_like(t), linewidth=1, zorder=1, color='k')
        ax.plot(numpy.zeros_like(t2), t2, linewidth=1, zorder=1, color='k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_yscale('symlog')
    # ax.set_xscale('symlog')

    if show_cbar:
        cmap = plt.get_cmap('Greys')
        cmap = truncate_colormap(cmap, 0.6, 1.0)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 120), cmap=cmap))
        cbar.set_label("Training Iteration", loc='center')
    if plot_title:
        ax.set_title(plot_title)
    # plt.legend()

    custom_lines = [Line2D([0], [0], color=legend_colors[0], lw=3),
                    Line2D([0], [0], color=legend_colors[1], lw=3)]
    ax.legend(custom_lines, ['Shared Vs. Shared', 'Shared Vs. Split (Red)'], loc=loc)


def main_plotting_fn(team_1, team_2, dtsl=None, dtsr=None, suptitle='None', symmetrical=True, team_1_name='Red',
                     team_2_name='Blue', savename=None, same_fig=True, three_fig=True):
    if savename:
        savename = ''.join(savename.split('.')[:-1])

    if same_fig:
        if three_fig:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.75), gridspec_kw={'width_ratios': [1, 1, 1.1]}, tight_layout=True)
            multi_sidebyside_reward_viz(ax1, team_1, team_2, dts=dtsl, d1_label=team_1_name, d2_label=team_2_name)

            comparative_reward_viz(fig, ax2, team_1, team_2, dts=['mean'], symmetrical=symmetrical,
                                   xlabel=f"Average {team_1_name} Reward", ylabel=f"Average {team_2_name} Reward", show_cbar=False)
            comparative_reward_viz(fig, ax3, team_1, team_2, dts=['max'], symmetrical=symmetrical,
                                   xlabel=f"Max {team_1_name} Reward", ylabel=f"Max {team_2_name} Reward")

            fig.suptitle(suptitle)
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.65, 4), gridspec_kw={'width_ratios': [1, 1.5]}, tight_layout=True)

            multi_sidebyside_reward_viz(ax1, team_1, team_2, dts=dtsl, d1_label=team_1_name, d2_label=team_2_name)

            comparative_reward_viz(fig, ax2, team_1, team_2, dts=dtsr, xlabel=f"{team_1_name} Reward", ylabel=f"{team_2_name} Reward",
                                   symmetrical=symmetrical)

            fig.suptitle(suptitle)
            if savename:
                plt.savefig(f'{savename}.png')
            plt.show()

    else:
        left_plotting_fn(team_1, team_2, dtsl=dtsl, dtsr=dtsr, suptitle=suptitle, team_1_name=team_1_name,
                         team_2_name=team_2_name, savename=savename)
        right_plotting_fn(team_1, team_2, dtsl=dtsl, dtsr=dtsr, suptitle=suptitle, team_1_name=team_1_name,
                          team_2_name=team_2_name, savename=savename)


def main_plotting_fn_2configs(team_1, team_2, team_1b, team_2b, suptitle='None', symmetrical=True, team_1_name='Red',
                     team_2_name='Blue', savename=None, same_fig=True, three_fig=False, loc='lower right'):
    if savename:
        savename = ''.join(savename.split('.')[:-1])

    team_1_group = {'shared-shared': team_1, 'shared-split': team_1b}
    team_2_group = {'shared-shared': team_2, 'shared-split': team_2b}

    if same_fig:
        if three_fig:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.75), gridspec_kw={'width_ratios': [1, 1, 1.1]}, tight_layout=True)
            multi_sidebyside_reward_viz(ax1, team_1, team_2, dts=['mean'], d1_label=team_1_name, d2_label=team_2_name)
            multi_sidebyside_reward_viz(ax1, team_1b, team_2b, dts=['mean'], d1_label=team_1_name, d2_label=team_2_name)

            comparative_reward_viz_axflip(fig, ax2, team_1_group, team_2_group, dts=['mean'], symmetrical=symmetrical,
                                   xlabel=f"Average {team_1_name} Reward", ylabel=f"Average {team_2_name} Reward", show_cbar=False,
                                          loc=loc)
            comparative_reward_viz_axflip(fig, ax3, team_1_group, team_2_group, dts=['max'], symmetrical=symmetrical,
                                   xlabel=f"Max {team_1_name} Reward", ylabel=f"Max {team_2_name} Reward", loc=loc)

            fig.suptitle(suptitle)
            plt.show()
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 3.75), gridspec_kw={'width_ratios': [1, 1, 1, 1.1]},
                                                tight_layout=True)
            multi_sidebyside_reward_viz_2config(ax1, team_1_group, label=team_1_name)
            multi_sidebyside_reward_viz_2config(ax2, team_2_group, label=team_2_name, color='blue')

            comparative_reward_viz_axflip(fig, ax3, team_1_group, team_2_group, dts=['mean'], symmetrical=symmetrical,
                                          xlabel=f"Average {team_1_name} Reward",
                                          ylabel=f"Average {team_2_name} Reward", show_cbar=False,
                                          loc=loc)
            comparative_reward_viz_axflip(fig, ax4, team_1_group, team_2_group, dts=['max'], symmetrical=symmetrical,
                                          xlabel=f"Max {team_1_name} Reward", ylabel=f"Max {team_2_name} Reward",
                                          loc=loc)

            fig.suptitle(suptitle)
            plt.show()

    else:
        # left_plotting_fn(team_1, team_2, dtsl=dtsl, dtsr=dtsr, suptitle=suptitle, team_1_name=team_1_name,
        #                  team_2_name=team_2_name, savename=savename)
        # right_plotting_fn(team_1, team_2, dtsl=dtsl, dtsr=dtsr, suptitle=suptitle, team_1_name=team_1_name,
        #                   team_2_name=team_2_name, savename=savename)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.75), gridspec_kw={'width_ratios': [1, 1.25]},
                                            tight_layout=True)
        # multi_sidebyside_reward_viz(ax1, team_1, team_2, dts=dtsl, d1_label=team_1_name, d2_label=team_2_name)

        comparative_reward_viz_axflip(fig, ax1, team_1_group, team_2_group, dts=['mean'], symmetrical=symmetrical,
                                      xlabel=f"Average {team_1_name} Reward", ylabel=f"Average {team_2_name} Reward",
                                      show_cbar=False, loc=loc)
        comparative_reward_viz_axflip(fig, ax2, team_1_group, team_2_group, dts=['max'], symmetrical=symmetrical,
                                      xlabel=f"Max {team_1_name} Reward", ylabel=f"Max {team_2_name} Reward", loc=loc)

        fig.suptitle(suptitle)
        if savename:
            plt.savefig(f'{savename}.png')
        plt.show()


def left_plotting_fn(team_1, team_2, dtsl=None, dtsr=None, suptitle='None', start_offsets=(0, 0), end_offsets=(0, 0), team_1_name='Red',
                     team_2_name='Blue', savename=None):
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    multi_sidebyside_reward_viz(ax, team_1, team_2, dts=dtsl, d1_label=team_1_name, d2_label=team_2_name)

    fig.suptitle(suptitle)
    if savename:
        plt.savefig(f'{savename}-left.png')
    plt.show()


def right_plotting_fn(team_1, team_2, dtsl=None, dtsr=None, suptitle='None', start_offsets=(0, 0), end_offsets=(0, 0), team_1_name='Red',
                     team_2_name='Blue', savename=None):
    fig, ax = plt.subplots(figsize=(5, 4), tight_layout=True)

    comparative_reward_viz(fig, ax, team_1, team_2, dts=dtsr,
                           xlabel=f"{team_1_name} Reward", ylabel=f"{team_2_name} Reward")
    ax.set_title(suptitle)
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    if savename:
        plt.savefig(f'{savename}-right.png')
    plt.show()


def main():
    # Battle shared shared plots
    battle_shared_shared = get_all_result_data("result_data/battle_shared_shared_ms30/result.json")
    # main_plotting_fn(battle_shared_shared['red'], battle_shared_shared['blue'],
    #                  suptitle='Battle Env - Red:Shared, Blue:Shared')
    #                  # savename='plots/Battle-ShSh.png', same_fig=False)

    # Battle shared split plots
    battle_shared_split = get_all_result_data("result_data/battle_shared_split_ms30/result.json")
    # main_plotting_fn(battle_shared_split['red'], battle_shared_split['blue'],
    #                  suptitle='Battle Env - Red:Split, Blue:Shared')
    #                  # savename='plots/Battle-ShSp.png', same_fig=False)

    main_plotting_fn_2configs(battle_shared_shared['red'], battle_shared_shared['blue'],
                              battle_shared_split['red'], battle_shared_split['blue'],
                              suptitle='Battle - Reward Breakdown per Training Iteration', same_fig=True,
                              savename='plots/Battle-trio.png')

    # # Battle selfplay plots
    # battle_selfplay = get_result_data("result_data/battle_selfplay/result.json")
    # simple_reward_viz(list(range(len(battle_selfplay['all']))), battle_selfplay['all'],
    #                   plot_title='Battle Env - Self-Play')

    # AP shared shared plots
    ap_shared_shared = get_all_result_data("result_data/ap_shared_shared_ms19/result.json")
    # main_plotting_fn(ap_shared_shared['predator'], ap_shared_shared['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Shared, Prey:Shared', symmetrical=False,
    #                  team_1_name='Predator', team_2_name='Prey', dtsl=['max', 'mean'], dtsr=['max', 'mean'])
    #                  # savename='plots/AP-ShSh.png', same_fig=False)
    # #
    # # AP shared split plots
    ap_shared_split = get_all_result_data("result_data/ap_shared_split_ms19/result.json")
    # main_plotting_fn(ap_shared_split['predator'], ap_shared_split['prey'],
    #                  suptitle='Adversarial Pursuit - Predator:Split, Prey:Shared',
    #                  team_1_name='Predator', team_2_name='Prey', dtsl=['max', 'mean'], dtsr=['max', 'mean'])
    #                  # savename='plots/AP-ShSp.png', same_fig=False)
    #
    main_plotting_fn_2configs(ap_shared_shared['predator'], ap_shared_shared['prey'],
                              ap_shared_split['predator'], ap_shared_split['prey'],
                              suptitle='Adversarial Pursuit - Reward Breakdown per Training Iteration', same_fig=True, symmetrical=False,
                              team_1_name='Predator', team_2_name='Prey', dtsl=['max', 'mean'], dtsr=['max', 'mean'],
                              loc='lower left',
                              savename='plots/AP-trio.png')


if __name__ == '__main__':
    main()
