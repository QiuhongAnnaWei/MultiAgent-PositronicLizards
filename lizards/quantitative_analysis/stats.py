from typing import Iterator
from main_utils import *
import ray.rllib.agents.ppo as ppo
import supersuit as ss
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path
import pytz



def get_timestamp():
    tz = pytz.timezone('US/Eastern')
    short_timestamp = datetime.now(tz).strftime("%H.%M.%S")
    return short_timestamp


def save_eval_viz(trial_path, diff_frame_list, width, height):

    trial_path.mkdir(parents=True, exist_ok=True) 

    get_path_str = lambda suffix: str((trial_path / suffix).resolve())
    mp4_save_path, gif_save_path = get_path_str("viz.mp4"), get_path_str("viz.gif")

    print(f"# Saving gif to: {gif_save_path}")
    diff_frame_list[0].save(gif_save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)

    print(f"# Saving gif to: {mp4_save_path}")
    video = cv2.VideoWriter(mp4_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    for i, image in enumerate(diff_frame_list):
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    for i in [0, len(diff_frame_list)-1]:
        frame_path = get_path_str(f"frame{i}.jpg")
        diff_frame_list[i].save(frame_path)


def record_attacks(action, agent, team_attack_statuses, agent_attack_statuses, agent_to_team_func):
    agent_attacked =  (12 < action <= 20) # (Currently only recognizes battle attacks).

    agent_attack_statuses[agent].append(agent_attacked)
    team_attack_statuses[agent_to_team_func(agent)].append(agent_attacked)


# def package_attacks_data(total_attacks_per_agent, team_attack_statuses, agent_attack_statuses):
def package_attacks_data(team_attack_statuses, agent_attack_statuses, team_hp_values):
    list_of_atks_across_timesteps_per_agent_series = [pd.Series(atks_across_timesteps, name=agent_nm) for agent_nm, atks_across_timesteps in agent_attack_statuses.items()]
    agent_attacks_df = pd.concat(list_of_atks_across_timesteps_per_agent_series, axis=1)
    agent_attacks_df.index.name = "Timesteps"

    list_of_atks_across_timesteps_per_team_series = [pd.Series(atks_across_timesteps, name=team_nm) for team_nm, atks_across_timesteps in team_attack_statuses.items()]
    team_attacks_df = pd.concat(list_of_atks_across_timesteps_per_team_series, axis=1)
    team_attacks_df.index.name = "Timesteps"

    list_of_hps_across_timesteps_per_team_series = [pd.Series(hps_across_timesteps, name=team_nm) for team_nm, hps_across_timesteps in team_hp_values.items()]
    team_hps_df = pd.concat(list_of_hps_across_timesteps_per_team_series, axis=1)
    team_hps_df.index.name = "Timesteps"

    return agent_attacks_df, team_attacks_df, team_hps_df


def record_hp(env_state, team_hp_values):
    hp_red = env_state[:,:,2]
    hp_blue = env_state[:,:,4]
    # not summing b/c don't want to lose individual agents' info (and can sum later)

    team_hp_values["red"].append(hp_red)
    team_hp_values["blue"].append(hp_blue)


def collect_stats_from_eval(representative_trainer, env, env_config, policy_fn, trial_path, save_viz=False, is_battle=True, max_iter=2 ** 16):
    """
    Runs an eval and collects stats (attacks, hp) from it 
    eval_id is an id tt we can use to distinguish between eval runs
    saves video of the eval run if save_viz == True
    """

    # Specific to battle
    agent_to_team_func = lambda a: agent.split("_")[0]

    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env.reset()

    if save_viz:
        diff_frame_list = []
        width, height, img = None, None, None

    
    # Stats to track:
    agent_attack_statuses = defaultdict(list)
    team_attack_statuses = defaultdict(list)
    team_hp_values = defaultdict(list)
    losing_team = None
    team_elim_counts = defaultdict(int) # (informs the eventual value of losing_team).

    timeline = defaultdict(dict)
    # 'absolute' timeline 
    # agent_iter idx =>  dict with keys agent, action, team_red_hp (scalar), team_blue_hp (scalar)

    def log_action_to_timeline(iter_idx, action, agent_id):        
        timeline_at_iter_idx = timeline[iter_idx]
        timeline_at_iter_idx["agent_id"] = agent_id
        timeline_at_iter_idx["action"] = action

    def log_scalar_hps_to_timeline(iter_idx, env_state):
        timeline_at_iter_idx = timeline[iter_idx]
        timeline_at_iter_idx["hp_red"] = env_state[:,:,2].sum()
        timeline_at_iter_idx["hp_blue"] = env_state[:,:,4].sum()


    # Loop through all agents until a team has lost:
    for i, agent in enumerate(env.agent_iter(max_iter=max_iter)):
        if i % 1000 == 0:
            # Printing this helps see iteration status over time.
            print(f"getting actions for: iter {i} ...")

        # Get info for current agent.
        observation, _, done, _ = env.last() # (observation[:,:,3/4]==0).sum()

        if done:
            curr_agent_action = None
            team_elim_counts[agent.split("_")[0]] += 1

            log_action_to_timeline(i, "died", agent)
        else:
            # Find action for agent, from current policy and observation.
            curr_agent_policy = representative_trainer.get_policy(policy_fn(agent, None))
            batch_obs = { 'obs': np.expand_dims(observation, 0)} # (10,10,5) -> (1,10,10,5)
            batched_action, _, _ = curr_agent_policy.compute_actions_from_input_dict(batch_obs)
            curr_agent_action = batched_action[0]

            record_attacks(curr_agent_action, agent, team_attack_statuses, agent_attack_statuses, agent_to_team_func)
            log_action_to_timeline(i, curr_agent_action, agent)
       
        # Get current state (or, if game is over via exception).
        env_state = None
        try:
            env_state = env.state() # (map_size, map_size, 5)
            record_hp(env_state, team_hp_values)
            log_scalar_hps_to_timeline(i, env_state)
            # print("env_state", len(env_state))
        except Exception as e:
            print("Team has lost; indicated by exception:", e)
            print(f"\nAt {i}: one team eliminated - env.agents = {env.agents}")
            if losing_team is None:
                # A bit of a hack, but the losing team should be the team that had the most agents die.
                losing_team = sorted([(-1*count, team) for team, count in team_elim_counts.items()])[0][1]
            break
            # I have no idea why `break` needs to be commented out when im running it
            # (Eli): I don't need to comment it out?
            
        # Take a step with the current action:
        env.step(curr_agent_action)
    
        if save_viz:
            img2 = PIL.Image.fromarray(env.render(mode='rgb_array'))
            if img is None:
                width, height = img2.width, img2.height
                img = np.zeros((width, height))
            if np.array_equal(np.array(img), np.array(img2)) == False:
                img = img2.copy()
                ImageDraw.Draw(img2).text( (2, height-10), f"iter={i}", (0,0,0))
                if is_battle:
                    ImageDraw.Draw(img2).text( (2, 13), f"HP={str(round((env_state[:,:,2]).sum(), 2))}",  (0,0,0)) 
                    ImageDraw.Draw(img2).text( (width-55, 13), f"HP={str(round((env_state[:,:,4]).sum(), 2))}",  (0,0,0)) # "{:.4f}".format()
                diff_frame_list.append(img2)
    env.close()

    if save_viz: 
        # Stack each frame as a video:
        save_eval_viz(trial_path, diff_frame_list, width, height)

    agent_attacks_df, team_attacks_df, team_hps_df = package_attacks_data(team_attack_statuses, agent_attack_statuses, team_hp_values)
    timeline_df = pd.DataFrame.from_dict(timeline, orient='index')

    # print("red hps:", team_hp_values["red"][-1])
    # print("blue hps:", team_hp_values["blue"][-1])
    # log(f"{log_dir}.txt", f"attacks per agent {total_attacks_per_agent}") 
    return losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df


# YM: I'm too sleepy to be able to think of the best abstractions, so just going to make adapted variants of the logging infrastructure for the evals we need to run
def iterate_BA_trial_stats(num_trials, run_path, trainer, env_config, policy_fn, gpu = False, save_viz=True):
    """Gets the stats for Battle, given a checkpoint and trainer.
    Outputs these as CSVs within the checkpoint directory that is supplied.

    - CSV descriptions (every trial has their own folder of CSVs):
        1. "AGENT_ATTACKS_TEMPORAL.csv" - Cells are true/false for attack/not attack when it is that agent's turn.
                Columns are specific agents, and columns correspond to a __time__ in the game. 
                In other words, the actions corresponding all of the cells in any row form the only player-actions 
                that occurred within some specific window of time (this window shortens as players are eliminated).

        2. "TEAM_ATTACKS_ORDERED.csv" - Each column is an ordered boolean list representing the exact sequence of attack/not attack
                decisions that a certain team made throughout the game. Note that when a team has more players than the other, it will 
                have more attack opportunities per unit of time than the other team, so rows do not necessarily map to specific moments.

        3. "TEAM_HPS_COMBINED_TEMPORAL.csv" - Each column is an ordered vector list represented each team's HP state
                 __after any action from any agent__. Since all team's HP are written and read at the same time throughout the game, 
                 the data has both temporal and ordered meaning. All of the HP values in any row correspond to the exact same moment in time.

        4. "ABSOLUTE_TIMELINE.csv" - Logs what's happening at every iteration of the agent_iter loop. That is, each row consists of (i) the id of the agent whose turn it is, (ii) what action the agent took (if it died on that iteration, that’s recorded as “died”), and (iii) the scalar hps of the teams at that iteration. The index of this csv corresponds to the iteration idx of the agent_iter loop.

    - Also, a "LOSING_TEAM_ACROSS_TRIALS.csv" exists in the upper directory that simply lists which team lost for each trial.
    """

    for trial_i in range(num_trials):
        trial_path = run_path / ("trial_" + str(trial_i))
        losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df = collect_stats_from_eval(trainer, battle_v3, env_config, policy_fn, trial_path, save_viz=save_viz)
        yield losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df


def write_BA_stats_CSVs(num_trials, trainer, log_dir, env_config, policy_fn, save_viz=True, gpu=False, run_name_from_user=""):
    """
    *Given trainer*, gets the stats for Battle; outputs these as CSVs within `log_dir`.
    See docstring of the other write_BA_stats function for descriptions of CSVs.
    """
    # Create directory for this run:
    log_dir = Path(log_dir)
    unique_run_ID = run_name_from_user + f"{num_trials}trials" + get_timestamp()
    run_path = log_dir / "eval_stats" / unique_run_ID
    run_path.mkdir(parents=True, exist_ok=True)

    losing_teams = []
    # Populate each trial-folder with CSVs:
    for i, (losing_team, agent_attacks_df, team_attacks_df, team_hps_df, timeline_df) in enumerate(iterate_BA_trial_stats(num_trials, run_path, trainer, env_config, policy_fn, gpu=gpu, save_viz=save_viz)):
        print("Running/saving trial", i)
        trial_path = run_path / ("trial_" + str(i))
        trial_path.mkdir(exist_ok=True)
        agent_attacks_df.to_csv((trial_path / "AGENT_ATTACKS_TEMPORAL.csv").resolve())
        team_attacks_df.to_csv((trial_path / "TEAM_ATTACKS_ORDERED.csv").resolve())
        team_hps_df.to_csv((trial_path / "TEAM_HPS_COMBINED_TEMPORAL.csv").resolve())
        timeline_df.to_csv((trial_path / "ABSOLUTE_TIMELINE.csv").resolve())

        losing_teams.append(losing_team)
        # ideally we would save the vizes here as well (instead of within `collect_stats_from_eval` as we are now), but don't have enough time to do the refactoring
    
    losing_teams_series = [pd.Series(losing_teams, name="Losing Team")]
    losing_team_df = pd.concat(losing_teams_series, axis=1)
    losing_team_df.index.name = "Trial #"
    losing_team_df.to_csv((run_path / "LOSING_TEAM_ACROSS_TRIALS.csv").resolve())


############
# GRAPHING #
############

def find_attack_prob_with_team_progress(team_attacks_arrs: Iterator[np.array]):
    total_attack_opportunities_per_timestep = defaultdict(int)
    total_attacks_per_timestep = defaultdict(int)
    for team_attack_arr in team_attacks_arrs:
        for i, attacked in team_attack_arr:
            total_attack_opportunities_per_timestep[i] += 1
            if attacked:
                total_attacks_per_timestep[i] += 1
    xs = list(sorted(total_attack_opportunities_per_timestep.keys()))
    ys = [(total_attacks_per_timestep[x] / total_attack_opportunities_per_timestep[x]) for x in xs]
    return np.array(xs), np.array(ys)

def find_attack_prob_with_overall_progress(list_of_attacks_list: Iterator[Iterator[np.array]]):
    total_attack_opportunities_per_timestep = defaultdict(int)
    total_attacks_per_timestep = defaultdict(int)
    for attacks_list in list_of_attacks_list:
        for agent_attacks_arr in attacks_list:
            for i, attacked in agent_attacks_arr:
                total_attack_opportunities_per_timestep[i] += 1
                if attacked:
                    total_attacks_per_timestep[i] += 1
    xs = list(sorted(total_attack_opportunities_per_timestep.keys()))
    ys = [(total_attacks_per_timestep[x] / total_attack_opportunities_per_timestep[x]) for x in xs]
    return np.array(xs), np.array(ys)

def render_graph(out_filepath, team1_xs, team2_xs, team1_ys, team2_ys, team1_name, team2_name, x_title, y_title, graph_title):
    pass