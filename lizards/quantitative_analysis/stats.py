from main_utils import *
import ray.rllib.agents.ppo as ppo
import supersuit as ss
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path

""" TO DOs:
1. Replace print statements with an actual logger
"""

def save_eval_viz(save_dir, save_file_prefix, diff_frame_list, width, height):
    save_dir = Path(save_dir)

    mp4_save_path = str(save_dir.joinpath(f"{save_file_prefix}.mp4"))
    gif_save_path = str(save_dir.joinpath(f"{save_file_prefix}.gif"))

    print(f"# Saving gif to: {gif_save_path}")
    diff_frame_list[0].save(gif_save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)

    print(f"# Saving gif to: {mp4_save_path}")
    video = cv2.VideoWriter(mp4_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    for i, image in enumerate(diff_frame_list):
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    for i in [0, len(diff_frame_list)-1]:
        frame_path = str(save_dir.joinpath(f"{save_file_prefix}_{i}.jpg"))
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


def collect_stats_from_eval(representative_trainer, env, env_config, policy_fn, log_dir, max_iter=2 ** 16, is_battle=True, eval_id="", save_viz=False):
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
    most_recent_done_agent = None # (informs the eventual value of losing_team).

    # Loop through all agents until a team has lost:
    for i, agent in enumerate(env.agent_iter(max_iter=max_iter)):
        if i % 1000 == 0:
            # Printing this helps see iteration status over time.
            print(f"getting actions for: iter {i} ...")

        # Get info for current agent.
        observation, _, done, _ = env.last() # (observation[:,:,3/4]==0).sum()

        if done:
            curr_agent_action = None
            most_recent_done_agent = agent
        else:
            # Find action for agent, from current policy and observation.
            curr_agent_policy = representative_trainer.get_policy(policy_fn(agent, None))
            batch_obs = { 'obs': np.expand_dims(observation, 0)} # (10,10,5) -> (1,10,10,5)
            batched_action, _, _ = curr_agent_policy.compute_actions_from_input_dict(batch_obs)
            curr_agent_action = batched_action[0]
            record_attacks(curr_agent_action, agent, team_attack_statuses, agent_attack_statuses, agent_to_team_func)
       
        # Get current state (or, if game is over via exception).
        env_state = None
        try:
            env_state = env.state() # (map_size, map_size, 5)
            record_hp(env_state, team_hp_values)
            # print("env_state", len(env_state))
        except Exception as e:
            print("Team has lost; indicated by exception:", e)
            print(f"\nAt {i}: one team eliminated - env.agents = {env.agents}")
            if losing_team is None:
                losing_team = most_recent_done_agent.split("_")[0]
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
        save_eval_viz(log_dir, f"viz_{eval_id}", diff_frame_list, width, height)

    agent_attacks_df, team_attacks_df, team_hps_df = package_attacks_data(team_attack_statuses, agent_attack_statuses, team_hp_values)

    # print("red hps:", team_hp_values["red"][-1])
    # print("blue hps:", team_hp_values["blue"][-1])
    # log(f"{log_dir}.txt", f"attacks per agent {total_attacks_per_agent}") 
    return losing_team, agent_attacks_df, team_attacks_df, team_hps_df



# Will prob move this df-making code out once I add the HP code
# make the df; using the concat method coz arrays not of same len

# list_of_atks_across_timesteps_per_agent_series = [pd.Series(atks_across_timesteps, name=agent_nm) for agent_nm, atks_across_timesteps in agent_attack_statuses.items()]
# attacks_df = pd.concat(list_of_atks_across_timesteps_per_agent_series, axis=1)
# attacks_df.index.name = "Timesteps"

# if save_df:
#     df_csv_savepath = log_dir.joinpath("attacks_data.csv")
#     attacks_df.to_csv(df_csv_savepath)
#     print(f"attacks df saved at {df_csv_savepath}")

