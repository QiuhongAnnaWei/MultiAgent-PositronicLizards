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

def save_eval_viz(save_dir, save_file_prefix, diff_frame_list):
    save_dir = Path(save_dir)

    mp4_save_path = save_dir.joinpath("{save_file_prefix}.mp4")
    gif_save_path = save_dir.joinpath("{save_file_prefix}.gif")

    print(f"# Saving gif to: {gif_save_path}")
    diff_frame_list[0].save(gif_save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)

    print(f"# Saving gif to: {mp4_save_path}")
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    for i, image in enumerate(diff_frame_list):
        video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    for i in [0, len(diff_frame_list)-1]:
        diff_frame_list[i].save(f"{save_file_prefix}_{i}.jpg")

def record_attacks(is_battle, action, agent, attacks_per_agent, attacks_per_team, attacks_total):
    action_is_atk = 12 < action <= 20

    if is_battle and action_is_atk:
        # 0-20 action space
        attacks_per_agent[agent] += 1
        if agent.startswith("blue"):
            attacks_per_team["blue"] += 1
        else:
            attacks_per_team["red"] += 1
    attacks_total[agent].append(action_is_atk)


def package_attacks_data(attacks_per_agent, attacks_per_team, attacks_total):
    list_of_atks_across_timesteps_per_agent_series = [pd.Series(atks_across_timesteps, name=agent_nm) for agent_nm, atks_across_timesteps in attacks_total.items()]

    attacks_df = pd.concat(list_of_atks_across_timesteps_per_agent_series, axis=1)
    attacks_df.index.name = "Timesteps"

    attacks_data =  {"df": attacks_df,
                     "attacks_per_agent": attacks_per_agent,
                     "attacks_per_team": attacks_per_team,
                     "attacks_total": attacks_total}

    return attacks_data


def record_hp(env_state, team_red_hps, team_blue_hps):
    hp_red = env_state[:,:,2]
    hp_blue = env_state[:,:,4]
    # not summing b/c don't want to lose individual agents' info (and can sum later)

    team_red_hps.append(hp_red)
    team_blue_hps.append(hp_blue)


def collect_stats_from_eval(checkpoint_path, trainer, env, env_config, policy_fn, max_iter=2 ** 8, is_battle=True, log_dir=None, eval_id="", save_viz=False):
    """
    Runs an eval and collects stats (attacks, hp) from it 
    eval_id is an id tt we can use to distinguish between eval runs
    saves video of the eval run if save_viz == True
    """
    if checkpoint_path:
        trainer.restore(checkpoint_path)

    if log_dir is None:
        log_dir = Path(checkpoint_path).parents[0]
    if not log_dir.is_dir(): log_dir.mkdir()
    
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env.reset()

    if save_viz:
        diff_frame_list = []
        width, height, img = None, None, None

    attacks_per_agent, attacks_total, attacks_per_team, losing_team = defaultdict(int), defaultdict(list), defaultdict(int), None
    team_red_hps, team_blue_hps = [], [] 

    i = 0
    done_agents = set()
    most_recent_done_agent = None
    for agent in env.agent_iter(max_iter=max_iter):
        # print(i,":", agent)
        if i % 1000 == 0:
            print(f"getting actions for: iter {i} ...")
        observation, reward, done, info = env.last() # (observation[:,:,3/4]==0).sum()

        if agent in done_agents:
            print(agent, "is already done.")
        if done:
            action = None
            attacks_total[agent].append(action)
            print("Adding", agent, "to done_agents.")
            done_agents.add(agent)
            most_recent_done_agent = agent
        else:
            agentpolicy = policy_fn(agent, None)  # map agent id to policy id
            policy = trainer.get_policy(agentpolicy)
            batch_obs = { 'obs': np.expand_dims(observation, 0)} # (10,10,5) -> (1,10,10,5)
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action 

            record_attacks(is_battle, action, agent, attacks_per_agent, attacks_per_team, attacks_total)

        try:
            env_state = env.state() # (map_size, map_size, 5)
            # print("env_state", len(env_state))
        except Exception as e:
            print("e:", e)
            env_state = None
            #log(f"{log_dir}.txt", f"\nAt {i}: one team eliminated - env.agents = {env.agents}") 
            print(f"\nAt {i}: one team eliminated - env.agents = {env.agents}")
            print("done_agents:", done_agents)
            if not losing_team:
                losing_team = most_recent_done_agent.split("_")[0]
            break
            # I have no idea why `break` needs to be commented out when im running it
            # (Eli): I don't need to comment it out?

        if env_state is not None:
            record_hp(env_state, team_red_hps, team_blue_hps)

        env.step(action)
    
        if save_viz:
            img2 = PIL.Image.fromarray(env.render(mode='rgb_array'))
            if img is None:
                width, height = img2.width, img2.height
                img = np.zeros((width, height))
            if np.array_equal(np.array(img), np.array(img2)) == False:
                img = img2.copy()
                ImageDraw.Draw(img2).text( (2, height-10), f"iter={i}", (0,0,0))
                if is_battle:
                    ImageDraw.Draw(img2).text( (2, 13), f"HP={str(round((s[:,:,2]).sum(), 2))}",  (0,0,0)) 
                    ImageDraw.Draw(img2).text( (width-55, 13), f"HP={str(round((s[:,:,4]).sum(), 2))}",  (0,0,0)) # "{:.4f}".format()
                diff_frame_list.append(img2)
        i += 1
    
    env.close()
    if save_viz: save_eval_viz(log_dir, f"viz_{eval_id}", diff_frame_list)

    attacks_data = package_attacks_data(attacks_per_agent, attacks_per_team, attacks_total)
    hp_data = {"team_red_hps": team_red_hps,
               "team_blue_hps": team_blue_hps}
    # print("red hps:", team_red_hps[-1])
    # print("blue hps:", team_blue_hps[-1])
    # log(f"{log_dir}.txt", f"attacks per agent {attacks_per_agent}") 
    return losing_team, attacks_data, hp_data



# Will prob move this df-making code out once I add the HP code
# make the df; using the concat method coz arrays not of same len

# list_of_atks_across_timesteps_per_agent_series = [pd.Series(atks_across_timesteps, name=agent_nm) for agent_nm, atks_across_timesteps in attacks_total.items()]
# attacks_df = pd.concat(list_of_atks_across_timesteps_per_agent_series, axis=1)
# attacks_df.index.name = "Timesteps"

# if save_df:
#     df_csv_savepath = log_dir.joinpath("attacks_data.csv")
#     attacks_df.to_csv(df_csv_savepath)
#     print(f"attacks df saved at {df_csv_savepath}")

