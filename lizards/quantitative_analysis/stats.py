from main_utils import *
import ray.rllib.agents.ppo as ppo
import supersuit as ss
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path

def save_eval_viz(save_dir, save_file_prefix, diff_frame_list):
    save_dir = Path(save_dir)
    mp4_save_path = save_dir.joinpath("{save_file_prefix}.mp4")

    # TO DO
    pass


    
    # save_path = f"{logname}.gif"
    # log(f"{logname}.txt", f"\n# Saving gif to: {save_path}")
    # diff_frame_list[0].save(save_path, save_all=True, append_images=diff_frame_list[1:], duration=100, loop=0)
    # save_path = f"{logname}.mp4"
    # log(f"{logname}.txt", f"\n# Saving video to: {save_path}\n")
    # video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))
    # for i, image in enumerate(diff_frame_list):
    #     video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # for i in [0, len(diff_frame_list)-1]:
    #     diff_frame_list[i].save(f"{logname}_{i}.jpg")


def collect_stats_from_eval(checkpoint_path, trainer, env, env_config, policy_fn, max_iter=2 ** 8, is_battle=True, log_dir=None, eval_id="", save_df=True, save_viz=False):
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

    attacksPerAgent, attacksTotal, attacksPerTeam = defaultdict(int), defaultdict(list), defaultdict(int)

    i = 0
    for agent in env.agent_iter(max_iter=max_iter):
        if i % 1000 == 0:
            print(f"getting actions for: iter {i} ...")
        observation, reward, done, info = env.last() # (observation[:,:,3/4]==0).sum()
        if done:
            action = None
            attacksTotal[agent].append(action)
        else:
            agentpolicy = policy_fn(agent, None)  # map agent id to policy id
            policy = trainer.get_policy(agentpolicy)
            batch_obs = { 'obs': np.expand_dims(observation, 0)} # (10,10,5) -> (1,10,10,5)
            batched_action, state_out, info = policy.compute_actions_from_input_dict(batch_obs)
            single_action = batched_action[0]
            action = single_action 
            if is_battle and 12 < action <= 20:
                # 0-20 action space
                attacksPerAgent[agent] += 1
                if agent.startswith("blue"):
                    attacksPerTeam["blue"] += 1
                else:
                    attacksPerTeam["red"] += 1
            attacksTotal[agent].append(12 < action <= 20)
        try:
            s = env.state() # (map_size, map_size, 5)
        except:
            #log(f"{log_dir}.txt", f"\nAt {i}: one team eliminated - env.agents = {env.agents}") 
            break
        # out = False
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
    if save_viz: save_eval_viz(save_dir, f"viz_{eval_id}", diff_frame_list)


    # Will prob move this df-making code out once I add the HP code
    # make the df; using the concat method coz arrays not of same len
    list_of_atks_across_timesteps_per_agent_series = [pd.Series(atks_across_timesteps, name=agent_nm) for agent_nm, atks_across_timesteps in attacksTotal.items()]
    attacksDf = pd.concat(list_of_atks_across_timesteps_per_agent_series, axis=1)
    attacksDf.index.name = "Timesteps"

    if save_df:
        df_csv_savepath = log_dir.joinpath("attacks_data.csv")
        attacksDf.to_csv(df_csv_savepath)
        print(f"attacks df saved at {df_csv_savepath}")
    # log(f"{log_dir}.txt", f"attacks per agent {attacksPerAgent}") 
    return attacksPerAgent, attacksTotal, attacksPerTeam, attacksDf


