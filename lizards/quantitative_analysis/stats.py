from main_utils import *
import ray.rllib.agents.ppo as ppo
import supersuit as ss
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path


def collect_stats_from_eval(checkpoint_path, trainer, env, env_config, policy_fn, max_iter=2 ** 8, is_battle=True, log_dir = None, save_df = True):
    if checkpoint_path:
        trainer.restore(checkpoint_path)

    if log_dir is None:
        log_dir = Path(checkpoint_path).parents[0]
    if not log_dir.is_dir(): log_dir.mkdir()
    
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    i = 0
    attacksPerAgent = defaultdict(int)
    attacksTotal = defaultdict(list)
    attacksPerTeam = defaultdict(int)

    env.reset()

    for agent in env.agent_iter(max_iter=max_iter):
        if i % 1000 == 0:
            print(f"getting actions for: iteration {i} ...")
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
        i += 1
    env.close()

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


