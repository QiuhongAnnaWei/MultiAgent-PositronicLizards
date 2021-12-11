from main_utils import *
import ray.rllib.agents.ppo as ppo
import supersuit as ss
import os
import numpy as np
from collections import defaultdict
import pandas as pd

def get_agent_attacks(checkpoint, trainer, env, env_config, policy_fn, max_iter=2 ** 8, is_battle=False, logname = None, save_df = True):
    if checkpoint:
        trainer.restore(checkpoint)

    if logname is None:
        logname = checkpoint
    else:
        if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    env = env.env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    i = 0
    attacksPerAgent = defaultdict(0)
    attacksTotal = defaultdict([])
    attacksPerTeam = defaultdict(0)

    for agent in env.agent_iter(max_iter=max_iter):
        if i % 1000 == 0:
            print(f"getting actions for: iteration {i} ...")
        observation, reward, done, info = env.last() # (observation[:,:,3/4]==0).sum()
        if done:
            action = None
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
                if agent.starts_with("blue"):
                    attacksPerTeam["blue"] += 1
                else:
                    attacksPerTeam["red"] += 1
            attacksTotal[agent].append(12 < action <= 20)
        try:
            s = env.state() # (map_size, map_size, 5)
        except:
            #log(f"{logname}.txt", f"\nAt {i}: one team eliminated - env.agents = {env.agents}") 
            break
        # out = False
        env.step(action)
        i += 1
    env.close()
    attacksDf = pd.DataFrame(attacksTotal)

    if save_df:
        attacksDf.to_csv(logname)
    log(f"{logname}.txt", f"attacks per agent {attacksPerAgent}") 
    return attacksPerAgent, attacksTotal, attacksPerTeam