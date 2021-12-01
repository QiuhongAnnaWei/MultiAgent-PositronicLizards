from ray.tune.trainable import Trainable


import pytz
from datetime import datetime
from copy import deepcopy
import pandas as pd
from typing import List, Dict
import numpy.typing as npt
import numpy as np
from functools import partial
from pathlib import Path

def list_map(*args): return list(map(*args))
def tup_map(*args): return tuple(map(*args))
def np_itermap(*args, dtype=bool): return np.fromiter(map(*args), dtype=dtype)

def get_weights(trainer: Trainable, team: str):
    """
    Gets the weights for a given team.
    trainer: RLib trainer
    team: team name for use in the dictionary
    """
    return trainer.get_policy(team).get_weights()

def get_timestamp():
    tz = pytz.timezone('US/Eastern')
    short_timestamp = datetime.now(tz).strftime("%d.%m_%H.%M")
    return short_timestamp

def pairwise_eq_chk(team_name, wt_dict1, wt_dict2):
    """
    Returns True iff wts for team_name in both dicts are the same  
    Assumes keys are the same for both dicts; see examples below 
    """
    d1 = wt_dict1[team_name]
    d2 = wt_dict2[team_name]

    return np.array([True if np.array_equal(d1[key], d2[key]) else False for key in d1]).all()

def check_eq_policy_wts_across_iters(pol_wts_across_iters: List[Dict[str, Dict[str, npt.ArrayLike]]], team_names: List[str]):
    """ 
    Given a list of policy weights across iterations, checks if the pol wts at each iteration is equal to the previous one
    pol_wts_across_iters's first value must be the initial random wts for each team; i.e., the wts at iteration 0 
    So at idx i of pol_wts_across_iters, we'll have the pol weights __at the end of__ the i-th iteration (where the idxing is 0-based)
    """
    return {team: tup_map(partial(pairwise_eq_chk, team), pol_wts_across_iters, pol_wts_across_iters[1:]) for team in team_names}

def get_changepoints(eq_chk_dict: dict):
    # False here means: the pol wts at that iteration not equal to those at previous iter
    return {team: np.nonzero(np.array(eq_chk_dict[team])==False) for team in eq_chk_dict}


