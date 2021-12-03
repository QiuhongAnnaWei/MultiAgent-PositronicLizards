from ray.tune.trainable import Trainable
from main_utils import *
import pytz
from datetime import datetime
from copy import deepcopy
import pandas as pd
from typing import List, Dict
from collections.abc import Iterable

import numpy.typing as npt
import numpy as np
from functools import partial
from pathlib import Path
import time

# Helper constants
# ================

convs = {"adversarial-pursuit": [[13, 10, 1]],
         "battle": [[21, 13, 1]],
         "battlefield": [[21, 13, 1]],
         "tiger-deer": [[9, 9, 1]],
         "combined-arms": [[25, 13, 1]]}


# Functions
# =========

# Generic helper utils
def map_wrapper(*args, collector_func): return collector_func(map(*args))
list_map = partial(map_wrapper, collector_func=list)
tup_map = partial(map_wrapper, collector_func=tuple)

def get_timestamp():
    tz = pytz.timezone('US/Eastern')
    short_timestamp = datetime.now(tz).strftime("%d.%m_%H.%M")
    return short_timestamp


# Training / logging related utils



# TO DO: This has not been tested yet, and likely has bugs
def save_results_dicts_pol_wts(results_dicts: List[Dict], policy_weights_for_iters: Iterable, policy_ids: Iterable, gen_dynamic_info):

    timestamp, log_dir = gen_dynamic_info["timestamp"], gen_dynamic_info["log_dir"]

    test_flag = "_TEST" if gen_dynamic_info["test_mode"] else ""
    r_num, b_num = gen_dynamic_info["r_num"], gen_dynamic_info["b_num"]
    full_log_dir = log_dir.joinpath(f"r{r_num}_b{b_num}_{timestamp}{test_flag}")
    if not full_log_dir.is_dir(): full_log_dir.mkdir()

    def savepath(suffix): return full_log_dir.joinpath(suffix)

    # Save results dict
    results_save_path = savepath("results_stats.csv")
    results_df = pd.DataFrame(results_dicts)
    results_df.to_csv(results_save_path)

    print(f"results_dicts saved to {results_save_path}")

    # Save raw pol wts
    policy_save_path = savepath("policy_stats.csv")
    pd.DataFrame(policy_weights_for_iters).to_csv(policy_save_path)
    
    # Save and print changepoints
    changepoints = get_changepoints(check_eq_policy_wts_across_iters(policy_weights_for_iters, policy_ids))
    changepts_save_path = savepath("changepoints.csv")
    
    list_of_changept_series = [pd.Series(changepoints[policy_id], name=policy_id) for policy_id in policy_ids]
    pd.concat(list_of_changept_series, axis=1).to_csv(changepts_save_path)

    return changepoints, results_df


def train_for_pol_wt_freezing(trainer: Trainable, const_exp_info, gen_dynamic_info):
    num_iters, log_intervals, log_dir = gen_dynamic_info["num_iters"], gen_dynamic_info["log_intervals"], gen_dynamic_info["log_dir"]

    timestamp = gen_dynamic_info["timestamp"] if gen_dynamic_info["timestamp"] is not None else get_timestamp()
    if not log_dir.is_dir(): log_dir.mkdir()

    policy_ids = list(trainer.get_config()["multiagent"]["policies"].keys())
    policyset_to_start_with = const_exp_info["policyset_to_start_with"]

    true_start = time.time()

    results_dicts = []
    policy_weights_for_iters = []

    # NOTE: We might want to stream the weights into a file, instead of trying to keep an array of all the wts across all the iters in memory
    def get_and_log_wts(trainer):
        copied_policy_wts_from_local_worker = deepcopy(trainer.get_weights())
        # there was an issue on Rllib github tt made me think they might not be careful enough about managing state and refs when it comes to policy wts
        policy_weights_for_iters.append(copied_policy_wts_from_local_worker)

        # for testing:
        changepoints = get_changepoints(check_eq_policy_wts_across_iters(policy_weights_for_iters, policy_ids))
        print(f"changepoints are:\n {changepoints}")
        return changepoints


    # Pre-training stuff
    # 1. Log the 0-th iteration
    get_and_log_wts(trainer) 

    # 2. Set trainable policy to only the policy to start with
    trainer.workers.foreach_worker(lambda worker: worker.set_policies_to_train(policyset_to_start_with))


    # Training loop
    for i in range(num_iters):
        print(f"Starting training on iter {i + 1}...")
        start = time.time()
        
        result = trainer.train()
        results_dicts.append(result)

        print(f"batch {i + 1}: took {time.time() - start} seconds")

        get_and_log_wts(trainer) 
        # could use a callback instead for this tho, e.g. https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        # but for now I like the idea of using a causal pathway distinct from tt which we are manipulating

        # TO DO: Chk log interval code
        if log_intervals is not None:
            if (i + 1) % log_intervals == 0:
                checkpoint_path = trainer.save(str(log_dir))
                print("checkpoint saved at", checkpoint_path)

    print(f"Full training took {(time.time() - true_start) / 60.0} min")
    checkpoint_path = trainer.save(str(log_dir)); print("checkpoint saved at", checkpoint_path)

    save_results_dicts_pol_wts(results_dicts, policy_weights_for_iters, policy_ids, gen_dynamic_info)
    
    return results_dicts, policy_weights_for_iters




# Weight getting and changepoint recording utils

def get_weights(trainer: Trainable, policy_id: str):
    """
    Gets the weights for a given policy_id.
    trainer: RLib trainer
    policy_id: policy_id for use in the dictionary
    """
    return trainer.get_policy(policy_id).get_weights()

def pairwise_eq_chk(policy_id, wt_dict1, wt_dict2):
    """
    Returns True iff wts for policy_id in both dicts are the same  
    Assumes keys are the same for both dicts; see examples below 
    """
    wt_dict1 = wt_dict1[policy_id]
    wt_dict2 = wt_dict2[policy_id]

    bool_list_from_wtname_map = [True if np.array_equal(wt_dict1[wt_name], wt_dict2[wt_name]) else False for wt_name in wt_dict1]

    return np.array(bool_list_from_wtname_map).all()

def get_test_base_egs():
    test_dict_blue_123 = {"blue": {'blue/conv_value_1/bias': np.array([1, 2, 3])}}
    test_dict_blue_123_copy = deepcopy(test_dict_blue_123)
    test_dict_blue_321 = {"blue": {'blue/conv_value_1/bias': np.array([3, 2, 1])}}
    test_dict_blue_321_copy = deepcopy(test_dict_blue_321)

    return test_dict_blue_123, test_dict_blue_123_copy, test_dict_blue_321, test_dict_blue_321_copy

def get_test_br_egs():
    test_dict_br_123 = {"blue": {'blue/conv_value_1/bias': np.array([1, 2, 3])},
                        "red": {'red/conv_value_1/bias': np.array([4, 9])}}
    test_dict_br_123_copy = deepcopy(test_dict_br_123)
    test_dict_br_321 = {"blue": {'blue/conv_value_1/bias': np.array([3, 2, 1])},
                        "red": {'red/conv_value_1/bias': np.array([4, 9])}}
    test_dict_br_321_copy = deepcopy(test_dict_br_321)

    return test_dict_br_123, test_dict_br_123_copy, test_dict_br_321, test_dict_br_321_copy


def test_pairwise_eq_chk():
    test_dict_blue_123, test_dict_blue_123_copy, test_dict_blue_321, test_dict_blue_321_copy = get_test_base_egs()

    assert pairwise_eq_chk("blue", test_dict_blue_123, test_dict_blue_123_copy) == True
    assert pairwise_eq_chk("blue", test_dict_blue_123, test_dict_blue_321) == False

test_pairwise_eq_chk()


def check_eq_policy_wts_across_iters(pol_wts_across_iters: List[Dict[str, Dict[str, npt.ArrayLike]]], policy_ids: List[str]):
    """ 
    Given a list of policy weights across iterations, checks if the pol wts at each iteration is equal to the previous one
    pol_wts_across_iters's first value must be the initial random wts for each team; i.e., the wts at iteration 0 
    So at idx i of pol_wts_across_iters, we'll have the pol weights from __the end of__ the i-th iteration (where the idxing is 0-based)
    """
    chk_equality_across_iters = lambda policy_id: tup_map(partial(pairwise_eq_chk, policy_id), pol_wts_across_iters, pol_wts_across_iters[1:])
    return {pol_id: chk_equality_across_iters(pol_id) for pol_id in policy_ids}

def test_check_eq_policy_wts_across_iters():
    test_dict_br_123, test_dict_br_123_copy, test_dict_br_321, test_dict_br_321_copy = get_test_br_egs()
    test_pw_across_iters = [test_dict_br_123, test_dict_br_123_copy, test_dict_br_321, test_dict_br_321_copy]
    test_pw_eq_chk_dict = {'blue': (True, False, True), 'red': (True, True, True)}
    assert check_eq_policy_wts_across_iters(test_pw_across_iters, ["blue", "red"]) == test_pw_eq_chk_dict

test_check_eq_policy_wts_across_iters()


def get_changepoints(eq_chk_dict: Dict[str, Iterable]):
    """
    Expects a dict with the structure: str (the policy id) => Tuple of bools
    False at i-th idx of tuple means: the pol wts at (i+1)th iteration not equal to those at previous iter,
    where training begins with iter 1.

    Returns dict of arrays of idxes of iterations at which policy wts changed
    """
    changepts_vec_for_policy_id = lambda policy_id: (np.flatnonzero(np.array(eq_chk_dict[policy_id])==False) + 1)

    return {policy_id: changepts_vec_for_policy_id(policy_id) for policy_id in eq_chk_dict}

def valarray_eq(dict1: Dict[str, npt.ArrayLike], dict2: Dict[str, npt.ArrayLike]):
    """ Given two dicts with same keys, check their associated np arrays for equality """
    bool_list_from_key_map = [True if np.array_equal(dict1[key], dict2[key]) else False for key in dict1]

    return np.array(bool_list_from_key_map).all()


def test_get_changepoints():
    test_pw_eq_chk_dict0 = {'blue': (), 'red': ()}
    test_pw_eq_chk_dict1 = {'blue': (True, False, True), 'red': (True, True, True)}
    test_pw_eq_chk_dict2 = {'blue': (True, False, True, False, False), 'red': (True, True, True, True, False)}

    assert valarray_eq(test_pw_eq_chk_dict0, {'blue': np.array([]), 'red': np.array([])}) == True 

    assert valarray_eq(get_changepoints(test_pw_eq_chk_dict1), {'blue': np.array([2]), 'red': np.array([])}) == True
    
    assert valarray_eq(get_changepoints(test_pw_eq_chk_dict2), {'blue': np.array([2, 4, 5]), 'red': np.array([5])}) == True

test_get_changepoints()




# red got trained on: iter 1, 4






