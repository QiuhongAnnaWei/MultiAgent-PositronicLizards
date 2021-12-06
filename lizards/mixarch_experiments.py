import os

from main_utils import *
from pettingzoo.magent import adversarial_pursuit_v3, tiger_deer_v3, battle_v3, battlefield_v3, combined_arms_v5
import ray.rllib.agents.ppo as ppo
import numpy as np
import json
from tensorflow import Tensor
import uuid
from datetime import datetime
from experiments import env_directory, env_spaces, TeamPolicyConfig


battle_arch = {
    "new_arch": [[7, [5, 5], 2], [21, [3, 3], 2], [21, [4,4], 1]], # (13,13,5) -> (7,5,5) -> (21,3,3) -> (21,1,1)
    "old_arch": [[21, 13, 1]] 
}

def ray_experiment_BF_training_arch(*args):
    env_name = 'battlefield'
    env_config = {'map_size': 55}
    print(f"\nCONFIG: env_config = {env_config}")
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    train_iters = 50
    log_intervals = 10
    gpu = False
    # [out_channels, kernel, stride] 
    new_arch = [[7, [3, 3], 1], [21, [3, 3], 2], [21, [7,7], 1]] # 7(13x13)-3(7x7)-1(1x1) filters
    old_arch = [[21, 13, 1]]
    # log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),  f"logs/PPO_battlefield_10-iters-test")
    # print(f"\n### (ray_experiment_BF_training_arch) `log_dir` has been set to {log_dir} ###\n")
    if False:
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        trainer_config["model"]["conv_filters"] = new_arch
        print(f"\nCONFIG: model-conv_filters = {trainer_config['model']['conv_filters']}")
        trainer = ppo.PPOTrainer(config=trainer_config)
        checkpoint = train_ray_trainer(trainer, num_iters=train_iters, log_intervals=log_intervals, log_dir=log_dir)
    else:        
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        trainer_config["model"]["conv_filters"] = new_arch
        temp_trainer = ppo.PPOTrainer(config=trainer_config)
        temp_trainer.restore('logs/ccv/PPO_battlefield_10-iters__85fbe/checkpoint_000090/checkpoint-90')
        red_new_weights = temp_trainer.get_policy("red_shared").get_weights()
        temp_trainer.stop()

        trainer_config["model"]["conv_filters"] = old_arch
        temp_trainer = ppo.PPOTrainer(config=trainer_config)
        temp_trainer.restore('logs/ccv/PPO_battlefield_10-iters__fa60e/checkpoint_000060/checkpoint-60')
        blue_old_weights = temp_trainer.get_policy("blue_shared").get_weights()
        temp_trainer.stop()

        policy_dict["red_shared"] = (policy_dict["red_shared"][0], policy_dict["red_shared"][1], policy_dict["red_shared"][2], 
                { "model": {  "conv_filters": new_arch, "conv_activation": "relu" }})
        policy_dict["blue_shared"] = (policy_dict["blue_shared"][0], policy_dict["blue_shared"][1], policy_dict["blue_shared"][2], 
                { "model": { "conv_filters": old_arch, "conv_activation": "relu" }})
        env_config = {'map_size': 100}
        trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
        del trainer_config["model"]
        trainer = ppo.PPOTrainer(config=trainer_config)
        trainer.get_policy("red_shared").set_weights(red_new_weights) # transfer the weights
        trainer.get_policy("blue_shared").set_weights(blue_old_weights)
        checkpoint = None
    
    render_from_checkpoint(checkpoint, trainer, env_directory[env_name], env_config, policy_fn, max_iter=10000, savefile=True) 
    rewards = evaluate_policies(checkpoint, trainer, battlefield_v3, env_config, policy_fn, max_iter=10000)
    print("\n### (ray_experiment_BF_training_arch) POLICY EVALUATION: REWARDS ###")
    for key in rewards:
        print(f"{key}: {rewards[key]}")


def ray_BA_selfplay_evaluate(env_name = 'battle', gpu = False):
    env_config = {'map_size': 30}
    policy_dict = {'all': (None, env_spaces[env_name]['obs_space'], env_spaces[env_name]['action_space'], dict())}
    policy_fn = lambda *args, **kwargs: 'all'
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    trainer = ppo.PPOTrainer(config=trainer_config)
    checkpoint = 'logs/battle/PPO_battle_self-play_120-iters__75eeb/checkpoint_000120/checkpoint-120'
    render_from_checkpoint(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True) 
    rewards, rewards_log = evaluate_policies(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000)
    print("\npolicy_evaluation_rewards = \n")
    for key in rewards:
        print(f"{key}: {rewards[key]}")

def ray_experiment_BA_arch_traineval(env_name = 'battle', gpu = False, evaluate=True):
    # old arch by default
    env_config = {'map_size': 19}
    train_iters = 80
    log_intervals = 20
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    trainer = ppo.PPOTrainer(config=trainer_config)
    if evaluate:
        checkpoint = 'logs/battle/PPO_battle_pretrained_200-iters__14b27/checkpoint_000200/checkpoint-200'
        render_from_checkpoint(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True) 
    else:
        policy_log_str = "".join([p.for_filename() for p in team_data])
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),f"logs/PPO_{env_name}{policy_log_str}_{train_iters}-iters__{uuid.uuid4().hex[:5]}")
        print(f"# log_dir set to {log_dir}")
        checkpoint = train_ray_trainer(trainer, num_iters=train_iters, log_intervals=log_intervals, log_dir=log_dir,
                        render=True, env=battle_v3, env_config=env_config, policy_fn=policy_fn, max_iter=10000, s_battle=True)
    rewards, rewards_log = evaluate_policies(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000)
    print("\npolicy_evaluation_rewards = \n")
    for key in rewards:
        print(f"{key}: {rewards[key]}")

def ray_experiment_BA_arch_traineval_pretrained(env_name = 'battle', gpu = False, evaluate=False):
    pt_ckpt = 'logs/PPO_battle_100-iters__ms19_cad08/checkpoint_000200/checkpoint-200'
    pt_team = "red_shared"
    env_config = {'map_size': 19}
    train_iters = 200
    log_intervals = 20
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    # trainer_config["model"]["conv_filters"] = battle_arch[eval_config["red_arch"]]
    # transfer the weights
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(pt_ckpt)
    pt_weights = temp_trainer.get_policy(pt_team).get_weights()
    temp_trainer.stop()
    trainer = ppo.PPOTrainer(config=trainer_config)
    trainer.get_policy(pt_team).set_weights(pt_weights)

    if evaluate:
        checkpoint = 'logs'
        trainer.restore(checkpoint)
        render_from_checkpoint(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True) 
    else:
        policy_log_str = "".join([p.for_filename() for p in team_data])
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),f"logs/PPO_{env_name}{policy_log_str}_pretrained_{train_iters}-iters__{uuid.uuid4().hex[:5]}")
        checkpoint = train_ray_trainer(trainer, num_iters=train_iters, log_intervals=log_intervals, log_dir=log_dir,
                        render=True, env=battle_v3, env_config=env_config, policy_fn=policy_fn, max_render_iter=10000, is_battle=True)
    rewards, rewards_log = evaluate_policies(checkpoint, trainer, battle_v3, env_config, policy_fn, max_iter=10000)
    print("\npolicy_evaluation_rewards = \n")
    for key in rewards:
        print(f"{key}: {rewards[key]}")
    print(f"# log_dir = {log_dir}")

def ray_BA_random_evaluate(env_name = 'battle', gpu = False):
    """Evaluate checkpoint (red) against a random policy (blue)"""
        # log set up
    timestamp = int(datetime.timestamp(datetime.now())) #%10000
    logname = f"logs/battle/evaluation/ms19_vsrandom/{timestamp}/{timestamp}.txt"
    if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    log(logname, logname)
    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    log(logname, [f"\nenv_config = {env_config}", f"\neval_env_config = {eval_env_config}"])
    eval_config = {
        "red_ckpt": "logs/battle/ccv_battle_newarch_ms19_ca3ee/checkpoint_000200/checkpoint-200",
        "red_load": "red_shared",
        "red_arch": "new_arch",
    }
    log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])
    # trainer_config set up
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    trainer_config["model"]["conv_filters"] = battle_arch[eval_config["red_arch"]]
    # transfer red weights (blue is random)
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["red_ckpt"])
    red_weights = temp_trainer.get_policy(eval_config["red_load"]).get_weights()
    temp_trainer.stop()
    trainer_config["env_config"] = eval_env_config
    evaluator = ppo.PPOTrainer(config=trainer_config)
    evaluator.get_policy("red_shared").set_weights(red_weights)
    # evaluate
    render_from_checkpoint(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True, logname=logname[:-4]) 
    rewards, rewards_log = evaluate_policies(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000)
    log(logname, ["\npolicy_evaluation_rewards = \n", json.dumps(rewards)])
    with open(os.path.join(os.path.split(logname)[0], "rewardslog.json"), "w") as f:
        f.write(json.dumps(rewards_log))
    print(f"\n{logname}")
    
def ray_BA_mixarch_evaluate(env_name = 'battle', gpu = False):
    # log set up
    timestamp = int(datetime.timestamp(datetime.now())) #%10000
    logname = f"logs/battle/evaluation/{timestamp}/{timestamp}.txt"
    if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    log(logname, logname)
    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    log(logname, [f"\nenv_config = {env_config}", f"\neval_env_config = {eval_env_config}"])
    team_data = [TeamPolicyConfig('red'), TeamPolicyConfig('blue')]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    eval_config = {
        "red_ckpt": "logs/battle/ccv_battle_newarch_ms19_ca3ee/checkpoint_000020/checkpoint-20",
        "red_load": "red_shared",
        "red_arch": "new_arch",
        "blue_ckpt": "logs/battle/ccv_battle_newarch_ms19_ca3ee/checkpoint_000200/checkpoint-200",
        "blue_load": "blue_shared",
        "blue_arch": "new_arch",
    }
    log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])
    # restore red weights
    trainer_config["model"]["conv_filters"] = battle_arch[eval_config["red_arch"]]
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["red_ckpt"])
    red_weights = temp_trainer.get_policy(eval_config["red_load"]).get_weights()
    temp_trainer.stop()
    # restore blue weights
    trainer_config["model"]["conv_filters"] = battle_arch[eval_config["blue_arch"]]
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["blue_ckpt"])
    blue_weights = temp_trainer.get_policy(eval_config["blue_load"]).get_weights()
    temp_trainer.stop()
    # evaluate
    policy_dict["red_shared"] = (policy_dict["red_shared"][0], policy_dict["red_shared"][1], policy_dict["red_shared"][2], 
            { "model": {  "conv_filters": battle_arch[eval_config["red_arch"]], "conv_activation": "relu" }})
    policy_dict["blue_shared"] = (policy_dict["blue_shared"][0], policy_dict["blue_shared"][1], policy_dict["blue_shared"][2], 
            { "model": { "conv_filters":  battle_arch[eval_config["blue_arch"]], "conv_activation": "relu" }})
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    del trainer_config["model"]
    trainer_config["env_config"] = eval_env_config
    evaluator = ppo.PPOTrainer(config=trainer_config)
    evaluator.get_policy("red_shared").set_weights(red_weights) # transfer the weights
    evaluator.get_policy("blue_shared").set_weights(blue_weights)
    render_from_checkpoint(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True, logname=logname[:-4]) 
    rewards, rewards_log = evaluate_policies(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000)
    log(logname, ["\npolicy_evaluation_rewards = \n", json.dumps(rewards)])
    with open(os.path.join(os.path.split(logname)[0], "rewardslog.json"), "w") as f:
        f.write(json.dumps(rewards_log))
    print(f"\n{logname}")



def ray_BA_split_evaluate(env_name = 'battle', gpu = False):
    timestamp = int(datetime.timestamp(datetime.now())) #%10000
    logname = f"logs/battle/evaluation/{timestamp}/{timestamp}.txt"
    if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    log(logname, logname)
    # config set up
    env_config = {'map_size': 19}
    log(logname, f"\nenv_config = {env_config}")
    counts = get_num_agents(env_directory[env_name], env_config)
    team_data = [ TeamPolicyConfig('red', method='split', count=counts['red']),
                  TeamPolicyConfig('blue', method='split', count=counts['blue']) ]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    eval_config = {
        "ckpt": "logs/battle/PPO_battle_red-split_blue-split_120-iters__dcd52/checkpoint_000120/checkpoint-120",
    }
    log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])

    trainer = ppo.PPOTrainer(config=trainer_config)
    render_from_checkpoint(eval_config["ckpt"], trainer, battle_v3, env_config, policy_fn, max_iter=10000, 
        savefile=True, is_battle=True, logname=logname[:-4]) 
    rewards, rewards_log = evaluate_policies(eval_config["ckpt"], trainer, battle_v3, env_config, policy_fn, max_iter=10000)
    log(logname, ["\npolicy_evaluation_rewards = \n", json.dumps(rewards)])
    with open(os.path.join(os.path.split(logname)[0], "rewardslog.json"), "w") as f:
        f.write(json.dumps(rewards_log))
    print(f"\n{logname}")

def ray_BA_random_split_evaluate(env_name = 'battle', gpu = False):
    """Evaluate checkpoint (red) against a random policy (blue)"""
        # log set up
    timestamp = int(datetime.timestamp(datetime.now())) #%10000
    logname = f"logs/battle/evaluation/ms19_vsrandom/{timestamp}/{timestamp}.txt"
    if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    log(logname, logname)
    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    log(logname, [f"\nenv_config = {env_config}", f"\neval_env_config = {eval_env_config}"])
    counts = get_num_agents(env_directory[env_name], env_config)
    team_data = [ TeamPolicyConfig('red', method='split', count=counts['red']),
                  TeamPolicyConfig('blue', method='split', count=counts['blue']) ]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    red_load_team = "blue"
    log(logname, f"\nred_load_team = {red_load_team}")
    eval_config = {
        "red_ckpt": "logs/battle/PPO_battle_red-split_blue-split_120-iters__dcd52/checkpoint_000120/checkpoint-120",
        "red_load": [f"{red_load_team}_{i}" for i in range(1)]
    }
    log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])
    # transfer red weights (blue is random)
    red_weights = []
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["red_ckpt"])
    for e in eval_config["red_load"]:
        red_weights.append(temp_trainer.get_policy(e).get_weights())
    temp_trainer.stop()
    trainer_config["env_config"] = eval_env_config
    evaluator = ppo.PPOTrainer(config=trainer_config)
    for i, e in enumerate(eval_config["red_load"]):
        evaluator.get_policy(f"red_{i}").set_weights(red_weights[i])
    # evaluate
    render_from_checkpoint(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True, logname=logname[:-4]) 
    rewards, rewards_log = evaluate_policies(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000)
    log(logname, ["\npolicy_evaluation_rewards = \n", json.dumps(rewards)])
    with open(os.path.join(os.path.split(logname)[0], "rewardslog.json"), "w") as f:
        f.write(json.dumps(rewards_log))
    print(f"\n{logname}")

def ray_BA_mixarch_split_evaluate(env_name = 'battle', gpu = False):
    # log set up
    timestamp = int(datetime.timestamp(datetime.now())) #%10000
    logname = f"logs/battle/evaluation/ms19_splitsplit/{timestamp}/{timestamp}.txt"
    if not os.path.exists(os.path.split(logname)[0]): os.makedirs(os.path.split(logname)[0])
    log(logname, logname)
    # config set up
    env_config = {'map_size': 19}
    eval_env_config = {'map_size': 19}
    log(logname, [f"\nenv_config = {env_config}", f"\neval_env_config = {eval_env_config}"])
    counts = get_num_agents(env_directory[env_name], env_config)
    team_data = [ TeamPolicyConfig('red', method='split', count=counts['red']),
                  TeamPolicyConfig('blue', method='split', count=counts['blue']) ]
    policy_dict, policy_fn = get_policy_config(**env_spaces[env_name], team_data=team_data)
    trainer_config = get_trainer_config(env_name, policy_dict, policy_fn, env_config, gpu=gpu)
    red_load_team = "red"
    blue_load_team = "blue"
    log(logname, [f"\nred_load_team = {red_load_team}", f"\nblue_load_team = {blue_load_team}"])
    eval_config = {
        "red_ckpt": "logs/battle/PPO_battle_red-split_blue-split_120-iters__dcd52/checkpoint_000120/checkpoint-120",
        "red_load":  [f"{red_load_team}_{i}" for i in range(counts[red_load_team])],
        "blue_ckpt": "logs/battle/PPO_battle_red-split_blue-split_120-iters__dcd52/checkpoint_000020/checkpoint-20",
        "blue_load": [f"{blue_load_team}_{i}" for i in range(counts[blue_load_team])]
    }
    log(logname, ["\neval_config = ", json.dumps(eval_config, indent=2)])
    # restore red weights
    red_weights, blue_weights = [], []
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["red_ckpt"])
    for e in eval_config["red_load"]:
        red_weights.append(temp_trainer.get_policy(e).get_weights())
    temp_trainer.stop()
    # restore blue weights
    temp_trainer = ppo.PPOTrainer(config=trainer_config)
    temp_trainer.restore(eval_config["blue_ckpt"])
    for e in eval_config["blue_load"]:
        red_weights.append(temp_trainer.get_policy(e).get_weights())
    temp_trainer.stop()
    # evaluate
    trainer_config["env_config"] = eval_env_config
    evaluator = ppo.PPOTrainer(config=trainer_config)
    for i, e in enumerate(eval_config["red_load"]):
        evaluator.get_policy(e).set_weights(red_weights[i])
    for i, e in enumerate(eval_config["blue_load"]):
        evaluator.get_policy(e).set_weights(red_weights[i])
    render_from_checkpoint(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000, savefile=True, is_battle=True, logname=logname[:-4]) 
    rewards, rewards_log = evaluate_policies(None, evaluator, battle_v3, env_config, policy_fn, max_iter=10000)
    log(logname, ["\npolicy_evaluation_rewards = \n", json.dumps(rewards)])
    with open(os.path.join(os.path.split(logname)[0], "rewardslog.json"), "w") as f:
        f.write(json.dumps(rewards_log))
    print(f"\n{logname}")

if __name__ == "__main__":
    # kwargs = parse_args()
    for env_name, env in env_directory.items():
        auto_register_env_ray(env_name, env)

    # print(os.environ.get("RLLIB_NUM_GPUS", "0"))
    # ray.init(num_gpus=1, local_mode=True)
    
    # ray_experiment_BF_training_arch()

    ray_BA_selfplay_evaluate()
    # ray_experiment_BA_arch_traineval()
    # ray_experiment_BA_arch_traineval_pretrained()
    # ray_BA_random_evaluate()
    # ray_BA_mixarch_evaluate()

    # ray_BA_split_evaluate()
    # ray_BA_random_split_evaluate()
    # ray_BA_mixarch_split_evaluate()