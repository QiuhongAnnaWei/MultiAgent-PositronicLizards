# MultiAgent-PositronicLizards

## Installation
### Option 1: venv (most recently tested)
```
python3 -m venv lizard-env
source lizard-env/bin/activate
pip install -r requirements.txt
```
### Option 2: Conda (NEEDS TO BE UPDATED FOR SB3)
```conda create -n lizards python=3.7``` <br>
```conda activate lizards```<br>
```pip install -r requirements.txt```<br>
<!-- ```pip install git+https://github.com/hill-a/stable-baselines``` (see [here](https://github.com/hill-a/stable-baselines/issues/849) for reason for separate pip)<br>  -->
```pip install git+https://github.com/DLR-RM/stable-baselines3``` (see [here](https://github.com/hill-a/stable-baselines/issues/849) for reason for separate pip)<br> 
```pip install magent```
 <!-- stable-baselic3nes==2.10.2 -->
<!-- simple-rl==0.811 -->

## Directory Structure:
```
root/
  lizards/
    trained_policies/
    experiments.py
    main_utils.py
    test.py (deprecated)
  README.md
  requirements.txt
```


Some helpful `pettingzoo` links:
* [API for interacting with environments](https://www.pettingzoo.ml/api#interacting-with-environments)
* We are running environments from the [MAgent](https://www.pettingzoo.ml/magent) codebase:
  * [Tiger Deer](https://www.pettingzoo.ml/magent/tiger_deer)
  * [Adversarial Pursuit](https://www.pettingzoo.ml/magent/adversarial_pursuit)


Contribution guidelines:
* Commit and push as soon as you can!
* Make the commit messages descriptive
* Update the requirements.txt
* Update the README as necessary

## 11/6/21 - Ben and Yongming
* Rendered random agents playing Adversarial Pursuit
* Rendered random agents playing Tiger Deer

## 11/13/21 - Ben
* Migrated to stable-baselines3. Some important links:
  * [Docs - examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)
  * [Docs - PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
  * [Github Repo](https://github.com/DLR-RM/stable-baselines3)
* Now using [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) to convert pettingzoo environments into sb3 parallel envs.
  * [Here is an important section on the Github readme](https://github.com/Farama-Foundation/SuperSuit#environment-vectorization)
* Followed [this TowardsDataScience article](https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b) for some healthy boilerplate.
  * Can now train PPO agents on [Battle](https://www.pettingzoo.ml/magent/battle) and [Battlefield](https://www.pettingzoo.ml/magent/battlefield) environments and observe their policies.
* Set up `main_utils.py` and boilerplate `experiments.py`
