# Generative Actor-Critic Algorithm

This is an implementation of a new off-policy algorithm, Generative Actor-Critic (GAC). (IJCAI Paper ID: 6795, Generative Actor-Critic: An Off-policy Algorithm Using the Push-forward Model)

For reproduction, this implementation depends on the open source project, [Spinningup](https://spinningup.openai.com/en/latest/user/introduction.html).
If you install Spinningup successfully, then you can run this implementation.

## Installation 

1. Install Spinningup: [tutorial](https://spinningup.openai.com/en/latest/user/installation.html#). (You have to install MuJoCo to reproduce the experiments of the paper.)
2. Copy the directory `gac` which includes two files `core.py` and `gac.py` to the Spinningup project directory `spinningup/spinup/algos/pytorch/`.
3. Add the following code into `spinningup/spinup/__init__.py`:

    ```python
    from spinup.algos.pytorch.gac.gac import gac as gac_pytorch
    ```

4. Change the list variable `BASE_ALGO_NAMES` at `spinningup/spinup/run.py:32`:

    ```python
    BASE_ALGO_NAMES = ['vpg', 'trpo', 'ppo', 'ddpg', 'td3', 'sac', 'gac']
    ```

5. Change the dict variable `DEFAULT_BACKEND` at `spinningup/spinnup/user_config.py:6`:

    ```python
    DEFAULT_BACKEND = {
        'vpg': 'pytorch',
        'trpo': 'tf1',
        'ppo': 'pytorch',
        'ddpg': 'pytorch',
        'td3': 'pytorch',
        'sac': 'pytorch',
        'gac': 'pytorch',
    }
    ```

## Reproduction

Here are some bash commands for experiments.

1. `Humanoid-v3`

    ```bash
    nohup python -m spinup.run gac_pytorch --env Humanoid-v3 --exp_name Humanoid_gac_auto --hid [400,300] --epochs 750 --alpha -1.2 --alpha_min 1.0 --alpha_max 1.8 --device cuda:0 --seed 123 > Humanoid_gac_auto_s123.log 2>&1 &
    ```

2. `HumanoidStandup-v2`

    ```bash
    nohup python -m spinup.run gac_pytorch --env HumanoidStandup-v2 --exp_name HumanoidStandup_gac_auto --hid [400,300] --epochs 750 --alpha -1.2 --alpha_min 1.0 --alpha_max 1.8 --reward_scale 0.05 --device cuda:0 --seed 123 > HumanoidStandup_gac_auto_s123.log 2>&1 &
    ```

3. `Ant-v3`

    ```bash
    nohup python -m spinup.run gac_pytorch --env Ant-v3 --exp_name Ant_gac_auto --hid [400,300] --epochs 750 --alpha -1.0 --alpha_min 0.7 --alpha_max 1.4 --device cuda:0 --seed 123 > Ant_gac_auto_s123.log 2>&1 &
    ```

4. `Hopper-v3`

    ```bash
    nohup python -m spinup.run gac_pytorch --env Hopper-v3 --exp_name Hopper_gac_auto --hid [400,300] --epochs 750 --alpha -0.5 --alpha_min 0.3 --alpha_max 0.8 --device cuda:0 --seed 123 > Hopper_gac_auto_s123.log 2>&1 &
    ```

5. `Walker2d-v3`

    ```bash
    nohup python -m spinup.run gac_pytorch --env Walker2d-v3 --exp_name Walker2d_gac_auto --hid [400,300] --epochs 750 --alpha -1.0 --alpha_min 0.7 --alpha_max 1.4 --device cuda:0 --seed 123 > Walker2d_gac_auto_s123.log 2>&1 &
    ```

6. `HalfCheetah-v3`

    ```bash
    nohup python -m spinup.run gac_pytorch --env HalfCheetah-v3 --exp_name HalfCheetah_gac_auto --hid [400,300] --epochs 750 --alpha -1.2 --alpha_min 1.0 --alpha_max 1.8 --device cuda:0 --seed 123 > HalfCheetah_gac_auto_s123.log 2>&1 &
    ```