# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quadcopter environment.
"""

import gymnasium as gym

from . import ball_catching_quadrotor_env
from .ball_catching_quadrotor_env import QuadrotorEnv, QuadrotorEnvCfg, QuadrotorManipulatorEnvCfg, QuadrotorManipulatorLongEnvCfg
from .ball_catching_quadrotor_env import BrushlessQuadrotorEnvCfg, BrushlessQuadrotorManipulatorEnvCfg
from . import agents

##
# Register Gym environments.
##



gym.register(
    id="Isaac-Crazyflie-Hover-v0",
    entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrotorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Crazyflie-CTBR-Hover-v0",
    entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BrushlessQuadrotorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Crazyflie-0DOF-Hover-v0",
    entry_point="envs.ball_catching.ball_catching_quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BrushlessQuadrotorManipulatorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Crazyflie-0DOF-Ball_Catch-v0",
#     entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": BrushlessQuadrotorManipulatorEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#     },
# )


gym.register(
    id="Isaac-CrazyflieManipulator-SRT-Hover-v0",
    entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrotorManipulatorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CrazyflieManipulator-CTBR-Hover-v0",
    entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrotorManipulatorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-CrazyflieManipulatorLong-SRT-Hover-v0",
    entry_point="envs.crazyflie_ctatt.quadrotor_env:QuadrotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadrotorManipulatorLongEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadrotorPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

