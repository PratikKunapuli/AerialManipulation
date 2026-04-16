# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial Manipulator ball-throw evaluation environment.
"""

import gymnasium as gym

from . import ball_throw_env
from .ball_throw_env import (
    BallThrowEnv,
    BallThrowEnvCfg,
    BallThrowWithMotorDynamicsCfg,
)
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-AerialManipulator-2DOF-BallThrow-v0",
    entry_point="envs.ball_throw.ball_throw_env:BallThrowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallThrowWithMotorDynamicsCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BallThrowPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)
