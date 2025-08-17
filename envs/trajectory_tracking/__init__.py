# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Aerial Manipulator environment for hovering.
"""

import gymnasium as gym

from . import trajectory_tracking_env
from .trajectory_tracking_env import AerialManipulatorTrajectoryTrackingEnv, AerialManipulator0DOFTrajectoryTrackingEnvCfg, AerialManipulator0DOFDebugTrajectoryTrackingEnvCfg
from .trajectory_tracking_env import AerialManipulator0DOFLongArmTrajectoryTrackingEnvCfg, AerialManipulator0DOFQuadOnlyTrajectoryTrackingEnvCfg
from .trajectory_tracking_env import AerialManipulator0DOFSmallArmCOMVehicleTrajectoryTrackingEnvCfg, AerialManipulator0DOFSmallArmCOMMiddleTrajectoryTrackingEnvCfg, AerialManipulator0DOFSmallArmCOMEndEffectorTrajectoryTrackingEnvCfg
from .trajectory_tracking_env_2dof import AerialManipulator2DOFTrajectoryTrackingEnvCfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-AerialManipulator-2DOF-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env_2dof:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator2DOFTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-LongArm-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFLongArmTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-AerialManipulator-QuadOnly-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFQuadOnlyTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-Vehicle-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMVehicleTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-Middle-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMMiddleTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-SmallArmCOM-EndEffector-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFSmallArmCOMEndEffectorTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-AerialManipulator-0DOF-Debug-TrajectoryTracking-v0",
    entry_point = "envs.trajectory_tracking.trajectory_tracking_env:AerialManipulatorTrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AerialManipulator0DOFDebugTrajectoryTrackingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "cleanrl_cfg_entry_point": f"{agents.__name__}:cleanrl_ppo_cfg.yaml",
    },
)





# gym.register(
#     id="Isaac-AerialManipulator-Hover-Vehicle-v0",
#     entry_point = "envs.hover.hover_env_vehicle:AerialManipulatorHoverEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": AerialManipulatorHoverEnvCfgVehicle,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml"
#     },
# )