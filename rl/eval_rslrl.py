import argparse
import sys 
from isaaclab.app import AppLauncher

# local imports
from utils import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL. ")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")
parser.add_argument("--baseline", action="store_true", default=False, help="Use baseline policy.")
parser.add_argument("--baseline_gains", type=str, default=None, help="Baseline gains to use.")
parser.add_argument("--use_integral_terms", type=bool, default=False, help="Use integral terms in the controller.")
parser.add_argument("--case_study", type=bool, default=False, help="Use case study policy.")
parser.add_argument("--save_prefix", type=str, default="", help="Prefix for saving files.")
parser.add_argument("--follow_robot", type=int, default=-1, help="Follow robot index.")
parser.add_argument(
    "--compare_b3_geometric",
    action="store_true",
    default=False,
    help=(
        "Additionally capture the exact followed-robot case for a "
        "side-by-side B3 geometric replay. Existing RSL-RL "
        "evaluation behavior is unchanged."
    ),
)


# RL_B3_GEOMETRIC_COMPARISON_BENCHMARK_V1
parser.add_argument(
    "--benchmark_pre_hover_s",
    type=float,
    default=2.0,
    help=(
        "Policy-controlled RL hover duration before the "
        "--compare_b3_geometric position step."
    ),
)
parser.add_argument(
    "--benchmark_goal_offset",
    type=float,
    nargs=3,
    default=(0.02416658, 0.37813187, 0.03000116),
    metavar=("DX", "DY", "DZ"),
    help=(
        "World XYZ goal offset used for the RL-vs-Mellinger "
        "comparison, relative to the initial end-effector target."
    ),
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


# ============================================================================
# B3_GEOMETRIC_COMPARE_SINGLE_ENTRYPOINT_V1
#
# User-facing behavior:
#
#   python -m rl.eval_rslrl ... --compare_b3_geometric
#
# launches:
#
#   1. this normal RL evaluator in its own Isaac process;
#   2. after that process has COMPLETELY exited, the exact matched B3 geometric
#      replay in a fresh Isaac process.
#
# This outer invocation itself never starts Isaac.
#
# Runs WITHOUT --compare_b3_geometric follow the original evaluator path exactly.
# ============================================================================

import json as _compare_json
import os as _compare_os
import subprocess as _compare_subprocess
import tempfile as _compare_tempfile
from pathlib import Path as _ComparePath


_compare_is_rl_child = (
    _compare_os.environ.get(
        "AERIAL_COMPARE_RL_CHILD",
        "0",
    )
    == "1"
)


if (
    args_cli.compare_b3_geometric
    and not _compare_is_rl_child
):
    if args_cli.follow_robot < 0:
        parser.error(
            "--compare_b3_geometric requires "
            "--follow_robot N."
        )

    if (
        args_cli.num_envs is not None
        and args_cli.follow_robot
        >= args_cli.num_envs
    ):
        parser.error(
            f"--follow_robot={args_cli.follow_robot} "
            f"but --num_envs={args_cli.num_envs}."
        )

    _compare_repo = _ComparePath(
        "/home/sumukh/AerialManipulation"
    )

    _compare_helper = (
        _compare_repo
        / "rl"
        / "b3_geometric_rslrl_compare.py"
    )

    if not _compare_helper.is_file():
        raise RuntimeError(
            "Internal B3 geometric replay helper is missing: "
            f"{_compare_helper}"
        )

    # Small temporary handoff file. The RL child writes the exact paths
    # of the case and RL trace it produced.
    _fd, _manifest_path = (
        _compare_tempfile.mkstemp(
            prefix="aerial_rl_b3_geometric_",
            suffix=".json",
        )
    )

    _compare_os.close(_fd)

    # Delete the empty file. Its later existence will prove that the
    # RL child actually completed the comparison capture.
    _compare_os.unlink(
        _manifest_path
    )

    _original_args = list(
        sys.argv[1:]
    )

    _rl_env = _compare_os.environ.copy()

    _rl_env[
        "AERIAL_COMPARE_RL_CHILD"
    ] = "1"

    _rl_env[
        "AERIAL_COMPARE_MANIFEST"
    ] = _manifest_path

    _rl_command = [
        sys.executable,
        "-m",
        "rl.eval_rslrl",
        *_original_args,
    ]

    print()
    print("=" * 100)
    print("RL + B3 BEHAVIOR-GEOMETRIC — SINGLE ENTRYPOINT")
    print("=" * 100)
    print(
        "Selected robot :",
        args_cli.follow_robot,
    )
    print(
        "Phase 1/2      : RL evaluation"
    )
    print("=" * 100)
    print()

    _rl_result = _compare_subprocess.run(
        _rl_command,
        cwd=str(_compare_repo),
        env=_rl_env,
    )

    if _rl_result.returncode != 0:
        raise RuntimeError(
            "RL evaluation child failed with "
            f"return code {_rl_result.returncode}."
        )

    # The subprocess is now completely gone. There is no live first
    # SimulationApp when Mellinger starts.
    if not _compare_os.path.isfile(
        _manifest_path
    ):
        raise RuntimeError(
            "RL evaluation finished, but the comparison "
            "handoff manifest was not produced."
        )

    with open(
        _manifest_path,
        "r",
        encoding="utf-8",
    ) as _manifest_file:
        _manifest = _compare_json.load(
            _manifest_file
        )

    required_manifest_keys = (
        "task",
        "robot_index",
        "case_path",
        "rl_trace_path",
        "output_dir",
        "device",
        "video",
    )

    for key in required_manifest_keys:
        if key not in _manifest:
            raise RuntimeError(
                "Comparison manifest is missing "
                f"{key!r}."
            )

    for filename in (
        _manifest["case_path"],
        _manifest["rl_trace_path"],
    ):
        if not _compare_os.path.isfile(
            filename
        ):
            raise RuntimeError(
                "Required RL -> Mellinger handoff "
                f"file does not exist: {filename}"
            )

    _mellinger_command = [
        sys.executable,
        str(_compare_helper),
        "--task",
        str(_manifest["task"]),
        "--case_path",
        str(_manifest["case_path"]),
        "--rl_trace_path",
        str(_manifest["rl_trace_path"]),
        "--output_dir",
        str(_manifest["output_dir"]),
        "--device",
        str(_manifest["device"]),
    ]

    if bool(_manifest["video"]):
        _mellinger_command.append(
            "--video"
        )

    print()
    print("=" * 100)
    print(
        "Phase 1/2 complete."
    )
    print(
        "The RL Isaac process has fully exited."
    )
    print()
    print(
        "Phase 2/2      : B3 behavior-geometric replay"
    )
    print(
        "Selected robot :",
        _manifest["robot_index"],
    )
    print(
        "Output         :",
        _manifest["output_dir"],
    )
    print("=" * 100)
    print()

    # ============================================================
    # RL -> B3 BEHAVIOR-GEOMETRIC REPLAY
    #
    # The RL child has already written the exact replay case and RL
    # trajectory.  The second process receives that captured task.
    #
    # No manually injected static goal is allowed here: the geometric
    # controller must receive the same relative EE displacement realized
    # by the RL benchmark.
    # ============================================================
    _geometric_env = _compare_os.environ.copy()

    # Prevent a manual debugging goal from leaking into the production
    # RL -> geometric comparison.
    _geometric_env.pop(
        "AERIAL_STATIC_GC_GOAL_OFFSET",
        None,
    )

    _geometric_env.update(
        {
            "B3_BEHAVIOR_GEOM_ENABLE": "1",
            "B3_PREHOVER_S": str(
                args_cli.benchmark_pre_hover_s
            ),
            "B3_CONTINUE_FROM_PREHOVER": "1",
            "B3_REAL_GOAL_DELIVERY_DELAY_S": "0",
            "B3_GYRO_LPF_ENABLE": "0",
            "B3_DIFF_ACTUATOR_ENABLE": "0",
            "B3_UNIFIED_ACTUATOR_ENABLE": "0",
            "B3_SIM_TRAJ_ENABLE": "0",
        }
    )

    try:
        _mellinger_result = (
            _compare_subprocess.run(
                _mellinger_command,
                cwd=str(_compare_repo),
                env=_geometric_env,
            )
        )

        if _mellinger_result.returncode != 0:
            raise RuntimeError(
                "B3 behavior-geometric replay failed with "
                f"return code "
                f"{_mellinger_result.returncode}."
            )

    finally:
        if _compare_os.path.exists(
            _manifest_path
        ):
            _compare_os.unlink(
                _manifest_path
            )

    print()
    print("=" * 100)
    print(
        "RL + B3 BEHAVIOR-GEOMETRIC COMPARISON COMPLETE"
    )
    print("=" * 100)
    print(
        "Robot     :",
        _manifest["robot_index"],
    )
    print(
        "Artifacts :",
        _manifest["output_dir"],
    )
    print("=" * 100)

    # Critical: the outer supervisor must never fall through to
    # AppLauncher and accidentally start a third Isaac application.
    raise SystemExit(0)

# always enable cameras to record video
# args_cli.enable_cameras = True
args_cli.enable_cameras = args_cli.video
args_cli.headless = True # make false to see the simulation

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os
import random
import time
from dataclasses import dataclass
import ast
import re
# import ruamel.yaml as yaml
import yaml

import gymnasium as gym
import envs
from controllers.decoupled_controller import DecoupledController
from controllers.gc_params import gc_params_dict
import utils.export_utilities as export_utils


from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab.utils.io import load_yaml

import numpy as np
import torch

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # torch.manual_seed(args_cli.seed)
    # env_cfg = parse_env_cfg(
    #     args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    # )

    
    # agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print("Resume path: ", resume_path)
    log_dir = os.path.dirname(resume_path)
    print("Log dir: ", log_dir)
    
    if not args_cli.baseline:
        policy_path = log_dir
    else:
        # policy_path = "./baseline_0dof/"
        # policy_path = "./baseline_0dof_com_lqr_tune/"
        # policy_path = "./baseline_0dof_com_reward_tune/"
        # policy_path = "./baseline_0dof_ee_reward_tune/"
        # policy_path = "./baseline_0dof_ee_lqr_tune/"

        
        # env_cfg.sim_rate_hz = 100
        # env_cfg.policy_rate_hz = 50
        # env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
        # env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
        # env_cfg.sim.render_interval = env_cfg.decimation
        env_cfg.gc_mode = True
        if "Crazyflie" in args_cli.task:
            env_cfg.task_body = "body"
            env_cfg.goal_body = "body"
            # env_cfg.reward_task_body = "endeffector"
            # env_cfg.reward_goal_body = "endeffector"
            env_cfg.reward_task_body = "body"
            env_cfg.reward_goal_body = "body"

            # policy_path = "./baseline_cf_0dof/"
        else:
            env_cfg.task_body = "COM"
            env_cfg.goal_body = "COM"
            env_cfg.reward_task_body = "root"
            env_cfg.reward_goal_body = "root"

            # policy_path = "./baseline_0dof_ee_reward_tune/"

        task_name = args_cli.task            
        if args_cli.use_integral_terms:
            task_name = args_cli.task + "-Integral"
        elif args_cli.baseline_gains is not None:
            task_name = args_cli.task + "-" + args_cli.baseline_gains
        
        if task_name in gc_params_dict.keys():
            policy_path = gc_params_dict[task_name]["log_dir"]
        else:
            print(f"[ERROR] Task name {task_name} not found in gc_params_dict.")
            print(f"Available tasks: {gc_params_dict.keys()}")
            return
            

        # env_cfg.yaw_distance_reward_scale = 5.0
    # else:
    #     print("\n\nSaved args: ", saved_args_cli)
    #     print("Keys: ", saved_args_cli.keys())
    #     env_cfg = update_env_cfg(env_cfg, saved_args_cli)

    env_cfg.eval_mode = True
    env_cfg.viewer.resolution = (1920, 1080)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # =================================================================
    # RL_B3_GEOMETRIC_COMPARISON_BENCHMARK_V1
    # =================================================================
    comparison_measurement_episode_length_s = float(
        env_cfg.episode_length_s
    )

    if args_cli.compare_b3_geometric:
        if args_cli.baseline:
            raise ValueError(
                "--compare_b3_geometric is for the trained RL policy, "
                "not --baseline."
            )

        num_envs_compare = int(env_cfg.scene.num_envs)

        if not (
            0 <= int(args_cli.follow_robot) < num_envs_compare
        ):
            raise ValueError(
                "--follow_robot must select a valid RL environment: "
                f"got {args_cli.follow_robot} for "
                f"{num_envs_compare} environment(s)."
            )

        if float(args_cli.benchmark_pre_hover_s) < 0.0:
            raise ValueError(
                "--benchmark_pre_hover_s must be >= 0."
            )

        # Static staging trajectory. The authoritative EE target is
        # installed after RSL-RL's reset.
        env_cfg.trajectory_type = "lissaajous"
        env_cfg.trajectory_horizon = 0
        env_cfg.random_shift_trajectory = False

        env_cfg.lissajous_amplitudes = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_amplitudes_rand_ranges = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_frequencies = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_frequencies_rand_ranges = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_phases = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_phases_rand_ranges = [
            0.0, 0.0, 0.0, 0.0
        ]
        env_cfg.lissajous_offsets = [
            0.0, 0.0, 3.0, 0.0
        ]
        env_cfg.lissajous_offsets_rand_ranges = [
            0.0, 0.0, 0.0, 0.0
        ]

        # "fixed" deliberately injects a -360 deg/s roll state in this
        # environment, so use rand with zero ranges instead.
        env_cfg.init_cfg = "rand"
        env_cfg.init_pos_ranges = [0.0, 0.0, 0.0]
        env_cfg.init_lin_vel_ranges = [0.0, 0.0, 0.0]
        env_cfg.init_yaw_ranges = [0.0]
        env_cfg.init_ang_vel_ranges = [0.0, 0.0, 0.0]

        if hasattr(env_cfg, "rotorpy_done"):
            env_cfg.rotorpy_done = False

        # The environment timeout includes the warm-up. Add the warm-up
        # so that the measured phase still has the original full horizon.
        env_cfg.episode_length_s = (
            comparison_measurement_episode_length_s
            + float(args_cli.benchmark_pre_hover_s)
        )
    

    # If ".hydra/config.yaml" is present, load some of the reward scalars from there
    if os.path.exists(os.path.join(log_dir, "params/env.yaml")):
        with open(os.path.join(log_dir, "params/env.yaml")) as f:
            hydra_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)
        # f = os.path.join(log_dir, "params/env.yaml")
        # loader = yaml.YAML(typ="safe")
        # hydra_cfg = loader.load(f)

        
        if "use_yaw_representation" in hydra_cfg:
            env_cfg.use_yaw_representation = hydra_cfg["use_yaw_representation"]
        if "use_full_ori_matrix" in hydra_cfg:
            env_cfg.use_full_ori_matrix = hydra_cfg["use_full_ori_matrix"]
        if not ("Ball" in args_cli.task):
            if "scale_reward_with_time" in hydra_cfg:
                env_cfg.scale_reward_with_time = hydra_cfg["scale_reward_with_time"]
            if "yaw_error_reward_scale" in hydra_cfg:
                env_cfg.yaw_error_reward_scale = hydra_cfg["yaw_error_reward_scale"]
            if "yaw_distance_reward_scale" in hydra_cfg:
                env_cfg.yaw_distance_reward_scale = hydra_cfg["yaw_distance_reward_scale"]
            if "yaw_smooth_transition_scale" in hydra_cfg:
                env_cfg.yaw_smooth_transition_scale = hydra_cfg["yaw_smooth_transition_scale"]
            if "yaw_radius" in hydra_cfg:
                env_cfg.yaw_radius = hydra_cfg["yaw_radius"]
            if "pos_distance_reward_scale" in hydra_cfg:
                env_cfg.pos_distance_reward_scale = hydra_cfg["pos_distance_reward_scale"]
            if "pos_error_reward_scale" in hydra_cfg:
                env_cfg.pos_error_reward_scale = hydra_cfg["pos_error_reward_scale"]
            if "lin_vel_reward_scale" in hydra_cfg:
                env_cfg.lin_vel_reward_scale = hydra_cfg["lin_vel_reward_scale"]
            if "ang_vel_reward_scale" in hydra_cfg:
                env_cfg.ang_vel_reward_scale = hydra_cfg["ang_vel_reward_scale"]
            if "combined_alpha" in hydra_cfg:
                env_cfg.combined_alpha = hydra_cfg["combined_alpha"]
            if "combined_tolerance" in hydra_cfg:
                env_cfg.combined_tolerance = hydra_cfg["combined_tolerance"]
            if "combined_reward_scale" in hydra_cfg:
                env_cfg.combined_reward_scale = hydra_cfg["combined_reward_scale"]

    # else:
    #     yaml_base = "./logs/rsl_rl/AM_0DOF_Hover/2024-09-14_14-38-12_rsl_rl_test_default_1024_env_pos_distance_15_yaw_error_-2.0_no_smooth_transition_full_ori"
    #     with open(os.path.join(yaml_base, "params/env.yaml"), "r") as f:
    #         hydra_cfg = yaml.load(f, Loader=yaml.FullLoader)
    #         if "use_yaw_representation" in hydra_cfg:
    #             env_cfg.use_yaw_representation = hydra_cfg["use_yaw_representation"]
    #         if "yaw_error_reward_scale" in hydra_cfg:
    #             env_cfg.yaw_error_reward_scale = hydra_cfg["yaw_error_reward_scale"]
    #         if "yaw_distance_reward_scale" in hydra_cfg:
    #             env_cfg.yaw_distance_reward_scale = hydra_cfg["yaw_distance_reward_scale"]
    #         if "yaw_smooth_transition_scale" in hydra_cfg:
    #             env_cfg.yaw_smooth_transition_scale = hydra_cfg["yaw_smooth_transition_scale"]
    #         if "yaw_radius" in hydra_cfg:
    #             env_cfg.yaw_radius = hydra_cfg["yaw_radius"]
            
    #         if "use_full_ori_matrix" in hydra_cfg:
    #             env_cfg.use_full_ori_matrix = hydra_cfg["use_full_ori_matrix"]
            
    #         if "scale_reward_with_time" in hydra_cfg:
    #             env_cfg.scale_reward_with_time = hydra_cfg["scale_reward_with_time"]

    # env_cfg.yaw_radius = 0.5
    
    if env_cfg.use_yaw_representation:
        # env_cfg.num_observations += 4
        env_cfg.num_observations += 1
    
    if env_cfg.use_full_ori_matrix:
        # env_cfg.num_observations += 6
        env_cfg.num_observations += 9

    if "Traj" in args_cli.task:
        env_cfg.goal_cfg = "rand"
        # env_cfg.trajectory_params["x_amp"] = 1.0
        # env_cfg.trajectory_params["x_freq"] = 0.5
        # env_cfg.trajectory_params["y_amp"] = 2.0
        # env_cfg.trajectory_params["y_freq"] = 1.0
        # env_cfg.trajectory_params["z_amp"] = 0.0
        # env_cfg.trajectory_params["z_offset"] = 0.5
        # env_cfg.trajectory_params["yaw_amp"] = 1.0
        # env_cfg.trajectory_params["yaw_freq"] = 1.0
        # env_cfg.traj_update_dt = 1.0
        # env_cfg.traj_update_dt = 2.0

    env_cfg.seed = args_cli.seed

    # import code; code.interact(local=locals())
    print("\n\nUpdated env cfg: ", env_cfg)

    robot_index_prefix = ""
    if args_cli.case_study:
        # Manual override of env cfg
        env_cfg.goal_cfg = "fixed"
        # env_cfg.goal_pos = [0.0, 0.0, 0.5]
        env_cfg.goal_pos = [0.0, 0.0, 3.0]
        env_cfg.goal_ori = [0.7071068, 0.0, 0.0, 0.7071068]
        env_cfg.init_cfg = "default"

        # Camera settings
        if "Crazyflie" in args_cli.task:
            env_cfg.viewer.eye = (0.25, 0.25, 3.25)
            # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        else:
            env_cfg.viewer.eye = (0.75, 0.75, 3.75)
            # env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 3.0)
        env_cfg.viewer.resolution = (1080, 1920)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
            
    else:
        if args_cli.follow_robot >= 0:
            if "Crazyflie" in args_cli.task:
                env_cfg.viewer.eye = (-0.5, 0.5, 0.5)
                env_cfg.viewer.resolution = (1920, 1080)
                env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
                env_cfg.viewer.origin_type = "asset_root"
                env_cfg.viewer.env_index = args_cli.follow_robot
                env_cfg.viewer.asset_name = "robot"

            else:
                if "Viz" in args_cli.save_prefix:
                    env_cfg.viewer.eye = (0, 0, 5.5)
                    env_cfg.viewer.lookat = (0, 0, 0)
                    env_cfg.viewer.resolution = (720, 720)
                    # env_cfg.viewer.origin_type = "asset_root"
                    env_cfg.viewer.origin_type = "env"
                    env_cfg.viewer.env_index = args_cli.follow_robot
                    env_cfg.viewer.asset_name = "robot"
                else:
                    env_cfg.viewer.eye = (0.75, 0.75, 0.75)
                    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
                    env_cfg.viewer.resolution = (1080, 1920)
                    env_cfg.viewer.origin_type = "asset_root"
                    env_cfg.viewer.env_index = args_cli.follow_robot
                    env_cfg.viewer.asset_name = "robot"




            robot_index_prefix = f"_robot_{args_cli.follow_robot}"


    
    # env_cfg.viewer.eye = (3.0, 1.5, 2.0)
    # env_cfg.viewer.resolution = (1920, 1080)
    # env_cfg.viewer.lookat = (0.0, 1.5, 0.5)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0

    # Manual override of env cfg
    # env_cfg.goal_pos_range = 2.0
    # env_cfg.goal_yaw_range = 0.0 #0.0 1.5708  3.14159



    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    save_prefix = args_cli.save_prefix
    if args_cli.case_study:
        save_prefix = "case_study_"
    
    if "Ball" in args_cli.task:
        save_prefix += "ball_catch_"

    if "Traj" in args_cli.task:
        save_prefix += "eval_traj_track_" + str(int(1/env_cfg.traj_update_dt)) + "Hz_"

    
    # save_prefix = "ball_catch_side_view_"
    if "Traj" in args_cli.task:
        viz_mode = env_cfg.viz_mode
    else:
        viz_mode = ""
        
    video_name = save_prefix + "_eval_video" + robot_index_prefix + "_viz_" + viz_mode
    if args_cli.baseline:
        video_folder_path = f"{policy_path}"
    else:
        video_folder_path = os.path.join(policy_path, "videos", "eval")

    video_kwargs = {
        "video_folder": video_folder_path,
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args_cli.video_length,
        "name_prefix": video_name
    }
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    device = envs.unwrapped.device


    if args_cli.baseline:
        env = envs.unwrapped
        vehicle_mass = envs.unwrapped.vehicle_mass
        arm_mass = envs.unwrapped.arm_mass
        inertia =  envs.unwrapped.quad_inertia
        arm_offset = envs.unwrapped.arm_offset
        pos_offset = envs.unwrapped.position_offset
        ori_offset = envs.unwrapped.orientation_offset

        if "Traj" in args_cli.task:
            feed_forward = True
        else:
            feed_forward = False

        # Hand-tuned gains
        # agent = DecoupledController(envs.num_envs, 0, envs.vehicle_mass, envs.arm_mass, envs.quad_inertia, envs.arm_offset, envs.orientation_offset, com_pos_w=None, device=device)
        
        
        if "Crazyflie" not in args_cli.task:
            # Optuna-tuned gains for EE-Reward
            # use_feed_forward = "Traj" in args_cli.task and "Integral" not in args_cli.task
            control_params_dict = gc_params_dict[task_name]["controller_params"]
            agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        **control_params_dict)
        else:
            # Crazyflie DC
            control_params_dict = gc_params_dict[task_name]["controller_params"]
            # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
            #                             kp_pos_gain_xy=6.5, kp_pos_gain_z=15.0, kd_pos_gain_xy=4.0, kd_pos_gain_z=9.0,
            #                             kp_att_gain_xy=544, kp_att_gain_z=544, kd_att_gain_xy=46.64, kd_att_gain_z=46.64, 
            #                             skip_precompute=True, vehicle="Crazyflie", control_mode="CTBM", print_debug=False, feed_forward=feed_forward)
            
            agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
                                        vehicle="Crazyflie", **control_params_dict)
            
        # Optuna-tuned gains for EE-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.675, kp_pos_gain_z=31.101, kd_pos_gain_xy=7.894, kd_pos_gain_z=8.207,
        #                             kp_att_gain_xy=950.228, kp_att_gain_z=10.539, kd_att_gain_xy=39.918, kd_att_gain_z=5.719)
        
        # Optuna-tuned gains for COM-Reward
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=38.704, kp_pos_gain_z=39.755, kd_pos_gain_xy=10.413, kd_pos_gain_z=13.509,
        #                             kp_att_gain_xy=829.511, kp_att_gain_z=1.095, kd_att_gain_xy=38.383, kd_att_gain_z=4.322)
        
        # Optuna-tuned gains for COM-LQR Cost (equal pos and yaw weight)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=49.960, kp_pos_gain_z=23.726, kd_pos_gain_xy=13.218, kd_pos_gain_z=6.878,
        #                             kp_att_gain_xy=775.271, kp_att_gain_z=3.609, kd_att_gain_xy=41.144, kd_att_gain_z=1.903)
        
        # Optuna-tuned gains for COM-LQR Cost (environment has further away goals)
        # agent = DecoupledController(env.num_envs, 0, env.vehicle_mass, env.arm_mass, env.quad_inertia, env.arm_offset, env.orientation_offset, com_pos_w=None, device=device,
        #                             kp_pos_gain_xy=24.172, kp_pos_gain_z=28.362, kd_pos_gain_xy=6.149, kd_pos_gain_z=8.881,
        #                             kp_att_gain_xy=955.034, kp_att_gain_z=14.370, kd_att_gain_xy=36.101, kd_att_gain_z=8.828)
    
    else:
        envs = RslRlVecEnvWrapper(envs) # This calls Reset!!
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(envs, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        agent = ppo_runner.get_inference_policy(device=envs.unwrapped.device)

        # actor_params =  sum(p.numel() for p in ppo_runner.alg.actor_critic.actor.parameters() if p.requires_grad)
        # critic_params = sum(p.numel() for p in ppo_runner.alg.actor_critic.critic.parameters() if p.requires_grad)
        # print(f"Actor params: {actor_params}, Critic params: {critic_params}")
        # print("Total params: ", actor_params + critic_params)
        # input()

        # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        # export_policy_as_jit(
        #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        # )

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        
        ee_offset = -envs.unwrapped.body_pos_ee_frame[args_cli.follow_robot].cpu().numpy() if "DOF" in args_cli.task else None
        export_utils.export_model_to_c(ppo_runner.alg.policy.actor, export_model_dir, policy_rate=env_cfg.policy_rate_hz, use_previous_action=env_cfg.use_previous_actions, ee_offset=ee_offset)

    if args_cli.baseline:
        obs_dict, info = envs.reset()
        obs = obs_dict["policy"]
        # print(envs.vehicle_mass)

        # Refresh GC models for mass, inertia and ttw
        # agent.reset_dr_terms(None, env.vehicle_mass, env.vehicle_inertia, env._thrust_to_weight)
    # else:
    #     # obs, dict_obs = envs.reset()
    #     obs_dict = envs.get_observations()
    #     # obs_dict = dict_obs['observations']
    else:
        # RslRlVecEnvWrapper returns a tuple: (obs, extras)
        obs, extras = envs.get_observations()
        # Extract the original dictionary from extras, or fallback to unwrapped env
        obs_dict = extras.get("observations", envs.unwrapped._get_observations())


    # =================================================================
    # RL_B3_GEOMETRIC_COMPARISON_BENCHMARK_V1 — POST-RESET PREPARATION
    # =================================================================
    if args_cli.compare_b3_geometric:
        benchmark_env = envs.unwrapped
        benchmark_device = benchmark_env.device
        benchmark_idx = int(args_cli.follow_robot)

        env_ids_b3 = torch.tensor(
            [benchmark_idx],
            dtype=torch.long,
            device=benchmark_device,
        )

        # -------------------------------------------------------------
        # Exact level/stationary body start at local (0,0,3).
        # -------------------------------------------------------------
        env_origin_b3 = (
            benchmark_env._terrain.env_origins[benchmark_idx]
            .detach()
            .clone()
        )

        root_pose_b3 = torch.zeros(
            (1, 7),
            dtype=benchmark_env._robot.data.root_state_w.dtype,
            device=benchmark_device,
        )

        root_pose_b3[0, :3] = (
            env_origin_b3
            + torch.tensor(
                [0.0, 0.0, 3.0],
                dtype=root_pose_b3.dtype,
                device=benchmark_device,
            )
        )

        # Quaternion WXYZ.
        root_pose_b3[0, 3] = 1.0

        root_velocity_b3 = torch.zeros(
            (1, 6),
            dtype=benchmark_env._robot.data.root_state_w.dtype,
            device=benchmark_device,
        )

        benchmark_env._robot.write_root_pose_to_sim(
            root_pose_b3,
            env_ids=env_ids_b3,
        )

        benchmark_env._robot.write_root_velocity_to_sim(
            root_velocity_b3,
            env_ids=env_ids_b3,
        )

        # Physically consistent hover rotor state.
        hover_motor_b3 = (
            float(
                benchmark_env._robot_mass[
                    benchmark_idx
                ].item()
            )
            * 9.81
            / (
                4.0
                * float(
                    benchmark_env._k_eta[
                        benchmark_idx
                    ].item()
                )
            )
        ) ** 0.5

        benchmark_env._motor_speeds[benchmark_idx].fill_(
            hover_motor_b3
        )

        if torch.is_tensor(
            getattr(
                benchmark_env,
                "_motor_speeds_des",
                None,
            )
        ):
            benchmark_env._motor_speeds_des[benchmark_idx].fill_(
                hover_motor_b3
            )

        # Zero command/history state at the beginning of policy warm-up.
        for name in (
            "_actions",
            "_previous_action",
            "_previous_omega_err",
        ):
            tensor = getattr(
                benchmark_env,
                name,
                None,
            )

            if torch.is_tensor(tensor):
                tensor[benchmark_idx].zero_()

        for name in (
            "_action_history",
            "_state_history",
        ):
            tensor = getattr(
                benchmark_env,
                name,
                None,
            )

            if torch.is_tensor(tensor):
                tensor[benchmark_idx].zero_()

        # Preserve the selected RL robot's realized/randomized
        # control latency. Only clear its queued commands before
        # policy-controlled pre-hover.
        latency_tensor_b3 = getattr(
            benchmark_env,
            "_control_latency_steps",
            None,
        )

        action_queue_b3 = getattr(
            benchmark_env,
            "_action_queue",
            None,
        )

        if torch.is_tensor(action_queue_b3):
            action_queue_b3[
                benchmark_idx
            ].zero_()

        # Invalidate body lazy buffers after direct state writes.
        robot_data_b3 = benchmark_env._robot.data

        for buffer_name in (
            "_body_com_vel_w",
            "_body_link_vel_w",
            "_body_state_w",
            "_body_link_state_w",
            "_body_com_state_w",
        ):
            buffer = getattr(
                robot_data_b3,
                buffer_name,
                None,
            )

            if buffer is not None:
                buffer.timestamp = -1.0

        # -------------------------------------------------------------
        # Establish the initial EE target.
        #
        # The RL policy remains an EE-goal policy. We do NOT convert it
        # to a body-goal task here.
        # -------------------------------------------------------------
        ee_frame_b3 = (
            "endeffector"
            if bool(benchmark_env.cfg.has_end_effector)
            else "body"
        )

        (
            ee_pos_b3,
            ee_quat_b3,
            _ee_vel_b3,
            _ee_ang_b3,
        ) = benchmark_env.get_frame_state_from_task(
            ee_frame_b3
        )

        benchmark_initial_ee_pos = (
            ee_pos_b3[
                benchmark_idx:benchmark_idx + 1
            ]
            .detach()
            .clone()
        )

        benchmark_initial_ee_ori = (
            ee_quat_b3[
                benchmark_idx:benchmark_idx + 1
            ]
            .detach()
            .clone()
        )

        benchmark_goal_state = {
            "pos": benchmark_initial_ee_pos.clone(),
            "ori": benchmark_initial_ee_ori.clone(),
        }

        original_update_goal_state_b3 = (
            benchmark_env.update_goal_state
        )

        def _write_benchmark_goal_b3():
            pos_b3 = benchmark_goal_state["pos"]
            ori_b3 = benchmark_goal_state["ori"]

            benchmark_env._desired_pos_w[
                benchmark_idx
            ].copy_(
                pos_b3[0]
            )

            benchmark_env._desired_ori_w[
                benchmark_idx
            ].copy_(
                ori_b3[0]
            )

            desired_pos_traj_b3 = getattr(
                benchmark_env,
                "_desired_pos_traj_w",
                None,
            )

            if torch.is_tensor(
                desired_pos_traj_b3
            ):
                desired_pos_traj_b3[
                    benchmark_idx
                ].copy_(
                    pos_b3[0]
                    .view(1, 3)
                    .expand_as(
                        desired_pos_traj_b3[
                            benchmark_idx
                        ]
                    )
                )

            desired_ori_traj_b3 = getattr(
                benchmark_env,
                "_desired_ori_traj_w",
                None,
            )

            if torch.is_tensor(
                desired_ori_traj_b3
            ):
                desired_ori_traj_b3[
                    benchmark_idx
                ].copy_(
                    ori_b3[0]
                    .view(1, 4)
                    .expand_as(
                        desired_ori_traj_b3[
                            benchmark_idx
                        ]
                    )
                )

        def _comparison_update_goal_state():
            original_update_goal_state_b3()
            _write_benchmark_goal_b3()

        benchmark_env.update_goal_state = (
            _comparison_update_goal_state
        )

        _write_benchmark_goal_b3()

        # Fresh observation after exact plant/state/goal installation.
        obs_dict = benchmark_env._get_observations()

        # -------------------------------------------------------------
        # RL-controlled pre-hover.
        #
        # No controller/policy/environment reset occurs at the command
        # switch. Action history and previous-action state are therefore
        # naturally populated by the policy itself.
        # -------------------------------------------------------------
        pre_hover_steps_b3 = int(
            round(
                float(
                    args_cli.benchmark_pre_hover_s
                )
                * float(env_cfg.policy_rate_hz)
            )
        )

        print()
        print("=" * 100)
        print("RL COMPARISON PRE-HOVER")
        print("=" * 100)
        print(
            "policy rate        :",
            env_cfg.policy_rate_hz,
            "Hz",
        )
        print(
            "pre-hover duration :",
            float(args_cli.benchmark_pre_hover_s),
            "s",
        )
        print(
            "pre-hover steps    :",
            pre_hover_steps_b3,
        )
        print(
            "initial EE goal    :",
            benchmark_initial_ee_pos[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print("=" * 100)

        with torch.no_grad():
            for pre_step_b3 in range(
                pre_hover_steps_b3
            ):
                pre_action_b3 = agent(
                    obs_dict["policy"]
                )

                (
                    _obs_wrapped_b3,
                    _reward_b3,
                    dones_b3,
                    extras_b3,
                ) = envs.step(
                    pre_action_b3
                )

                if bool(
                    dones_b3[
                        benchmark_idx
                    ].item()
                ):
                    raise RuntimeError(
                        "Selected RL benchmark robot "
                        f"{benchmark_idx} terminated during "
                        "pre-hover at step "
                        f"{pre_step_b3 + 1}/"
                        f"{pre_hover_steps_b3}."
                    )

                obs_dict = extras_b3.get(
                    "observations",
                    benchmark_env._get_observations(),
                )

        # Physical state immediately before the benchmark step.
        (
            pre_body_pos_b3,
            pre_body_quat_b3,
            pre_body_vel_b3,
            pre_body_ang_b3,
        ) = benchmark_env.get_frame_state_from_task(
            "body"
        )

        (
            pre_ee_pos_b3,
            _pre_ee_quat_b3,
            pre_ee_vel_b3,
            _pre_ee_ang_b3,
        ) = benchmark_env.get_frame_state_from_task(
            ee_frame_b3
        )

        # From here onward, benchmark tensors represent exactly one
        # robot: --follow_robot / benchmark_idx.
        pre_body_pos_b3 = pre_body_pos_b3[
            benchmark_idx:benchmark_idx + 1
        ]
        pre_body_quat_b3 = pre_body_quat_b3[
            benchmark_idx:benchmark_idx + 1
        ]
        pre_body_vel_b3 = pre_body_vel_b3[
            benchmark_idx:benchmark_idx + 1
        ]
        pre_body_ang_b3 = pre_body_ang_b3[
            benchmark_idx:benchmark_idx + 1
        ]

        pre_ee_pos_b3 = pre_ee_pos_b3[
            benchmark_idx:benchmark_idx + 1
        ]
        pre_ee_vel_b3 = pre_ee_vel_b3[
            benchmark_idx:benchmark_idx + 1
        ]

        # Apply the exact recorded real-flight XYZ displacement to the
        # EE goal. The reference point is the ORIGINAL hover target,
        # not whatever small physical drift exists after warm-up.
        goal_offset_b3 = torch.tensor(
            args_cli.benchmark_goal_offset,
            dtype=benchmark_initial_ee_pos.dtype,
            device=benchmark_device,
        ).view(1, 3)

        benchmark_final_ee_goal = (
            benchmark_initial_ee_pos
            + goal_offset_b3
        )

        benchmark_goal_state["pos"] = (
            benchmark_final_ee_goal
        )

        _write_benchmark_goal_b3()

        # This observation is t=0 of the measured transfer.
        obs_dict = benchmark_env._get_observations()

        print()
        print("=" * 100)
        print("RL COMPARISON BENCHMARK STEP — t = 0")
        print("=" * 100)
        print(
            "body position      :",
            pre_body_pos_b3[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "body speed         :",
            float(
                torch.linalg.norm(
                    pre_body_vel_b3[0]
                ).item()
            ),
            "m/s",
        )
        print(
            "EE position        :",
            pre_ee_pos_b3[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "EE speed           :",
            float(
                torch.linalg.norm(
                    pre_ee_vel_b3[0]
                ).item()
            ),
            "m/s",
        )
        print(
            "goal offset        :",
            goal_offset_b3[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "final EE goal      :",
            benchmark_final_ee_goal[0]
            .detach()
            .cpu()
            .numpy(),
        )
        print(
            "mass               :",
            float(
                benchmark_env._robot_mass[benchmark_idx].item()
            ),
        )
        print(
            "k_eta              :",
            float(
                benchmark_env._k_eta[benchmark_idx].item()
            ),
        )
        print(
            "tau_m              :",
            float(
                benchmark_env._tau_m[benchmark_idx].item()
            ),
        )
        print(
            "k_torque           :",
            float(
                benchmark_env._k_torque[benchmark_idx].item()
            ),
        )
        print("=" * 100)
        print()

    # MELLINGER_COMPARE_CAPTURE_V1
    # -----------------------------------------------------------------
    # OPT-IN ONLY.
    #
    # Capture the exact realized test case of --follow_robot AFTER the
    # normal RSL-RL reset path and BEFORE policy action #1.
    #
    # IMPORTANT:
    #   - no Isaac state is modified here;
    #   - no observations/actions are modified here;
    #   - normal eval_rslrl.py behavior is unchanged when the flag is
    #     absent;
    #   - this artifact will later be replayed in a separate Isaac
    #     process using the B3 geometric controller.
    # -----------------------------------------------------------------
    b3_geometric_compare_case_path = None

    if args_cli.compare_b3_geometric:
        if args_cli.baseline:
            raise ValueError(
                "--compare_b3_geometric is intended for an RSL-RL policy "
                "evaluation, not --baseline."
            )

        if args_cli.follow_robot < 0:
            raise ValueError(
                "--compare_b3_geometric requires --follow_robot N."
            )

        compare_env = envs.unwrapped
        robot = int(args_cli.follow_robot)

        if robot >= int(compare_env.num_envs):
            raise ValueError(
                f"--follow_robot={robot}, but the environment has "
                f"{compare_env.num_envs} robots."
            )

        if "Crazyflie" not in str(args_cli.task):
            raise ValueError(
                "--compare_b3_geometric currently supports the Crazyflie "
                "environment only."
            )

        def _cpu_clone(value):
            if torch.is_tensor(value):
                return value.detach().cpu().clone()
            return value

        def _env_tensor(name):
            value = getattr(compare_env, name, None)

            if not torch.is_tensor(value):
                return None

            if value.ndim == 0:
                return _cpu_clone(value)

            if value.shape[0] != compare_env.num_envs:
                raise RuntimeError(
                    f"Expected {name} first dimension to be num_envs="
                    f"{compare_env.num_envs}, got shape={tuple(value.shape)}"
                )

            return _cpu_clone(value[robot])

        # -------------------------------------------------------------
        # Exact physical/root state.
        # -------------------------------------------------------------
        root_state_w = _cpu_clone(
            compare_env._robot.data.root_state_w[robot]
        )

        root_pos_w = _cpu_clone(
            compare_env._robot.data.root_pos_w[robot]
        )
        root_quat_w = _cpu_clone(
            compare_env._robot.data.root_quat_w[robot]
        )
        root_lin_vel_w = _cpu_clone(
            compare_env._robot.data.root_lin_vel_w[robot]
        )
        root_ang_vel_w = _cpu_clone(
            compare_env._robot.data.root_ang_vel_w[robot]
        )

        (
            body_pos_w_all,
            body_quat_w_all,
            body_lin_vel_w_all,
            body_ang_vel_w_all,
        ) = compare_env.get_frame_state_from_task("body")

        body_pos_w = _cpu_clone(body_pos_w_all[robot])
        body_quat_w = _cpu_clone(body_quat_w_all[robot])
        body_lin_vel_w = _cpu_clone(body_lin_vel_w_all[robot])
        body_ang_vel_w = _cpu_clone(body_ang_vel_w_all[robot])

        ee_frame_name = (
            "endeffector"
            if bool(compare_env.cfg.has_end_effector)
            else "body"
        )

        (
            ee_pos_w_all,
            ee_quat_w_all,
            ee_lin_vel_w_all,
            ee_ang_vel_w_all,
        ) = compare_env.get_frame_state_from_task(
            ee_frame_name
        )

        ee_pos_w = _cpu_clone(ee_pos_w_all[robot])
        ee_quat_w = _cpu_clone(ee_quat_w_all[robot])
        ee_lin_vel_w = _cpu_clone(ee_lin_vel_w_all[robot])
        ee_ang_vel_w = _cpu_clone(ee_ang_vel_w_all[robot])

        # Environment origin is useful because the replay may use one
        # environment at origin zero. Translation relative to this origin
        # preserves the exact local case.
        env_origin_w = None

        if (
            hasattr(compare_env, "_terrain")
            and hasattr(compare_env._terrain, "env_origins")
            and torch.is_tensor(compare_env._terrain.env_origins)
        ):
            env_origin_w = _cpu_clone(
                compare_env._terrain.env_origins[robot]
            )
        elif (
            hasattr(compare_env, "scene")
            and hasattr(compare_env.scene, "env_origins")
            and torch.is_tensor(compare_env.scene.env_origins)
        ):
            env_origin_w = _cpu_clone(
                compare_env.scene.env_origins[robot]
            )

        # -------------------------------------------------------------
        # Goal.
        #
        # _desired_pos_w is the environment's authoritative task goal.
        #
        # For the manipulator task it represents the EE goal. Mellinger,
        # however, controls the quadrotor body/COM. Therefore also capture
        # the corresponding COM goal using the environment's OWN transform
        # rather than inventing an offset.
        # -------------------------------------------------------------
        desired_pos_w = _cpu_clone(
            compare_env._desired_pos_w[robot]
        )
        desired_ori_w = _cpu_clone(
            compare_env._desired_ori_w[robot]
        )

        (
            mellinger_goal_pos_all,
            mellinger_goal_ori_all,
        ) = compare_env.get_goal_state_from_task("COM")

        mellinger_goal_pos_w = _cpu_clone(
            mellinger_goal_pos_all[robot]
        )
        mellinger_goal_ori_w = _cpu_clone(
            mellinger_goal_ori_all[robot]
        )

        # -------------------------------------------------------------
        # Realized physical plant.
        # -------------------------------------------------------------
        plant = {
            "robot_mass_kg": _env_tensor("_robot_mass"),
            "robot_weight_n": _env_tensor("_robot_weight"),
            "robot_inertia": _env_tensor("_robot_inertia"),
            "inertia_tensor": _env_tensor("inertia_tensor"),
            "arm_length_m": _env_tensor("_arm_length"),
            "k_eta": _env_tensor("_k_eta"),
            "k_m": _env_tensor("_k_m"),
            "k_torque": _env_tensor("_k_torque"),
            "tau_m_s": _env_tensor("_tau_m"),
            "kp_att": _env_tensor("_kp_att"),
            "kd_att": _env_tensor("_kd_att"),
            "thrust_to_weight": _env_tensor(
                "_thrust_to_weight"
            ),
            "min_thrust": _env_tensor("min_thrust"),
            "max_thrust": _env_tensor("max_thrust"),
        }

        # Capture actual PhysX values too, not just the environment's
        # mirrored tensors.
        physx_masses = _cpu_clone(
            compare_env._robot.root_physx_view
            .get_masses()[robot]
        )

        physx_inertias = _cpu_clone(
            compare_env._robot.root_physx_view
            .get_inertias()[robot]
        )

        # Motor geometry/mixer after DR has been applied.
        rotor_positions = _env_tensor("_rotor_positions")
        rotor_directions = _env_tensor("_rotor_directions")
        f_to_TM = _env_tensor("f_to_TM")
        TM_to_f = _env_tensor("TM_to_f")

        # -------------------------------------------------------------
        # Exact actuator/controller internal state before RL action #1.
        # -------------------------------------------------------------
        motor_speeds = _env_tensor("_motor_speeds")
        motor_speeds_des = _env_tensor("_motor_speeds_des")
        previous_action = _env_tensor("_previous_action")
        previous_omega_err = _env_tensor("_previous_omega_err")
        action_history = _env_tensor("_action_history")

        action_queue = getattr(
            compare_env,
            "_action_queue",
            None,
        )

        if not torch.is_tensor(action_queue):
            raise RuntimeError(
                "Comparison requires environment _action_queue."
            )

        if action_queue.ndim != 3:
            raise RuntimeError(
                "Unexpected _action_queue shape: "
                f"{tuple(action_queue.shape)}"
            )

        if action_queue.shape[1] != compare_env.num_envs:
            raise RuntimeError(
                "_action_queue env dimension does not match num_envs."
            )

        robot_action_queue = _cpu_clone(
            action_queue[:, robot, :]
        )

        queue_length = int(action_queue.shape[0])

        raw_latency_index = int(
            compare_env._control_latency_steps[
                robot
            ].item()
        )

        selected_latency_index = max(
            0,
            min(
                raw_latency_index,
                queue_length - 1,
            ),
        )

        # _pre_physics_step rolls left and places newest action at [-1].
        # Consequently the physical age of the selected command is:
        physical_delay_steps = (
            (queue_length - 1)
            - selected_latency_index
        )

        rl_step_dt_s = float(compare_env.step_dt)

        physical_delay_s = (
            float(physical_delay_steps)
            * rl_step_dt_s
        )

        # -------------------------------------------------------------
        # Trajectory/reference state.
        # Capture this even though the initial comparison is intended for
        # the static hover/transfer task. This lets the replay validate
        # that assumption instead of silently guessing.
        # -------------------------------------------------------------
        desired_pos_traj_w = None
        desired_ori_traj_w = None
        pos_traj = None
        yaw_traj = None

        if torch.is_tensor(
            getattr(
                compare_env,
                "_desired_pos_traj_w",
                None,
            )
        ):
            desired_pos_traj_w = _cpu_clone(
                compare_env._desired_pos_traj_w[
                    robot
                ]
            )

        if torch.is_tensor(
            getattr(
                compare_env,
                "_desired_ori_traj_w",
                None,
            )
        ):
            desired_ori_traj_w = _cpu_clone(
                compare_env._desired_ori_traj_w[
                    robot
                ]
            )

        if torch.is_tensor(
            getattr(compare_env, "_pos_traj", None)
        ):
            pos_traj = _cpu_clone(
                compare_env._pos_traj[
                    :,
                    robot,
                    ...,
                ]
            )

        if torch.is_tensor(
            getattr(compare_env, "_yaw_traj", None)
        ):
            yaw_traj = _cpu_clone(
                compare_env._yaw_traj[
                    :,
                    robot,
                    ...,
                ]
            )

        # -------------------------------------------------------------
        # Save in a dedicated comparison directory.
        # -------------------------------------------------------------
        compare_dir = os.path.join(
            video_folder_path,
            f"b3_geometric_compare_robot_{robot}",
        )

        os.makedirs(
            compare_dir,
            exist_ok=True,
        )

        b3_geometric_compare_case_path = os.path.join(
            compare_dir,
            "rl_realized_case.pt",
        )

        compare_case = {
            "metadata": {
                "capture_version": (
                    "MELLINGER_COMPARE_CAPTURE_V1"
                ),
                "task": str(args_cli.task),
                "seed": int(args_cli.seed),
                "follow_robot": robot,
                "num_envs_in_rl_run": int(
                    compare_env.num_envs
                ),
                "rl_policy_rate_hz": float(
                    env_cfg.policy_rate_hz
                ),
                "rl_step_dt_s": rl_step_dt_s,
                "physics_dt_s": float(
                    compare_env.physics_dt
                ),
                "episode_length_s": float(
                    env_cfg.episode_length_s
                ),
                "rl_control_mode": str(
                    compare_env.cfg.control_mode
                ),
                "task_body": str(
                    compare_env.cfg.task_body
                ),
                "goal_body": str(
                    compare_env.cfg.goal_body
                ),
                "reward_task_body": str(
                    compare_env.cfg.reward_task_body
                ),
                "reward_goal_body": str(
                    compare_env.cfg.reward_goal_body
                ),
                "has_end_effector": bool(
                    compare_env.cfg.has_end_effector
                ),
                "video_requested": bool(
                    args_cli.video
                ),
                "video_length_rl_steps": int(
                    args_cli.video_length
                ),
                "policy_path": str(policy_path),
                "video_folder_path": str(
                    video_folder_path
                ),
                "save_prefix": str(save_prefix),
                "video_name": str(video_name),
            },

            "initial_state": {
                "env_origin_w": env_origin_w,

                "root_state_w": root_state_w,
                "root_pos_w": root_pos_w,
                "root_quat_wxyz": root_quat_w,
                "root_lin_vel_w": root_lin_vel_w,
                "root_ang_vel_w_radps": root_ang_vel_w,

                "body_pos_w": body_pos_w,
                "body_quat_wxyz": body_quat_w,
                "body_lin_vel_w": body_lin_vel_w,
                "body_ang_vel_w_radps": body_ang_vel_w,

                "ee_pos_w": ee_pos_w,
                "ee_quat_wxyz": ee_quat_w,
                "ee_lin_vel_w": ee_lin_vel_w,
                "ee_ang_vel_w_radps": ee_ang_vel_w,

                "full_state": _cpu_clone(
                    obs_dict["full_state"][robot]
                ),
            },

            "goal": {
                # Exact RL task goal, normally EE for manipulator.
                "desired_pos_w": desired_pos_w,
                "desired_ori_wxyz": desired_ori_w,

                # Exact corresponding COM/body goal for Mellinger.
                "b3_geometric_goal_pos_w": (
                    mellinger_goal_pos_w
                ),
                "mellinger_goal_ori_wxyz": (
                    mellinger_goal_ori_w
                ),
            },

            "plant": {
                **plant,
                "physx_masses": physx_masses,
                "physx_inertias": physx_inertias,
                "rotor_positions": rotor_positions,
                "rotor_directions": rotor_directions,
                "f_to_TM": f_to_TM,
                "TM_to_f": TM_to_f,

                # These CTBR cfg gains are included for provenance.
                # Mellinger itself uses SRT and therefore does not use
                # the environment's CTBR inner-loop gains.
                "cfg_kp_omega": float(
                    compare_env.cfg.kp_omega
                ),
                "cfg_kd_omega": float(
                    compare_env.cfg.kd_omega
                ),
                "cfg_body_rate_scale_xy": float(
                    compare_env.cfg.body_rate_scale_xy
                ),
                "cfg_body_rate_scale_z": float(
                    compare_env.cfg.body_rate_scale_z
                ),
                "cfg_motor_speed_min": float(
                    compare_env.cfg.motor_speed_min
                ),
                "cfg_motor_speed_max": float(
                    compare_env.cfg.motor_speed_max
                ),
                "dr_dict": dict(
                    compare_env.cfg.dr_dict
                ),
            },

            "actuator_state": {
                "motor_speeds": motor_speeds,
                "motor_speeds_des": (
                    motor_speeds_des
                ),
                "previous_action": previous_action,
                "previous_omega_err": (
                    previous_omega_err
                ),
                "action_history": action_history,
                "action_queue": robot_action_queue,

                "queue_length": queue_length,
                "raw_latency_index": (
                    raw_latency_index
                ),
                "selected_latency_index": (
                    selected_latency_index
                ),
                "physical_delay_steps_at_rl_rate": (
                    physical_delay_steps
                ),
                "physical_delay_s": physical_delay_s,
                "physical_delay_ms": (
                    1000.0 * physical_delay_s
                ),
            },

            "trajectory": {
                "trajectory_type": str(
                    compare_env.cfg.trajectory_type
                ),
                "trajectory_horizon": int(
                    compare_env.cfg.trajectory_horizon
                ),
                "traj_update_dt": float(
                    compare_env.cfg.traj_update_dt
                ),
                "desired_pos_traj_w": (
                    desired_pos_traj_w
                ),
                "desired_ori_traj_w": (
                    desired_ori_traj_w
                ),
                "pos_traj": pos_traj,
                "yaw_traj": yaw_traj,
            },
        }

        torch.save(
            compare_case,
            b3_geometric_compare_case_path,
        )

        print()
        print("=" * 100)
        print(
            "MELLINGER COMPARISON CASE CAPTURED "
            "(RL ROLLOUT UNCHANGED)"
        )
        print("=" * 100)
        print(
            "robot index             :",
            robot,
        )
        print(
            "case file               :",
            b3_geometric_compare_case_path,
        )
        print(
            "RL body start [m]       :",
            body_pos_w.numpy(),
        )
        print(
            "RL EE start [m]         :",
            ee_pos_w.numpy(),
        )
        print(
            "RL task goal [m]        :",
            desired_pos_w.numpy(),
        )
        print(
            "Mellinger body/COM goal :",
            mellinger_goal_pos_w.numpy(),
        )

        if plant["robot_mass_kg"] is not None:
            print(
                "mass [kg]               :",
                float(
                    plant[
                        "robot_mass_kg"
                    ].item()
                ),
            )

        if plant["robot_inertia"] is not None:
            print(
                "inertia                 :",
                plant["robot_inertia"].numpy(),
            )

        if plant["arm_length_m"] is not None:
            print(
                "arm length [m]          :",
                float(
                    plant[
                        "arm_length_m"
                    ].item()
                ),
            )

        if plant["k_eta"] is not None:
            print(
                "k_eta                   :",
                float(plant["k_eta"].item()),
            )

        if plant["k_torque"] is not None:
            print(
                "k_torque                :",
                float(
                    plant["k_torque"].item()
                ),
            )

        if plant["tau_m_s"] is not None:
            print(
                "tau_m [s]               :",
                float(
                    plant["tau_m_s"].item()
                ),
            )

        print(
            "queue length             :",
            queue_length,
        )
        print(
            "raw latency index        :",
            raw_latency_index,
        )
        print(
            "selected latency index   :",
            selected_latency_index,
        )
        print(
            "physical latency         :",
            f"{1000.0 * physical_delay_s:.3f} ms",
        )
        print("=" * 100)
        print()


    # B3_GEOMETRIC_COMPARE_REPLAY_CASE_V2
    # ------------------------------------------------------------
    # Standardized exact-state payload consumed by the independent
    # Mellinger replay process. This remains completely opt-in.
    # ------------------------------------------------------------
    if args_cli.compare_b3_geometric:
        if args_cli.baseline:
            raise RuntimeError(
                "--compare_b3_geometric is intended for an RL policy run, "
                "not --baseline."
            )

        if args_cli.follow_robot < 0:
            raise RuntimeError(
                "--compare_b3_geometric requires --follow_robot N."
            )

        compare_env_v2 = envs.unwrapped
        compare_robot_v2 = int(
            args_cli.follow_robot
        )

        compare_dir_v2 = os.path.join(
            video_folder_path,
            f"b3_geometric_compare_robot_{compare_robot_v2}",
        )
        os.makedirs(
            compare_dir_v2,
            exist_ok=True,
        )

        if (
            compare_robot_v2
            >= compare_env_v2.num_envs
        ):
            raise RuntimeError(
                f"Requested robot {compare_robot_v2}, "
                f"but num_envs={compare_env_v2.num_envs}."
            )

        def _compare_env_value_v2(
            name,
        ):
            value = getattr(
                compare_env_v2,
                name,
                None,
            )

            if not torch.is_tensor(
                value
            ):
                return None

            value = value.detach()

            if (
                value.ndim >= 1
                and value.shape[0]
                == compare_env_v2.num_envs
            ):
                value = value[
                    compare_robot_v2
                ]

            return (
                value
                .cpu()
                .clone()
            )

        def _compare_traj_value_v2(
            name,
        ):
            value = getattr(
                compare_env_v2,
                name,
                None,
            )

            if not torch.is_tensor(
                value
            ):
                return None

            value = value.detach()

            # _pos_traj/_yaw_traj use derivative-order first and
            # environment index second.
            if (
                value.ndim >= 2
                and value.shape[1]
                == compare_env_v2.num_envs
            ):
                value = value[
                    :,
                    compare_robot_v2,
                ]
            elif (
                value.ndim >= 1
                and value.shape[0]
                == compare_env_v2.num_envs
            ):
                value = value[
                    compare_robot_v2
                ]

            return (
                value
                .cpu()
                .clone()
            )

        (
            compare_body_pos_v2,
            compare_body_quat_v2,
            compare_body_vel_v2,
            compare_body_ang_v2,
        ) = compare_env_v2.get_frame_state_from_task(
            "body"
        )

        if bool(
            getattr(
                compare_env_v2.cfg,
                "has_end_effector",
                False,
            )
        ):
            (
                compare_ee_pos_v2,
                compare_ee_quat_v2,
                compare_ee_vel_v2,
                compare_ee_ang_v2,
            ) = compare_env_v2.get_frame_state_from_task(
                "endeffector"
            )
        else:
            compare_ee_pos_v2 = (
                compare_body_pos_v2
            )
            compare_ee_quat_v2 = (
                compare_body_quat_v2
            )
            compare_ee_vel_v2 = (
                compare_body_vel_v2
            )
            compare_ee_ang_v2 = (
                compare_body_ang_v2
            )

        (
            compare_b3_geometric_goal_pos_v2,
            compare_b3_geometric_goal_ori_v2,
        ) = compare_env_v2.get_goal_state_from_task(
            "COM"
        )

        terrain_origins_v2 = getattr(
            compare_env_v2._terrain,
            "env_origins",
            None,
        )

        if not torch.is_tensor(
            terrain_origins_v2
        ):
            raise RuntimeError(
                "Cannot capture exact environment origin."
            )

        env_origin_v2 = (
            terrain_origins_v2[
                compare_robot_v2
            ]
            .detach()
            .cpu()
            .clone()
        )

        queue_v2 = getattr(
            compare_env_v2,
            "_action_queue",
            None,
        )
        latency_index_v2 = getattr(
            compare_env_v2,
            "_control_latency_steps",
            None,
        )

        if (
            not torch.is_tensor(queue_v2)
            or not torch.is_tensor(
                latency_index_v2
            )
        ):
            raise RuntimeError(
                "Exact action latency state is unavailable."
            )

        queue_length_v2 = int(
            queue_v2.shape[0]
        )

        # MELLINGER_COMPARE_EFFECTIVE_LATENCY_FIX_V3
        #
        # Preserve EXACT environment semantics.
        #
        # quadrotor_env.py may domain-randomize the raw latency index
        # outside the currently allocated queue range. During the real
        # RL rollout, _pre_physics_step() clamps that raw value to:
        #
        #     [0, queue_length - 1]
        #
        # before indexing _action_queue.
        #
        # Therefore distinguish:
        #   raw_latency_index_v2      = sampled DR value
        #   selected_latency_index_v2 = value ACTUALLY applied by env
        #
        # The comparison must reproduce the latter.
        raw_latency_index_v2 = int(
            latency_index_v2[
                compare_robot_v2
            ].item()
        )

        selected_latency_index_v2 = max(
            0,
            min(
                raw_latency_index_v2,
                queue_length_v2 - 1,
            ),
        )

        physical_delay_steps_v2 = (
            queue_length_v2
            - 1
            - selected_latency_index_v2
        )

        rl_step_dt_v2 = float(
            compare_env_v2.step_dt
        )

        if (
            raw_latency_index_v2
            != selected_latency_index_v2
        ):
            print(
                "[B3 geometric comparison] RL latency index "
                f"clamped exactly as environment does: "
                f"raw={raw_latency_index_v2}, "
                f"effective={selected_latency_index_v2}, "
                f"queue_length={queue_length_v2}"
            )

        physical_delay_seconds_v2 = (
            physical_delay_steps_v2
            * rl_step_dt_v2
        )

        motor_speeds_v2 = (
            _compare_env_value_v2(
                "_motor_speeds"
            )
        )

        if motor_speeds_v2 is None:
            raise RuntimeError(
                "Cannot capture exact physical motor state."
            )

        root_state_v2 = (
            compare_env_v2._robot.data
            .root_state_w[
                compare_robot_v2
            ]
            .detach()
            .cpu()
            .clone()
        )

        physx_masses_v2 = (
            compare_env_v2._robot
            .root_physx_view
            .get_masses()[
                compare_robot_v2
            ]
            .detach()
            .cpu()
            .clone()
        )

        physx_inertias_v2 = (
            compare_env_v2._robot
            .root_physx_view
            .get_inertias()[
                compare_robot_v2
            ]
            .detach()
            .cpu()
            .clone()
        )

        replay_case_v2 = {
            "metadata": {
                "schema": (
                    "mellinger_rslrl_exact_replay_v2"
                ),
                "task": args_cli.task,
                "robot_index": (
                    compare_robot_v2
                ),
                "seed": int(
                    args_cli.seed
                ),
                "policy_rate_hz": float(
                    env_cfg.policy_rate_hz
                ),
                "step_dt": (
                    rl_step_dt_v2
                ),
                "sim_dt": float(
                    env_cfg.sim.dt
                ),
                "control_mode": str(
                    compare_env_v2.cfg
                    .control_mode
                ),
                "task_body": getattr(
                    compare_env_v2.cfg,
                    "task_body",
                    None,
                ),
                "goal_body": getattr(
                    compare_env_v2.cfg,
                    "goal_body",
                    None,
                ),
                "reward_task_body": getattr(
                    compare_env_v2.cfg,
                    "reward_task_body",
                    None,
                ),
                "reward_goal_body": getattr(
                    compare_env_v2.cfg,
                    "reward_goal_body",
                    None,
                ),
                "visualization_body": getattr(
                    compare_env_v2.cfg,
                    "visualization_body",
                    None,
                ),
                "rotorpy_done": bool(
                    getattr(
                        compare_env_v2.cfg,
                        "rotorpy_done",
                        False,
                    )
                ),
            },

            "env_origin_w": (
                env_origin_v2
            ),

            "initial_state": {
                "root_state_w": (
                    root_state_v2
                ),
                "body_pos_w": (
                    compare_body_pos_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "body_quat_w": (
                    compare_body_quat_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "body_lin_vel_w": (
                    compare_body_vel_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "body_ang_vel_w": (
                    compare_body_ang_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "ee_pos_w": (
                    compare_ee_pos_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "ee_quat_w": (
                    compare_ee_quat_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "ee_lin_vel_w": (
                    compare_ee_vel_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "ee_ang_vel_w": (
                    compare_ee_ang_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
            },

            "goal": {
                "desired_pos_w": (
                    compare_env_v2
                    ._desired_pos_w[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "desired_ori_w": (
                    compare_env_v2
                    ._desired_ori_w[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "b3_geometric_goal_pos_w": (
                    compare_b3_geometric_goal_pos_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
                "b3_geometric_goal_ori_w": (
                    compare_b3_geometric_goal_ori_v2[
                        compare_robot_v2
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
            },

            "plant": {
                "robot_mass": (
                    _compare_env_value_v2(
                        "_robot_mass"
                    )
                ),
                "robot_weight": (
                    _compare_env_value_v2(
                        "_robot_weight"
                    )
                ),
                "robot_inertia": (
                    _compare_env_value_v2(
                        "_robot_inertia"
                    )
                ),
                "inertia_tensor": (
                    _compare_env_value_v2(
                        "inertia_tensor"
                    )
                ),
                "arm_length": (
                    _compare_env_value_v2(
                        "_arm_length"
                    )
                ),
                "k_eta": (
                    _compare_env_value_v2(
                        "_k_eta"
                    )
                ),
                "k_m": (
                    _compare_env_value_v2(
                        "_k_m"
                    )
                ),
                "k_torque": (
                    _compare_env_value_v2(
                        "_k_torque"
                    )
                ),
                "tau_m": (
                    _compare_env_value_v2(
                        "_tau_m"
                    )
                ),
                "kp_att": (
                    _compare_env_value_v2(
                        "_kp_att"
                    )
                ),
                "kd_att": (
                    _compare_env_value_v2(
                        "_kd_att"
                    )
                ),
                "thrust_to_weight": (
                    _compare_env_value_v2(
                        "_thrust_to_weight"
                    )
                ),
                "min_thrust": (
                    _compare_env_value_v2(
                        "min_thrust"
                    )
                ),
                "max_thrust": (
                    _compare_env_value_v2(
                        "max_thrust"
                    )
                ),
                "rotor_positions": (
                    _compare_env_value_v2(
                        "_rotor_positions"
                    )
                ),
                "rotor_directions": (
                    _compare_env_value_v2(
                        "_rotor_directions"
                    )
                ),
                "f_to_TM": (
                    _compare_env_value_v2(
                        "f_to_TM"
                    )
                ),
                "TM_to_f": (
                    _compare_env_value_v2(
                        "TM_to_f"
                    )
                ),
                "physx_masses": (
                    physx_masses_v2
                ),
                "physx_inertias": (
                    physx_inertias_v2
                ),
            },

            "actuator": {
                "motor_speeds": (
                    motor_speeds_v2
                ),
                "motor_speeds_des": (
                    _compare_env_value_v2(
                        "_motor_speeds_des"
                    )
                ),
                "previous_action": (
                    _compare_env_value_v2(
                        "_previous_action"
                    )
                ),
                "action_history": (
                    _compare_env_value_v2(
                        "_action_history"
                    )
                ),
                "action_queue_rl_control_space": (
                    queue_v2[
                        :,
                        compare_robot_v2,
                        :,
                    ]
                    .detach()
                    .cpu()
                    .clone()
                ),
            },

            "latency": {
                "rl_queue_length": (
                    queue_length_v2
                ),
                "rl_raw_queue_index": (
                    raw_latency_index_v2
                ),
                "rl_selected_queue_index": (
                    selected_latency_index_v2
                ),
                "physical_delay_steps_at_rl_rate": (
                    physical_delay_steps_v2
                ),
                "physical_delay_seconds": (
                    physical_delay_seconds_v2
                ),
            },

            "trajectory": {
                "pos_traj": (
                    _compare_traj_value_v2(
                        "_pos_traj"
                    )
                ),
                "yaw_traj": (
                    _compare_traj_value_v2(
                        "_yaw_traj"
                    )
                ),
                "desired_pos_traj_w": (
                    _compare_env_value_v2(
                        "_desired_pos_traj_w"
                    )
                ),
                "desired_ori_traj_w": (
                    _compare_env_value_v2(
                        "_desired_ori_traj_w"
                    )
                ),
            },
        }

        b3_geometric_replay_case_path = (
            os.path.join(
                compare_dir_v2,
                "b3_geometric_replay_case_v2.pt",
            )
        )

        torch.save(
            replay_case_v2,
            b3_geometric_replay_case_path,
        )

        print()
        print(
            "[B3 geometric comparison] Exact replay case:",
            b3_geometric_replay_case_path,
        )

    print("Starting obs: ", obs_dict["full_state"])

    ee_start = obs_dict["full_state"][:, 13:16]
    # goal_start = obs_dict["full_state"][:, 26:26 + 3]
    goal_start = obs_dict["full_state"][:, 30:33]
    # print("starting norm: ", torch.norm(ee_start - goal_start, dim=1))
    # input("Check and press Enter to continue...")
    # import code; code.interact(local=locals())
    if args_cli.compare_b3_geometric:
        max_steps = int(
            comparison_measurement_episode_length_s
            * env_cfg.policy_rate_hz
        )
    else:
        max_steps = int(
            env_cfg.episode_length_s
            * env_cfg.policy_rate_hz
        )


    full_state_size = obs_dict["full_state"].shape[1]
    full_states = torch.zeros((args_cli.num_envs, max_steps, full_state_size), dtype=torch.float32).to(device)
    rewards = torch.zeros((args_cli.num_envs, max_steps), dtype=torch.float32).to(device)
    actions_log = torch.zeros((args_cli.num_envs, max_steps, 4), dtype=torch.float32).to(device)


    steps = 0
    done = False
    done_count = 0
    times = []
    # input("Press Enter to continue...")
    with torch.no_grad():
        while simulation_app.is_running():
            while steps < max_steps and not done:
                obs_tensor = obs_dict["policy"]
                full_states[:, steps, :] = obs_dict["full_state"]
                # print("Full State: ", obs_dict["full_state"][0, 33:])

                start = time.time()
                # if args_cli.baseline:
                #     actions = agent.get_action(obs_dict["gc"])
                # else:
                #     actions = agent(obs_dict)
                if args_cli.baseline:
                    actions = agent.get_action(obs_dict["gc"])
                else:
                    # Pass the policy tensor instead of the full dictionary
                    actions = agent(obs_dict["policy"])
                actions_log[:, steps] = actions
                times.append(time.time() - start)

                if args_cli.baseline:
                    obs_dict, reward, terminated, truncated, info = envs.step(actions)
                    done_count += terminated.sum().item() + truncated.sum().item()
                    # done = bool((terminated | truncated).any().item())  # ADD THIS
                # else:
                #     obs_dict, reward, dones, extras = envs.step(actions)
                #     # print("Reward: ", reward)
                #     done_count += dones.sum().item()
                #     # obs_dict = extras["observations"]
                #     info = extras
                else:
                    # Wrapper returns 4 values in your rsl_rl version: obs, reward, dones, extras
                    obs, reward, dones, extras = envs.step(actions)
                    done_count += dones.sum().item()
                    # done = bool(dones.any().item())  # ADD THIS
                    # Extract the original dictionary for the next iteration
                    obs_dict = extras.get("observations", envs.unwrapped._get_observations())
                    info = extras
                rewards[:, steps] = reward.detach()

                steps += 1
                print("Step: ", steps)

            print("Full states shape: ", full_states.shape)
            torch.save(full_states, os.path.join(policy_path, save_prefix + "eval_full_states.pt"))
            torch.save(rewards, os.path.join(policy_path, save_prefix + "eval_rewards.pt"))

            print("Final Info: \n\n", info, "\n")

            print("\nAverage inference time: ", np.mean(times))

            quad_pos = full_states[args_cli.follow_robot, :-1, 0:3].cpu().numpy()
            quad_quat = full_states[args_cli.follow_robot, :-1, 3:7].cpu().numpy()
            quad_vel = full_states[args_cli.follow_robot, :-1, 7:10].cpu().numpy()
            quad_ang_vel = full_states[args_cli.follow_robot, :-1, 10:13].cpu().numpy()
            ee_pos = full_states[args_cli.follow_robot, :-1, 13:16].cpu().numpy()
            goal_pos = full_states[args_cli.follow_robot, :-1, 26:26 + 3].cpu().numpy()
            # goal_pos = full_states[args_cli.follow_robot, :-1, 30:33].cpu().numpy()
            # Grab the timer from the exact end!
            time_to_catch = full_states[args_cli.follow_robot, :-1, -1].cpu().numpy()
            



            if args_cli.follow_robot >= 0:
                import matplotlib.pyplot as plt
                import isaaclab.utils.math as isaac_math_utils
                quad_euler = isaac_math_utils.euler_xyz_from_quat(full_states[args_cli.follow_robot, :-1, 3:7])
                quad_roll = quad_euler[0].cpu().numpy()
                quad_pitch = quad_euler[1].cpu().numpy()
                quad_yaw = quad_euler[2].cpu().numpy()
                T = rewards.shape[1] - 1
                x =  np.arange(T) * (1/env_cfg.policy_rate_hz)

                save_path = video_folder_path +"/" + video_name + ".png"
                
                fig = plt.figure(figsize=(10, 10))
                
                plt.subplot(4, 3, 1)                
                plt.plot(x, quad_pos[:, 0], label="Quad X")
                plt.plot(x, ee_pos[:, 0], label="EE X", linestyle="--")
                plt.plot(x, goal_pos[:, 0], label="Goal X")
                plt.legend(loc="best")
                
                plt.subplot(4, 3, 2)
                plt.plot(x, quad_pos[:, 1], label="Quad Y")
                plt.plot(x, ee_pos[:, 1], label="EE Y", linestyle="--")
                plt.plot(x, goal_pos[:, 1], label="Goal Y")
                plt.legend(loc="best")
                
                plt.subplot(4, 3, 3)
                plt.plot(x, quad_pos[:, 2], label="Quad Z")
                plt.plot(x, ee_pos[:, 2], label="EE Z", linestyle="--")
                plt.plot(x, goal_pos[:, 2], label="Goal Z")
                plt.legend(loc="best")

                plt.subplot(4, 3, 4)
                plt.plot(x, quad_vel[:, 0], label="Quad Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 5)
                plt.plot(x, quad_vel[:, 1], label="Quad Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 6)
                plt.plot(x, quad_vel[:, 2], label="Quad Vel Z")
                plt.legend(loc="best")
                plt.subplot(4, 3, 7)
                plt.plot(x, quad_roll, label="Quad Roll")
                plt.legend(loc="best")
                plt.subplot(4, 3, 8)
                plt.plot(x, quad_pitch, label="Quad Pitch")
                plt.legend(loc="best")
                plt.subplot(4, 3, 9)
                plt.plot(x, quad_yaw, label="Quad Yaw")
                plt.legend(loc="best")
                plt.subplot(4, 3, 10)
                plt.plot(x, quad_ang_vel[:, 0], label="Quad Ang Vel X")
                plt.legend(loc="best")
                plt.subplot(4, 3, 11)
                plt.plot(x, quad_ang_vel[:, 1], label="Quad Ang Vel Y")
                plt.legend(loc="best")
                plt.subplot(4, 3, 12)
                plt.plot(x, quad_ang_vel[:, 2], label="Quad Ang Vel Z")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
                plt.close(fig)

                # Plot actions
                fig = plt.figure(figsize=(10, 10))
                actions_log = actions_log.cpu().numpy()
                plt.subplot(2, 2, 1)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 0], label="Action 1")
                plt.legend(loc="best")
                plt.subplot(2, 2, 2)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 1], label="Action 2")
                plt.legend(loc="best")
                plt.subplot(2, 2, 3)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 2], label="Action 3")
                plt.legend(loc="best")
                plt.subplot(2, 2, 4)
                plt.plot(x, actions_log[args_cli.follow_robot, :-1, 3], label="Action 4")
                plt.legend(loc="best")
                plt.tight_layout()
                save_path = video_folder_path + "/" + video_name + "_actions.png"
                plt.savefig(save_path)
                print(f"Saved actions plot to {save_path}")
                plt.close(fig)

                # Plot Time to Catch
                fig = plt.figure(figsize=(8, 4))
                plt.plot(x, time_to_catch, label="Time to Catch (s)", color='red')
                plt.axhline(0.0, color="k", linestyle="--", label="Catch Moment")
                plt.xlabel("Time (s)")
                plt.ylabel("Seconds Remaining")
                plt.title("Time-to-Catch Countdown")
                plt.legend(loc="best")
                plt.tight_layout()
                save_path = video_folder_path + "/" + video_name + "_time_to_catch.png"
                plt.savefig(save_path)
                print(f"Saved timer plot to {save_path}")
                plt.close(fig)


            # MELLINGER_COMPARE_REPLAY_LAUNCH_V2
            # --------------------------------------------------------
            # Save only the selected RL trajectory needed by the
            # independent comparison process. Existing RL outputs above
            # remain untouched.
            # --------------------------------------------------------
            if args_cli.compare_b3_geometric:
                compare_robot_v2 = int(
                    args_cli.follow_robot
                )

                compare_dir_v2 = os.path.join(
                    video_folder_path,
                    f"b3_geometric_compare_robot_{compare_robot_v2}",
                )

                os.makedirs(
                    compare_dir_v2,
                    exist_ok=True,
                )

                rl_trace_path_v2 = os.path.join(
                    compare_dir_v2,
                    "rl_trace.pt",
                )

                torch.save(
                    {
                        "schema": (
                            "mellinger_rslrl_rl_trace_v2"
                        ),
                        "robot_index": (
                            compare_robot_v2
                        ),
                        "policy_rate_hz": float(
                            env_cfg.policy_rate_hz
                        ),
                        "steps": int(
                            steps
                        ),
                        "physical_duration_s": (
                            float(steps)
                            / float(
                                env_cfg.policy_rate_hz
                            )
                        ),
                        "video_length_rl_steps": int(
                            args_cli.video_length
                        ),
                        "full_state": (
                            full_states[
                                compare_robot_v2,
                                :steps,
                            ]
                            .detach()
                            .cpu()
                            .clone()
                        ),
                    },
                    rl_trace_path_v2,
                )

                print()
                print(
                    "[B3 geometric comparison] RL trace:",
                    rl_trace_path_v2,
                )


                # ============================================================
                # B3_GEOMETRIC_COMPARE_SINGLE_ENTRYPOINT_V1
                #
                # If this evaluation was launched by the outer supervisor,
                # publish only the paths required by the second Isaac process.
                # No Mellinger code executes in this RL process.
                # ============================================================
                comparison_manifest_path_v2 = (
                    os.environ.get(
                        "AERIAL_COMPARE_MANIFEST"
                    )
                )

                if comparison_manifest_path_v2:
                    import json

                    comparison_manifest_v2 = {
                        "task": str(
                            args_cli.task
                        ),
                        "robot_index": int(
                            compare_robot_v2
                        ),
                        "case_path": str(
                            b3_geometric_replay_case_path
                        ),
                        "rl_trace_path": str(
                            rl_trace_path_v2
                        ),
                        "output_dir": str(
                            compare_dir_v2
                        ),
                        "device": str(
                            env_cfg.sim.device
                        ),
                        "video": bool(
                            args_cli.video
                        ),
                    }

                    with open(
                        comparison_manifest_path_v2,
                        "w",
                        encoding="utf-8",
                    ) as comparison_manifest_file_v2:
                        json.dump(
                            comparison_manifest_v2,
                            comparison_manifest_file_v2,
                            indent=2,
                        )

                    print(
                        "[B3 geometric comparison] "
                        "single-entrypoint handoff ready:",
                        comparison_manifest_path_v2,
                    )

            envs.close()
            simulation_app.close()

            if (
                args_cli.compare_b3_geometric
                and not _compare_is_rl_child
            ):
                # Legacy direct-launch path.  The normal single-entrypoint
                # pipeline never enters this branch: the outer supervisor
                # owns phase-2 launch after the RL child has exited.
                # Launching the replay as a subprocess prevents the
                # second controller from sharing mutable simulator state
                # with the RL rollout.
                import subprocess

                helper_path_v2 = os.path.join(
                    "/home/sumukh/AerialManipulation",
                    "rl",
                    "b3_geometric_rslrl_compare.py",
                )

                replay_command_v2 = [
                    sys.executable,
                    helper_path_v2,
                    "--task",
                    str(args_cli.task),
                    "--case_path",
                    str(
                        b3_geometric_replay_case_path
                    ),
                    "--rl_trace_path",
                    str(
                        rl_trace_path_v2
                    ),
                    "--output_dir",
                    str(
                        compare_dir_v2
                    ),
                ]

                if args_cli.video:
                    replay_command_v2.append(
                        "--video"
                    )

                replay_device_v2 = str(
                    env_cfg.sim.device
                )

                if replay_device_v2:
                    replay_command_v2.extend(
                        [
                            "--device",
                            replay_device_v2,
                        ]
                    )

                print()
                print("=" * 100)
                print(
                    "STARTING INDEPENDENT FROZEN MELLINGER REPLAY"
                )
                print("=" * 100)
                print(
                    " ".join(
                        replay_command_v2
                    )
                )
                print("=" * 100)

                subprocess.run(
                    replay_command_v2,
                    cwd=(
                        "/home/sumukh/AerialManipulation"
                    ),
                    check=True,
                )

                print()
                print(
                    "[B3 geometric comparison] All artifacts:",
                    compare_dir_v2,
                )


    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()