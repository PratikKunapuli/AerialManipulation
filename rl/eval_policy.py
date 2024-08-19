import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CleanRL. ")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--load_path", type=str, default=None, required=True, help="Policy to evaluate.")
parser.add_argument("--goal_task", type=str, default="rand", help="Goal task for the environment.")
parser.add_argument("--frame", type=str, default="root", help="Frame of the task.")

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True # make false to see the simulation

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
import time
from dataclasses import dataclass
import ast
import re

import gymnasium as gym
import envs
from policies import Agent

from omni.isaac.lab_tasks.utils import parse_env_cfg


import numpy as np
import torch

def process_args(args_string):
    clean_str = args_string.strip()[10:-1]
    clean_str = clean_str.replace("=", ":")
    clean_str = clean_str.replace("None", "None").replace("True", "True").replace("False", "False")
    clean_str = re.sub(r'(\w+):', r'"\1":', clean_str)

    try:
        result_dict = ast.literal_eval(f"{{{clean_str}}}")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error converting string to dictionary: {e}")
    
    return result_dict

def update_env_cfg(env_cfg, args_cli):
    env_cfg.goal_cfg = args_cli['goal_task']
    env_cfg.task_body = args_cli['frame']
    env_cfg.sim_rate_hz = args_cli['sim_rate_hz']
    env_cfg.policy_rate_hz = args_cli['policy_rate_hz']
    env_cfg.sim.dt = 1/env_cfg.sim_rate_hz
    env_cfg.decimation = env_cfg.sim_rate_hz // env_cfg.policy_rate_hz
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.pos_radius = args_cli['pos_radius']

    return env_cfg

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["policy"]

def main():
    # find the policy to load
    policy_path = args_cli.load_path
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy {policy_path} not found.")

    if "model" in policy_path:
        model_path = policy_path
        policy_path = os.path.dirname(policy_path)
    else:
        model_path = os.path.join(policy_path, "best_cleanrl_model.pt")


    with open(os.path.join(policy_path, "args_cli.txt"), "r") as f:
        raw_string = f.read()
        if "Namespace" in raw_string:
            saved_args_cli = process_args(raw_string)
        else:
            saved_args_cli = ast.literal_eval(raw_string)


    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=True, num_envs=args_cli.num_envs, use_fabric=True
    )

    print("\n\nSaved args: ", saved_args_cli)
    print("Keys: ", saved_args_cli.keys())
    env_cfg = update_env_cfg(env_cfg, saved_args_cli)

    # Manual override of env cfg
    env_cfg.goal_cfg = "fixed"
    env_cfg.goal_pos = [0.0, 0.0, 1.0]
    env_cfg.goal_ori = [1.0, 0.0, 0.0, 0.0]
    env_cfg.init_cfg = "rand"
    env_cfg.scene.env_spacing = 0.0

    envs = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

   

    video_kwargs = {
        "video_folder": f"{policy_path}",
        "step_trigger": lambda step: step == 0,
        # "episode_trigger": lambda episode: (episode % args.save_interval) == 0,
        "video_length": args_cli.video_length,
        "name_prefix": "eval_video"
    }
    envs = gym.wrappers.RecordVideo(envs, **video_kwargs)
    envs = ExtractObsWrapper(envs)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space

    envs.single_action_space_shape = (np.array(envs.single_action_space.shape[1]).prod(),)
    envs.single_observation_space_shape = (np.array(envs.single_observation_space.shape[1]).prod(),)


    # import code; code.interact(local=locals())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path))

    import code; code.interact(local=locals())

    obs, info = envs.reset()
    steps = 0
    done = False
    # input("Press Enter to continue...")
    
    while steps < 1001:
        # obs_tensor = obs_dict["policy"]
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = envs.step(action)
        steps += 1

    envs.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()