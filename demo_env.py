# Launch Sim window
import argparse
# from isaacsim import SimulationApp
from omni.isaac.lab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run demo with Isaac Sim")
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-AerialManipulator-Hover-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# simulation_app = SimulationApp(vars(args_cli))

args_cli.headless=False
# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab_tasks.utils import parse_env_cfg
import gymnasium as gym
import torch
# from envs.hover import hover_env
import envs
# import envs.hover
# from AerialManipulation.envs.hover import hover_env

def main():

    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs= args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.viewer.eye = (2.0, 1.0, 0.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.5)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0

    print(env_cfg.robot.spawn)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": "videos",
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": "demo_env"
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


    import code; code.interact(local=locals())

    obs_dict, info = env.reset()
    done = False
    
    while simulation_app.is_running():
        obs_tensor = obs_dict["policy"]
        # action = env.action_space.sample()
        action = torch.tensor([(-1.0/3.0), 0.0, 0.0, 0.0, 0.0, 0.0]) # nominal hover action with gravity enabled
        # action = torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # nominal hover action with gravity disabled.


        action = torch.tile(action, (args_cli.num_envs, 1))

        obs_dict, reward, terminated, truncated, info = env.step(action)

    env.close()

if __name__ == "__main__":
    main()

    simulation_app.close()