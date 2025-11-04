"""
Train the trajectory tracking environment with a coarse entropy scheduler.
Needed since RslRlOnPolicyRunnerCfg class does not seem to support entropy scheduling.
"""
import argparse
import subprocess
import os
import sys
from numpy import linspace
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser(description="Train the trajectory tracking environment with a coarse entropy scheduler.")
parser.add_argument("--experiment_name", type=str, default="2DOF-Trajectory", help="The name of the experiment.")
parser.add_argument("--run_name", type=str, required=True, help="The name of the run.")
parser.add_argument("--iters-per-run", type=int, default=100, help="The number of iterations per run.")
parser.add_argument("--num-runs", type=int, default=8, help="The number of runs.")
parser.add_argument("--start-entropy", type=float, default=0.1, help="The starting entropy coefficient.")
parser.add_argument("--end-entropy", type=float, default=-0.1, help="The ending entropy coefficient.")

if __name__ == "__main__":
    args = parser.parse_args()
    entropy_schedule = linspace(args.start_entropy, args.end_entropy, args.num_runs)
    # train first run
    base_command = (
        f"{sys.executable} train_rslrl.py --experiment_name {args.experiment_name} --run_name {args.run_name} --task Isaac-AerialManipulator-2DOF-TrajectoryTracking-v0 --num_envs 4096 "
        f"agent.max_iterations={args.iters_per_run}"
    )
    base_command = base_command.split(" ") 
    command = base_command + [f"agent.algorithm.entropy_coef={entropy_schedule[0]}"]
    print(command)
    subprocess.run(command)
    
    # train remaining runs with metrics from the end of each previous run
    for i in range(1, args.num_runs):
        last_run_logdir = f"logs/rsl_rl/{args.experiment_name}/{sorted(os.listdir(f'logs/rsl_rl/{args.experiment_name}/'))[-1]}"
        event_acc = EventAccumulator(last_run_logdir)
        event_acc.Reload()
        ee_pos_radius_start = event_acc.scalars.Items("Metrics/EE Position Radius")[-1].value
        body_pos_radius_start = event_acc.scalars.Items("Metrics/Quad Position Radius")[-1].value
        ori_radius_start = event_acc.scalars.Items("Metrics/Ori Radius")[-1].value
        wrist_radius_start = event_acc.scalars.Items("Metrics/Wrist Radius")[-1].value
        
        new_command = (
            f"agent.resume=true env.ee_pos_radius_start={ee_pos_radius_start} env.body_pos_radius_start={body_pos_radius_start} "
            f"env.ori_radius_start={ori_radius_start} env.wrist_radius_start={wrist_radius_start} agent.algorithm.entropy_coef={entropy_schedule[i]}"
        )
        command = base_command + new_command.split(" ")
        subprocess.run(command)

