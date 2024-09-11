import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml

import omni.isaac.lab.utils.math as isaac_math_utils

import math_utilities as math_utils

def plot_data(rl_eval_path, dc_eval_path=None, save_prefix=""):
    """
    Make plots for the evaluation.

    full_states: torch.Tensor (num_envs, 500, full_state_size) [33]
    policy_path: str, path to the policy for saving the plots
    """

    """Plot the errors from the RL policy."""
    # Load the RL policy errors
    rl_state_path = os.path.join(rl_eval_path, save_prefix +  "eval_full_states.pt")
    rl_rewards_path = os.path.join(rl_eval_path, save_prefix +  "eval_rewards.pt")
    rl_full_states = torch.load(rl_state_path)
    rl_rewards = torch.load(rl_rewards_path)
    rl_save_path = rl_eval_path

    if dc_eval_path is not None:
        dc_state_path = os.path.join(dc_eval_path, save_prefix + "eval_full_states.pt")
        dc_rewards_path = os.path.join(dc_eval_path, save_prefix + "eval_rewards.pt")
        dc_full_states = torch.load(dc_state_path)
        dc_rewards = torch.load(dc_rewards_path)
        dc_save_path = dc_eval_path
    else:
        dc_full_states = torch.rand(rl_full_states.shape).to(rl_full_states.device)
        dc_rewards = torch.rand(rl_rewards.shape).to(rl_rewards.device)
        dc_save_path = rl_save_path
    
    goal_pos_w = rl_full_states[:, :-1, -7:-4]
    goal_ori_w = rl_full_states[:, :-1, -4:]
    ee_pos_w = rl_full_states[:, :-1, 13:16]
    ee_ori_w = rl_full_states[:, :-1, 16:20]
    
    dc_goal_pos_w = dc_full_states[:, :-1, -7:-4]
    dc_goal_ori_w = dc_full_states[:, :-1, -4:]
    dc_ee_pos_w = dc_full_states[:, :-1, 13:16]
    dc_ee_ori_w = dc_full_states[:, :-1, 16:20]

    

    pos_error = torch.norm(goal_pos_w - ee_pos_w, dim=-1).cpu()
    print("Individual converged error for RL: ", pos_error[:, -1])

    pos_error_df = pd.DataFrame(pos_error.numpy().T, columns=[f"env_{i}" for i in range(pos_error.shape[0])])
    pos_error_df = pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    pos_error_df['Timesteps'] = pos_error_df['Timesteps'] * 0.02



    mean_pos_error = pos_error.mean(dim=0).numpy()
    std_pos_error = pos_error.std(dim=0).numpy()
    time_axis = torch.arange(mean_pos_error.shape[0]) * 0.02


    print("Mean converged error for RL: ", mean_pos_error[-1])
    print("Std converged error for RL: ", std_pos_error[-1])

    

    dc_pos_error = torch.norm(dc_goal_pos_w - dc_ee_pos_w, dim=-1).cpu()
    dc_mean_pos_error = dc_pos_error.mean(dim=0).numpy()
    dc_std_pos_error = dc_pos_error.std(dim=0).numpy()

    dc_pos_error_df = pd.DataFrame(dc_pos_error.numpy().T, columns=[f"env_{i}" for i in range(dc_pos_error.shape[0])])
    dc_pos_error_df = dc_pos_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Position Error')
    dc_pos_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_pos_error_df['Timesteps'] = dc_pos_error_df['Timesteps'] * 0.02

    print("Mean converged error for Decoupled: ", dc_mean_pos_error[-1])
    print("Std converged error for Decoupled: ", dc_std_pos_error[-1])

    sns.set_theme()
    sns.set_context("paper")


    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_pos_error, label="RL Controller", ci='sd')
    plt.fill_between(time_axis, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.2)
    plt.plot([0, time_axis[-1]], [0.05, 0.05], 'r--', label="Threshold")
    sns.lineplot(x=time_axis, y=dc_mean_pos_error, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_pos_error - dc_std_pos_error, dc_mean_pos_error + dc_std_pos_error, alpha=0.2)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Position Error', hue='Environment', data=pos_error_df, legend=True)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')


    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Position Error', hue='Environment', data=dc_pos_error_df, legend=True)
    plt.title("End Effector Position Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "pos_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    

    # We want to use the isaac_math_utils to convert the quaternion to a pure yaw quaternion
    goal_ori_yaw_quat = isaac_math_utils.yaw_quat(goal_ori_w)
    ee_ori_yaw_quat = isaac_math_utils.yaw_quat(ee_ori_w)

    dc_goal_ori_yaw_quat = isaac_math_utils.yaw_quat(dc_goal_ori_w)
    dc_ee_ori_yaw_quat = isaac_math_utils.yaw_quat(dc_ee_ori_w)

    # yaw_error = (isaac_math_utils.wrap_to_pi(math_utils.yaw_from_quat(goal_ori_yaw_quat)) - isaac_math_utils.wrap_to_pi(math_utils.yaw_from_quat(ee_ori_yaw_quat))).cpu()
    # yaw_error = isaac_math_utils.quat_error_magnitude(goal_ori_yaw_quat, ee_ori_yaw_quat).cpu()
    # dc_yaw_error = isaac_math_utils.quat_error_magnitude(dc_goal_ori_yaw_quat, dc_ee_ori_yaw_quat).cpu()

    yaw_error = math_utils.yaw_error_from_quats(goal_ori_w, ee_ori_w, 0).cpu()
    dc_yaw_error = math_utils.yaw_error_from_quats(dc_goal_ori_w, dc_ee_ori_w, 0).cpu()

    yaw_error_df = pd.DataFrame(yaw_error.numpy().T, columns=[f"env_{i}" for i in range(yaw_error.shape[0])])
    yaw_error_df = yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    yaw_error_df['Timesteps'] = yaw_error_df['Timesteps'] * 0.02

    dc_yaw_error_df = pd.DataFrame(dc_yaw_error.numpy().T, columns=[f"env_{i}" for i in range(dc_yaw_error.shape[0])])
    dc_yaw_error_df = dc_yaw_error_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Yaw Error')
    dc_yaw_error_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_yaw_error_df['Timesteps'] = dc_yaw_error_df['Timesteps'] * 0.02


    mean_yaw_error = yaw_error.mean(dim=0).numpy()
    std_yaw_error = yaw_error.std(dim=0).numpy()

    dc_mean_yaw_error = dc_yaw_error.mean(dim=0).numpy()
    dc_std_yaw_error = dc_yaw_error.std(dim=0).numpy()

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_yaw_error, label="RL Controller", ci='sd')
    plt.fill_between(time_axis, mean_yaw_error - std_yaw_error, mean_yaw_error + std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_yaw_error, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_yaw_error - dc_std_yaw_error, dc_mean_yaw_error + dc_std_yaw_error, alpha=0.2)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Yaw Error', hue='Environment', data=yaw_error_df, legend=True)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Yaw Error', hue='Environment', data=dc_yaw_error_df, legend=True)
    plt.title("End Effector Yaw Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (rad)")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "yaw_error.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    rewards = rl_rewards[:, :-1].cpu()
    dc_rewards = dc_rewards[:, :-1].cpu()

    # Look at hydra config for the reward scales
    if os.path.exists(os.path.join(rl_eval_path, ".hydra/config.yaml")):
        with open(os.path.join(rl_eval_path, ".hydra/config.yaml"), "r") as f:
            hydra_cfg = yaml.safe_load(f)
            max_reward = 0.0
            for key in hydra_cfg["env"]:
                if "distance_reward_scale" in key:
                    max_reward += hydra_cfg["env"][key]
    else:
        max_reward = 15.0

    # # max_reward = 15.0
    # max_reward = 20.0
    min_reward = 0.0
    # normalize the rewards
    rewards = (rewards - min_reward) / (max_reward - min_reward)
    dc_rewards = (dc_rewards - min_reward) / (max_reward - min_reward)

    rewards_df = pd.DataFrame(rewards.numpy().T, columns=[f"env_{i}" for i in range(rewards.shape[0])])
    rewards_df = rewards_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Reward')
    rewards_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    rewards_df['Timesteps'] = rewards_df['Timesteps'] * 0.02

    dc_rewards_df = pd.DataFrame(dc_rewards.numpy().T, columns=[f"env_{i}" for i in range(dc_rewards.shape[0])])
    dc_rewards_df = dc_rewards_df.reset_index().melt(id_vars='index', var_name='Environment', value_name='Reward')
    dc_rewards_df.rename(columns={'index': 'Timesteps'}, inplace=True)
    dc_rewards_df['Timesteps'] = dc_rewards_df['Timesteps'] * 0.02

    mean_rewards = rewards.mean(dim=0).numpy()
    std_rewards = rewards.std(dim=0).numpy()
    time_axis = torch.arange(mean_rewards.shape[0]) * 0.02

    dc_mean_rewards = dc_rewards.mean(dim=0).numpy()
    dc_std_rewards = dc_rewards.std(dim=0).numpy()

    sns.set_theme()
    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_rewards, label="RL Controller", ci='sd')
    plt.fill_between(time_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_rewards, label="Decoupled Controller")
    plt.fill_between(time_axis, dc_mean_rewards - dc_std_rewards, dc_mean_rewards + dc_std_rewards, alpha=0.2)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.legend()
    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Reward', hue='Environment', data=rewards_df, legend=True)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.savefig(os.path.join(rl_save_path, save_prefix + "rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')

    plt.figure(figsize=(3.25, 2.5), dpi=300)
    sns.lineplot(x='Timesteps', y='Reward', hue='Environment', data=dc_rewards_df, legend=True)
    plt.title("Rewards")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Reward")
    plt.savefig(os.path.join(dc_save_path, save_prefix + "rewards.pdf"), dpi=300, format='pdf', bbox_inches='tight')


    # Make one plot that is 1x3 of the mean pos error, mean yaw error, and mean reward, and keep the legend outside the plot at the bottom below all the plots
    fig, axs = plt.subplots(1, 3, figsize=(9.75, 2.5), dpi=300)
    sns.lineplot(x=time_axis, y=mean_pos_error, label="RL Controller", ci='sd', ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, mean_pos_error - std_pos_error, mean_pos_error + std_pos_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_pos_error, label="Decoupled Controller", ax=axs[0], legend=False)
    axs[0].fill_between(time_axis, dc_mean_pos_error - dc_std_pos_error, dc_mean_pos_error + dc_std_pos_error, alpha=0.2)
    axs[0].set_title("End Effector Position Error")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Error (m)")
    if "case_study" in save_prefix:
        axs[0].set_ylim(-0.02, 0.3)
    else:
        axs[0].set_ylim(-0.1, 2.0)
    
    sns.lineplot(x=time_axis, y=mean_yaw_error, label="RL Controller", ci='sd', ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, mean_yaw_error - std_yaw_error, mean_yaw_error + std_yaw_error, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_yaw_error, label="Decoupled Controller", ax=axs[1], legend=False)
    axs[1].fill_between(time_axis, dc_mean_yaw_error - dc_std_yaw_error, dc_mean_yaw_error + dc_std_yaw_error, alpha=0.2)
    axs[1].set_title("End Effector Yaw Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error (rad)")
    axs[1].set_ylim(-0.1, 3.141)
    
    sns.lineplot(x=time_axis, y=mean_rewards, label="RL Controller", ci='sd', ax=axs[2], legend=False)
    axs[2].fill_between(time_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    sns.lineplot(x=time_axis, y=dc_mean_rewards, label="Decoupled Controller", ax=axs[2], legend=False)
    axs[2].fill_between(time_axis, dc_mean_rewards - dc_std_rewards, dc_mean_rewards + dc_std_rewards, alpha=0.2)
    axs[2].set_title("Rewards")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Normalized Reward")
    axs[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    plt.savefig(os.path.join(rl_save_path, save_prefix + "mean_combined_plots.pdf"), dpi=300, format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(dc_save_path, save_prefix + "mean_combined_plots.pdf"), dpi=300, format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_error_1/"
    # rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_error_scale_-0.1_custom_yaw_func_anneal_lr_1/"
    rl_eval_path = "../rl/runs/AM_0DOF_hover_pos_and_yaw_yaw_distance_scale_5_pos_distance_15_smooth_transition_1"
    dc_eval_path = "../rl/baseline_0dof/"
    plot_data(rl_eval_path, dc_eval_path)
    plot_data(rl_eval_path, dc_eval_path, "case_study_")