import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import itertools

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


import omni.isaac.lab.utils.math as isaac_math_utils
import utils.math_utilities as math_utils


from matplotlib import rc
rc('font', size=8)
rc('legend', fontsize=8)
rc('ytick', labelsize=6)
rc('xtick', labelsize=6)
sns.set_context("paper")
sns.set_theme()

import plotting.plotting_utils as plotting_utils
from plotting.plotting_utils import params


rl_model_base_paths = {
    "RL-Simple": "BrushlessCrazyflie_DR_NoMotorDynamics/2025-04-23_12-34-19_CTBM_TrajTrack_DR_0",
    "RL-Realistic": "BrushlessCrazyflie_DR/2025-03-22_14-44-55_CTBM_TrajTrack_DR_all_latency_0ms",
}

gc_model_base_paths = {
    "GC-Simple": "../rl/baseline_cf_ctbm/",
    "GC-Realistic": "../rl/baseline_cf_ctbm_DR/",
}

eval_setting= ""

@torch.no_grad()
def print_avg_rewards():
    for rl_model in rl_model_base_paths.keys():
        print("-"*20)
        print("RL Model: {}".format(rl_model))
        
        load_path = "../rl/logs/rsl_rl/{}/{}eval_rewards.pt".format(rl_model_base_paths[rl_model], eval_setting)
        eval_rewards = torch.load(load_path, weights_only=True) / 0.02
        # eval_rewards = torch.load(load_path, weights_only=True)
        eval_rewards = torch.mean(eval_rewards, dim=1)
        avg_reward = torch.mean(eval_rewards)
        std_reward = torch.std(eval_rewards)
        print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))

    for gc_model in gc_model_base_paths.keys():
        print("-"*20)
        print("GC Model: {}".format(gc_model))
        
        load_path = "../rl/{}/{}eval_rewards.pt".format(gc_model_base_paths[gc_model], eval_setting)
        eval_rewards = torch.load(load_path, weights_only=True) / 0.02
        # eval_rewards = torch.load(load_path, weights_only=True) 
        eval_rewards = torch.mean(eval_rewards, dim=1)
        avg_reward = torch.mean(eval_rewards)
        std_reward = torch.std(eval_rewards)
        print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))

@torch.no_grad()
def print_rmse():
    for rl_model in rl_model_base_paths.keys():
        print("-"*20)
        print("RL Model: {}".format(rl_model))
        
        load_path = "../rl/logs/rsl_rl/{}/{}eval_full_states.pt".format(rl_model_base_paths[rl_model], eval_setting)
        data = torch.load(load_path, weights_only=True)
        pos_error, yaw_error = plotting_utils.get_errors(data)
        pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
        yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
        pos_rmse_mean = torch.mean(pos_rmse)
        pos_rmse_std = torch.std(pos_rmse)
        yaw_rmse_mean = torch.mean(yaw_rmse)
        yaw_rmse_std = torch.std(yaw_rmse)
        print("{}: pos: {:.3f} $\pm$ {:.2f}, yaw: {:.3f} $\pm$ {:.2f}".format(eval_setting, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))

    for gc_model in gc_model_base_paths.keys():
        print("-"*20)
        print("GC Model: {}".format(gc_model))
        
        load_path = "../rl/{}/{}eval_full_states.pt".format(gc_model_base_paths[gc_model], eval_setting)
        pos_error, yaw_error = plotting_utils.get_errors(data)
        pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
        yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
        pos_rmse_mean = torch.mean(pos_rmse)
        pos_rmse_std = torch.std(pos_rmse)
        yaw_rmse_mean = torch.mean(yaw_rmse)
        yaw_rmse_std = torch.std(yaw_rmse)
        print("{}: pos: {:.3f} $\pm$ {:.2f}, yaw: {:.3f} $\pm$ {:.2f}".format(eval_setting, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))
            
def print_combined():
    for rl_model in rl_model_base_paths.keys():
        # print("-"*20)
        # print("RL Model: {}".format(rl_model))
        eval_setting = ""
        
        load_path = "../rl/logs/rsl_rl/{}/{}eval_rewards.pt".format(rl_model_base_paths[rl_model], eval_setting)
        eval_rewards = torch.load(load_path, weights_only=True) / 0.02
        # eval_rewards = torch.load(load_path, weights_only=True)
        eval_rewards = torch.mean(eval_rewards, dim=1)
        avg_reward = torch.mean(eval_rewards)
        std_reward = torch.std(eval_rewards)
        load_path = "../rl/logs/rsl_rl/{}/{}eval_full_states.pt".format(rl_model_base_paths[rl_model], eval_setting)
        data = torch.load(load_path, weights_only=True)
        pos_error, yaw_error = plotting_utils.get_errors(data)
        pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
        yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
        pos_rmse_mean = torch.mean(pos_rmse)
        pos_rmse_std = torch.std(pos_rmse)
        yaw_rmse_mean = torch.mean(yaw_rmse)
        yaw_rmse_std = torch.std(yaw_rmse)
        # print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))
        print("{} & {:.3f} $\pm$ {:.2f} & {:.3f} $\pm$ {:.2f} & {:.3f} $\pm$ {:.2f} \\\\  ".format(rl_model, avg_reward, std_reward, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))

    for gc_model in gc_model_base_paths.keys():
        # print("-"*20)
        # print("GC Model: {}".format(gc_model))
        if gc_model == "GC-Realistic":
            eval_setting = "control_delay_0ms"
        
        load_path = "../rl/{}/{}eval_rewards.pt".format(gc_model_base_paths[gc_model], eval_setting)
        eval_rewards = torch.load(load_path, weights_only=True) / 0.02
        # eval_rewards = torch.load(load_path, weights_only=True) 
        eval_rewards = torch.mean(eval_rewards, dim=1)
        avg_reward = torch.mean(eval_rewards)
        std_reward = torch.std(eval_rewards)
        load_path = "../rl/{}/{}eval_full_states.pt".format(gc_model_base_paths[gc_model], eval_setting)
        data = torch.load(load_path, weights_only=True)
        pos_error, yaw_error = plotting_utils.get_errors(data)
        pos_rmse = plotting_utils.get_RMSE_from_error(pos_error)
        yaw_rmse = plotting_utils.get_RMSE_from_error(yaw_error)
        pos_rmse_mean = torch.mean(pos_rmse)
        pos_rmse_std = torch.std(pos_rmse)
        yaw_rmse_mean = torch.mean(yaw_rmse)
        yaw_rmse_std = torch.std(yaw_rmse)
        # print("{}: {:.3f} $\pm$ {:.2f} ".format(eval_setting, avg_reward, std_reward))
        print("{} & {:.3f} $\pm$ {:.2f} & {:.3f} $\pm$ {:.2f} & {:.3f} $\pm$ {:.2f} \\\\  ".format(gc_model, avg_reward, std_reward, pos_rmse_mean, pos_rmse_std, yaw_rmse_mean, yaw_rmse_std))



if __name__ == "__main__":
    # print_avg_rewards()
    # print_rmse()
    print_combined()