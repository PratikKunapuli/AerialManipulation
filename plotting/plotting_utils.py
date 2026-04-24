import torch

import utils.math_utilities as math_utils
import isaaclab.utils.math as isaac_math_utils

# Description: Parameters for plotting
params = {
    # Data indicies
    "quad_pos_slice": slice(0,3),
    "quad_ori_slice" : slice(3,7),
    "ee_pos_slice" : slice(13,16),
    "ee_ori_slice" : slice(16,20),
    "goal_pos_slice" : slice(30,33),
    "goal_ori_slice" : slice(33,37),
    "crash_rate_slice" : slice(61,62),
    "lin_vel_des_slice" : slice(63,66),
    "ang_vel_des_slice" : slice(66,69),

    # Colors
    "rl_ee_color": "#56B4E9",
    "rl_com_color": "#CC79A7",
    "gc_color": "#E69F00",
    "c3_color": "#D55E00",
    "violin_color_1": "#009E73",
    "violin_color_2": "#0072B2",
}

@torch.no_grad()
def get_quantiles_error(data, quantiles):
    N = data.shape[0]
    T = data.shape[1]-1

    pos_error = torch.norm(data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1)
    yaw_error = isaac_math_utils.quat_error_magnitude(data[:,:T,params["goal_ori_slice"]], data[:,:T,params["ee_ori_slice"]])

    pos_quantiles = torch.quantile(pos_error, torch.tensor(quantiles, device=data.device), dim=0).cpu()
    yaw_quantiles = torch.quantile(yaw_error, torch.tensor(quantiles, device=data.device), dim=0).cpu()

    return pos_quantiles, yaw_quantiles

@torch.no_grad()
def get_quantiles(data, quantiles):
    quantiles = torch.tensor(quantiles, device=data.device)
    return torch.quantile(data, quantiles, dim=0).cpu()

def get_error_bars_from_quantiles(quantiles):
    return torch.abs(quantiles[::2] - quantiles[1]).numpy().reshape((2,1))

@torch.no_grad()
def get_errors(data):
    N = data.shape[0]
    T = data.shape[1]-1
    # print(data.shape)

    pos_error = torch.norm(data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1)
    yaw_error = isaac_math_utils.quat_error_magnitude(data[:,:T,params["goal_ori_slice"]], data[:,:T,params["ee_ori_slice"]])
    print("best ori error end idx: ", yaw_error[:,-1].argmin())
    print("worst ori error end idx: ", yaw_error[:,-1].argmax())
    # worst = yaw_error[:,-1].argmax()
    # print(yaw_error[worst])

    return pos_error, yaw_error

@torch.no_grad()
def get_RMSE_from_error(error):
    # Assumes error is of shape (N, T)
    return torch.sqrt(torch.mean(error**2, dim=1))


@torch.no_grad()
def get_vel_error_scatter(data):
    """Return flattened (N*T,) arrays of lin_vel_norm, ang_vel_norm, pos_error, ori_error."""
    T = data.shape[1] - 1
    pos_error = torch.norm(
        data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1
    )
    ori_error = isaac_math_utils.quat_error_magnitude(
        data[:, :T, params["goal_ori_slice"]], data[:, :T, params["ee_ori_slice"]]
    )
    lin_vel_norm = torch.norm(data[:, :T, params["lin_vel_des_slice"]], dim=-1)
    ang_vel_norm = torch.norm(data[:, :T, params["ang_vel_des_slice"]], dim=-1)
    return (
        lin_vel_norm.flatten().cpu(),
        ang_vel_norm.flatten().cpu(),
        pos_error.flatten().cpu(),
        ori_error.flatten().cpu(),
    )


@torch.no_grad()
def get_peak_vel_rmse_per_trajectory(data):
    """Return per-trajectory (N,) arrays: peak desired velocity magnitudes and RMSE of error.

    Each of the N environments contributes one data point: the peak lin/ang velocity
    magnitude seen across its rollout, and its RMSE computed over all timesteps.
    """
    T = data.shape[1] - 1
    pos_error = torch.norm(
        data[:, :T, params["goal_pos_slice"]] - data[:, :T, params["ee_pos_slice"]], dim=-1
    )
    ori_error = isaac_math_utils.quat_error_magnitude(
        data[:, :T, params["goal_ori_slice"]], data[:, :T, params["ee_ori_slice"]]
    )
    lin_vel_norm = torch.norm(data[:, :T, params["lin_vel_des_slice"]], dim=-1)
    ang_vel_norm = torch.norm(data[:, :T, params["ang_vel_des_slice"]], dim=-1)
    lin_vel_peak = lin_vel_norm.max(dim=1).values.cpu()
    ang_vel_peak = ang_vel_norm.max(dim=1).values.cpu()
    pos_rmse = torch.sqrt(torch.mean(pos_error ** 2, dim=1)).cpu()
    ori_rmse = torch.sqrt(torch.mean(ori_error ** 2, dim=1)).cpu()
    return lin_vel_peak, ang_vel_peak, pos_rmse, ori_rmse


@torch.no_grad()
def get_crash_rate_series(data):
    """Return crash-rate series over time from a single environment rollout."""
    T = data.shape[1] - 1
    # Crash rate is duplicated across environments; use the first environment.
    return data[0, :T, params["crash_rate_slice"]].squeeze(-1).cpu()