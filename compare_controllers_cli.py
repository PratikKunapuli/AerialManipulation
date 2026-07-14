import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from matplotlib import rc
rc('font', size=8)
rc('legend', fontsize=8)
rc('ytick', labelsize=6)
rc('xtick', labelsize=6)
sns.set_context("paper")
sns.set_theme()

import plotting.plotting_utils as plotting_utils
from plotting.plotting_utils import params


def parse_args():
    parser = argparse.ArgumentParser(description="Compare up to three controllers using eval_full_states.pt files.")
    parser.add_argument("-c1", "--controller1-path", type=str, required=True,
                        help="Path to the first controller's eval_full_states.pt file")
    parser.add_argument("-c1n", "--controller1-name", type=str, default="RL",
                        help="Display name for the first controller (e.g., 'RL')")
    parser.add_argument("-c1s", "--controller1-shortname", type=str, default="RL",
                        help="Short name for the first controller used in violin plots (e.g., 'RL')")
    parser.add_argument("-c2", "--controller2-path", type=str, default=None,
                        help="Path to the second controller's eval_full_states.pt file")
    parser.add_argument("-c2n", "--controller2-name", type=str, default="Decoupled Controller",
                        help="Display name for the second controller (e.g., 'Decoupled Controller')")
    parser.add_argument("-c2s", "--controller2-shortname", type=str, default="DC",
                        help="Short name for the second controller used in violin plots (e.g., 'DC')")
    parser.add_argument("-c3", "--controller3-path", type=str, default=None,
                        help="Path to the third controller's eval_full_states.pt file")
    parser.add_argument("-c3n", "--controller3-name", type=str, default="Controller 3",
                        help="Display name for the third controller (e.g., 'Controller 3')")
    parser.add_argument("-c3s", "--controller3-shortname", type=str, default="C3",
                        help="Short name for the third controller used in violin plots (e.g., 'C3')")
    parser.add_argument("--output", type=str, default="hover_error_violin_v2",
                        help="Output filename without extension (default: hover_error_violin_v2)")
    parser.add_argument("--summary-output", type=str, default=None,
                        help="Optional summary text filename without extension; defaults to '<output>_summary'")
    parser.add_argument("--vel-analysis", action="store_true", default=False,
                        help="Generate 3D and contour plots of error vs desired velocity magnitude")
    parser.add_argument("--vel-bins-lin", type=int, default=20,
                        help="Number of bins for the linear velocity axis (default: 20)")
    parser.add_argument("--vel-bins-ang", type=int, default=20,
                        help="Number of bins for the angular velocity axis (default: 20)")
    return parser.parse_args()


def load_data(path1, path2=None, path3=None):
    data1 = torch.load(path1, weights_only=True)
    data2 = torch.load(path2, weights_only=True) if path2 is not None else None
    data3 = torch.load(path3, weights_only=True) if path3 is not None else None
    T = min(data1.shape[1], 250)
    data1 = data1[:, :T]
    if data2 is not None:
        data2 = data2[:, :T]
    if data3 is not None:
        data3 = data3[:, :T]
    return data1, data2, data3


@torch.no_grad()
def plot_error_pos_yaw(data1, data2, name1, name2=None, data3=None, name3=None, axs=None):
    N = data1.shape[0]
    T = data1.shape[1] - 1
    pos_quantiles_1, yaw_quantiles_1 = plotting_utils.get_quantiles_error(data1, [0.25, 0.5, 0.75])
    has_second = data2 is not None
    has_third = data3 is not None
    if has_second:
        # data2 = data2[:, :T+1]
        pos_quantiles_2, yaw_quantiles_2 = plotting_utils.get_quantiles_error(data2, [0.25, 0.5, 0.75])
    if has_third:
        # data3 = data3[:, :T+1]
        pos_quantiles_3, yaw_quantiles_3 = plotting_utils.get_quantiles_error(data3, [0.25, 0.5, 0.75])

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 7), dpi=300)
    axs = np.array(axs).reshape(-1)
    if axs.size != 2:
        raise ValueError("plot_error_pos_yaw expects exactly two axes.")
    x_axis = np.arange(T) * 0.02
    plot_clip_time = (T+1) * 0.02

    sns.lineplot(x=x_axis, y=pos_quantiles_1[1], ax=axs[0], label=name1, color=params["rl_ee_color"], legend=False)
    axs[0].fill_between(x_axis, pos_quantiles_1[0], pos_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    if has_second:
        sns.lineplot(x=x_axis, y=pos_quantiles_2[1], ax=axs[0], label=name2, color=params["gc_color"], legend=False)
        axs[0].fill_between(x_axis, pos_quantiles_2[0], pos_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    if has_third:
        sns.lineplot(x=x_axis, y=pos_quantiles_3[1], ax=axs[0], label=name3, color=params["c3_color"], legend=False)
        axs[0].fill_between(x_axis, pos_quantiles_3[0], pos_quantiles_3[2], alpha=0.2, color=params["c3_color"])
    axs[0].set_ylabel("Position Error (m)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_xlim(0, plot_clip_time)
    axs[0].set_xticks(np.linspace(0, plot_clip_time, 3))
    axs[0].set_xticklabels([np.round(x, 2) for x in np.linspace(0, plot_clip_time, 3)])
    axs[0].set_yticks(np.linspace(0, 2.0, 4))
    axs[0].set_yticklabels(np.round(np.linspace(0, 2.0, 4), 2))

    # x1, x2, y1, y2 = 25.0, 30.0, 0.0, 0.3
    # ins_ax = axs[0].inset_axes([0.55, 0.55, 0.4, 0.4], xlim=(x1, x2), ylim=(y1, y2))
    # axs[0].indicate_inset_zoom(ins_ax, edgecolor="grey")
    # sns.lineplot(x=x_axis, y=pos_quantiles_1[1], ax=ins_ax, label=name1, color=params["rl_ee_color"], legend=False)
    # ins_ax.fill_between(x_axis, pos_quantiles_1[0], pos_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    # if has_second:
    #     sns.lineplot(x=x_axis, y=pos_quantiles_2[1], ax=ins_ax, label=name2, color=params["gc_color"], legend=False)
    #     ins_ax.fill_between(x_axis, pos_quantiles_2[0], pos_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    # if has_third:
    #     sns.lineplot(x=x_axis, y=pos_quantiles_3[1], ax=ins_ax, label=name3, color=params["c3_color"], legend=False)
    #     ins_ax.fill_between(x_axis, pos_quantiles_3[0], pos_quantiles_3[2], alpha=0.2, color=params["c3_color"])
    # ins_ax.yaxis.label.set_visible(False)
    # ins_ax.set_xlim([x1, x2])
    # ins_ax.set_xticks(np.linspace(x1, x2, 2))
    # ins_ax.set_xticklabels([np.round(x, 2) for x in np.linspace(x1, x2, 2)])
    # ins_ax.set_ylim([y1, y2])
    # ins_ax.set_yticks(np.linspace(y1, y2, 4))
    # ins_ax.set_yticklabels(np.round(np.linspace(y1, y2, 4), 2))

    sns.lineplot(x=x_axis, y=yaw_quantiles_1[1], ax=axs[1], label=name1, color=params["rl_ee_color"], legend=False)
    axs[1].fill_between(x_axis, yaw_quantiles_1[0], yaw_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    if has_second:
        sns.lineplot(x=x_axis, y=yaw_quantiles_2[1], ax=axs[1], label=name2, color=params["gc_color"], legend=False)
        axs[1].fill_between(x_axis, yaw_quantiles_2[0], yaw_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    if has_third:
        sns.lineplot(x=x_axis, y=yaw_quantiles_3[1], ax=axs[1], label=name3, color=params["c3_color"], legend=False)
        axs[1].fill_between(x_axis, yaw_quantiles_3[0], yaw_quantiles_3[2], alpha=0.2, color=params["c3_color"])
    axs[1].set_ylabel("Orientation Error (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(0, plot_clip_time)
    axs[1].set_xticks(np.linspace(0, plot_clip_time, 3))
    axs[1].set_xticklabels([np.round(x, 2) for x in np.linspace(0, plot_clip_time, 3)])
    axs[1].set_yticks(np.linspace(0, 2.5, 4))
    axs[1].set_yticklabels(np.round(np.linspace(0, 2.5, 4), 2))


@torch.no_grad()
def compute_settling_time(errors, tolerance=0.05, dt=1.0):
    initial_values = errors[:, 0].unsqueeze(1)
    final_values = errors[:, -1].unsqueeze(1)

    error_range = (initial_values - final_values).abs()
    lower_bound = final_values - (tolerance * error_range)
    upper_bound = final_values + (tolerance * error_range)

    within_band = (errors >= lower_bound) & (errors <= upper_bound)
    outside_band = (errors < lower_bound) | (errors > upper_bound)
    outside_band_int = outside_band.int()

    last_outside_index = outside_band.size(1) - torch.argmax(outside_band_int.flip(dims=[1]), dim=1) - 1

    never_outside_mask = ~outside_band.any(dim=1)
    last_outside_index[never_outside_mask] = -1

    settling_times = (last_outside_index + 1).float() * dt
    settling_times[last_outside_index == -1] = 0.0
    return settling_times


def _compute_metrics(data):
    # T = min(data.shape[1] - 1, 500-1)
    # data = data[:, :T+1]
    pos_error, yaw_error = plotting_utils.get_errors(data)
    pos_rmse = torch.sqrt(torch.mean(pos_error**2, dim=1)).cpu()
    yaw_rmse = torch.sqrt(torch.mean(yaw_error**2, dim=1)).cpu()
    pos_rmse_total = torch.sqrt(torch.mean(pos_error**2)).cpu()
    yaw_rmse_total = torch.sqrt(torch.mean(yaw_error**2)).cpu()
    final_pos_error = pos_error[:, -1]
    final_yaw_error = yaw_error[:, -1]
    final_pos_rmse = torch.sqrt(torch.mean(final_pos_error**2)).cpu()
    final_yaw_rmse = torch.sqrt(torch.mean(final_yaw_error**2)).cpu()
    pos_settling = compute_settling_time(pos_error, tolerance=0.05, dt=0.02).cpu()
    yaw_settling = compute_settling_time(yaw_error, tolerance=0.05, dt=0.02).cpu()
    final_crash_rate = plotting_utils.get_crash_rate_series(data)[-1].item()
    return {
        "pos_rmse": pos_rmse,
        "yaw_rmse": yaw_rmse,
        "pos_rmse_total": pos_rmse_total,
        "yaw_rmse_total": yaw_rmse_total,
        "final_pos_rmse": final_pos_rmse,
        "final_yaw_rmse": final_yaw_rmse,
        "pos_settling": pos_settling,
        "yaw_settling": yaw_settling,
        "final_crash_rate": final_crash_rate,
    }


def _infer_trajectory_name(path):
    basename = os.path.basename(path)
    match = re.search(r"eval_trajectory_(.+?)(?:_\d+_envs_|_eval_full_states|\.pt|$)", basename)
    if match:
        return match.group(1)
    return "unspecified"


def write_summary_stats(
    output_prefix,
    summary_output_prefix,
    controller_entries,
):
    save_dir = "controller_stats"
    os.makedirs(save_dir, exist_ok=True)
    summary_name = f"{output_prefix}_summary" if summary_output_prefix is None else summary_output_prefix
    summary_path = os.path.join(save_dir, f"{summary_name}.txt")
    lines = []
    lines.append("Controller RMSE summary statistics")
    lines.append("=" * 40)
    lines.append("")
    lines.append("RMSE over all timesteps (per-trajectory RMSE mean/std over environments):")
    for entry in controller_entries:
        pos_rmse_mean = entry["metrics"]["pos_rmse"].mean().item()
        yaw_rmse_mean = entry["metrics"]["yaw_rmse"].mean().item()
        lines.append(f"- {entry['name']} ({entry['shortname']}), trajectory={entry['trajectory']}")
        lines.append(f"  Position RMSE [m]: {pos_rmse_mean:.6f}")
        lines.append(f"  Position RMSE std over envs [m]: {entry['metrics']['pos_rmse'].std().item():.6f}")
        lines.append(
            f"  Orientation RMSE [rad]: {yaw_rmse_mean:.6f}"
        )
        lines.append(
            f"  Orientation RMSE std over envs [rad]: {entry['metrics']['yaw_rmse'].std().item():.6f}"
        )
        lines.append(
            f"  Orientation RMSE [deg]: "
            f"{(yaw_rmse_mean * 180.0 / np.pi):.6f}"
        )
        lines.append(
            f"  Orientation RMSE std over envs [deg]: "
            f"{(entry['metrics']['yaw_rmse'].std().item() * 180.0 / np.pi):.6f}"
        )
    lines.append("")
    lines.append("Average RMSE at end of rollouts:")
    for entry in controller_entries:
        lines.append(f"- {entry['name']} ({entry['shortname']}), trajectory={entry['trajectory']}")
        lines.append(f"  Position end RMSE [m]: {entry['metrics']['final_pos_rmse'].item():.6f}")
        lines.append(f"  Orientation end RMSE [rad]: {entry['metrics']['final_yaw_rmse'].item():.6f}")
        lines.append(
            f"  Orientation end RMSE [deg]: {(entry['metrics']['final_yaw_rmse'].item() * 180.0 / np.pi):.6f}"
        )
    lines.append("")
    lines.append("Crash rate at final timestep:")
    for entry in controller_entries:
        lines.append(f"- {entry['name']} ({entry['shortname']}), trajectory={entry['trajectory']}")
        lines.append(f"  Crash rate: {entry['metrics']['final_crash_rate']:.6f}")
    lines.append("")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary stats written to: {summary_path}")


def plot_violin_rmse_settling_time(data1, data2, shortname1, shortname2=None, data3=None, shortname3=None, axs=None):
    N = data1.shape[0]
    T = data1.shape[1] - 1
    # data1 = data1[:, :T+1]
    metrics_1 = _compute_metrics(data1)
    has_second = data2 is not None
    has_third = data3 is not None
    if has_second:
        # data2 = data2[:, :T+1]
        metrics_2 = _compute_metrics(data2)
    if has_third:
        # data3 = data3[:, :T+1]
        metrics_3 = _compute_metrics(data3)

    print("RMSEs")
    rmse_vals = [metrics_1["pos_rmse_total"], metrics_1["yaw_rmse_total"] * 180.0 / np.pi]
    if has_second:
        rmse_vals += [metrics_2["pos_rmse_total"], metrics_2["yaw_rmse_total"] * 180.0 / np.pi]
    if has_third:
        rmse_vals += [metrics_3["pos_rmse_total"], metrics_3["yaw_rmse_total"] * 180.0 / np.pi]
    print(*rmse_vals)

    print("RMSEs at the end:")
    end_vals = [metrics_1["final_pos_rmse"], metrics_1["final_yaw_rmse"] * 180.0 / np.pi]
    if has_second:
        end_vals += [metrics_2["final_pos_rmse"], metrics_2["final_yaw_rmse"] * 180.0 / np.pi]
    if has_third:
        end_vals += [metrics_3["final_pos_rmse"], metrics_3["final_yaw_rmse"] * 180.0 / np.pi]
    print(*end_vals)

    data_dict = {
        "Settling Time": torch.cat([metrics_1["pos_settling"], metrics_1["yaw_settling"]]),
        "RMSE": torch.cat([metrics_1["pos_rmse"], metrics_1["yaw_rmse"]]),
        "Type": (["Position"] * N + ["Orientation"] * N),
        "Method": ([shortname1] * 2 * N),
    }
    if has_second:
        N2 = data2.shape[0]
        data_dict["Settling Time"] = torch.cat(
            [data_dict["Settling Time"], metrics_2["pos_settling"], metrics_2["yaw_settling"]]
        )
        data_dict["RMSE"] = torch.cat([data_dict["RMSE"], metrics_2["pos_rmse"], metrics_2["yaw_rmse"]])
        data_dict["Type"] += (["Position"] * N2 + ["Orientation"] * N2)
        data_dict["Method"] += ([shortname2] * 2 * N2)
    if has_third:
        N3 = data3.shape[0]
        data_dict["Settling Time"] = torch.cat(
            [data_dict["Settling Time"], metrics_3["pos_settling"], metrics_3["yaw_settling"]]
        )
        data_dict["RMSE"] = torch.cat([data_dict["RMSE"], metrics_3["pos_rmse"], metrics_3["yaw_rmse"]])
        data_dict["Type"] += (["Position"] * N3 + ["Orientation"] * N3)
        data_dict["Method"] += ([shortname3] * 2 * N3)
    data = pd.DataFrame(data_dict)

    n_methods = 1 + int(has_second) + int(has_third)
    split = n_methods == 2

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 7), dpi=300)
    sns.violinplot(data=data, x="Method", y="Settling Time", hue="Type", inner="quart", split=split,
                   palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[0])
    axs[0].set_ylabel("Settling Time (s)")
    sns.violinplot(data=data, x="Method", y="RMSE", hue="Type", inner="quart", split=split,
                   palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[1])
    axs[1].set_ylabel("RMSE (m/rad)")


def plot_crash_rate_over_time(data1, data2, name1, name2=None, data3=None, name3=None, ax=None):
    has_second = data2 is not None
    has_third = data3 is not None
    crash_rate_1 = plotting_utils.get_crash_rate_series(data1).numpy()
    T = crash_rate_1.shape[0]
    x_axis = np.arange(T) * 0.02
    plot_clip_time = (T+1) * 0.02
    # data1 = data1[:T]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.3), dpi=300)
    sns.lineplot(x=x_axis, y=crash_rate_1, ax=ax, label=name1, color=params["rl_ee_color"], legend=False)
    if has_second:
        # data2 = data2[:T+1]
        crash_rate_2 = plotting_utils.get_crash_rate_series(data2).numpy()
        sns.lineplot(x=x_axis, y=crash_rate_2, ax=ax, label=name2, color=params["gc_color"], legend=False)
    if has_third:
        # data3 = data3[:T+1]
        crash_rate_3 = plotting_utils.get_crash_rate_series(data3).numpy()
        sns.lineplot(x=x_axis, y=crash_rate_3, ax=ax, label=name3, color=params["c3_color"], legend=False)

    ax.set_ylabel("Crash Rate")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, plot_clip_time)
    ax.set_xticks(np.linspace(0, plot_clip_time, 3))
    ax.set_xticklabels([np.round(x, 2) for x in np.linspace(0, plot_clip_time, 3)])
    ax.grid(True, linestyle="--", alpha=0.35)


_MAX_SCATTER_PTS = 5_000
_SCATTER_PTS_PER_BIN = 5


def _stratified_scatter_idx(lin_vel_np, ang_vel_np, n_bins, pts_per_bin, rng):
    """Sample up to pts_per_bin points from each occupied velocity bin so that
    sparse high-velocity regions are represented equally to the dense low-velocity region."""
    x_edges = np.linspace(lin_vel_np.min(), lin_vel_np.max(), n_bins + 1)
    y_edges = np.linspace(ang_vel_np.min(), ang_vel_np.max(), n_bins + 1)
    xi = np.clip(np.searchsorted(x_edges[1:-1], lin_vel_np), 0, n_bins - 1)
    yi = np.clip(np.searchsorted(y_edges[1:-1], ang_vel_np), 0, n_bins - 1)
    # Group point indices by bin
    bin_key = xi * n_bins + yi
    order = np.argsort(bin_key)
    keys_sorted = bin_key[order]
    selected = []
    start = 0
    while start < len(keys_sorted):
        end = start
        while end < len(keys_sorted) and keys_sorted[end] == keys_sorted[start]:
            end += 1
        bin_indices = order[start:end]
        n_pick = min(pts_per_bin, len(bin_indices))
        selected.append(rng.choice(bin_indices, n_pick, replace=False))
        start = end
    return np.concatenate(selected)


def _bin_error_on_velocity_grid(lin_vel_np, ang_vel_np, error_np, n_bins_lin, n_bins_ang):
    """Bin error values onto a 2D velocity-magnitude grid; returns (X, Y, X_edge, Y_edge, Z)."""
    x_edges = np.linspace(lin_vel_np.min(), lin_vel_np.max(), n_bins_lin + 1)
    y_edges = np.linspace(ang_vel_np.min(), ang_vel_np.max(), n_bins_ang + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xi = np.clip(np.searchsorted(x_edges[1:-1], lin_vel_np), 0, n_bins_lin - 1)
    yi = np.clip(np.searchsorted(y_edges[1:-1], ang_vel_np), 0, n_bins_ang - 1)
    sums = np.zeros((n_bins_lin, n_bins_ang))
    counts = np.zeros((n_bins_lin, n_bins_ang))
    np.add.at(sums, (xi, yi), error_np)
    np.add.at(counts, (xi, yi), 1)
    Z = np.full((n_bins_lin, n_bins_ang), np.nan)
    filled = counts > 0
    Z[filled] = sums[filled] / counts[filled]
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    X_edge, Y_edge = np.meshgrid(x_edges, y_edges, indexing='ij')
    return X, Y, X_edge, Y_edge, Z


def _style_3d_axes(ax):
    """Force white 3D background panes and black grid lines."""
    ax.set_facecolor("white")
    # Make pane backgrounds white
    pane_white = (1.0, 1.0, 1.0, 1.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "set_pane_color"):
            axis.set_pane_color(pane_white)
    # 3D grid style (private but stable across mpl versions used in practice)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "_axinfo") and "grid" in axis._axinfo:
            axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.4)
            axis._axinfo["grid"]["linewidth"] = 0.6
            axis._axinfo["grid"]["linestyle"] = "-"
    ax.grid(True)


def _ratio_norm(z):
    """TwoSlopeNorm centered at 1.0 so ratio==1 (equal RMSE) is the neutral color,
    or None if the data doesn't straddle 1.0 (in which case a plain Normalize is used)."""
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return None
    vmin, vmax = finite.min(), finite.max()
    if vmin < 1.0 < vmax:
        from matplotlib.colors import TwoSlopeNorm
        return TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    return None


def _plot_error_vs_desired_grid(lin_np, ang_np, pos_error_np, ori_error_np, xlabel, ylabel,
                                 output_prefix, suffix, color=None, is_ratio=False,
                                 n_bins_lin=20, n_bins_ang=20):
    """3D surface+scatter and contour figures of a per-trajectory quantity vs peak desired
    velocity/acceleration magnitude. Used both for raw RMSE (color=solid color) and for the
    RMSE ratio between two controllers (is_ratio=True, diverging colormap centered at 1.0)."""
    if is_ratio:
        pos_label_3d, ori_label_3d = "Position RMSE Ratio (C2/C1)", "Orientation RMSE Ratio (C2/C1)"
        pos_label_c, ori_label_c = "Mean Position RMSE Ratio (C2/C1)", "Mean Orientation RMSE Ratio (C2/C1)"
        surf_cmap = contour_cmap = "coolwarm"
    else:
        pos_label_3d, ori_label_3d = "Position RMSE (m)", "Orientation RMSE (rad)"
        pos_label_c, ori_label_c = "Mean Position RMSE (m)", "Mean Orientation RMSE (rad)"
        surf_cmap, contour_cmap = "magma", "rainbow"

    # --- 3D figure: all N scatter points + binned mean surface ---
    error_specs_3d = [(pos_error_np, pos_label_3d), (ori_error_np, ori_label_3d)]
    fig_3d = plt.figure(figsize=(4.5, 10), dpi=300)
    fig_3d.patch.set_facecolor("white")
    for col_idx, (error_np, zlabel) in enumerate(error_specs_3d):
        ax = fig_3d.add_subplot(2, 1, col_idx + 1, projection='3d')
        X, Y, X_edge, Y_edge, Z = _bin_error_on_velocity_grid(lin_np, ang_np, error_np, n_bins_lin, n_bins_ang)
        Z_surf = np.pad(Z, ((0, 1), (0, 1)), mode='edge')
        Z_surf_masked = np.ma.array(Z_surf, mask=np.isnan(Z_surf))
        norm = _ratio_norm(error_np) if is_ratio else None
        if is_ratio:
            ax.scatter(lin_np, ang_np, error_np, alpha=0.3, s=3, c=error_np, cmap=surf_cmap, norm=norm)
        else:
            ax.scatter(lin_np, ang_np, error_np, alpha=0.3, s=3, c=color)
        ax.plot_surface(X_edge, Y_edge, Z_surf_masked, alpha=0.65, cmap=surf_cmap, norm=norm)
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_zlabel(zlabel, labelpad=6)
        _style_3d_axes(ax)
    fig_3d.tight_layout(pad=2.0)
    fig_3d.subplots_adjust(left=0.08, right=0.96, bottom=0.04, top=0.96, hspace=0.22)
    _save_fig(fig_3d, f"{output_prefix}_{suffix}_3d", use_tight_bbox=False)

    # --- Contour figure ---
    error_specs_c = [(pos_error_np, pos_label_c), (ori_error_np, ori_label_c)]
    fig_c, axs_c = plt.subplots(2, 1, figsize=(4, 9), dpi=300)
    n_levels = 20
    for col_idx, (error_np, clabel) in enumerate(error_specs_c):
        X, Y, X_edge, Y_edge, Z = _bin_error_on_velocity_grid(lin_np, ang_np, error_np, n_bins_lin, n_bins_ang)
        Z_masked = np.ma.array(Z, mask=np.isnan(Z))
        norm = _ratio_norm(Z_masked.compressed()) if is_ratio else None
        cf = axs_c[col_idx].contourf(X, Y, Z_masked, cmap=contour_cmap, norm=norm, levels=n_levels)
        axs_c[col_idx].contour(X, Y, Z_masked, colors='k', linewidths=0.4, levels=n_levels, alpha=0.35)
        plt.colorbar(cf, ax=axs_c[col_idx], label=clabel)
        axs_c[col_idx].set_xlabel(xlabel)
        axs_c[col_idx].set_ylabel(ylabel)
    fig_c.tight_layout()
    _save_fig(fig_c, f"{output_prefix}_{suffix}_contour")


_VEL_XLABEL = r"$\max\|\mathbf{v}^{\mathrm{des}}\|$ (m/s)"
_VEL_YLABEL = r"$\max\|\boldsymbol{\omega}^{\mathrm{des}}\|$ (rad/s)"
_ACC_XLABEL = r"$\max\|\mathbf{a}^{\mathrm{des}}\|$ (m/s$^2$)"
_ACC_YLABEL = r"$\max\|\dot{\boldsymbol{\omega}}^{\mathrm{des}}\|$ (rad/s$^2$)"


def plot_velocity_error_analysis(data, name, shortname, color, output_prefix, n_bins_lin=20, n_bins_ang=20):
    """3D surface+scatter and contour figures of trajectory RMSE vs peak desired velocity magnitude.

    Each of the N trajectories contributes one point: its peak desired lin/ang velocity magnitude
    as the horizontal axes, and its per-trajectory RMSE as the vertical axis.
    """
    lin_vel_peak, ang_vel_peak, pos_rmse, ori_rmse = \
        plotting_utils.get_peak_vel_rmse_per_trajectory(data)
    _plot_error_vs_desired_grid(
        lin_vel_peak.numpy(), ang_vel_peak.numpy(), pos_rmse.numpy(), ori_rmse.numpy(),
        _VEL_XLABEL, _VEL_YLABEL, output_prefix, f"{shortname}_vel", color=color,
        n_bins_lin=n_bins_lin, n_bins_ang=n_bins_ang,
    )


def plot_acceleration_error_analysis(data, name, shortname, color, output_prefix, n_bins_lin=20, n_bins_ang=20):
    """3D surface+scatter and contour figures of trajectory RMSE vs peak desired acceleration magnitude."""
    lin_acc_peak, ang_acc_peak, pos_rmse, ori_rmse = \
        plotting_utils.get_peak_acc_rmse_per_trajectory(data)
    _plot_error_vs_desired_grid(
        lin_acc_peak.numpy(), ang_acc_peak.numpy(), pos_rmse.numpy(), ori_rmse.numpy(),
        _ACC_XLABEL, _ACC_YLABEL, output_prefix, f"{shortname}_acc", color=color,
        n_bins_lin=n_bins_lin, n_bins_ang=n_bins_ang,
    )


def plot_velocity_error_ratio_analysis(data1, data2, shortname1, shortname2, output_prefix,
                                        n_bins_lin=20, n_bins_ang=20):
    """3D surface+scatter and contour figures of the RMSE ratio (controller2 / controller1)
    vs peak desired velocity magnitude, for exactly two controllers."""
    lin_vel_peak, ang_vel_peak, pos_rmse_1, ori_rmse_1 = \
        plotting_utils.get_peak_vel_rmse_per_trajectory(data1)
    _, _, pos_rmse_2, ori_rmse_2 = plotting_utils.get_peak_vel_rmse_per_trajectory(data2)
    pos_ratio = (pos_rmse_2 / pos_rmse_1).numpy()
    ori_ratio = (ori_rmse_2 / ori_rmse_1).numpy()
    _plot_error_vs_desired_grid(
        lin_vel_peak.numpy(), ang_vel_peak.numpy(), pos_ratio, ori_ratio,
        _VEL_XLABEL, _VEL_YLABEL, output_prefix, f"{shortname1}_vs_{shortname2}_vel_ratio",
        is_ratio=True, n_bins_lin=n_bins_lin, n_bins_ang=n_bins_ang,
    )


def plot_acceleration_error_ratio_analysis(data1, data2, shortname1, shortname2, output_prefix,
                                            n_bins_lin=20, n_bins_ang=20):
    """3D surface+scatter and contour figures of the RMSE ratio (controller2 / controller1)
    vs peak desired acceleration magnitude, for exactly two controllers."""
    lin_acc_peak, ang_acc_peak, pos_rmse_1, ori_rmse_1 = \
        plotting_utils.get_peak_acc_rmse_per_trajectory(data1)
    _, _, pos_rmse_2, ori_rmse_2 = plotting_utils.get_peak_acc_rmse_per_trajectory(data2)
    pos_ratio = (pos_rmse_2 / pos_rmse_1).numpy()
    ori_ratio = (ori_rmse_2 / ori_rmse_1).numpy()
    _plot_error_vs_desired_grid(
        lin_acc_peak.numpy(), ang_acc_peak.numpy(), pos_ratio, ori_ratio,
        _ACC_XLABEL, _ACC_YLABEL, output_prefix, f"{shortname1}_vs_{shortname2}_acc_ratio",
        is_ratio=True, n_bins_lin=n_bins_lin, n_bins_ang=n_bins_ang,
    )


def _save_fig(fig, output_name, use_tight_bbox=True):
    save_dir = "controller_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_kwargs = {"dpi": 500, "format": "png"}
    if use_tight_bbox:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(os.path.join(save_dir, f"{output_name}.png"), **save_kwargs)
    # fig.savefig(os.path.join(save_dir, f"{output_name}.pdf"), bbox_inches='tight', dpi=500, format='pdf')
    plt.close(fig)


def gen_separate_layouts(data1, data2, name1, name2, shortname1, shortname2, output,
                         data3=None, name3=None, shortname3=None,
                         vel_analysis=False, vel_bins_lin=20, vel_bins_ang=20):
    has_second = data2 is not None
    has_third = data3 is not None
    error_legend_elements = [Line2D([0], [0], color=params["rl_ee_color"], label=name1)]
    if has_second:
        error_legend_elements.append(Line2D([0], [0], color=params["gc_color"], label=name2))
    if has_third:
        error_legend_elements.append(Line2D([0], [0], color=params["c3_color"], label=name3))
    violin_legend_elements = [
        Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Position'),
        Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='Orientation'),
    ]

    fig_error, axs_error = plt.subplots(2, 1, figsize=(3.5, 7), dpi=300)
    plot_error_pos_yaw(data1, data2, name1, name2, data3, name3, axs_error)
    n_controllers = 1 + int(has_second) + int(has_third)
    if n_controllers > 1:
        # Legend entries stack vertically (ncol=1), so reserve bottom margin proportional to
        # the number of rows to keep the legend from overlapping the bottom subplot.
        n_legend_rows = len(error_legend_elements)
        bottom_margin = 0.05 * n_legend_rows
        fig_error.legend(handles=error_legend_elements, loc='lower center', ncol=1, bbox_to_anchor=(0.5, 0.0))
        fig_error.tight_layout(rect=[0, bottom_margin, 1, 1])
    else:
        fig_error.tight_layout()
    _save_fig(fig_error, f"{output}_trajectory_tracking")

    # fig_violin, axs_violin = plt.subplots(2, 1, figsize=(3.5, 7), dpi=300)
    # plot_violin_rmse_settling_time(data1, data2, shortname1, shortname2, data3, shortname3, axs_violin)
    # fig_violin.legend(handles=violin_legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    # fig_violin.tight_layout(rect=[0, 0.03, 1, 1])
    # _save_fig(fig_violin, f"{output}_violin")

    # fig_crash, ax_crash = plt.subplots(1, 1, figsize=(3.5, 3.5), dpi=300)
    # plot_crash_rate_over_time(data1, data2, name1, name2, data3, name3, ax_crash)
    # if n_controllers > 1:
    #     fig_crash.legend(handles=error_legend_elements, loc='lower center', ncol=len(error_legend_elements), bbox_to_anchor=(0.5, -0.12))
    #     fig_crash.tight_layout(rect=[0, 0.07, 1, 1])
    # else:
    #     fig_crash.tight_layout()
    # _save_fig(fig_crash, f"{output}_crash_rate")

    crash_rate_1_final = plotting_utils.get_crash_rate_series(data1)[-1].item()
    print(f"{name1} crash rate at final timestep: {crash_rate_1_final:.6f}")
    if has_second:
        crash_rate_2_final = plotting_utils.get_crash_rate_series(data2)[-1].item()
        print(f"{name2} crash rate at final timestep: {crash_rate_2_final:.6f}")
    if has_third:
        crash_rate_3_final = plotting_utils.get_crash_rate_series(data3)[-1].item()
        print(f"{name3} crash rate at final timestep: {crash_rate_3_final:.6f}")

    if vel_analysis:
        if has_second and not has_third:
            # Exactly two controllers: plot RMSE ratio (C2/C1) instead of one set per controller.
            plot_velocity_error_ratio_analysis(
                data1, data2, shortname1, shortname2, output, n_bins_lin=vel_bins_lin, n_bins_ang=vel_bins_ang
            )
            plot_acceleration_error_ratio_analysis(
                data1, data2, shortname1, shortname2, output, n_bins_lin=vel_bins_lin, n_bins_ang=vel_bins_ang
            )
        else:
            controllers_vel = [(data1, name1, shortname1, params["rl_ee_color"])]
            if has_second:
                controllers_vel.append((data2, name2, shortname2, params["gc_color"]))
            if has_third:
                controllers_vel.append((data3, name3, shortname3, params["c3_color"]))
            for d, n, sn, col in controllers_vel:
                plot_velocity_error_analysis(d, n, sn, col, output, n_bins_lin=vel_bins_lin, n_bins_ang=vel_bins_ang)
                plot_acceleration_error_analysis(d, n, sn, col, output, n_bins_lin=vel_bins_lin, n_bins_ang=vel_bins_ang)


if __name__ == "__main__":
    args = parse_args()
    data1, data2, data3 = load_data(args.controller1_path, args.controller2_path, args.controller3_path)
    controller_entries = [
        {
            "name": args.controller1_name,
            "shortname": args.controller1_shortname,
            "path": args.controller1_path,
            "trajectory": _infer_trajectory_name(args.controller1_path),
            "metrics": _compute_metrics(data1),
        }
    ]
    if data2 is not None:
        controller_entries.append(
            {
                "name": args.controller2_name,
                "shortname": args.controller2_shortname,
                "path": args.controller2_path,
                "trajectory": _infer_trajectory_name(args.controller2_path),
                "metrics": _compute_metrics(data2),
            }
        )
    if data3 is not None:
        controller_entries.append(
            {
                "name": args.controller3_name,
                "shortname": args.controller3_shortname,
                "path": args.controller3_path,
                "trajectory": _infer_trajectory_name(args.controller3_path),
                "metrics": _compute_metrics(data3),
            }
        )
    gen_separate_layouts(
        data1, data2,
        args.controller1_name, args.controller2_name,
        args.controller1_shortname, args.controller2_shortname,
        args.output,
        data3=data3, name3=args.controller3_name, shortname3=args.controller3_shortname,
        vel_analysis=args.vel_analysis, vel_bins_lin=args.vel_bins_lin, vel_bins_ang=args.vel_bins_ang,
    )
    write_summary_stats(args.output, args.summary_output, controller_entries)
