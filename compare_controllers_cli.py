import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import itertools

from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import isaaclab.utils.math as isaac_math_utils
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two controllers using eval_full_states.pt files.")
    parser.add_argument("-c1", "--controller1-path", type=str, required=True,
                        help="Path to the first controller's eval_full_states.pt file")
    parser.add_argument("-c1n", "--controller1-name", type=str, default="RL",
                        help="Display name for the first controller (e.g., 'RL')")
    parser.add_argument("-c1s", "--controller1-shortname", type=str, default="RL",
                        help="Short name for the first controller used in violin plots (e.g., 'RL')")
    parser.add_argument("-c2", "--controller2-path", type=str, required=True,
                        help="Path to the second controller's eval_full_states.pt file")
    parser.add_argument("-c2n", "--controller2-name", type=str, default="Decoupled Controller",
                        help="Display name for the second controller (e.g., 'Decoupled Controller')")
    parser.add_argument("-c2s", "--controller2-shortname", type=str, default="DC",
                        help="Short name for the second controller used in violin plots (e.g., 'DC')")
    parser.add_argument("--output", type=str, default="hover_error_violin_v2",
                        help="Output filename without extension (default: hover_error_violin_v2)")
    return parser.parse_args()


def load_data(path1, path2):
    data1 = torch.load(path1, weights_only=True)
    data2 = torch.load(path2, weights_only=True)
    return data1, data2


@torch.no_grad()
def plot_error_pos_yaw(data1, data2, name1, name2, axs=None):
    N = data1.shape[0]
    T = data1.shape[1] - 1
    pos_quantiles_1, yaw_quantiles_1 = plotting_utils.get_quantiles_error(data1, [0.25, 0.5, 0.75])
    pos_quantiles_2, yaw_quantiles_2 = plotting_utils.get_quantiles_error(data2, [0.25, 0.5, 0.75])

    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), dpi=300)
    x_axis = np.arange(T) * 0.02
    plot_clip_time = 10

    sns.lineplot(x=x_axis, y=pos_quantiles_1[1], ax=axs[0], label=name1, color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=pos_quantiles_2[1], ax=axs[0], label=name2, color=params["gc_color"], legend=False)
    axs[0].fill_between(x_axis, pos_quantiles_1[0], pos_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    axs[0].fill_between(x_axis, pos_quantiles_2[0], pos_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    axs[0].set_ylabel("Position Error (m)")
    plt.setp(axs[0].get_xticklabels(), visible=False)
    axs[0].set_xlim(0, plot_clip_time)
    axs[0].set_xticks(np.linspace(0, plot_clip_time, 3))
    axs[0].set_xticklabels([np.round(x, 2) for x in np.linspace(0, plot_clip_time, 3)])
    axs[0].set_yticks(np.linspace(0, 2.0, 4))
    axs[0].set_yticklabels(np.round(np.linspace(0, 2.0, 4), 2))

    x1, x2, y1, y2 = 9.0, 10.0, 0.0, 0.3
    ins_ax = axs[0].inset_axes([1.0, 0.4, 0.6, 0.6], xlim=(x1, x2), ylim=(y1, y2))
    axs[0].indicate_inset_zoom(ins_ax, edgecolor="grey")
    sns.lineplot(x=x_axis, y=pos_quantiles_1[1], ax=ins_ax, label=name1, color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=pos_quantiles_2[1], ax=ins_ax, label=name2, color=params["gc_color"], legend=False)
    ins_ax.fill_between(x_axis, pos_quantiles_1[0], pos_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    ins_ax.fill_between(x_axis, pos_quantiles_2[0], pos_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    ins_ax.yaxis.label.set_visible(False)
    ins_ax.set_xlim([x1, x2])
    ins_ax.set_xticks(np.linspace(x1, x2, 2))
    ins_ax.set_xticklabels([np.round(x, 2) for x in np.linspace(x1, x2, 2)])
    ins_ax.set_ylim([y1, y2])
    ins_ax.set_yticks(np.linspace(y1, y2, 4))
    ins_ax.set_yticklabels(np.round(np.linspace(y1, y2, 4), 2))

    sns.lineplot(x=x_axis, y=yaw_quantiles_1[1], ax=axs[1], label=name1, color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=yaw_quantiles_2[1], ax=axs[1], label=name2, color=params["gc_color"], legend=False)
    axs[1].fill_between(x_axis, yaw_quantiles_1[0], yaw_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    axs[1].fill_between(x_axis, yaw_quantiles_2[0], yaw_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    axs[1].set_ylabel("Orientation Error (rad)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_xlim(0, plot_clip_time)
    axs[1].set_xticks(np.linspace(0, plot_clip_time, 3))
    axs[1].set_xticklabels([np.round(x, 2) for x in np.linspace(0, plot_clip_time, 3)])
    axs[1].set_yticks(np.linspace(0, 2.5, 4))
    axs[1].set_yticklabels(np.round(np.linspace(0, 2.5, 4), 2))

    x1, x2, y1, y2 = 9.0, 10.0, 0.0, 0.3
    ins_ax = axs[1].inset_axes([1.0, 0.4, 0.6, 0.6], xlim=(x1, x2), ylim=(y1, y2))
    axs[1].indicate_inset_zoom(ins_ax, edgecolor="grey")
    sns.lineplot(x=x_axis, y=yaw_quantiles_1[1], ax=ins_ax, label=name1, color=params["rl_ee_color"], legend=False)
    sns.lineplot(x=x_axis, y=yaw_quantiles_2[1], ax=ins_ax, label="", color=params["gc_color"], legend=False)
    ins_ax.fill_between(x_axis, yaw_quantiles_1[0], yaw_quantiles_1[2], alpha=0.2, color=params["rl_ee_color"])
    ins_ax.fill_between(x_axis, yaw_quantiles_2[0], yaw_quantiles_2[2], alpha=0.2, color=params["gc_color"])
    ins_ax.yaxis.label.set_visible(False)
    ins_ax.set_xlim([x1, x2])
    ins_ax.set_xticks(np.linspace(x1, x2, 2))
    ins_ax.set_xticklabels([np.round(x, 2) for x in np.linspace(x1, x2, 2)])
    ins_ax.set_ylim([y1, y2])
    ins_ax.set_yticks(np.linspace(y1, y2, 4))
    ins_ax.set_yticklabels(np.round(np.linspace(y1, y2, 4), 2))


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


def plot_violin_rmse_settling_time(data1, data2, shortname1, shortname2, axs=None):
    N = data1.shape[0]
    T = data1.shape[1] - 1

    pos_error_1, yaw_error_1 = plotting_utils.get_errors(data1)
    pos_error_2, yaw_error_2 = plotting_utils.get_errors(data2)

    pos_rmse_1 = torch.sqrt(torch.mean(pos_error_1**2, dim=1)).cpu()
    pos_rmse_2 = torch.sqrt(torch.mean(pos_error_2**2, dim=1)).cpu()
    yaw_rmse_1 = torch.sqrt(torch.mean(yaw_error_1**2, dim=1)).cpu()
    yaw_rmse_2 = torch.sqrt(torch.mean(yaw_error_2**2, dim=1)).cpu()

    print("RMSEs")
    print(pos_rmse_1.mean(), yaw_rmse_1.mean() * 180.0 / np.pi,
          pos_rmse_2.mean(), yaw_rmse_2.mean() * 180.0 / np.pi)

    final_pos_error_1 = pos_error_1[:, -1]
    final_yaw_error_1 = yaw_error_1[:, -1] * 180.0 / np.pi
    final_pos_error_2 = pos_error_2[:, -1]
    final_yaw_error_2 = yaw_error_2[:, -1] * 180.0 / np.pi

    final_pos_rmse_1 = torch.sqrt(torch.mean(final_pos_error_1**2)).cpu()
    final_yaw_rmse_1 = torch.sqrt(torch.mean(final_yaw_error_1**2)).cpu()
    final_pos_rmse_2 = torch.sqrt(torch.mean(final_pos_error_2**2)).cpu()
    final_yaw_rmse_2 = torch.sqrt(torch.mean(final_yaw_error_2**2)).cpu()

    print("RMSEs at the end:")
    print(final_pos_rmse_1, final_yaw_rmse_1, final_pos_rmse_2, final_yaw_rmse_2)

    pos_settling_1 = compute_settling_time(pos_error_1, tolerance=0.05, dt=0.02).cpu()
    pos_settling_2 = compute_settling_time(pos_error_2, tolerance=0.05, dt=0.02).cpu()
    yaw_settling_1 = compute_settling_time(yaw_error_1, tolerance=0.05, dt=0.02).cpu()
    yaw_settling_2 = compute_settling_time(yaw_error_2, tolerance=0.05, dt=0.02).cpu()

    data = pd.DataFrame({
        "Settling Time": torch.cat([pos_settling_1, yaw_settling_1, pos_settling_2, yaw_settling_2]),
        "RMSE": torch.cat([pos_rmse_1, yaw_rmse_1, pos_rmse_2, yaw_rmse_2]),
        "Type": (["Position"] * N + ["Orientation"] * N + ["Position"] * N + ["Orientation"] * N),
        "Method": ([shortname1] * 2 * N + [shortname2] * 2 * N)
    })

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(3.5, 3.5), dpi=300)
    sns.violinplot(data=data, x="Method", y="Settling Time", hue="Type", inner="quart", split=True,
                   palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[0])
    axs[0].set_ylabel("Settling Time (s)")
    sns.violinplot(data=data, x="Method", y="RMSE", hue="Type", inner="quart", split=True,
                   palette=[params['violin_color_1'], params['violin_color_2']], legend=False, ax=axs[1])
    axs[1].set_ylabel("RMSE (m/rad)")


def gen_combined_layout(data1, data2, name1, name2, shortname1, shortname2, output):
    error_legend_elements = [
        Line2D([0], [0], color=params["rl_ee_color"], label=name1),
        Line2D([0], [0], color=params["gc_color"], label=name2),
    ]
    violin_legend_elements_v2 = [
        Patch(facecolor=params["violin_color_1"], edgecolor=params["violin_color_1"], fill=True, label='Position'),
        Patch(facecolor=params["violin_color_2"], edgecolor=params["violin_color_2"], fill=True, label='Orientation'),
        Patch(facecolor='none', edgecolor='none', fill=False, label=''),
    ]

    fig = plt.figure(layout="constrained", dpi=300, figsize=(7, 4))
    axd = fig.subplot_mosaic(
        """
        ACD
        BCD
        """
    )
    plot_error_pos_yaw(data1, data2, name1, name2, [axd["A"], axd["B"]])
    plot_violin_rmse_settling_time(data1, data2, shortname1, shortname2, [axd["C"], axd["D"]])

    legend_elements = list(itertools.chain.from_iterable(zip(error_legend_elements, violin_legend_elements_v2)))
    legend_elements[1], legend_elements[2] = legend_elements[2], legend_elements[1]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))
    plt.savefig(f"{output}.png", bbox_inches='tight', dpi=500, format='png')
    plt.savefig(f"{output}.pdf", bbox_inches='tight', dpi=500, format='pdf')


if __name__ == "__main__":
    args = parse_args()
    data1, data2 = load_data(args.controller1_path, args.controller2_path)
    gen_combined_layout(
        data1, data2,
        args.controller1_name, args.controller2_name,
        args.controller1_shortname, args.controller2_shortname,
        args.output,
    )
