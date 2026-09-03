"""
Clean RL-vs-B3-geometric replay backend.

Purpose
-------
Replay a captured RL benchmark case on the fixed physical B3 plant using
B3BehaviorGeometricController and direct body-wrench authority.

There is deliberately NO:
- Crazyflie firmware
- PWM
- motor model
- motor lag
- actuator SYSID
- firmware pre-hover
- firmware hidden state
- Mellinger SIL controller

The only legacy compatibility retained is reading the old replay-case field
name ``mellinger_goal_pos_w`` for the already-captured COM/GC target.
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

# Always resolve project-local imports from AerialManipulation first.
# This prevents another editable robotics codebase from shadowing
# packages such as `envs`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)

if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)

sys.path.insert(0, _REPO_ROOT_STR)

from isaaclab.app import AppLauncher


# =============================================================================
# CLI / ISAAC LAUNCH
# =============================================================================

parser = argparse.ArgumentParser(
    description="Clean RL vs B3 geometric benchmark replay."
)

parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Crazyflie-0DOF-Hover-v0",
)
parser.add_argument(
    "--case_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--rl_trace_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--benchmark_horizon_s",
    type=float,
    default=4.0,
)

AppLauncher.add_app_launcher_args(parser)

# Parse this script's CLI while preserving any Hydra overrides.
args_cli, hydra_args = parser.parse_known_args()

# hydra_task_config parses sys.argv again. Remove the arguments already
# consumed by argparse and leave only genuine Hydra overrides.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# ISAAC / NUMERIC IMPORTS
# =============================================================================

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import envs  # noqa: F401 - registers project environments

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from controllers.b3_behavior_geometric_controller import (
    B3BehaviorGeometricController,
)


# =============================================================================
# FIXED B3 PLANT
# =============================================================================

B3_MASS_KG = 0.046

B3_INERTIA_KGM2 = torch.tensor(
    [
        2.4255e-05,
        1.8650e-05,
        3.9300e-05,
    ],
    dtype=torch.float64,
)

GEOMETRIC_RATE_HZ = 500.0

XY_SETTLE_POSITION_THRESHOLD_M = 0.15
SETTLE_3D_POSITION_THRESHOLD_M = 0.25
SETTLE_SPEED_THRESHOLD_MPS = 0.20
SETTLE_HOLD_TIME_S = 0.50


# =============================================================================
# HELPERS
# =============================================================================

def _load_pt(path):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def _tensor3(value, *, device):
    return (
        torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        .reshape(3)
        .clone()
    )


def _quat4(value, *, device):
    return (
        torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        .reshape(4)
        .clone()
    )


def _first_tensor(mapping, names, *, what):
    for name in names:
        value = mapping.get(name)

        if value is not None:
            return value, name

    raise RuntimeError(
        f"Could not find {what}. "
        f"Tried {names}. "
        f"Available keys: {sorted(mapping.keys())}"
    )


def _first_scalar(mapping, names, default=None):
    for name in names:
        value = mapping.get(name)

        if value is None:
            continue

        if torch.is_tensor(value):
            return float(
                value.detach().cpu().reshape(-1)[0].item()
            )

        return float(value)

    return default


def _yaw_from_quat_wxyz(q):
    w, x, y, z = [
        float(v)
        for v in q.detach().cpu().reshape(4)
    ]

    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _ee_speed_from_position(position, dt):
    position = np.asarray(
        position,
        dtype=np.float64,
    )

    if position.shape[0] < 2:
        return np.zeros(
            position.shape[0],
            dtype=np.float64,
        )

    velocity = np.gradient(
        position,
        float(dt),
        axis=0,
    )

    return np.linalg.norm(
        velocity,
        axis=1,
    )


def _tilt_deg_from_quaternion_wxyz(quaternion):
    quaternion = np.asarray(
        quaternion,
        dtype=np.float64,
    )

    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    norm = np.sqrt(
        w * w
        + x * x
        + y * y
        + z * z
    )

    norm = np.maximum(
        norm,
        1.0e-12,
    )

    x = x / norm
    y = y / norm

    cos_tilt = (
        1.0
        - 2.0 * (
            x * x
            + y * y
        )
    )

    return np.degrees(
        np.arccos(
            np.clip(
                cos_tilt,
                -1.0,
                1.0,
            )
        )
    )


def _first_settling_time(
    position_error,
    speed,
    dt,
    position_threshold,
    speed_threshold,
    hold_time,
):
    position_error = np.asarray(
        position_error,
        dtype=np.float64,
    )

    speed = np.asarray(
        speed,
        dtype=np.float64,
    )

    good = (
        (position_error < float(position_threshold))
        & (speed < float(speed_threshold))
    )

    hold_samples = max(
        1,
        int(
            np.ceil(
                float(hold_time)
                / float(dt)
            )
        ),
    )

    if good.shape[0] < hold_samples:
        return False, None

    run = 0

    for index, value in enumerate(good):
        if value:
            run += 1
        else:
            run = 0

        if run >= hold_samples:
            first_index = (
                index
                - hold_samples
                + 1
            )

            return (
                True,
                float(
                    first_index
                    * float(dt)
                ),
            )

    return False, None


def _integral(error, dt):
    error = np.asarray(
        error,
        dtype=np.float64,
    )

    if error.shape[0] < 2:
        return 0.0

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(
                error,
                dx=float(dt),
            )
        )

    return float(
        np.trapz(
            error,
            dx=float(dt),
        )
    )


def _max_along_track_overshoot(
    ee_position,
    initial_ee_position,
    goal_position,
):
    ee_position = np.asarray(
        ee_position,
        dtype=np.float64,
    )

    initial_ee_position = np.asarray(
        initial_ee_position,
        dtype=np.float64,
    ).reshape(3)

    goal_position = np.asarray(
        goal_position,
        dtype=np.float64,
    ).reshape(3)

    direction = (
        goal_position
        - initial_ee_position
    )

    distance = float(
        np.linalg.norm(direction)
    )

    if distance < 1.0e-12:
        return 0.0

    unit_direction = (
        direction
        / distance
    )

    beyond_goal = (
        ee_position
        - goal_position[None, :]
    ) @ unit_direction

    return float(
        max(
            0.0,
            float(np.max(beyond_goal)),
        )
    )


def _compute_common_metrics(
    ee_position,
    body_velocity,
    body_quaternion,
    goal_position,
    initial_ee_position,
    dt,
    xy_position_threshold,
    position_3d_threshold,
    speed_threshold,
    hold_time,
):
    ee_position = np.asarray(
        ee_position,
        dtype=np.float64,
    )

    body_velocity = np.asarray(
        body_velocity,
        dtype=np.float64,
    )

    body_quaternion = np.asarray(
        body_quaternion,
        dtype=np.float64,
    )

    goal_position = np.asarray(
        goal_position,
        dtype=np.float64,
    ).reshape(3)

    error_vector = (
        ee_position
        - goal_position[None, :]
    )

    error_xy = np.linalg.norm(
        error_vector[:, 0:2],
        axis=1,
    )

    error_z = np.abs(
        error_vector[:, 2]
    )

    error_3d = np.linalg.norm(
        error_vector,
        axis=1,
    )

    ee_speed = _ee_speed_from_position(
        ee_position,
        dt,
    )

    body_speed = np.linalg.norm(
        body_velocity,
        axis=1,
    )

    tilt_deg = _tilt_deg_from_quaternion_wxyz(
        body_quaternion
    )

    (
        xy_settled,
        xy_settling_time,
    ) = _first_settling_time(
        error_xy,
        ee_speed,
        dt,
        xy_position_threshold,
        speed_threshold,
        hold_time,
    )

    (
        settled_3d,
        settling_time_3d,
    ) = _first_settling_time(
        error_3d,
        ee_speed,
        dt,
        position_3d_threshold,
        speed_threshold,
        hold_time,
    )

    return {
        "samples": int(
            ee_position.shape[0]
        ),
        "duration_s": float(
            max(
                0,
                ee_position.shape[0] - 1,
            )
            * float(dt)
        ),
        "initial_3d_error_m": float(
            error_3d[0]
        ),
        "minimum_3d_error_m": float(
            np.min(error_3d)
        ),
        "minimum_xy_error_m": float(
            np.min(error_xy)
        ),
        "final_3d_error_m": float(
            error_3d[-1]
        ),
        "final_xy_error_m": float(
            error_xy[-1]
        ),
        "final_abs_z_error_m": float(
            error_z[-1]
        ),
        "peak_ee_speed_mps": float(
            np.max(ee_speed)
        ),
        "peak_body_speed_mps": float(
            np.max(body_speed)
        ),
        "peak_tilt_deg": float(
            np.max(tilt_deg)
        ),
        "integrated_3d_error_m_s": _integral(
            error_3d,
            dt,
        ),
        "integrated_xy_error_m_s": _integral(
            error_xy,
            dt,
        ),
        "max_along_track_overshoot_m": (
            _max_along_track_overshoot(
                ee_position,
                initial_ee_position,
                goal_position,
            )
        ),
        "xy_settled_success": bool(
            xy_settled
        ),
        "xy_settling_time_s": (
            xy_settling_time
        ),
        "settled_3d_success": bool(
            settled_3d
        ),
        "settling_3d_time_s": (
            settling_time_3d
        ),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _jsonable(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        (
            np.floating,
            np.integer,
        ),
    ):
        return value.item()

    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    return value


def _compact_metrics(metrics):
    return {
        k: v
        for k, v in metrics.items()
        if k
        not in {
            "distance_3d_m",
            "distance_xy_m",
            "ee_speed_mps",
            "tilt_deg",
        }
    }


def _invalidate_robot_buffers(env):
    robot_data = env._robot.data

    for buffer_name in (
        "_body_com_vel_w",
        "_body_link_vel_w",
        "_body_state_w",
        "_body_link_state_w",
        "_body_com_state_w",
    ):
        buffer = getattr(
            robot_data,
            buffer_name,
            None,
        )

        if buffer is not None:
            buffer.timestamp = -1.0


def _install_exact_b3_plant(
    env,
    *,
    device,
):
    """
    Collapse all physical mass/inertia onto the main rigid body.

    The non-body links are retained kinematically but receive only tiny
    numerical mass/inertia values, matching the validated B3 replay plant.
    """

    body_names = list(
        env._robot.body_names
    )

    if "body" not in body_names:
        raise RuntimeError(
            f"Main rigid body 'body' not found: {body_names}"
        )

    body_idx = body_names.index(
        "body"
    )

    env_ids = torch.tensor(
        [0],
        device=device,
        dtype=torch.long,
    )

    masses = (
        env._robot.root_physx_view
        .get_masses()
        .clone()
    )

    masses[0].fill_(
        1.0e-13
    )

    masses[
        0,
        body_idx,
    ] = B3_MASS_KG

    env._robot.root_physx_view.set_masses(
        masses.cpu(),
        env_ids.cpu(),
    )

    inertias = (
        env._robot.root_physx_view
        .get_inertias()
        .clone()
    )

    inertias[0].zero_()

    tiny_I = torch.eye(
        3,
        dtype=inertias.dtype,
        device=inertias.device,
    ) * 1.0e-13

    for idx in range(
        inertias.shape[1]
    ):
        inertias[
            0,
            idx,
            :,
        ] = tiny_I.reshape(9)

    body_I = torch.diag(
        B3_INERTIA_KGM2.to(
            dtype=inertias.dtype,
            device=inertias.device,
        )
    )

    inertias[
        0,
        body_idx,
        :,
    ] = body_I.reshape(9)

    env._robot.root_physx_view.set_inertias(
        inertias.cpu(),
        env_ids.cpu(),
    )

    mass_readback = (
        env._robot.root_physx_view
        .get_masses()[0]
        .detach()
        .cpu()
    )

    inertia_readback = (
        env._robot.root_physx_view
        .get_inertias()[
            0,
            body_idx,
        ]
        .detach()
        .cpu()
        .reshape(3, 3)
    )

    total_mass = float(
        mass_readback.sum().item()
    )

    mass_error = abs(
        total_mass
        - B3_MASS_KG
    )

    inertia_error = float(
        torch.max(
            torch.abs(
                inertia_readback
                - torch.diag(
                    B3_INERTIA_KGM2
                ).to(
                    dtype=inertia_readback.dtype
                )
            )
        ).item()
    )

    if mass_error > 1.0e-7:
        raise RuntimeError(
            "B3 mass installation failed: "
            f"error={mass_error:.9e}"
        )

    if inertia_error > 1.0e-9:
        raise RuntimeError(
            "B3 inertia installation failed: "
            f"error={inertia_error:.9e}"
        )

    print()
    print("=" * 80)
    print("FIXED B3 PHYSICAL PLANT")
    print("=" * 80)
    print(
        f"mass       : {total_mass:.9f} kg"
    )
    print(
        "inertia    :",
        B3_INERTIA_KGM2.tolist(),
        "kg m^2",
    )
    print(
        f"mass error : {mass_error:.3e}"
    )
    print(
        f"I error    : {inertia_error:.3e}"
    )
    print("=" * 80)


def _install_direct_wrench_authority(
    env,
):
    """
    Replace only this environment instance's actuator application.

    env.step() still advances the normal Isaac environment, but the plant
    receives exactly [collective thrust, body torque] from the geometric
    controller instead of an SRT/PWM/motor interpretation.
    """

    env._b3_geometric_wrench = torch.zeros(
        (
            env.num_envs,
            4,
        ),
        dtype=torch.float32,
        device=env.device,
    )

    def _apply_direct_wrench(self):
        wrench = (
            self._b3_geometric_wrench
            .to(
                dtype=self._thrust.dtype,
                device=self._thrust.device,
            )
        )

        self._thrust.zero_()
        self._moment.zero_()

        self._thrust[
            :,
            0,
            2,
        ] = wrench[:, 0]

        self._moment[
            :,
            0,
            :,
        ] = wrench[:, 1:4]

        self._robot.set_external_force_and_torque(
            self._thrust,
            self._moment,
            body_ids=self._body_id,
        )

    env._apply_action = types.MethodType(
        _apply_direct_wrench,
        env,
    )


# =============================================================================
# LOAD CAPTURED BENCHMARK
# =============================================================================

CASE = _load_pt(
    args_cli.case_path
)

RL_TRACE = _load_pt(
    args_cli.rl_trace_path
)


# =============================================================================
# MAIN
# =============================================================================

@hydra_task_config(
    args_cli.task,
    "rsl_rl_cfg_entry_point",
)
def main(
    env_cfg: (
        ManagerBasedRLEnvCfg
        | DirectRLEnvCfg
    ),
    agent_cfg,
):
    del agent_cfg

    meta = CASE["metadata"]
    initial = CASE["initial_state"]
    goal = CASE["goal"]
    latency = CASE.get(
        "latency",
        {},
    )

    device = torch.device(
        args_cli.device
        if args_cli.device is not None
        else env_cfg.sim.device
    )

    horizon_s = float(
        args_cli.benchmark_horizon_s
    )

    if horizon_s <= 0.0:
        raise RuntimeError(
            "--benchmark_horizon_s must be positive."
        )

    # -------------------------------------------------------------------------
    # This clean backend currently reproduces the validated zero-latency B3
    # geometric benchmark. Do not silently ignore future nonzero-latency cases.
    # -------------------------------------------------------------------------
    physical_delay_s = float(
        latency.get(
            "physical_delay_seconds",
            0.0,
        )
    )

    if abs(
        physical_delay_s
    ) > 1.0e-9:
        raise RuntimeError(
            "Clean B3 geometric backend currently supports the validated "
            "zero-latency benchmark only. Captured delay was "
            f"{physical_delay_s:.9f} s."
        )

    sim_dt = float(
        env_cfg.sim.dt
    )

    decimation_float = (
        1.0
        / (
            sim_dt
            * GEOMETRIC_RATE_HZ
        )
    )

    decimation = int(
        round(
            decimation_float
        )
    )

    if abs(
        decimation_float
        - decimation
    ) > 1.0e-9:
        raise RuntimeError(
            "500 Hz geometric control cannot be represented exactly "
            f"with sim dt={sim_dt}."
        )

    # -------------------------------------------------------------------------
    # One independent B3 plant.
    # -------------------------------------------------------------------------
    env_cfg.scene.num_envs = 1

    if args_cli.device is not None:
        env_cfg.sim.device = (
            args_cli.device
        )

    env_cfg.eval_mode = True
    env_cfg.gc_mode = True
    env_cfg.control_mode = "SRT"

    env_cfg.policy_rate_hz = int(
        GEOMETRIC_RATE_HZ
    )

    env_cfg.decimation = (
        decimation
    )

    env_cfg.sim.render_interval = (
        decimation
    )

    for name in (
        "task_body",
        "goal_body",
        "reward_task_body",
        "reward_goal_body",
        "visualization_body",
    ):
        value = meta.get(name)

        if (
            value is not None
            and hasattr(
                env_cfg,
                name,
            )
        ):
            setattr(
                env_cfg,
                name,
                value,
            )

    envs_gym = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    env = envs_gym.unwrapped

    envs_gym.reset()

    env_ids = torch.tensor(
        [0],
        device=device,
        dtype=torch.long,
    )

    # -------------------------------------------------------------------------
    # Fixed physical B3.
    # -------------------------------------------------------------------------
    _install_exact_b3_plant(
        env,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Restore the EXACT captured RL kinematic start.
    #
    # We deliberately keep the original world coordinates here. Translation
    # of the entire experiment is physically irrelevant, and avoiding it makes
    # the clean benchmark independent of the old multi-env origin machinery.
    # -------------------------------------------------------------------------
    root_state = (
        initial["root_state_w"]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .clone()
        .reshape(13)
    )

    env._robot.write_root_pose_to_sim(
        root_state[
            0:7
        ].view(1, 7),
        env_ids=env_ids,
    )

    env._robot.write_root_velocity_to_sim(
        root_state[
            7:13
        ].view(1, 6),
        env_ids=env_ids,
    )

    env.episode_length_buf[0] = 0

    _invalidate_robot_buffers(
        env
    )

    # -------------------------------------------------------------------------
    # Captured EE target.
    # -------------------------------------------------------------------------
    desired_ee_raw, desired_ee_key = (
        _first_tensor(
            goal,
            (
                "authoritative_ee_goal_pos_w",
                "desired_pos_w",
                "task_goal_pos_w",
                "ee_goal_pos_w",
            ),
            what="captured EE goal",
        )
    )

    desired_ee_goal_w = _tensor3(
        desired_ee_raw,
        device=device,
    )

    initial_ee_w = _tensor3(
        initial["ee_pos_w"],
        device=device,
    )

    commanded_ee_delta_w = (
        desired_ee_goal_w
        - initial_ee_w
    )

    # -------------------------------------------------------------------------
    # Controller COM/GC target.
    #
    # Compatibility:
    # Existing captured V2 cases call this `mellinger_goal_pos_w`.
    # It is simply the already-solved COM/GC target corresponding to the
    # captured EE transfer. No firmware/Mellinger code is used here.
    # -------------------------------------------------------------------------
    gc_goal_raw, gc_goal_key = (
        _first_tensor(
            goal,
            (
                "b3_geometric_goal_pos_w",
                "controller_goal_com_w",
                "gc_goal_pos_w",
                "com_goal_pos_w",
                "mellinger_goal_pos_w",
            ),
            what="controller COM/GC goal",
        )
    )

    gc_goal_w = _tensor3(
        gc_goal_raw,
        device=device,
    )

    # Goal yaw. Existing benchmark commands do not request a yaw step.
    # Prefer an explicit captured field; otherwise use zero, matching the
    # validated point-to-point benchmark contract.
    goal_yaw_rad = _first_scalar(
        goal,
        (
            "b3_geometric_goal_yaw_rad",
            "controller_goal_yaw_rad",
            "mellinger_goal_yaw_rad",
            "goal_yaw_rad",
            "desired_yaw_rad",
            "desired_yaw_w",
        ),
        default=0.0,
    )

    # Keep environment bookkeeping aligned with the actual benchmark EE goal
    # where possible. The geometric controller itself does NOT consume this.
    if hasattr(
        env,
        "_desired_pos_w",
    ):
        env._desired_pos_w[
            0
        ].copy_(
            desired_ee_goal_w
        )

    # -------------------------------------------------------------------------
    # Validate restored state.
    # -------------------------------------------------------------------------
    (
        body_pos,
        body_quat,
        body_vel,
        body_ang,
    ) = env.get_frame_state_from_task(
        "body"
    )

    (
        ee_pos,
        ee_quat,
        _,
        _,
    ) = env.get_frame_state_from_task(
        "endeffector"
    )

    expected_body_pos = _tensor3(
        initial["body_pos_w"],
        device=device,
    )

    expected_body_quat = _quat4(
        initial["body_quat_w"],
        device=device,
    )

    expected_body_vel = _tensor3(
        initial["body_lin_vel_w"],
        device=device,
    )

    expected_body_ang = _tensor3(
        initial["body_ang_vel_w"],
        device=device,
    )

    start_errors = {
        "body_position": float(
            torch.max(
                torch.abs(
                    body_pos[0]
                    - expected_body_pos
                )
            ).item()
        ),
        "body_quaternion": float(
            torch.min(
                torch.stack(
                    (
                        torch.max(
                            torch.abs(
                                body_quat[0]
                                - expected_body_quat
                            )
                        ),
                        torch.max(
                            torch.abs(
                                body_quat[0]
                                + expected_body_quat
                            )
                        ),
                    )
                )
            ).item()
        ),
        "body_linear_velocity": float(
            torch.max(
                torch.abs(
                    body_vel[0]
                    - expected_body_vel
                )
            ).item()
        ),
        "body_angular_velocity": float(
            torch.max(
                torch.abs(
                    body_ang[0]
                    - expected_body_ang
                )
            ).item()
        ),
        "ee_position": float(
            torch.max(
                torch.abs(
                    ee_pos[0]
                    - initial_ee_w
                )
            ).item()
        ),
    }

    for name, error in (
        start_errors.items()
    ):
        if error > 1.0e-4:
            raise RuntimeError(
                "Captured-state restoration failed for "
                f"{name}: {error:.9e}"
            )

    # -------------------------------------------------------------------------
    # Geometric controller + direct plant authority.
    # -------------------------------------------------------------------------
    controller = (
        B3BehaviorGeometricController(
            device=device,
        )
    )

    _install_direct_wrench_authority(
        env
    )

    print()
    print("=" * 80)
    print("RL -> B3 GEOMETRIC BENCHMARK")
    print("=" * 80)
    print(
        "source robot index :",
        meta.get(
            "robot_index",
            "unknown",
        ),
    )
    print(
        "control rate       :",
        GEOMETRIC_RATE_HZ,
        "Hz",
    )
    print(
        "initial EE         :",
        initial_ee_w.detach().cpu(),
    )
    print(
        "commanded EE delta :",
        commanded_ee_delta_w
        .detach()
        .cpu(),
    )
    print(
        "EE goal            :",
        desired_ee_goal_w
        .detach()
        .cpu(),
        f"(case key: {desired_ee_key})",
    )
    print(
        "COM/GC goal        :",
        gc_goal_w
        .detach()
        .cpu(),
        f"(case key: {gc_goal_key})",
    )
    print(
        "goal yaw           :",
        goal_yaw_rad,
        "rad",
    )
    print(
        "firmware           : NONE",
    )
    print(
        "motor/PWM model    : NONE",
    )
    print(
        "pre-hover          : NONE",
    )
    print(
        "plant authority    : direct body wrench",
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Rollout.
    # -------------------------------------------------------------------------
    steps = int(
        round(
            horizon_s
            * GEOMETRIC_RATE_HZ
        )
    )

    body_pos_log = []
    body_quat_log = []
    body_vel_log = []
    body_ang_log = []

    ee_pos_log = []
    ee_quat_log = []

    wrench_log = []

    terminated = False
    truncated = False

    dummy_action = torch.zeros_like(
        env._motor_speeds_des
    )

    for step in range(
        steps
    ):
        (
            body_pos,
            body_quat,
            body_vel,
            body_ang,
        ) = env.get_frame_state_from_task(
            "body"
        )

        (
            ee_pos,
            ee_quat,
            _,
            _,
        ) = env.get_frame_state_from_task(
            "endeffector"
        )

        behavior_out = (
            controller.step(
                pos_w=body_pos[0],
                quat_wxyz=body_quat[0],
                lin_vel_w=body_vel[0],
                ang_vel_w=(
                    env._robot.data
                    .root_ang_vel_w[0]
                ),
                goal_pos_w=gc_goal_w,
                goal_yaw_rad=(
                    goal_yaw_rad
                ),
            )
        )

        wrench = (
            behavior_out[
                "wrench_body"
            ]
            .detach()
            .clone()
            .view(1, 4)
        )

        env._b3_geometric_wrench.copy_(
            wrench
        )

        body_pos_log.append(
            body_pos[0]
            .detach()
            .cpu()
            .clone()
        )
        body_quat_log.append(
            body_quat[0]
            .detach()
            .cpu()
            .clone()
        )
        body_vel_log.append(
            body_vel[0]
            .detach()
            .cpu()
            .clone()
        )
        body_ang_log.append(
            body_ang[0]
            .detach()
            .cpu()
            .clone()
        )
        ee_pos_log.append(
            ee_pos[0]
            .detach()
            .cpu()
            .clone()
        )
        ee_quat_log.append(
            ee_quat[0]
            .detach()
            .cpu()
            .clone()
        )
        wrench_log.append(
            wrench[0]
            .detach()
            .cpu()
            .clone()
        )

        if step == 0:
            print()
            print("=" * 80)
            print(
                "B3 GEOMETRIC FIRST STEP"
            )
            print("=" * 80)
            print(
                "body pos W     :",
                body_pos[0]
                .detach()
                .cpu(),
            )
            print(
                "COM/GC goal W  :",
                gc_goal_w
                .detach()
                .cpu(),
            )
            print(
                "position error :",
                behavior_out[
                    "position_error_w"
                ]
                .detach()
                .cpu(),
            )
            print(
                "target force W :",
                behavior_out[
                    "target_force_w"
                ]
                .detach()
                .cpu(),
                "N",
            )
            print(
                "desired tilt   :",
                float(
                    torch.rad2deg(
                        behavior_out[
                            "desired_tilt_rad"
                        ]
                    ).cpu()
                ),
                "deg",
            )
            print(
                "thrust         :",
                float(
                    behavior_out[
                        "thrust_N"
                    ].cpu()
                ),
                "N",
            )
            print(
                "torque         :",
                behavior_out[
                    "torque_Nm"
                ]
                .detach()
                .cpu(),
                "Nm",
            )
            print("=" * 80)

        (
            _obs,
            _reward,
            term,
            trunc,
            _info,
        ) = envs_gym.step(
            dummy_action
        )

        terminated = bool(
            term[0].item()
        )

        truncated = bool(
            trunc[0].item()
        )

        if (
            terminated
            or truncated
        ):
            print(
                "[B3 geometric] rollout ended "
                f"at t={(step + 1) / GEOMETRIC_RATE_HZ:.3f}s "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )
            break

    if not ee_pos_log:
        raise RuntimeError(
            "B3 geometric rollout produced no samples."
        )

    # -------------------------------------------------------------------------
    # Tensors.
    # -------------------------------------------------------------------------
    b3_body_pos = torch.stack(
        body_pos_log,
        dim=0,
    )
    b3_body_quat = torch.stack(
        body_quat_log,
        dim=0,
    )
    b3_body_vel = torch.stack(
        body_vel_log,
        dim=0,
    )
    b3_body_ang = torch.stack(
        body_ang_log,
        dim=0,
    )
    b3_ee_pos = torch.stack(
        ee_pos_log,
        dim=0,
    )
    b3_ee_quat = torch.stack(
        ee_quat_log,
        dim=0,
    )
    b3_wrench = torch.stack(
        wrench_log,
        dim=0,
    )

    # -------------------------------------------------------------------------
    # RL trace over the exact same physical horizon.
    # -------------------------------------------------------------------------
    rl_rate_hz = float(
        RL_TRACE[
            "policy_rate_hz"
        ]
    )

    rl_full_state = (
        RL_TRACE[
            "full_state"
        ]
        .detach()
        .cpu()
        .to(
            torch.float32
        )
    )

    rl_steps = min(
        int(
            RL_TRACE[
                "steps"
            ]
        ),
        int(
            rl_full_state.shape[0]
        ),
        int(
            round(
                horizon_s
                * rl_rate_hz
            )
        ),
    )

    rl_full_state = (
        rl_full_state[
            :rl_steps
        ]
    )

    if (
        rl_full_state.ndim != 2
        or rl_full_state.shape[1] < 16
    ):
        raise RuntimeError(
            "Unexpected RL full-state layout."
        )

    rl_body_quat = (
        rl_full_state[
            :,
            3:7,
        ]
        .numpy()
    )

    rl_body_vel = (
        rl_full_state[
            :,
            7:10,
        ]
        .numpy()
    )

    rl_ee_pos = (
        rl_full_state[
            :,
            13:16,
        ]
        .numpy()
    )

    # -------------------------------------------------------------------------
    # Common metrics.
    # -------------------------------------------------------------------------
    goal_np = (
        desired_ee_goal_w
        .detach()
        .cpu()
        .numpy()
    )

    initial_ee_np = (
        initial_ee_w
        .detach()
        .cpu()
        .numpy()
    )

    rl_metrics = _compute_common_metrics(
        ee_position=rl_ee_pos,
        body_velocity=rl_body_vel,
        body_quaternion=rl_body_quat,
        goal_position=goal_np,
        initial_ee_position=initial_ee_np,
        dt=1.0 / rl_rate_hz,
        xy_position_threshold=XY_SETTLE_POSITION_THRESHOLD_M,
        position_3d_threshold=SETTLE_3D_POSITION_THRESHOLD_M,
        speed_threshold=SETTLE_SPEED_THRESHOLD_MPS,
        hold_time=SETTLE_HOLD_TIME_S,
    )

    b3_metrics = _compute_common_metrics(
        ee_position=b3_ee_pos.numpy(),
        body_velocity=b3_body_vel.numpy(),
        body_quaternion=b3_body_quat.numpy(),
        goal_position=goal_np,
        initial_ee_position=initial_ee_np,
        dt=1.0 / GEOMETRIC_RATE_HZ,
        xy_position_threshold=XY_SETTLE_POSITION_THRESHOLD_M,
        position_3d_threshold=SETTLE_3D_POSITION_THRESHOLD_M,
        speed_threshold=SETTLE_SPEED_THRESHOLD_MPS,
        hold_time=SETTLE_HOLD_TIME_S,
    )

    # -------------------------------------------------------------------------
    # Output artifacts.
    # -------------------------------------------------------------------------
    output_dir = (
        Path(
            args_cli.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "benchmark_metrics.json"
    )

    trace_path = (
        output_dir
        / "rl_vs_geometric_trace.pt"
    )

    plot_path = (
        output_dir
        / "distance_to_goal.png"
    )

    manifest = {
        "benchmark_contract": (
            "DEPLOYMENT_RELATIVE_EE_TRANSFER_V1"
        ),
        "source_rl_robot_index": (
            int(
                meta.get(
                    "robot_index",
                    -1,
                )
            )
        ),
        "benchmark_horizon_s": (
            horizon_s
        ),
        "rl_rate_hz": (
            rl_rate_hz
        ),
        "b3_geometric_rate_hz": (
            GEOMETRIC_RATE_HZ
        ),
        "initial_state_errors": (
            start_errors
        ),
        "initial_ee_pos_w": (
            initial_ee_np
        ),
        "commanded_ee_displacement_w": (
            commanded_ee_delta_w
            .detach()
            .cpu()
            .numpy()
        ),
        "ee_goal_w": (
            goal_np
        ),
        "controller_com_goal_w": (
            gc_goal_w
            .detach()
            .cpu()
            .numpy()
        ),
        "controller_goal_yaw_rad": (
            goal_yaw_rad
        ),
        "plant": {
            "mass_kg": (
                B3_MASS_KG
            ),
            "inertia_kgm2": (
                B3_INERTIA_KGM2
                .numpy()
            ),
        },
        "controller": {
            "name": (
                "B3BehaviorGeometricController"
            ),
            "outer_loop": (
                "real_B3_Mellinger_position_gains"
            ),
            "inner_loop": (
                "SI_SO3"
            ),
            "actuation": (
                "direct_body_wrench"
            ),
            "firmware": False,
            "pwm": False,
            "motor_model": False,
            "prehover": False,
        },
        "rl": (
            _compact_metrics(
                rl_metrics
            )
        ),
        "b3_geometric": (
            _compact_metrics(
                b3_metrics
            )
        ),
        "terminated": (
            terminated
        ),
        "truncated": (
            truncated
        ),
    }

    metrics_path.write_text(
        json.dumps(
            _jsonable(
                manifest
            ),
            indent=2,
        )
        + "\n"
    )

    torch.save(
        {
            "metadata": {
                "benchmark_contract": (
                    "DEPLOYMENT_RELATIVE_EE_TRANSFER_V1"
                ),
                "controller": (
                    "B3BehaviorGeometricController"
                ),
                "b3_geometric_rate_hz": (
                    GEOMETRIC_RATE_HZ
                ),
                "rl_rate_hz": (
                    rl_rate_hz
                ),
                "firmware_used": False,
                "motor_model_used": False,
                "prehover_used": False,
                "direct_body_wrench": True,
            },
            "goal": {
                "initial_ee_pos_w": (
                    initial_ee_w
                    .detach()
                    .cpu()
                ),
                "commanded_ee_displacement_w": (
                    commanded_ee_delta_w
                    .detach()
                    .cpu()
                ),
                "ee_goal_w": (
                    desired_ee_goal_w
                    .detach()
                    .cpu()
                ),
                "controller_com_goal_w": (
                    gc_goal_w
                    .detach()
                    .cpu()
                ),
                "controller_goal_yaw_rad": (
                    goal_yaw_rad
                ),
            },
            "rl": {
                "full_state": (
                    rl_full_state
                ),
            },
            "b3_geometric": {
                "body_position_w": (
                    b3_body_pos
                ),
                "body_quaternion_wxyz": (
                    b3_body_quat
                ),
                "body_linear_velocity_w": (
                    b3_body_vel
                ),
                "body_angular_velocity_w": (
                    b3_body_ang
                ),
                "ee_position_w": (
                    b3_ee_pos
                ),
                "ee_quaternion_wxyz": (
                    b3_ee_quat
                ),
                "applied_body_wrench": (
                    b3_wrench
                ),
            },
        },
        trace_path,
    )

    # Simple, correctly-labelled 3D distance-to-EE-goal plot.
    rl_distance_3d = np.linalg.norm(
        rl_ee_pos - goal_np[None, :],
        axis=1,
    )

    b3_distance_3d = np.linalg.norm(
        b3_ee_pos.numpy() - goal_np[None, :],
        axis=1,
    )

    rl_t = (
        np.arange(
            len(rl_distance_3d),
            dtype=np.float64,
        )
        / rl_rate_hz
    )

    b3_t = (
        np.arange(
            len(b3_distance_3d),
            dtype=np.float64,
        )
        / GEOMETRIC_RATE_HZ
    )

    plt.figure(
        figsize=(8.0, 4.8)
    )

    plt.plot(
        rl_t,
        rl_distance_3d,
        label="RL",
    )

    plt.plot(
        b3_t,
        b3_distance_3d,
        label="B3 Geometric",
    )

    plt.axhline(
        SETTLE_3D_POSITION_THRESHOLD_M,
        linestyle="--",
        linewidth=1.0,
        label="3D settle threshold",
    )

    plt.xlabel(
        "Time [s]"
    )
    plt.ylabel(
        "3D EE distance to goal [m]"
    )
    plt.title(
        "RL vs B3 Geometric"
    )
    plt.grid(
        True,
        alpha=0.25,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=160,
    )
    plt.close()

    # -------------------------------------------------------------------------
    # Concise final report.
    # -------------------------------------------------------------------------
    def _fmt_t(value):
        return (
            "None"
            if value is None
            else f"{value:.3f}"
        )

    print()
    print("=" * 80)
    print(
        "COMMON APPLES-TO-APPLES BENCHMARK METRICS"
    )
    print("=" * 80)

    print(
        "RL            "
        f"min3D={rl_metrics['minimum_3d_error_m']:.4f} m  "
        f"final3D={rl_metrics['final_3d_error_m']:.4f} m  "
        f"finalXY={rl_metrics['final_xy_error_m']:.4f} m  "
        f"finalZ={rl_metrics['final_abs_z_error_m']:.4f} m"
    )

    print(
        "              "
        f"peakEEv={rl_metrics['peak_ee_speed_mps']:.4f} m/s  "
        f"peakTilt={rl_metrics['peak_tilt_deg']:.2f} deg  "
        f"t3D={_fmt_t(rl_metrics['settling_3d_time_s'])}"
    )

    print(
        "B3 Geometric  "
        f"min3D={b3_metrics['minimum_3d_error_m']:.4f} m  "
        f"final3D={b3_metrics['final_3d_error_m']:.4f} m  "
        f"finalXY={b3_metrics['final_xy_error_m']:.4f} m  "
        f"finalZ={b3_metrics['final_abs_z_error_m']:.4f} m"
    )

    print(
        "              "
        f"peakEEv={b3_metrics['peak_ee_speed_mps']:.4f} m/s  "
        f"peakTilt={b3_metrics['peak_tilt_deg']:.2f} deg  "
        f"t3D={_fmt_t(b3_metrics['settling_3d_time_s'])}"
    )

    print("-" * 80)
    print(
        "metrics :",
        metrics_path,
    )
    print(
        "trace   :",
        trace_path,
    )
    print(
        "plot    :",
        plot_path,
    )
    print("=" * 80)

    envs_gym.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
