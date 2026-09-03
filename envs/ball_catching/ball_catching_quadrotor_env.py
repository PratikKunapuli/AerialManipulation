# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, random_yaw_orientation, matrix_from_quat, matrix_from_euler, quat_apply_inverse, quat_apply, normalize, wrap_to_pi, quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from utils.assets import MODELS_PATH
from configs.aerial_manip_asset import CRAZYFLIE_MANIPULATOR_0DOF_CFG, CRAZYFLIE_MANIPULATOR_0DOF_LONG_CFG, CRAZYFLIE_BRUSHLESS_CFG, CRAZYFLIE_BRUSHLESS_MANIPULATOR_CFG
from utils.math_utilities import yaw_from_quat, yaw_error_from_quats, quat_from_yaw, compute_desired_pose_from_transform, vee_map, exp_so3, hat_map
import utils.flatness_utilities as flatness_utils
import utils.trajectory_utilities as traj_utils
import utils.math_utilities as math_utils
import isaaclab.utils.math as isaac_math_utils


##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadrotorEnvWindow(BaseEnvWindow):
    """Window manager for the Quadrotor environment."""

    def __init__(self, env: QuadrotorEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadrotorEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    action_space = 4
    observation_space = 18

    # Catch tolerance curriculum
    catch_tolerance_min: float = 0.08       # tightest it will ever get
    catch_tolerance_max: float = 0.50       # starts here
    catch_curriculum_tighten_rate: float = 0.005   # subtract per curriculum step
    catch_curriculum_loosen_rate: float = 0.002    # add per curriculum step
    catch_curriculum_tighten_threshold: float = 0.45  # success rate needed to tighten
    catch_curriculum_loosen_threshold: float = 0.20   # success rate that triggers loosening
    catch_curriculum_update_interval: int = 50        # update every N iterations

    catch_tolerance = 0.50
    
    # --- Time-to-catch curriculum ---
    time_to_catch_max: float = 1.5       # starts here, tightens over training
    time_to_catch_min: float = 0.60       # tightest it will ever get
    time_to_catch_tighten_rate: float = 0.02
    time_to_catch_tighten_threshold: float = 0.40  # success rate needed to tighten
    time_to_catch_loosen_threshold: float = 0.15   # success rate that triggers loosening
    time_to_catch_update_interval: int = 50        # update every N iterations

    # --- Continuous yaw alignment ---
    yaw_continuous_reward_scale: float = -1.5

    state_space = 0
    debug_vis = True
    sim_rate_hz = 1000
    policy_rate_hz = 250
    pd_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz

    num_actions = action_space
    num_observations = observation_space

    ui_window_class_type = QuadrotorEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)


    # Trajectory
    traj_update_dt = 0.02

    trajectory_type = "lissaajous"
    trajectory_horizon = 0
    random_shift_trajectory = False

    lissajous_amplitudes = [0, 0, 0, 0]
    lissajous_amplitudes_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_frequencies = [0, 0, 0, 0]
    lissajous_frequencies_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_phases = [0, 0, 0, 0]
    lissajous_phases_rand_ranges = [0.0, 0.0, 0.0, 0.0]
    lissajous_offsets = [0, 0,1.0, 0]
    lissajous_offsets_rand_ranges = [0.0, 0.0, 0.0, 0.0]

    init_cfg = "rand"
    init_pos_ranges=[0.0, 0.0, 0.0]
    init_lin_vel_ranges=[0.0, 0.0, 0.0]
    init_yaw_ranges=[0.0]
    init_euler_ranges = [0.5236, 0.5236, 1.5708]  
    init_ang_vel_ranges=[0.0, 0.0, 0.0]
    goal_pos_range = 2.0
    goal_yaw_range = 3.14159


    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    end_effector_mass = float(1e-9)
    mass = 0.03 # 30 grams
    Ixx = 1.35e-5
    Iyy = 1.35e-5
    Izz = 2.9e-5
    thrust_to_weight = 1.8
    # thrust_to_weight = 1.9
    moment_scale = 0.01
    # attitude_scale = (3.14159/180.0) * 30.0
    # attitude_scale = (3.14159)/2.0
    attitude_scale = 3.14159 / 6.0
    attitude_scale_z = torch.pi - 1e-6
    attitude_scale_xy = 0.2
    has_end_effector = False
    num_joints = 0
    skip_motor_dynamics = False

    control_mode = "CTATT" # "CTBM" or "CTATT" or "CTBR"
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate


    # Flag to use Rotopy obs and reward
    rotorpy_obs = False
    rotorpy_reward = False
    rotorpy_done = False
    action_history_length = 1
    state_history_length = 0

    # reward scales
    pos_distance_reward_scale = 15.0
    pos_radius = 0.8
    pos_corse_reward_scale = 0.0
    pos_corse_radius = 0.8
    pos_fine_reward_scale= 0.0
    pos_fine_radius = 0.1
    pos_radius_curriculum = 50_000_000
    pos_error_reward_scale= 0.0
    catch_miss_penalty_scale = -8.0  # <--- NEW: Massive penalty for missing
    catch_success_bonus_scale = 5.0
    yaw_catch_penalty_scale = -2.0
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    ori_error_reward_scale = -2.0
    previous_thrust_reward_scale = -0.1
    previous_attitude_reward_scale = -0.1
    previous_action_reward_scale = [7e-3, 3e-3, 3e-3, 3e-3] # [thrust, roll, pitch, yaw]
    action_vec_norm_reward_scale = [ 0.0, 0.0, 0.0, 0.0] # [thrust, roll, pitch, yaw]
    action_norm_reward_scale = 0.0
    stay_alive_reward = 0.0
    crash_penalty = -50.0
    scale_reward_with_time = True
    # reward / history
    rotorpy_obs = False
    rotorpy_reward = False
    rotorpy_done = False

    action_history_length = 5
    state_history_length = 0
    action_smooth_window = 4

    # position shaping
    square_pos_error = False
    pos_distance_reward_scale = 15.0
    pos_radius = 0.15

    pos_corse_reward_scale = 0.0
    pos_corse_radius = 0.8

    pos_fine_reward_scale = 2.5
    pos_fine_radius = 0.25

    pos_error_reward_scale = -1.0
    pos_radius_curriculum = 50_000_000

    # stabilization
    lin_vel_reward_scale = -0.03
    ang_vel_reward_scale = -0.04
    ori_error_reward_scale = -1.25

    # action penalties
    previous_thrust_reward_scale = -0.08
    previous_attitude_reward_scale = -0.10
    previous_action_norm_reward_scale = -0.05
    action_norm_reward_scale = -0.003

    # optional rotorpy-only terms
    previous_action_reward_scale = [7e-3, 3e-3, 3e-3, 3e-3]
    action_vec_norm_reward_scale = [0.0, 0.0, 0.0, 0.0]
    yaw_error_reward_scale = -2.0

    # episode terms
    stay_alive_reward = 0.02
    crash_penalty = 0.0
    scale_reward_with_time = True

    # near-goal weighting
    near_goal_radius = 0.12
    inner_hover_radius = 0.07

    vel_weight_base = 0.5
    vel_weight_near = 0.35
    vel_weight_inner = 0.5

    ori_weight_base = 0.65
    ori_weight_near = 0.35
    ori_weight_inner = 0.5

    smooth_weight_base = 0.15
    smooth_weight_near = 0.75
    smooth_weight_inner = 1.0

    # hover quality
    hover_lin_denom = 0.12
    hover_ang_denom = 0.18
    hover_ori_denom = 0.12

    # reward mixing
    pos_fine_mix = 0.75
    hover_quality_mix = 0.75
    stay_alive_hover_mix = 0.25
    action_norm_near_mix = 0.3


    ori_error_reward_scale = 0.0 # -0.5
    joint_vel_reward_scale = 0.0 # -0.01
    previous_action_norm_reward_scale = 0.0 # -0.01
    yaw_distance_reward_scale = 0.0 # -0.01
    yaw_radius = 0.2
    yaw_smooth_transition_scale = 0.0
    square_reward_errors = False
    square_pos_error = True
    penalize_action = False
    penalize_previous_action = False
    combined_alpha = 0.0
    combined_tolerance = 0.0
    combined_scale = 0.0

    # observation modifiers
    use_yaw_representation = False
    use_full_ori_matrix = True
    use_grav_vector = True
    use_previous_actions = False
    use_yaw_representation_for_trajectory=True
    use_ang_vel_from_trajectory=True

    eval_mode = False
    gc_mode = False
    goal_cfg= "rand"
    goal_pos = [0.0, 0.0, 1.0]
    goal_ori = [1.0, 0.0, 0.0, 0.0]

    task_body = "body"
    goal_body = "body"
    reward_task_body = "body"
    reward_goal_body = "body"
    visualization_body= "body"


    seed = 0


    # Motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0
    init_motor_speed = 1788.53

    kp_att = 1575.0 # 544
    kd_att = 229.93 # 46.64

    # CTBR Parameters
    kp_omega = 1 # default taken from RotorPy, needs to be checked on hardware. 
    kd_omega = 0.1 # default taken from RotorPy, needs to be checked on hardware.
    body_rate_scale_xy = 10.0
    body_rate_scale_z = 2.5

    # Domain Randomization
    dr_dict = {
        'thrust_to_weight':  0.0,
        'mass': 0.0,
        'inertia': 0.0,
        'arm_length': 0.0,
        'k_eta': 0.0,
        'k_m': 0.0,
        'tau_m': 0.0,
        'kp_att': 0.0,
        'kd_att': 0.0,
        'control_latency_steps': 0,
        }
    control_latency_steps = 12 # number of timesteps for control latency (estimated time delay 40ms)

    # Visualizations
    viz_mode = "triad" # or robot
    viz_history_length = 100
    robot_color=[0.0, 0.0, 0.0]
    viz_ref_offset=[0.0,0.0,0.0]

@configclass
class QuadrotorManipulatorEnvCfg(QuadrotorEnvCfg):
    # robot
    robot: ArticulationCfg = CRAZYFLIE_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.spawn.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")

    has_end_effector=True
    task_body = "endeffector"
    goal_body = "endeffector"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"
    visualization_body= "endeffector"


    dr_dict = {'thrust_to_weight':  False}

@configclass
class QuadrotorManipulatorLongEnvCfg(QuadrotorEnvCfg):
    # robot
    robot: ArticulationCfg = CRAZYFLIE_MANIPULATOR_0DOF_LONG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.spawn.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = (CRAZYFLIE_CFG.replace(usd_path=f"{MODELS_PATH}/crazyflie_manipulator.usd")).replace(prim_path="/World/envs/env_.*/Robot")

    has_end_effector=True
    task_body = "endeffector"
    goal_body = "endeffector"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"
    visualization_body= "endeffector"


    dr_dict = {'thrust_to_weight':  False}

@configclass
class BrushlessQuadrotorEnvCfg(QuadrotorEnvCfg):
    """
    Cfg for the brushless quadrotor environment simulating the Brushless Crazyflie 2.1
    """

    robot: ArticulationCfg = CRAZYFLIE_BRUSHLESS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    thrust_to_weight = 3.5
    mass = 0.041 # 39 grams
    Ixx = 1.5417e-5
    Iyy = 1.5904e-5
    Izz = 2.869e-5
    sim_rate_hz = 1000
    decimation = 10 # 10x decimation from sim physics rate
    pd_loop_rate_hz = 500
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate

    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Reward Terms
    previous_thrust_reward_scale = -0.01
    previous_attitude_reward_scale = -0.01

    # Motor dynamics
    arm_length = 0.05
    k_eta = 0.31615 # Measured from thrust stand data
    k_m = 7.8e-10 #unchanged
    k_torque = 2.567e-3 # Computed from sysID
    tau_m = 0.045 # slower motor dynamics
    motor_speed_min = 0.0
    motor_speed_max = 1.0
    init_motor_speed = 0.1 # should be 0.1 if using SRT control mode

    kp_att = 3264.54 # 544
    kd_att = 361.58 # 46.64

    # CTBR Parameters
    # kp_omega = 5.27 # measured on static test stand
    # kd_omega = 1.0
    kp_omega = 75.0 # measured from real-data
    kd_omega = 10.0
    body_rate_scale_xy = 6.28319
    body_rate_scale_z = 1.5708

    control_mode = "CTBR" # "CTBM" or "CTATT" or "CTBR" or "SRT"

    # task_body = "body"
    # goal_body = "body"
    # reward_task_body = "body"
    # reward_goal_body = "body"

    dr_dict = {
        'thrust_to_weight':  0.0,
        'mass': 0.0,
        'inertia': 0.0,
        'arm_length': 0.0,
        'k_eta': 0.0,
        'k_m': 0.0,
        'k_torque': 0.0,
        'tau_m': 0.0,
        'kp_att': 0.0,
        'kd_att': 0.0,
    }


@configclass
class BrushlessQuadrotorManipulatorEnvCfg(QuadrotorEnvCfg):
    """
    Cfg for the brushless quadrotor environment simulating the Brushless Crazyflie 2.1 with a manipulator
    """
    robot: ArticulationCfg = CRAZYFLIE_BRUSHLESS_MANIPULATOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    thrust_to_weight = 3.5
    mass = 0.046                   # 46.0 grams measured in real life with battery plugged in
    # Ixx = 2.68015e-5               # Calculated from sysID done on collected data
    # Iyy = 2.06085e-5               # Calculated from sysID done on collected data
    # Izz = 4.34276e-5               # Calculated from sysID done on collected data
    Ixx = 2.4255e-5             # Calculated from sysID done on collected data
    Iyy = 1.865e-5              # Calculated from sysID done on collected data
    Izz = 3.93e-5               # Calculated from sysID done on collected data
    sim_rate_hz = 1000
    decimation = 10 # 10x decimation from sim physics rate
    pd_loop_rate_hz = 500
    pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz # decimation from sim physics rate

    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Reward Terms
    previous_thrust_reward_scale = -0.01
    previous_attitude_reward_scale = -0.01

    # Motor dynamics
    arm_length = 0.05
    k_eta = 0.31615 # Measured from thrust stand data
    k_m = 7.8e-10 #unchanged
    # k_torque = 2.567e-3 # Computed from sysID
    k_torque=3.987e-3 # Computed from sysID
    tau_m = 0.050 # slower motor dynamics (Computed from sysID)
    motor_speed_min = 0.0
    motor_speed_max = 1.0
    init_motor_speed = 0.1 # should be 0.1 if using SRT control mode

    kp_att = 3264.54 # 544
    kd_att = 361.58 # 46.64

    # CTBR Parameters
    kp_omega = 75.0             # measured on static test stand
    kd_omega = 10.0             
    body_rate_scale_xy = 10.0
    body_rate_scale_z = 2.5

    control_mode = "CTBR" # "CTBM" or "CTATT" or "CTBR" or "SRT"

    task_body = "endeffector"
    goal_body = "endeffector"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"
    visualization_body= "endeffector"

    # dr_dict = {
    #     'thrust_to_weight':  0.0,
    #     'mass': 0.0,
    #     'inertia': 0.0,
    #     'arm_length': 0.0,
    #     'k_eta': 0.0,
    #     'k_m': 0.0,
    #     'k_torque': 0.0,
    #     'tau_m': 0.0,
    #     'kp_att': 0.0,
    #     'kd_att': 0.0,
    # }
    dr_dict = {
        "mass": 0.20,              # +/- 20%
        "inertia": 0.20,           # +/- 20%
        "arm_length": 0.05,        # +/- 5%
        "k_eta": 0.20,             # +/- 20%
        "k_m": 0.1,                # +/- 10%
        "k_torque": 0.20,          # +/- 20%
        "tau_m": 0.20,             # +/- 20%
        "kp_att": 0.20,            # +/- 20%
        "kd_att": 0.20,            # +/- 20%
        "thrust_to_weight": 0.20,  # +/- 20%
        "kp_omega": 0.10,          # +/- 10%
        "kd_omega": 0.10,          # +/- 10%
        'control_latency_steps': 4,
    }

    has_end_effector = True


class QuadrotorEnv(DirectRLEnv):
    cfg: QuadrotorEnvCfg

    def __init__(self, cfg: QuadrotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        torch.manual_seed(self.cfg.seed)

        # Total thrust and moment applied to the base of the quadrotor
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._hover_thrust = 2.0 / self.cfg.thrust_to_weight - 1.0
        self.min_thrust = torch.zeros(self.num_envs, device=self.device)
        self.max_thrust = torch.ones(self.num_envs, device=self.device) * (2.0 / self.cfg.thrust_to_weight - 1.0)
        self._nominal_action = torch.tensor([self._hover_thrust, 0.0, 0.0, 0.0], device=self.device).tile((self.num_envs, 1))
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)
        self._action_queue = torch.tensor([self._hover_thrust, 0.0, 0.0, 0.0], device=self.device).tile((self.cfg.control_latency_steps+1, self.num_envs, 1)) # for control latency
        self._control_latency_steps = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.cfg.control_latency_steps

        # --- NEW: Time-to-catch and success tracking buffers ---
        self.time_to_catch = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.is_success = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.has_evaluated_catch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._catch_curriculum_step = 0
        self._ttc_curriculum_step = 0

        self._action_history = torch.zeros(self.num_envs, self.cfg.action_history_length, self.cfg.action_space, device=self.device)
        self._state_history = torch.zeros(self.num_envs, self.cfg.state_history_length, 3, device=self.device)

        # Parameters for potential Domain Randomization
        self._thrust_to_weight = self.cfg.thrust_to_weight * torch.ones(self.num_envs, device=self.device)
        self._tau_m = self.cfg.tau_m * torch.ones(self.num_envs, device=self.device)
        self._arm_length = self.cfg.arm_length * torch.ones(self.num_envs, device=self.device)
        self._k_m = self.cfg.k_m * torch.ones(self.num_envs, device=self.device)
        self._k_eta = self.cfg.k_eta * torch.ones(self.num_envs, device=self.device)
        self._kp_att = self.cfg.kp_att * torch.ones(self.num_envs, device=self.device)
        self._kd_att = self.cfg.kd_att * torch.ones(self.num_envs, device=self.device)



        # Trajectory initialization
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_pos_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._desired_ori_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 4, device=self.device)
        self._pos_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._yaw_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, device=self.device)
        self._pos_shift = torch.zeros(self.num_envs, 3, device=self.device)
        self._yaw_shift = torch.zeros(self.num_envs, 1, device=self.device)
        # self.amplitudes = torch.zeros(self.num_envs, 4, device=self.device)
        # self.frequencies = torch.zeros(self.num_envs, 4, device=self.device)
        # self.phases = torch.zeros(self.num_envs, 4, device=self.device)
        # self.offsets = torch.zeros(self.num_envs, 4, device=self.device)
        self.lissajous_amplitudes = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_amplitudes_rand_ranges = torch.tensor(self.cfg.lissajous_amplitudes_rand_ranges, device=self.device).float()
        self.lissajous_frequencies = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_frequencies_rand_ranges = torch.tensor(self.cfg.lissajous_frequencies_rand_ranges, device=self.device).float()
        self.lissajous_phases = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_phases_rand_ranges = torch.tensor(self.cfg.lissajous_phases_rand_ranges, device=self.device).float()
        self.lissajous_offsets = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_offsets_rand_ranges = torch.tensor(self.cfg.lissajous_offsets_rand_ranges, device=self.device).float()

        self._time = torch.zeros(self.num_envs, 1, device=self.device)
        self.pos_radius_start = self.cfg.pos_radius

        # Things necessary for motor dynamics
        r2o2 = math.sqrt(2.0) / 2.0
        self._rotor_positions = torch.cat(
            [
                (self._arm_length.unsqueeze(1) * torch.tensor([r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length.unsqueeze(1) * torch.tensor([r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length.unsqueeze(1) * torch.tensor([-r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length.unsqueeze(1) * torch.tensor([-r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
            ],
            dim=1, 
        ).to(self.device)
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device).tile(self.num_envs, 1)
        # if self.cfg.k_torque > 0.0:
        #     self.k = self.cfg.k_torque * torch.ones(self.num_envs, device=self.device)
        # else:
        #     self.k = self._k_m / self._k_eta
        if getattr(self.cfg, "k_torque", 0.0) > 0.0:
            self._k_torque = self.cfg.k_torque * torch.ones(self.num_envs, device=self.device)
        else:
            self._k_torque = self._k_m / self._k_eta

        # test_cross =  torch.linalg.cross(self._rotor_positions, torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(self.num_envs, 1, 1))[:,:, 0:2].transpose(-2,-1)
        # print(test_cross.shape)

        self.f_to_TM = torch.cat(
            [
                torch.ones(self.num_envs, 1, 4, device=self.device),
                # torch.cat(
                #     [
                #         torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                #     ], 
                #     dim=1
                # ).to(self.device),
                torch.linalg.cross(self._rotor_positions, torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(self.num_envs, 1, 1))[:,:, 0:2].transpose(-2,-1),
                self._k_torque.view(self.num_envs, 1, 1) * self._rotor_directions.view(self.num_envs, 1, 4),
            ],
            dim=1
        )

        self.TM_to_f = torch.linalg.inv(self.f_to_TM)


        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "pos_distance",
                "pos_corse",
                "pos_fine",
                "pos_error",
                "ori_error",
                "previous_thrust",
                "previous_attitude",
                "previous_action",
                "action_norm",
                "crash_penalty",
                "stay_alive",
                "catch_miss_penalty",
                "catch_success_bonus",
                "early_arrival",
                "yaw_catch_penalty",
                'yaw_continuous'
            ]
        }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0]
            default_masses = self._robot.root_physx_view.get_masses()[0]
            # default_masses[self._ee_id] = 1e-9 # 0 gram end effector
            default_masses[self._ee_id] = self.cfg.end_effector_mass # 5 gram end effector
            new_masses = default_masses.tile((self.num_envs, 1))
            self._robot.root_physx_view.set_masses(new_masses, torch.arange(self.num_envs))

            ee_pos = self._robot.data.body_pos_w[0, self._ee_id]
            ee_ori = self._robot.data.body_quat_w[0, self._ee_id]
            quad_pos = self._robot.data.body_pos_w[0, self._body_id]
            quad_ori = self._robot.data.body_quat_w[0, self._body_id]
            self.body_pos_ee_frame, self.body_ori_ee_frame = subtract_frame_transforms(ee_pos, ee_ori, quad_pos, quad_ori)
            # self.body_pos_ee_frame = ee_pos - quad_pos
            
            # print("ee_pos: ", ee_pos)
            # print("quad_pos: ", quad_pos)
            # print("Body pos in EE frame: ", self.body_pos_ee_frame)
            # print("Body ori in EE frame: ", self.body_ori_ee_frame)
            # import code; code.interact(local=locals())
            self.body_pos_ee_frame = self.body_pos_ee_frame.tile((self.num_envs, 1))
            self.com_pos_e = self.body_pos_ee_frame.clone().to(self.device)  # Center of mass in end effector frame

        ## INTRODUCED FIXES TO GET THE SRT HOVER EXAMPLE TO WORK
        else:
            self._ee_id = self._robot.find_bodies("body")[0]
            quad_pos = self._robot.data.body_pos_w[0, self._body_id]
            quad_ori = self._robot.data.body_quat_w[0, self._body_id]
            self.body_pos_ee_frame, self.body_ori_ee_frame = subtract_frame_transforms(quad_pos, quad_ori, quad_pos, quad_ori)
            self.body_pos_ee_frame = self.body_pos_ee_frame.tile((self.num_envs, 1))
            self.com_pos_e = torch.zeros(3, device=self.device).tile((self.num_envs, 1))

        if "mass" in self.cfg.to_dict().keys():
            self._robot_mass = self.cfg.mass * torch.ones(self.num_envs, device=self.device)
        else:
            self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum() * torch.ones(self.num_envs, device=self.device)
        self._default_masses = self._robot.root_physx_view.get_masses()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))
        self._robot_weight = (self._robot_mass * self._gravity_magnitude)

        self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)


        if "Ixx" in self.cfg.to_dict().keys():
            self.inertia_tensor = torch.diag(
                torch.tensor(
                    [
                        self.cfg.Ixx,
                        self.cfg.Iyy,
                        self.cfg.Izz,
                    ],
                    device=self.device,
                )
            ).unsqueeze(0).tile(self.num_envs, 1, 1)
            self.default_inertia = self.inertia_tensor[0].clone().to(self.device)
        else:
            self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)
            self.default_inertia = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].to(self.device)
        self._robot_inertia = self.inertia_tensor.clone().to(self.device)

        # Visualization setup
        if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame":
            self._frame_positions = torch.zeros(self.num_envs, 2, 3, device=self.device)
            self._frame_orientations = torch.zeros(self.num_envs, 2, 4, device=self.device)
        elif self.cfg.viz_mode == "robot":
            self._robot_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self._robot_orientations = torch.zeros(self.num_envs, 4, device=self.device)
            self._robot_pos_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 3, device=self.device)
            self._robot_ori_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 4, device=self.device)
            self._goal_pos_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 3, device=self.device)
            self._goal_ori_history = torch.zeros(self.num_envs, self.cfg.viz_history_length, 4, device=self.device)
        elif self.cfg.viz_mode == "viz":
            self._robot_positions = torch.zeros(self.num_envs, 3, device=self.device)
            self._robot_orientations = torch.zeros(self.num_envs, 4, device=self.device)
        else:
            raise ValueError("Visualization mode not recognized: ", self.cfg.viz_mode)

        self.local_num_envs = self.num_envs
        self.set_debug_vis(self.cfg.debug_vis)


        self.vehicle_mass = self._robot_mass
        self.arm_mass = 0.0
        self.quad_inertia =  self.inertia_tensor[0]
        self.arm_offset = torch.zeros(3, device=self.device)
        self.position_offset = torch.zeros(3, device=self.device)
        self.orientation_offset = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # print("Crazyflie mass: ", self._robot_mass)
        # print("Crazyflie inertia: ", self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).squeeze())
        # print("TM_to_f: \n", self.TM_to_f)
        # print("f_to_TM: \n", self.f_to_TM)
        # import code; code.interact(local=locals())


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_motor_speeds(self, wrench_des):
        # f_des = torch.matmul(self.TM_to_f, wrench_des.t()).t()
        f_des = torch.bmm(self.TM_to_f, wrench_des.unsqueeze(2)).squeeze(2) # (n_envs, 4)
        # print("Desired force: ", f_des.shape)
        motor_speed_squared = f_des / self._k_eta.unsqueeze(1) # (n_envs, 4)
        # print("Desired motor speed squared: ", motor_speed_squared.shape)
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        # print("Desired motor speed: ", motor_speeds_des.shape)
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
        # print("Clamped desired motor speed: ", motor_speeds_des.shape)
        return motor_speeds_des
    
    def _get_moment_from_ctatt(self, actions):
        ori_matrix = matrix_from_quat(self._robot.data.root_quat_w)
        # old version with euler angles
        # euler_des = actions[:, 1:] * self.cfg.attitude_scale
        # ori_des_matrix = matrix_from_euler(euler_des, "XYZ")
        # print("Env R des: ", ori_des_matrix)

        # Exp Hat Map
        # ori_des_matrix = exp_so3(hat_map(actions[:, 1:] * self.cfg.attitude_scale))
        
        # Flatness based control
        shape_des = flatness_utils.s2_projection(actions[:, 1]* self.cfg.attitude_scale_xy, actions[:, 2]* self.cfg.attitude_scale_xy)
        psi_des = actions[:,3] * self.cfg.attitude_scale_z
        ori_des_matrix = flatness_utils.getRotationFromShape(shape_des, psi_des)

        euler_des = actions[:, 1:] * self.cfg.attitude_scale
        ori_des_matrix = isaac_math_utils.matrix_from_euler(euler_des, "XYZ") # (n_envs, 3, 3) 




        S_err = 0.5 * (torch.bmm(ori_des_matrix.transpose(-2, -1), ori_matrix) - torch.bmm(ori_matrix.transpose(-2, -1), ori_des_matrix)) # (n_envs, 3, 3)
        att_err = vee_map(S_err) # (n_envs, 3)
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        # omega_des[:, 2] = self._actions[3] * self.cfg.moment_scale
        omega_err = self._robot.data.root_ang_vel_b - omega_des # (n_envs, 3)

        # att_pd = -self.cfg.kp_att * att_err - self.cfg.kd_att * omega_err


        att_pd = torch.bmm(att_err.unsqueeze(2), -self._kp_att.reshape(-1, 1, 1)).squeeze(2) - torch.bmm(omega_err.unsqueeze(2), self._kd_att.reshape(-1, 1, 1)).squeeze(2) # (n_envs, 3)
        I_omega = torch.bmm(self.inertia_tensor, self._robot.data.root_ang_vel_b.unsqueeze(2)).squeeze(2).to(self.device)
        cmd_moment = torch.bmm(self.inertia_tensor, att_pd.unsqueeze(2)).squeeze(2) + \
                    torch.cross(self._robot.data.root_ang_vel_b, I_omega, dim=1) 
        return cmd_moment
    
    def _get_moment_from_ctbr(self, actions):
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, :2] = self.cfg.body_rate_scale_xy * actions[:, 1:3]
        omega_des[:, 2] = self.cfg.body_rate_scale_z * actions[:, 3]
        
        omega_err = self._robot.data.root_ang_vel_b - omega_des
        omega_dot_err = (omega_err - self._previous_omega_err) / self.cfg.pd_loop_rate_hz
        omega_dot = -self.cfg.kp_omega * omega_err - self.cfg.kd_omega * omega_dot_err
        self._previous_omega_err = omega_err

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    def _pre_physics_step(self, actions: torch.Tensor):

        # Decrement the timer by the policy step time, bounding it at zero
        self.time_to_catch = torch.clamp(self.time_to_catch - self.step_dt, min=0.0)
        # self._actions = actions.clone().clamp(-1.0, 1.0)
        self._action_queue = torch.roll(self._action_queue, shifts=-1, dims=0)
        self._action_queue[-1] = actions.clone().clamp(-1.0, 1.0)

        delay_idx = torch.clamp(
            self._control_latency_steps,
            min=0,
            max=self._action_queue.shape[0] - 1,
        )
        env_arange = torch.arange(self.num_envs, device=self.device)
        self._actions = self._action_queue[delay_idx, env_arange]

        self._action_history = torch.roll(self._action_history, shifts=1, dims=1) # roll the action history to make room for the new action
        self._action_history[:, 0] = self._actions.clone().clamp(-1.0, 1.0) # add the new action to the history
        self.pd_loop_counter = 0



        # 0th action is collective thrust
        # self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self._thrust_to_weight)
        self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (4.0 * self.max_thrust - 4.0 * self.min_thrust) + self.min_thrust # scale thrust to the range [min_thrust, max_thrust]

        if self.cfg.control_mode == "CTBM":
            self._wrench_des[:, 1:] = self.cfg.moment_scale * self._actions[:, 1:]
        elif self.cfg.control_mode == "CTATT":
            # 1st and 2nd action are desired attitude for pitch and roll
            # 3rd action is desired yaw rate
            # compute wrench from desired attitude and current attitude using PD controller
            self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
        elif self.cfg.control_mode == "CTBR":
            # 1st and 2nd action are desired body rates for pitch and roll
            # 3rd action is desired yaw rate            
            # compute wrench from desired body rates and current body rates using PD controller
            self._wrench_des[:,1:] = self._get_moment_from_ctbr(self._actions)
        elif self.cfg.control_mode == "SRT":
            # The SRT control mode uses the actions directly as motor speeds
            # This is a simplified control mode for testing purposes
            self._motor_speeds_des = ((self._actions.clone() + 1.0) / 2.0) # rescale from [-1, 1] to [0, 1]
            return

            
        else:
            raise NotImplementedError(f"Control mode {self.cfg.control_mode} is not implemented.")

        self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        # print("Desired Motor Speeds: ", self._motor_speeds_des[0])
        # print("Current Motor Speeds: ", self._motor_speeds[0])

        # import code; code.interact(local=locals())

        # print("\n--- CONTROL LATENCY DEBUG Pre Physics Step ---")
        # print(f"Policy action (new):      {self._action_queue[-1, 0]}")
        # print(f"Delayed action (applied): {self._actions[0]}")
        # print(f"Queue state:\n{self._action_queue[:, 0]}")
        # print(f"Wrench desired:           {self._wrench_des[0]}")
        # print(f"Motor speeds desired:     {self._motor_speeds_des[0]}")
        # print(f"Actual motor speeds:      {self._motor_speeds[0]}")
        # print("--------------------------\n")

    def _apply_action(self):
        # Skip low-level motor dynamics
        if self.cfg.skip_motor_dynamics:
            self._thrust[:, 0, 2] = self._wrench_des[:, 0]
            self._moment[:, 0, :] = self._wrench_des[:, 1:]
            self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
            return

        # Update PD loop at the appropriate rate (100Hz or whatever pd_loop_rate_hz is)
        if self.pd_loop_counter % self.cfg.pd_loop_decimation == 0 and self.cfg.control_mode != "SRT":
            # Recompute wrench using CURRENT state but DELAYED actions
            if self.cfg.control_mode == "CTATT":
                self._wrench_des[:,1:] = self._get_moment_from_ctatt(self._actions)
            elif self.cfg.control_mode == "CTBR":
                self._wrench_des[:,1:] = self._get_moment_from_ctbr(self._actions)
                
            # Recompute motor speeds based on fresh wrench calculation
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)

        # print("\n--- CONTROL LATENCY DEBUG Apply Action ---")
        # print(f"Policy action (new):      {self._action_queue[-1, 0]}")
        # print(f"Delayed action (applied): {self._actions[0]}")
        # print(f"Queue state:\n{self._action_queue[:, 0]}")
        # print(f"Wrench desired:           {self._wrench_des[0]}")
        # print(f"Motor speeds desired:     {self._motor_speeds_des[0]}")
        # print(f"Actual motor speeds:      {self._motor_speeds[0]}")
        # print("--------------------------\n")

        self.pd_loop_counter += 1


        # Old Motor Dynamics
        # if self.cfg.control_mode != "SRT":       
        #     motor_accel = torch.bmm((1.0/self._tau_m).reshape(self.num_envs, 1, 1), (self._motor_speeds_des - self._motor_speeds).unsqueeze(1)).squeeze(1) # (n_envs, 4)
        #     self._motor_speeds += motor_accel * self.physics_dt
        #     self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        # else:
        #     # New Motor Dynamics --> Assumes motor speeds are between 0 and 1 (used for SRT control mode)
        #     alpha = torch.exp(-self.physics_dt / self._tau_m).unsqueeze(-1) # Exponential decay factor
        #     self._motor_speeds = alpha * self._motor_speeds + (1 - alpha) * self._motor_speeds_des # Update motor speeds with exponential decay
        
        alpha = torch.exp(-self.physics_dt / self._tau_m).unsqueeze(-1) # Exponential decay factor
        self._motor_speeds = alpha * self._motor_speeds + (1 - alpha) * self._motor_speeds_des # Update motor speeds with exponential decay

        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation

        motor_forces = self._k_eta.unsqueeze(1) * self._motor_speeds ** 2
   
        # wrench = torch.matmul(self.f_to_TM, motor_forces.t()).t()
        wrench = torch.bmm(self.f_to_TM, motor_forces.unsqueeze(2)).squeeze(2) # (n_envs, 4)
        # print("wrench: ", wrench.shape)
        
        # print("Motor acceleration: ", motor_accel[0])
        # print("Current motor speed (post update): ", self._motor_speeds[0])
        # print("Wrench resconstruction error: ", torch.norm(wrench[0] - self._wrench_des[0]))
        # print("Output wrench: ", wrench[0])
        self._thrust[:, 0, 2] = wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _apply_curriculum(self, total_timesteps):
        """
        Apply the curriculum to the environment.
        """
        # print("[Isaac Env: Curriculum] Total Timesteps: ", total_timesteps, " Pos Radius: ", self.cfg.pos_radius)
        if self.cfg.pos_radius_curriculum > 0:
            # half the pos radius every pos_radius_curriculum timesteps
            # self.cfg.pos_radius = 0.8 * (0.25 ** (total_timesteps // self.cfg.pos_radius_curriculum))
            self.cfg.pos_radius = self.pos_radius_start * (0.5 ** (total_timesteps // self.cfg.pos_radius_curriculum))
            
    def _update_catch_tolerance_curriculum(self) -> None:
        """Tighten or loosen catch_tolerance based on rolling success rate."""
        self._catch_curriculum_step += 1
        if self._catch_curriculum_step % self.cfg.catch_curriculum_update_interval != 0:
            return

        success_rate = self.is_success.mean().item()

        if success_rate >= self.cfg.catch_curriculum_tighten_threshold:
            new_tol = max(
                self.cfg.catch_tolerance_min,
                self.cfg.catch_tolerance - self.cfg.catch_curriculum_tighten_rate
            )
            if new_tol != self.cfg.catch_tolerance:
                self.get_logger().info(
                    f"[CatchCurriculum] Tightening tolerance: "
                    f"{self.cfg.catch_tolerance:.4f} -> {new_tol:.4f} "
                    f"(success_rate={success_rate:.3f})"
                ) if hasattr(self, 'get_logger') else None
            self.cfg.catch_tolerance = new_tol

        elif success_rate < self.cfg.catch_curriculum_loosen_threshold:
            new_tol = min(
                self.cfg.catch_tolerance_max,
                self.cfg.catch_tolerance + self.cfg.catch_curriculum_loosen_rate
            )
            self.cfg.catch_tolerance = new_tol
    def _update_time_to_catch_curriculum(self) -> None:
        """Tighten or loosen time_to_catch_max based on rolling success rate."""
        self._ttc_curriculum_step += 1
        if self._ttc_curriculum_step % self.cfg.time_to_catch_update_interval != 0:
            return

        success_rate = self.is_success.mean().item()

        if success_rate >= self.cfg.time_to_catch_tighten_threshold:
            new_ttc = max(
                self.cfg.time_to_catch_min,
                self.cfg.time_to_catch_max - self.cfg.time_to_catch_tighten_rate
            )
            if new_ttc != self.cfg.time_to_catch_max:
                self.get_logger().info(
                    f"[TTCCurriculum] Tightening time_to_catch: "
                    f"{self.cfg.time_to_catch_max:.3f}s -> {new_ttc:.3f}s "
                    f"(success_rate={success_rate:.3f})"
                ) if hasattr(self, 'get_logger') else None
            self.cfg.time_to_catch_max = new_ttc


        elif success_rate < self.cfg.time_to_catch_loosen_threshold:
            new_ttc = min(
                1.5,
                self.cfg.time_to_catch_max + self.cfg.time_to_catch_tighten_rate
            )
            if new_ttc != self.cfg.time_to_catch_max:
                self.get_logger().info(
                    f"[TTCCurriculum] Loosening time_to_catch: "
                    f"{self.cfg.time_to_catch_max:.3f}s -> {new_ttc:.3f}s "
                    f"(success_rate={success_rate:.3f})"
                ) if hasattr(self, 'get_logger') else None
            self.cfg.time_to_catch_max = new_ttc

    def update_goal_state(self):
        # env_ids = (self.episode_length_buf % int(self.cfg.traj_update_dt*self.cfg.policy_rate_hz)== 0).nonzero(as_tuple=False)
        env_ids = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        
        if len(env_ids) == 0 or env_ids.size(0) == 0:
            return
        

        # current_time = self.episode_length_buf[env_ids]
        current_time = self.episode_length_buf.view(self.num_envs, 1)
        future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        # future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        time = (current_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt
        time = time.view(self.num_envs, -1)
        # env_ids =  # need to squeeze after getting current time

        # Update the desired position and orientation based on the trajectory
        # Traj Util functions return a position and a yaw trajectory as tensors of the following shape:
        # pos: Tensor containing the evaluated curves and their derivatives.
        #      Shape: (num_derivatives + 1, n_envs, 3, n_samples).
        # yaw: Tensor containing the yaw angles of the curves.
        #      Shape: (num_derivatives + 1, n_envs, n_samples).
        if self.cfg.trajectory_type == "lissaajous":
            # print("Time: ", time.shape)
            # print("Amp: ", self.lissajous_amplitudes.shape)
            # print("Freq: ", self.lissajous_frequencies.shape)
            # print("Phase: ", self.lissajous_phases.shape)
            # print("Offset: ", self.lissajous_offsets.shape)
            pos_traj, yaw_traj = traj_utils.eval_lissajous_curve(time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4)
        elif self.cfg.trajectory_type == "polynomial":
            pos_traj, yaw_traj = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
        elif self.cfg.trajectory_type == "combined":
            pos_lissajous, yaw_lissajous = traj_utils.eval_lissajous_curve(time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4)
            pos_poly, yaw_poly = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
            pos_traj = pos_lissajous + pos_poly
            yaw_traj = yaw_lissajous + yaw_poly

            # print("Poly coefficients: ", self.polynomial_coefficients[0, :, :])
            # print("Pos Poly: ", pos_poly[:2, 0, :, 0])
            # print("Yaw poly: ", yaw_poly[:2, 0, 0])
        elif self.cfg.trajectory_type == "random_walk":
            pass
        else:
            raise NotImplementedError("Trajectory type not implemented")
    
        self._pos_traj = pos_traj
        self._yaw_traj = yaw_traj
        
        if self.cfg.random_shift_trajectory:
            # Ensure the shapes are compatible for broadcasting
            pos_shift = self._pos_shift.unsqueeze(-1)
            yaw_shift = self._yaw_shift

            pos_traj[0, :, :, :] += pos_shift
            yaw_traj[0, :, :] += yaw_shift

        # we need to switch the last two dimensions of pos_traj since the _desired_pos_w is of shape (num_envs, horizon, 3) instead of (num_envs, 3, horizon)
        # print(self._desired_pos_traj_w.shape, pos_traj[0,env_ids.squeeze(1)].shape)
        self._desired_pos_traj_w[env_ids.squeeze(1)] = (pos_traj[0,env_ids.squeeze(1)]).transpose(1,2)
        # self._desired_pos_traj_w[env_ids.squeeze(1),:, :2] += self._terrain.env_origins[env_ids, :2] # shift the trajectory to the correct position for each environment
        # we need to convert from the yaw angle to a quaternion representation
        # print("Yaw Traj: ", yaw_traj[0, 0, :2])
        self._desired_ori_traj_w[env_ids.squeeze(1)] = quat_from_yaw(yaw_traj[0,env_ids.squeeze(1)])
        # print("desired ori traj: ", self._desired_ori_traj_w[0,:2])

        # print("pos traj: ", pos_traj[0, 0, :, :2])
        # print("desired pos traj: ", self._desired_pos_traj_w[0,:2])

        # print("Traj shape: ", self._pos_traj.shape)
        # print("Traj velocity: ", self._pos_traj[1, 0, :, 0])
        # print("Traj acceleration: ", self._pos_traj[2, 0, :, 0])
        # print("Traj yaw: ", self._yaw_traj[0, 0, 0])
        # print("Traj yaw velocity: ", self._yaw_traj[1, 0, 0])


        self._desired_pos_w[env_ids] = self._desired_pos_traj_w[env_ids, 0]
        self._desired_ori_w[env_ids] = self._desired_ori_traj_w[env_ids, 0]
        # print("0th env: ", self._desired_pos_w[0], self._desired_ori_w[0])
        # print("[Isaac Env: Update Goal State] Desired Pos: ", self._desired_pos_w[env_ids[:5,0]])

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """
        self._apply_curriculum(self.common_step_counter * self.num_envs)
        self.update_goal_state()
        
        
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.goal_body)

        # ====================================================================
        # ADDED: Sim-to-Real Tracker Convergence Noise (15cm -> 0cm)
        # ====================================================================
        if not self.cfg.eval_mode:
            # 1. Lazily initialize a persistent drift vector so the noise is smooth, not jerky
            if not hasattr(self, '_tracker_drift'):
                self._tracker_drift = torch.zeros_like(goal_pos_w)
                
            # 2. Smooth random walk (Exponential Moving Average)
            # This causes the "predicted point" to wander smoothly over time
            self._tracker_drift = 0.95 * self._tracker_drift + 0.05 * torch.randn_like(goal_pos_w)
            
            # 3. Distance-based decay envelope WITH 10cm DEADZONE
            # If distance < 10cm, envelope is exactly 0.0. Ramps up to full 15cm at 1.5m.
            dist_to_goal = torch.linalg.norm(goal_pos_w - base_pos_w, dim=1, keepdim=True)
            noise_envelope = torch.clamp((dist_to_goal - 0.10) / 1.5, 0.0, 1.0) * 0.15
            
            # 4. Apply the noise
            drift_norm = torch.linalg.norm(self._tracker_drift, dim=1, keepdim=True) + 1e-6
            actual_noise = (self._tracker_drift / drift_norm) * noise_envelope
            
            goal_pos_w = goal_pos_w + actual_noise
        # ====================================================================


        # Find the error of the end-effector to the desired position and orientation
        # The root state of the robot is the end-effector frame in this case
        # Batched over number of environments, returns (num_envs, 3) and (num_envs, 4) tensors
        # pos_error_b, ori_error_b = subtract_frame_transforms(self._desired_pos_w, self._desired_ori_w, 
        #                                                      base_pos, base_ori)
        pos_error_b, ori_error_b = subtract_frame_transforms(
            base_pos_w, base_ori_w, 
            # self._desired_pos_w, self._desired_ori_w
            goal_pos_w, goal_ori_w
        )

        future_pos_error_b = []
        future_ori_error_b = []
        for i in range(self.cfg.trajectory_horizon):
            goal_pos_traj_w, goal_ori_traj_w = self.convert_ee_goal_from_task(self._desired_pos_traj_w[:, i+1].squeeze(1), self._desired_ori_traj_w[:, i+1].squeeze(1), self.cfg.goal_body)

            waypoint_pos_error_b, waypoint_ori_error_b = subtract_frame_transforms(base_pos_w, base_ori_w, goal_pos_traj_w, goal_ori_traj_w)
            future_pos_error_b.append(waypoint_pos_error_b) # append (n, 3) tensor
            future_ori_error_b.append(waypoint_ori_error_b) # append (n, 4) tensor
        if len(future_pos_error_b) > 0:
            future_pos_error_b = torch.stack(future_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor
            future_ori_error_b = torch.stack(future_ori_error_b, dim=1) # stack to (n, horizon, 4) tensor
 
            if self.cfg.use_yaw_representation_for_trajectory:
                future_ori_error_b = math_utils.yaw_from_quat(future_ori_error_b).reshape(self.num_envs, self.cfg.trajectory_horizon, 1)
        else:
            future_pos_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
            future_ori_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 4, device=self.device)

        

        # Compute the orientation error as a yaw error in the body frame
        # goal_yaw_w = yaw_quat(self._desired_ori_w)
        goal_yaw_w = isaac_math_utils.yaw_quat(goal_ori_w)
        current_yaw_w = isaac_math_utils.yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        yaw_error_w = yaw_error_from_quats(current_yaw_w, goal_yaw_w, dof=self.cfg.num_joints).view(self.num_envs, 1)
        
        if self.cfg.use_yaw_representation:
            yaw_representation = yaw_error_w
        else:
            yaw_representation = torch.zeros(self.num_envs, 0, device=self.device)
        

        if self.cfg.use_full_ori_matrix:
            ori_representation_b = matrix_from_quat(ori_error_b).flatten(-2, -1)
        else:
            ori_representation_b = torch.zeros(self.num_envs, 0, device=self.device)

        if self.cfg.use_grav_vector:
            grav_vector_b = quat_apply_inverse(base_ori_w, self._grav_vector_unit) # projected gravity vector in the cfg frame
        else:
            grav_vector_b = torch.zeros(self.num_envs, 0, device=self.device)


        if self.cfg.action_history_length > 1:
            action_history = self._action_history[:,1:].view(self.num_envs, -1) # flatten the action history. Most recent action is not included since it is already part of the observation
        else:
            action_history = torch.zeros(self.num_envs, 0, device=self.device)
        
        # Compute the linear and angular velocities of the end-effector in body frame
        # if self.cfg.trajectory_horizon > 0:
        #     lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        # else:
        #     lin_vel_error_w = torch.zeros_like(lin_vel_w, device=self.device) - lin_vel_w
        # lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_error_w)
        # if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0:
        #     ang_vel_des = torch.zeros_like(ang_vel_w)
        #     ang_vel_des[:,2] = self._yaw_traj[1, :, 0]
        #     ang_vel_error_w = ang_vel_des - ang_vel_w
        # else:
        #     ang_vel_error_w = torch.zeros_like(ang_vel_w) - ang_vel_w
        # ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_error_w)
        
        
        # Old computation for lin and ang vel:
        lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_error_w)
        ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_w)


        # Compute the joint states
        shoulder_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        shoulder_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        if self.cfg.num_joints > 0:
            shoulder_joint_pos = self._robot.data.joint_pos[:, self._shoulder_joint_idx].unsqueeze(1)
            shoulder_joint_vel = self._robot.data.joint_vel[:, self._shoulder_joint_idx].unsqueeze(1)
        if self.cfg.num_joints > 1:
            wrist_joint_pos = self._robot.data.joint_pos[:, self._wrist_joint_idx].unsqueeze(1)
            wrist_joint_vel = self._robot.data.joint_vel[:, self._wrist_joint_idx].unsqueeze(1)

        # Previous Action
        if self.cfg.use_previous_actions:
            previous_actions = self._previous_action
        else:
            previous_actions = torch.zeros(self.num_envs, 0, device=self.device)

        # # Legacy Obs
        # pos_error_b, ori_error_b = subtract_frame_transforms(
        #     base_pos_w, base_ori_w, goal_pos_w, goal_ori_w
        # )

        # if self.cfg.use_full_ori_matrix:
        #     ori_error_b = matrix_from_quat(ori_error_b).view(-1, 9)

        # yaw_error = yaw_error_from_quats(self._robot.data.root_quat_w, goal_ori_w, 0)

        # lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_w)
        # ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_w)
        # grav_vector_b = quat_apply_inverse(base_ori_w, self._grav_vector_unit)
        # obs = torch.cat(
        #     [
        #         lin_vel_b, # 3
        #         ang_vel_b, # 3
        #         grav_vector_b, # 3
        #         pos_error_b, # 3
        #         ori_error_b, # 4 or 9 if use_full_ori_matrix
        #         yaw_error.unsqueeze(-1), # 1
        #         self._previous_action, # 4
        #     ],
        #     dim=-1,
        # )

        if self.cfg.rotorpy_obs:
            future_pos_error_b = []
            for i in range(self.cfg.trajectory_horizon):
                goal_pos_traj_w = self._desired_pos_traj_w[:, i+1]
                waypoint_pos_error_b = goal_pos_traj_w - base_pos_w  # (num_envs, 3)
                future_pos_error_b.append(waypoint_pos_error_b) # append (n, 3) tensor

            if len(future_pos_error_b) > 0:
                future_pos_error_b = torch.stack(future_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor

            current_state = torch.cat( [
                base_pos_w - goal_pos_w,                                        # (num_envs, 3)
                lin_vel_w - self._pos_traj[1, :, :, 0].view(self.num_envs, 3),  # (num_envs, 3)
                base_ori_w,                                                     # (num_envs, 4) 
                isaac_math_utils.quat_apply_inverse(base_ori_w, ang_vel_w),    # (num_envs, 3) 
            ], dim=-1)

            if self.cfg.state_history_length > 0:
                # import code; code.interact(local=locals())
                previous_states = (self._state_history - base_pos_w.unsqueeze(2)).view(self.num_envs, -1)   # (num_envs, state_history_length * 13)
            else:
                previous_states = torch.zeros(self.num_envs, 0, device=self.device)

            obs = torch.cat(
                [
                    current_state,                                  # (num_envs, 13)
                    previous_states,                                # (num_envs, state_history_length * 13)
                    self._action_history.view(self.num_envs, -1),   # (num_envs, 4 * action_history_length)
                    future_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                ],
                dim=-1                                          # (num_envs, 25 if action_history_length=3)
            )

            if self.cfg.state_history_length > 0:
                # Update the state history
                self._state_history = torch.roll(self._state_history, shifts=1, dims=1)
                self._state_history[:, 0] = base_pos_w  # Update the last state in the history
        else:
            obs = torch.cat(
                [
                    pos_error_b,                                # (num_envs, 3)
                    ori_representation_b,                       # (num_envs, 0) if not using full ori matrix, (num_envs, 9) if using full ori matrix
                    yaw_representation,                         # (num_envs, 4) if using yaw representation (quat), 0 otherwise
                    grav_vector_b,                              # (num_envs, 3) if using gravity vector, 0 otherwise
                    lin_vel_b,                                  # (num_envs, 3)
                    ang_vel_b,                                  # (num_envs, 3)
                    previous_actions,                           # (num_envs, 4)
                    action_history,                             # (num_envs, 4 * action_history_length) if action_history_length > 1, else 0
                    future_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                    future_ori_error_b.flatten(-2, -1),         # (num_envs, horizon * 4) if use_yaw_representation_for_trajectory, else (num_envs, horizon, 1)
                    self.time_to_catch.unsqueeze(-1)            # [N, 1] <--- NEW
                ],
                dim=-1                                          # (num_envs, 22 + 7*horizon)
            )
        # import code; code.interact(local=locals())


        
        
        # We also need the state information for other controllers like the decoupled controller.
        # This is the full state of the robot
        # print("[Isaac Env: Observations] \"Frame\" Pos: ", base_pos_w)
        # quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("vehicle")
        # quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("COM")
        quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("body")

        if self.cfg.has_end_effector:
            ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_frame_state_from_task(self.cfg.task_body)
        else:
            ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_frame_state_from_task("body")
        # print("[Isaac Env: Observations] Quad pos: ", quad_pos_w)
        # print("[Isaac Env: Observations] EE pos: ", ee_pos_w)

        if self.cfg.gc_mode:
            future_com_pos_w = []
            future_com_ori_w = []
            for i in range(self.cfg.trajectory_horizon):
                des_com_pos_w, des_com_ori_w = self.convert_ee_goal_to_com_goal(self._desired_pos_traj_w[:, i].squeeze(1), self._desired_ori_traj_w[:, i].squeeze(1))
                future_com_pos_w.append(des_com_pos_w)
                future_com_ori_w.append(des_com_ori_w)

            if len(future_com_pos_w) > 0:
                future_com_pos_w = torch.stack(future_com_pos_w, dim=1)
                future_com_ori_w = torch.stack(future_com_ori_w, dim=1)
            else:
                future_com_pos_w = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
                future_com_ori_w = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 4, device=self.device)

            goal_pos_w, goal_ori_w = self.get_goal_state_from_task("COM")

            # Convert angular velocity to deg/s to match gyro sensor output (it gets converted to body-frame in the controller)
            quad_ang_vel_w = quad_ang_vel_w * (180.0 / math.pi)

            gc_obs = torch.cat(
                [
                    quad_pos_w,                                 # (num_envs, 3) 
                    quad_ori_w,                                 # (num_envs, 4)
                    quad_lin_vel_w,                             # (num_envs, 3)
                    quad_ang_vel_w,                             # (num_envs, 3)
                    goal_pos_w,                                 # (num_envs, 3)
                    yaw_from_quat(goal_ori_w).unsqueeze(1),     # (num_envs, 1)
                    future_com_pos_w.flatten(-2, -1),            # (num_envs, horizon * 3)
                    future_com_ori_w.flatten(-2, -1)            # (num_envs, horizon * 4)
                ],
                dim=-1                                          # (num_envs, 17 + 3*horizon)
            )
        else:
            gc_obs = None

        if self.cfg.eval_mode:
            pos_traj = self._pos_traj[:3,:,:,0].permute(1,0,2).reshape(self.num_envs, -1)
            yaw_traj = self._yaw_traj[:2,:,0].permute(1,0).reshape(self.num_envs, -1)
            full_state = torch.cat(
                [
                    quad_pos_w,                                 # (num_envs, 3) [0-3]
                    quad_ori_w,                                 # (num_envs, 4) [3-7]
                    quad_lin_vel_w,                             # (num_envs, 3) [7-10]
                    quad_ang_vel_w,                             # (num_envs, 3) [10-13]
                    ee_pos_w,                                   # (num_envs, 3) [13-16]
                    ee_ori_w,                                   # (num_envs, 4) [16-20]
                    ee_lin_vel_w,                               # (num_envs, 3) [20-23]
                    ee_ang_vel_w,                               # (num_envs, 3) [23-26]
                    shoulder_joint_pos,                         # (num_envs, 1) [26] 
                    wrist_joint_pos,                            # (num_envs, 1) [27]
                    shoulder_joint_vel,                         # (num_envs, 1) [28]
                    wrist_joint_vel,                            # (num_envs, 1) [29]
                    self._desired_pos_w,                        # (num_envs, 3) [30-33] [26-29]
                    self._desired_ori_w,                        # (num_envs, 4) [33-37] [29-33]
                    pos_traj,
                    yaw_traj,
                    self.time_to_catch.unsqueeze(-1),           # (num_envs, 1) [48] 
                ],
                dim=-1                                          # (num_envs, 18)
            )
            self._state = full_state
        else:
            full_state = None

        if torch.any(torch.isnan(obs)):
            print("[Isaac Env: Observations] NaN in observations")
            # Find where nan is
            nan_indices = torch.isnan(obs).nonzero(as_tuple=True)
            for i in range(len(nan_indices[0])):
                env_id = nan_indices[0][i]
                index = nan_indices[1][i]
                print(f"Env ID: {env_id}, Index: {index}, Value: {obs[env_id, index]}")
            
        return {"policy": obs, "gc": gc_obs, "full_state": full_state}

    def _get_rewards(self) -> torch.Tensor:
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.reward_task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.reward_goal_body)

        pos_error = torch.linalg.norm(goal_pos_w - base_pos_w, dim=1)

        if self.cfg.square_pos_error:
            pos_distance = torch.exp(-(pos_error ** 2) / self.cfg.pos_radius)
            pos_corse_distance = torch.exp(-(pos_error ** 2) / self.cfg.pos_corse_radius)
            pos_fine_distance = torch.exp(-(pos_error ** 2) / self.cfg.pos_fine_radius)
        else:
            pos_distance = torch.exp(-pos_error / self.cfg.pos_radius)
            pos_corse_distance = torch.exp(-pos_error / self.cfg.pos_corse_radius)
            pos_fine_distance = torch.exp(-pos_error / self.cfg.pos_fine_radius)

        ori_error = isaac_math_utils.quat_error_magnitude(goal_ori_w, base_ori_w)

        if self.cfg.trajectory_horizon > 0:
            lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        else:
            lin_vel_error_w = -lin_vel_w
        lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_error_w)

        if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_des[:, 2] = self._yaw_traj[1, :, 0]
            ang_vel_error_w = ang_vel_des - ang_vel_w
        else:
            ang_vel_error_w = -ang_vel_w
        ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_error_w)

        lin_vel_error = torch.norm(lin_vel_b, dim=1)
        ang_vel_error = torch.norm(ang_vel_b, dim=1)

        action_delta = self._actions - self._previous_action
        action_thrust_error = torch.square(action_delta[:, 0])
        action_att_error = torch.sum(torch.square(action_delta[:, 1:]), dim=1)
        previous_action_error = torch.sum(torch.square(action_delta), dim=1)
        action_norm_error = torch.sum(torch.square(self._actions[:, 1:]), dim=1)

        window = min(self.cfg.action_smooth_window, self.cfg.action_history_length)

        if window >= 2:
            hist = self._action_history[:, :window]
            mean_action = torch.mean(hist, dim=1)
            action_mean_error = torch.sum(torch.square(self._actions - mean_action), dim=1)

            if window >= 3:
                hist_deltas = hist[:, :-1] - hist[:, 1:]
                mean_delta = torch.mean(hist_deltas, dim=1)
                action_delta_mean_error = torch.sum(torch.square(action_delta - mean_delta), dim=1)
            else:
                action_delta_mean_error = torch.zeros_like(pos_error)
        else:
            action_mean_error = torch.zeros_like(pos_error)
            action_delta_mean_error = torch.zeros_like(pos_error)

        self._previous_action = self._actions.clone()
        crash_penalty_time = self.cfg.crash_penalty * (self.max_episode_length - self.episode_length_buf)

        near_goal_weight = torch.exp(-(pos_error ** 2) / self.cfg.near_goal_radius)
        inner_hover_weight = torch.exp(-(pos_error ** 2) / self.cfg.inner_hover_radius)

        vel_weight = (
            self.cfg.vel_weight_base
            + self.cfg.vel_weight_near * near_goal_weight
            + self.cfg.vel_weight_inner * inner_hover_weight
        )
        ori_weight = (
            self.cfg.ori_weight_base
            + self.cfg.ori_weight_near * near_goal_weight
            + self.cfg.ori_weight_inner * inner_hover_weight
        )
        smooth_weight = (
            self.cfg.smooth_weight_base
            + self.cfg.smooth_weight_near * near_goal_weight
            + self.cfg.smooth_weight_inner * inner_hover_weight
        )

        hover_lin = torch.exp(-(lin_vel_error ** 2) / self.cfg.hover_lin_denom)
        hover_ang = torch.exp(-(ang_vel_error ** 2) / self.cfg.hover_ang_denom)
        hover_ori = torch.exp(-(ori_error ** 2) / self.cfg.hover_ori_denom)
        at_target_gate = (pos_error < self.cfg.pos_fine_radius * 3.0).float()
        hover_quality = pos_fine_distance * hover_lin * hover_ang * at_target_gate

        if self.cfg.scale_reward_with_time:
            time_scale = self.step_dt
        else:
            time_scale = 1.0

        # time_weight is 0 when time is high, and 1.0 when time runs out
        # Narrowed the window so the drone doesn't hover early
        time_weight = torch.exp(-self.time_to_catch / 0.25)

        # --- UNIFIED CATCH LOGIC ---
        # 1. Define the exact moment the "catch event" is evaluated. 
        # Evaluate it at 0.02s before 0.0, to guarantee it registers in the buffer before the environment resets on timeout.
        catch_event_triggered = (self.time_to_catch <= 0.02)
        
        # 2. ONLY evaluate if we haven't already evaluated this episode
        evaluate_catch_now = torch.logical_and(catch_event_triggered, torch.logical_not(self.has_evaluated_catch))
        
        missed_catch = pos_error > self.cfg.catch_tolerance
        caught_ball = pos_error <= self.cfg.catch_tolerance
        
        # Cap the miss distance at 2.0 meters to prevent gradient explosion
        capped_miss_distance = torch.clamp(pos_error, max=2.0)
        
        # 3. Apply Penalty (Only when evaluate_catch_now is True AND it missed)
        catch_miss_penalty = torch.where(
            torch.logical_and(evaluate_catch_now, missed_catch),
            capped_miss_distance, 
            torch.zeros_like(pos_error)
        )

        # 4. Apply Bonus (Only when evaluate_catch_now is True AND it caught it)
        # Using a massive +50.0 bonus
        catch_success_bonus = torch.where(
            torch.logical_and(evaluate_catch_now, caught_ball),
            torch.ones_like(pos_error),  
            torch.zeros_like(pos_error)
        )
        
        # 5. Yaw mismatch penalty
        current_yaw = yaw_from_quat(base_ori_w)
        goal_yaw = yaw_from_quat(goal_ori_w)
        yaw_err_at_catch = torch.abs(wrap_to_pi(current_yaw - goal_yaw))

        # Penalty: fired only when evaluate_catch_now is True, regardless of success/miss
        yaw_catch_penalty = torch.where(evaluate_catch_now, yaw_err_at_catch, torch.zeros_like(pos_error))
        
        # Continuous yaw alignment — active from step 1, no proximity gate
        yaw_error_continuous = torch.abs(wrap_to_pi(current_yaw - goal_yaw))

        # 6. Track metric using the same logic
        self.is_success = torch.where(
            torch.logical_and(evaluate_catch_now, caught_ball),
            torch.ones_like(self.is_success),
            self.is_success
        )
        
        # 7. Mark these environments as evaluated so they don't get the reward/penalty again
        self.has_evaluated_catch = torch.logical_or(self.has_evaluated_catch, evaluate_catch_now)
        # -------------------------------------

        if self.cfg.rotorpy_reward:
            rewards = {
                "pos_error": torch.linalg.norm(base_pos_w - goal_pos_w, dim=1) * self.cfg.pos_error_reward_scale * time_scale,
                "lin_vel": torch.linalg.norm(lin_vel_w, dim=1) * self.cfg.lin_vel_reward_scale * time_scale,
                "ang_vel": torch.linalg.norm(isaac_math_utils.quat_apply_inverse(base_ori_w, ang_vel_w), dim=1) * self.cfg.ang_vel_reward_scale * time_scale,
                "yaw_error": torch.abs(math_utils.yaw_from_quat(base_ori_w)) * self.cfg.yaw_error_reward_scale * time_scale,
                "previous_action": ((self._actions - torch.mean(self._action_history, dim=1)) ** 2).reshape((-1, 4)) @ torch.tensor(self.cfg.previous_action_reward_scale, device=self.device).reshape((4,)) * time_scale,
                "action_norm": torch.abs(self._actions).view((-1, 4)) @ torch.tensor(self.cfg.action_vec_norm_reward_scale, device=self.device).reshape((4,)) * time_scale,
                "stay_alive": torch.ones_like(pos_error) * self.cfg.stay_alive_reward,
            }
        else:
            rewards = {
                "lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * vel_weight * time_scale,
                "ang_vel": ang_vel_error * self.cfg.ang_vel_reward_scale * vel_weight * time_scale,
                "pos_distance": pos_distance * self.cfg.pos_distance_reward_scale * time_scale,
                "pos_corse": pos_corse_distance * self.cfg.pos_corse_reward_scale * time_scale,
                "pos_fine": (self.cfg.pos_fine_mix * pos_fine_distance + self.cfg.hover_quality_mix * hover_quality) * self.cfg.pos_fine_reward_scale * time_scale,
                "pos_error": pos_error * self.cfg.pos_error_reward_scale * time_scale,
                "ori_error": ori_error * self.cfg.ori_error_reward_scale * ori_weight * time_scale,
                "previous_thrust": action_thrust_error * self.cfg.previous_thrust_reward_scale * smooth_weight * time_scale,
                "previous_attitude": action_att_error * self.cfg.previous_attitude_reward_scale * smooth_weight * time_scale,
                "previous_action": (0.50 * previous_action_error + 0.25 * action_mean_error + 0.25 * action_delta_mean_error) * self.cfg.previous_action_norm_reward_scale * smooth_weight * time_scale,
                "action_norm": action_norm_error * self.cfg.action_norm_reward_scale * (0.25 + self.cfg.action_norm_near_mix * near_goal_weight) * time_scale,
                "crash_penalty": self.reset_terminated.float() * crash_penalty_time * time_scale,
                "stay_alive": (self.cfg.stay_alive_reward + self.cfg.stay_alive_hover_mix * hover_quality) * time_scale,
                # --- NEW: Add the penalty to the dictionary ---
                "catch_miss_penalty": catch_miss_penalty * self.cfg.catch_miss_penalty_scale,
                "catch_success_bonus": catch_success_bonus * self.cfg.catch_success_bonus_scale,
                # ADD this new term to the rewards dict:
                "early_arrival": pos_fine_distance * time_weight * 2.0 * time_scale,
                # High when drone is at target AND timer still has time left — rewards arriving early, not late-dashing
                "yaw_catch_penalty" : yaw_catch_penalty * self.cfg.yaw_catch_penalty_scale,
                "yaw_continuous" : yaw_error_continuous * self.cfg.yaw_continuous_reward_scale * (1 - near_goal_weight) * time_scale,
            }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        self._update_catch_tolerance_curriculum()
        self._update_time_to_catch_curriculum()
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tensors corresponding to termination and truncation. 
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check if end effector or body has collided with the ground
        if self.cfg.has_end_effector:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.0, self._robot.data.body_state_w[:, self._body_id, 2].squeeze() < 0.0)
        else:
            if self.cfg.rotorpy_done:
                died = torch.logical_or(torch.any(self._robot.data.root_vel_w[:,:].abs() > 100.0, dim=1), torch.any(self._robot.data.root_ang_vel_w[:,:].abs() > 100.0, dim=1))


                # Check if the robot has moved too far from the trajectory
                died = torch.logical_or(died, torch.any(torch.abs(self._robot.data.root_pos_w[:, :] - self._pos_traj[0, :, :, 0]) > 4.0, dim=1))

                # died = torch.logical_or(died, torch.abs(self._robot.data.root_pos_w[:, 2] ))

                # import code; code.interact(local=locals())
                # died = torch.logical_or(died, torch.any(torch.abs(self._robot.data.root_pos_w[:, :2] - self._terrain.env_origins[:,:2]) > 4.0, dim=1))
            else:
                died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.body_state_w[:, self._body_id, 2].squeeze() < 0.0)

        # Check if the robot is too high
        # died = torch.logical_or(died, self._robot.data.root_pos_w[:, 2] > 10.0)

        if died[0] or time_out[0]:
            print("[Isaac Env: Dones] Robot has died: ", died[0].item(), " Time out: ", time_out[0].item())
            print("[Isaac Env: Dones] Robot position: ", self._robot.data.root_pos_w[0] - self._pos_traj[0, 0, :, 0])
            print("[Isaac Env: Dones] Robot velocity: ", self._robot.data.root_lin_vel_w[0])
            print("[Isaac Env: Dones] Robot angular velocity: ", self._robot.data.root_ang_vel_w[0])

        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.reward_task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.reward_goal_body)
        final_distance_to_goal = torch.linalg.norm(
            goal_pos_w[env_ids] - base_pos_w[env_ids], dim=1
        ).mean()
        final_yaw_error = yaw_error_from_quats(
            base_ori_w[env_ids], goal_ori_w[env_ids], 0
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_yaw_error_to_goal"] = final_yaw_error.item()
        extras["Metrics/pos_radius"] = self.cfg.pos_radius
        extras['Metrics/catch_tolerance'] = self.cfg.catch_tolerance
        extras['Metrics/time_to_catch_max'] = self.cfg.time_to_catch_max
        extras["Metrics/catch_success_rate"] = self.is_success[env_ids].mean().item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.eval_mode:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        elif self.cfg.eval_mode:
            self.episode_length_buf[env_ids] = 0

        # --- NEW: Reset timer and success flags for the new episode ---
        # Give the drone a random flight time between 0.5s and 1.5s
        ttc_low = max(self.cfg.time_to_catch_min, self.cfg.time_to_catch_max - 0.30)
        self.time_to_catch[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
    	    ttc_low, self.cfg.time_to_catch_max
	    )
        self.is_success[env_ids] = 0.0
        self.has_evaluated_catch[env_ids] = False


        self._actions[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._previous_omega_err[env_ids] = 0.0
        # Update the trajectories for the reset environments
        self.initialize_trajectories(env_ids)
        self.update_goal_state()
        
        # Sample new commands
        # if self.cfg.goal_cfg == "rand":
        #     self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        #     self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        #     self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(2.0, 4.0)
        #     self._desired_ori_w[env_ids] = random_yaw_orientation(len(env_ids), device=self.device) 
        # elif self.cfg.goal_cfg == "fixed":
        #     self._desired_pos_w[env_ids] = torch.tensor(self.cfg.goal_pos, device=self.device).tile((env_ids.size(0), 1))
        #     self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        #     self._desired_ori_w[env_ids] = torch.tensor(self.cfg.goal_ori, device=self.device).tile((env_ids.size(0), 1))
        # # Reset robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, 2] = 3.0 # start at 3m height
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.cfg.init_cfg == "rand":
            default_root_state = self._robot.data.default_root_state[env_ids]
            # Initialize the robot on the trajectory with the correct velocity
            traj_pos_start = self._pos_traj[0, env_ids, :, 0]
            traj_vel_start = self._pos_traj[1, env_ids, :, 0]
            traj_yaw_start = self._yaw_traj[0, env_ids, 0]
            pos_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_pos_ranges, device=self.device).float()
            vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_lin_vel_ranges, device=self.device).float()
            yaw_rand = (torch.rand(len(env_ids), 1, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_yaw_ranges, device=self.device).float()
            ang_vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_ang_vel_ranges, device=self.device).float()
            init_yaw = math_utils.quat_from_yaw(traj_yaw_start + yaw_rand.squeeze(1))

            default_root_state[:, :3] = traj_pos_start + pos_rand
            default_root_state[:, 3:7] = init_yaw
            default_root_state[:, 7:10] = traj_vel_start + vel_rand
            default_root_state[:, 10:13] = ang_vel_rand
            # default_root_state[:, :3] = traj_pos_start
            # default_root_state[:, 3:7] = math_utils.quat_from_yaw(traj_yaw_start)
            # default_root_state[:, 7:10] = traj_vel_start
            # default_root_state[:, 10:13] = torch.zeros_like(traj_vel_start)
        elif self.cfg.init_cfg == "fixed":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, 2] = 3.0

            # Initialize with -360 deg/s roll rate to test stabilization
            init_vel = default_root_state[:, 7:10]
            init_ang_vel_w = torch.zeros_like(default_root_state[:, 10:13], device=self.device)
            init_ang_vel_w[:, 0] = -2.0 * math.pi  # -360 deg/s roll rate
            init_ang_vel_b = isaac_math_utils.quat_apply_inverse(default_root_state[:, 3:7], init_ang_vel_w)
            default_root_state[:, 7:10] = init_vel
            default_root_state[:, 10:13] = init_ang_vel_b
            print("[Isaac Env: Reset] Initial angular velocity (world frame): ", init_ang_vel_w[0])
            print("[Isaac Env: Reset] Initial angular velocity (body frame): ", init_ang_vel_b[0])
            # Initialize the robot on the trajectory with the correct velocity
            # default_root_state[:, :3] = self._desired_pos_w[env_ids]
            # default_root_state[:, 3:7] = self._desired_ori_w[env_ids]
        elif self.cfg.init_cfg == "rotorpy":
            default_root_state = self._robot.data.default_root_state[env_ids]
            
            traj_pos_start = self._pos_traj[0, env_ids, :, 0]

            default_root_state[:, :3] = (torch.rand(len(env_ids), 3, device=self.device) * 4.0 - 2.0) + traj_pos_start
            # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            random_roll = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.init_euler_ranges[0] * 2.0 - self.cfg.init_euler_ranges[0]
            random_pitch = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.init_euler_ranges[1] * 2.0 - self.cfg.init_euler_ranges[1]
            random_yaw = torch.rand(len(env_ids), 1, device=self.device) * self.cfg.init_euler_ranges[2] * 2.0 - self.cfg.init_euler_ranges[2]
            default_root_state[:, 3:7] = isaac_math_utils.quat_from_euler_xyz(
                random_roll.squeeze(), random_pitch.squeeze(), random_yaw.squeeze())

            default_root_state[:, 7:10] = torch.rand(len(env_ids), 3, device=self.device) * 3.0 - 1.5
            default_root_state[:, 10:13] = torch.zeros(len(env_ids), 3, device=self.device)

            self._action_history[env_ids] = torch.zeros(len(env_ids), self.cfg.action_history_length, 4, device=self.device)
            self._state_history[env_ids] = torch.zeros(len(env_ids), self.cfg.state_history_length, 3, device=self.device)

            if 0 in env_ids:
                print("[Isaac Env: Reset] Default root state: ", default_root_state[0])
                print("[Isaac Env: Reset] Desired pos: ", self._desired_pos_w[env_ids][0])
                print("[Isaac Env: Reset] Action history: ", self._action_history[env_ids][0])
                print("[Isaac Env: Reset] State history: ", self._state_history[env_ids][0])
        else:
            default_root_state = self._robot.data.default_root_state[env_ids]
            # Initialize the robot on the trajectory with the correct velocity
            default_root_state[:, :3] = self._desired_pos_w[env_ids]
            default_root_state[:, 3:7] = self._desired_ori_w[env_ids]
            default_root_state[:, 7:10] = self._pos_traj[1, env_ids, :, 0]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        
        self._motor_speeds[env_ids] = self.cfg.init_motor_speed * torch.ones_like(self._motor_speeds[env_ids])
        
        # Domain Randomization
        self.domain_randomization(env_ids)

        # Reset action queue
        # hover_thrust = 2.0 / self._thrust_to_weight[env_ids] - 1.0
        # hover_actions = torch.zeros(len(env_ids), 4, device=self.device)
        # hover_actions[:, 0] = hover_thrust
        # self._action_queue[:,env_ids,:] = hover_actions.unsqueeze(0).repeat(self._action_queue.shape[0], 1, 1)
        hover_actions = torch.zeros(len(env_ids), 4, device=self.device)
        hover_actions[:, 0] = 2.0 * (self._robot_weight[env_ids] / (4.0 * self.max_thrust[env_ids])) - 1.0
        hover_actions[:, 0] = hover_actions[:, 0].clamp(-1.0, 1.0)
        self._action_queue[:, env_ids, :] = hover_actions.unsqueeze(0).repeat(self._action_queue.shape[0], 1, 1)

    def initialize_trajectories(self, env_ids):
        """
        Initializes the trajectory for the environment ids.
        """
        num_envs = env_ids.size(0)

        # Randomize Lissajous parameters
        random_amplitudes = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_amplitudes_rand_ranges
        random_frequencies = ((torch.rand(num_envs, 4, device=self.device))) * self.lissajous_frequencies_rand_ranges
        random_phases = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_phases_rand_ranges
        random_offsets = ((torch.rand(num_envs, 4, device=self.device)) * 2.0 - 1.0) * self.lissajous_offsets_rand_ranges

        terrain_offsets = torch.zeros_like(random_offsets, device=self.device)
        terrain_offsets[:, :2] = self._terrain.env_origins[env_ids, :2]
        
        self.lissajous_amplitudes[env_ids] = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((num_envs, 1)).float() + random_amplitudes
        self.lissajous_frequencies[env_ids] = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((num_envs, 1)).float() + random_frequencies
        self.lissajous_phases[env_ids] = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((num_envs, 1)).float() + random_phases
        self.lissajous_offsets[env_ids] = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((num_envs, 1)).float() + random_offsets + terrain_offsets

    def domain_randomization(self, env_ids: torch.Tensor | None):
        if env_ids is None or env_ids.shape[0] == 0:
            return
        
        reinit_motor_dynamics = False

        if self.cfg.dr_dict.get("thrust_to_weight", 0.0) > 0:
            self._thrust_to_weight[env_ids] = torch.zeros_like(self._thrust_to_weight[env_ids]).normal_(mean=0.0, std=0.4) + self.cfg.thrust_to_weight

        if self.cfg.dr_dict.get("mass", 0.0) > 0:
            dr_range = self.cfg.dr_dict["mass"]
            self._robot_mass[env_ids] = torch.zeros_like(self._robot_mass[env_ids]).uniform_(1-dr_range, 1+dr_range) * self.cfg.mass
            self._robot_weight[env_ids] = self._robot_mass[env_ids] * self._gravity_magnitude
            new_masses = self._default_masses.clone().to(device=self.device)
            new_masses[env_ids, self._body_id] = self._robot_mass[env_ids]
            self._robot.root_physx_view.set_masses(new_masses.cpu(), env_ids.cpu())
            self._robot_masses = self._robot.root_physx_view.get_masses().to(self.device)

        if self.cfg.dr_dict.get("inertia", 0.0) > 0:
            dr_range = self.cfg.dr_dict["inertia"]
            self._robot_inertia[env_ids] = torch.zeros_like(self._robot_inertia[env_ids]).uniform_(1-dr_range, 1+dr_range) * self.default_inertia.view(-1, 3, 3).tile(env_ids.shape[0], 1, 1)
            new_inertia = self._robot.root_physx_view.get_inertias().clone().to(self.device)
            new_inertia[env_ids, self._body_id] = self._robot_inertia[env_ids].view(env_ids.shape[0], 9)
            # import code; code.interact(local=locals())
            self._robot.root_physx_view.set_inertias(new_inertia.cpu(), env_ids.cpu())
            self.inertia_tensor[env_ids] = self._robot_inertia[env_ids]
            # self.inertia = self._robot.root_physx_view.get_inertia(env_ids.cpu()).to(self.device)
            # self._robot_inertia = self._robot.root_physx_view.get_inertias().to(self.device)

        if self.cfg.dr_dict.get("tau_m", 0.0) > 0:
            dr_range = self.cfg.dr_dict["tau_m"]
            self._tau_m[env_ids] = torch.zeros_like(self._tau_m[env_ids], device=self.device).uniform_(1-dr_range, 1+dr_range) * self.cfg.tau_m

        if self.cfg.dr_dict.get("k_eta", 0.0) > 0:
            dr_range = self.cfg.dr_dict["k_eta"]
            self._k_eta[env_ids] = torch.zeros_like(self._k_eta[env_ids], device=self.device).uniform_(1-dr_range, 1+dr_range) * self.cfg.k_eta
            reinit_motor_dynamics = True

        if self.cfg.dr_dict.get("k_m", 0.0) > 0.0:
            dr_range = self.cfg.dr_dict["k_m"]
            self._k_m[env_ids] = (torch.zeros_like(self._k_m[env_ids], device=self.device).uniform_(1 - dr_range, 1 + dr_range)* self.cfg.k_m)
            reinit_motor_dynamics = True

        # if self.cfg.dr_dict.get("k_torque", 0.0) < 0:
        #     self._k_torque[env_ids] = torch.zeros_like(self._k_torque[env_ids], device=self.device).uniform_(1-self.cfg.dr_dict["k_torque"], 1+self.cfg.dr_dict["k_torque"]) * self.cfg.k_torque
        if self.cfg.dr_dict.get("k_torque", 0.0) > 0.0:
            drrange = self.cfg.dr_dict["k_torque"]
            self._k_torque[env_ids] = (torch.zeros_like(self._k_torque[env_ids], device=self.device).uniform_(1 - drrange, 1 + drrange) * self.cfg.k_torque)
            reinit_motor_dynamics = True

        # if self.cfg.dr_dict.get("arm_length", 0.0) > 0:
        #     self._arm_length[env_ids] = torch.zeros_like(self._arm_length[env_ids], device=self.device).uniform_(-0.01, 0.01) + self.cfg.arm_length*torch.ones(self._arm_length[env_ids].shape, device=self.device)
        #     reinit_motor_dynamics = True
        if self.cfg.dr_dict.get("arm_length", 0.0) > 0.0:
            drrange = self.cfg.dr_dict["arm_length"]
            self._arm_length[env_ids] = (torch.zeros_like(self._arm_length[env_ids], device=self.device).uniform_(1 - drrange, 1 + drrange) * self.cfg.arm_length)
            reinit_motor_dynamics = True

        if self.cfg.dr_dict.get("kp_att", 0.0) > 0:
            self._kp_att[env_ids] = torch.zeros_like(self._kp_att[env_ids], device=self.device).uniform_(1-self.cfg.dr_dict["kp_att"], 1+self.cfg.dr_dict["kp_att"]) * self.cfg.kp_att
        
        if self.cfg.dr_dict.get("kd_att", 0.0) > 0:
            self._kd_att[env_ids] = torch.zeros_like(self._kd_att[env_ids], device=self.device).uniform_(1-self.cfg.dr_dict["kd_att"], 1+self.cfg.dr_dict["kd_att"]) * self.cfg.kd_att

        if reinit_motor_dynamics:
            self.reinitialize_motor_dynamics(env_ids)


        self._motor_speeds[env_ids] = torch.sqrt(self._robot_weight[env_ids] / (4 * self._k_eta[env_ids])).unsqueeze(1).tile((1, 4)).to(self.device)
        self.max_thrust[env_ids] = self.cfg.motor_speed_max**2 * self._k_eta[env_ids]

        if self.cfg.dr_dict.get("control_latency_steps", 0) > 0:
            dr_range = int(self.cfg.dr_dict["control_latency_steps"])
            low = max(0, int(self.cfg.control_latency_steps) - dr_range)
            high = int(self.cfg.control_latency_steps) + dr_range + 1
            self._control_latency_steps[env_ids] = torch.randint(
                low, high, (env_ids.shape[0],), device=self.device
            )
        else:
            self._control_latency_steps[env_ids] = int(self.cfg.control_latency_steps)

        if 0 in env_ids:
            print("[Isaac Env: Domain Randomization] Domain randomization applied:")
            print("[Isaac Env: Domain Randomization] Robot mass: ", self._robot_mass[env_ids][0])
            print("[Isaac Env: Domain Randomization] Robot inertia: ", self._robot_inertia[env_ids][0])
            print("[Isaac Env: Domain Randomization] k_eta: ", self._k_eta[env_ids][0])
            print("[Isaac Env: Domain Randomization] Tau_m: ", self._tau_m[env_ids][0])
            print("[Isaac Env: Domain Randomization] kp_att: ", self._kp_att[env_ids][0])
            print("[Isaac Env: Domain Randomization] kd_att: ", self._kd_att[env_ids][0])
            print("[Isaac Env: Domain Randomization] motor speeds: ", self._motor_speeds[env_ids][0])
            print("[Isaac Env: Domain Randomization] max thrust: ", self.max_thrust[env_ids][0])
            print("[Isaac Env: Domain Randomization] control latency steps: ", self._control_latency_steps[env_ids][0])

    
    def reinitialize_motor_dynamics(self, env_ids: torch.Tensor | None = None):
        if env_ids is None or env_ids.shape[0] == 0:
            return
        r2o2 = math.sqrt(2.0) / 2.0
        num_envs = len(env_ids)
        self._rotor_positions[env_ids] = torch.cat(
            [
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([-r2o2, -r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
                (self._arm_length[env_ids].unsqueeze(1) * torch.tensor([-r2o2, r2o2, 0], device=self.device).unsqueeze(0)).unsqueeze(1),
            ],
            dim=1, 
        )

        # if self.cfg.k_torque > 0.0:
        #     self.k[env_ids] = self.cfg.k_torque * torch.ones(num_envs, device=self.device)
        # else:
        #     self.k[env_ids] = self._k_m / self._k_eta
        if getattr(self.cfg, "k_torque", 0.0) > 0.0:
            k_torque = self._k_torque[env_ids]
        else:
            k_torque = self._k_m[env_ids] / self._k_eta[env_ids]
        
        # self.f_to_TM[env_ids] = torch.cat(
        #     [
        #         torch.ones(num_envs, 1, 4, device=self.device),
        #         # torch.cat(
        #         #     [
        #         #         torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
        #         #     ], 
        #         #     dim=1
        #         # ).to(self.device),
        #         torch.linalg.cross(self._rotor_positions[env_ids], torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(num_envs, 1, 1))[:,:, 0:2].transpose(-2,-1),
        #         self.k[env_ids].view(num_envs, 1, 1) * self._rotor_directions[env_ids].view(num_envs, 1, 4),
        #     ],
        #     dim=1
        # )

        # self.TM_to_f[env_ids] = torch.linalg.inv(self.f_to_TM[env_ids])
        self.f_to_TM[env_ids] = torch.cat(
            [
                torch.ones(num_envs, 1, 4, device=self.device),
                torch.linalg.cross(
                    self._rotor_positions[env_ids],
                    torch.tensor([0.0, 0.0, 1.0], device=self.device).tile(num_envs, 1, 1),
                )[:, :, 0:2].transpose(-2, -1),
                k_torque.view(num_envs, 1, 1) * self._rotor_directions[env_ids].view(num_envs, 1, 4),
            ],
            dim=1,
        )
        self.TM_to_f[env_ids] = torch.linalg.inv(self.f_to_TM[env_ids])


    def get_body_state_by_name(self, body_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if body_name == "body":
            pos = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            ori = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            vel = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        elif body_name == "root":
            pos = self._robot.data.root_pos_w
            ori = self._robot.data.root_quat_w
            vel = self._robot.data.root_lin_vel_w
            ang_vel = self._robot.data.root_ang_vel_w
        elif body_name == "endeffector":
            pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            vel = self._robot.data.body_lin_vel_w[:, self._ee_id].squeeze(1)
            ang_vel = self._robot.data.body_ang_vel_w[:, self._ee_id].squeeze(1)
        else:
            raise NotImplementedError(f"Body name {body_name} is not implemented.")
        return pos, ori, vel, ang_vel

    def get_ang_vel_body(self, body_name: str) -> torch.Tensor:
        if body_name == "body":
            ang_vel_b = self._robot.data.root_ang_vel_b
        elif body_name == "root":
            ang_vel_b = self._robot.data.root_ang_vel_b
        elif body_name == "endeffector":
            ang_vel_b = quat_apply_inverse(self._robot.data.body_quat_w[:, self._ee_id].squeeze(1), self._robot.data.body_ang_vel_w[:, self._ee_id].squeeze(1))
        else:
            raise NotImplementedError(f"Body name {body_name} is not implemented.")
        
        return ang_vel_b
    
    def get_frame_state_from_task(self, task_body:str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if task_body == "root":
            base_pos_w = self._robot.data.root_pos_w
            base_ori_w = self._robot.data.root_quat_w
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif task_body == "endeffector":
            base_pos_w = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            lin_vel_w = self._robot.data.root_lin_vel_w
            ang_vel_w = self._robot.data.root_ang_vel_w
        elif task_body == "vehicle" or task_body == "body":
            base_pos_w = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, self._body_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, self._body_id].squeeze(1)
        elif task_body == "COM":
            frame_id = self._robot.find_bodies("COM")[0]
            base_pos_w = self._robot.data.body_pos_w[:, frame_id].squeeze(1)
            base_ori_w = self._robot.data.body_quat_w[:, frame_id].squeeze(1)
            lin_vel_w = self._robot.data.body_lin_vel_w[:, frame_id].squeeze(1)
            ang_vel_w = self._robot.data.body_ang_vel_w[:, frame_id].squeeze(1)
        else:
            raise ValueError("Invalid task body: ", self.cfg.task_body)

        return base_pos_w, base_ori_w, lin_vel_w, ang_vel_w
    
    def compute_desired_pose_from_transform(self, goal_pos_w, goal_ori_w, pos_transform):
        # Find b2 in the ori frame, set z component to 0 and the desired yaw is the atan2 of the x and y components
        b2 = quat_apply(goal_ori_w, torch.tensor([[0.0, 1.0, 0.0]], device=goal_ori_w.device).tile(goal_ori_w.shape[0], 1))
        if self.cfg.num_joints == 0:
            b2[:, 2] = 0.0
        b2 = normalize(b2)
        
        # Yaw is the angle between b2 and the y-axis
        yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi/2
        yaw_desired = wrap_to_pi(yaw_desired)

        # Position desired is the pos_transform along -b2 direction
        pos_desired = goal_pos_w + torch.bmm(torch.linalg.norm(pos_transform, dim=1).tile(self.num_envs, 1, 1), -1*b2.unsqueeze(1)).squeeze(1)

        
        return pos_desired, yaw_desired

    def get_goal_state_from_task(self, goal_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_body == "root" or goal_body == "body":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "endeffector":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "COM":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            desired_pos, desired_yaw = compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e, 0)
            goal_pos_w = desired_pos
            goal_ori_w = quat_from_yaw(desired_yaw)
        elif goal_body == "vehicle":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            # desired_pos, desired_yaw = compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.body_pos_ee_frame, 0)
            # desired_pos_w is the ee in world frame, we want the corresponding body pos in world frame
            desired_pos = self._desired_pos_w + quat_apply(self._desired_ori_w, self.body_pos_ee_frame)
            # desired_pos = self._desired_pos_w + self.body_pos_ee_frame
            # desired_pos = self._desired_pos_w
            goal_pos_w = desired_pos
            goal_ori_w = self._desired_ori_w
        else:
            raise ValueError("Invalid goal body: ", self.cfg.goal_body)

        return goal_pos_w, goal_ori_w

    def convert_ee_goal_from_task(self, ee_pos_w, ee_ori_w, task_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.has_end_effector:
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "root":
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "endeffector":
            desired_pos, desired_ori = ee_pos_w, ee_ori_w
        elif task_body == "vehicle":
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self._robot.data.body_pos_w[:, self._body_id].squeeze(1), 0)
            desired_ori = quat_from_yaw(desired_yaw)
        elif task_body == "COM":
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self.com_pos_e, 0)
            desired_ori = quat_from_yaw(desired_yaw)
        else:
            raise ValueError("Invalid task body: ", task_body)

        return desired_pos, desired_ori
    
    def convert_ee_goal_to_com_goal(self, ee_pos_w: torch.Tensor, ee_ori_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(ee_pos_w, ee_ori_w, self.com_pos_e, 0)
        return desired_pos, quat_from_yaw(desired_yaw)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame": 
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "frame": sim_utils.UsdFileCfg(
                                                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                                scale=(0.03, 0.03, 0.03),
                                            ),})
                elif self.cfg.viz_mode == "robot":
                    history_color = tuple(self.cfg.robot_color) if sum(self.cfg.robot_color) > 0 else (0.05, 0.05, 0.05)
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "robot_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),
                                            "robot_history": sim_utils.SphereCfg(
                                                radius=0.005,
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=history_color),
                                            ),
                                            "goal_history": sim_utils.SphereCfg(
                                                radius=0.005,
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),})
                elif self.cfg.viz_mode == "viz":
                    robot_color = tuple(self.cfg.robot_color) if sum(self.cfg.robot_color) > 0 else (0.05, 0.05, 0.05)
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "goal_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
                                            ),
                                            "robot_mesh": sim_utils.UsdFileCfg(
                                                usd_path=self.cfg.robot.spawn.usd_path,
                                                scale=(1.0, 1.0, 1.0),
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=robot_color),
                                            ),
                                            })
                else:
                    raise ValueError("Visualization mode not recognized: ", self.cfg.viz_mode)
    
                self.frame_visualizer = VisualizationMarkers(frame_marker_cfg)
                # set their visibility to true
                self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        # update the markers
        # Update frame positions for debug visualization
        if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame":
            self._frame_positions[:, 0] = self._robot.data.root_pos_w
            self._frame_positions[:, 1] = self._desired_pos_w
            # self._frame_positions[:, 2] = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            # self._frame_positions[:, 2] = com_pos_w
            self._frame_orientations[:, 0] = self._robot.data.root_quat_w
            self._frame_orientations[:, 1] = self._desired_ori_w
            # self._frame_orientations[:, 2] = self._robot.data.body_quat_w[:, self._body_id].squeeze(1)
            # self._frame_orientations[:, 2] = com_ori_w
            self.frame_visualizer.visualize(self._frame_positions.flatten(0, 1), self._frame_orientations.flatten(0,1))
        elif self.cfg.viz_mode == "robot":
            self._robot_positions = self._desired_pos_w + torch.tensor(self.cfg.viz_ref_offset, device=self.device).unsqueeze(0).tile((self.num_envs, 1))
            self._robot_orientations = self._desired_ori_w
            # self.frame_visualizer.visualize(self._robot_positions, self._robot_orientations, marker_indices=[0]*self.num_envs)

            self._goal_pos_history = self._goal_pos_history.roll(1, dims=1)
            self._goal_pos_history[:, 0] = self._desired_pos_w
            self._goal_ori_history = self._goal_ori_history.roll(1, dims=1)
            self._goal_ori_history[:, 0] = self._desired_ori_w
            # self.frame_visualizer.visualize(self._goal_pos_history.flatten(0, 1), self._goal_ori_history.flatten(0, 1),  marker_indices=[2]*self.num_envs*10)

            self._robot_pos_history = self._robot_pos_history.roll(1, dims=1)
            self._robot_pos_history[:, 0] = self._robot.data.root_pos_w
            self._robot_ori_history = self._robot_ori_history.roll(1, dims=1)
            self._robot_ori_history[:, 0] = self._robot.data.root_quat_w
            # self.frame_visualizer.visualize(self._robot_pos_history.flatten(0, 1), self._robot_ori_history.flatten(0, 1),  marker_indices=[1]*self.num_envs*10)

            translation_pos = torch.cat([self._robot_positions, self._robot_pos_history.flatten(0, 1), self._goal_pos_history.flatten(0, 1)], dim=0)
            translation_ori = torch.cat([self._robot_orientations, self._robot_ori_history.flatten(0, 1), self._goal_ori_history.flatten(0, 1)], dim=0)
            marker_indices = [0]*self.num_envs + [1]*self.num_envs*self.cfg.viz_history_length + [2]*self.num_envs*self.cfg.viz_history_length
            self.frame_visualizer.visualize(translation_pos, translation_ori, marker_indices=marker_indices)
        elif self.cfg.viz_mode == "viz":
            self._robot_positions = self._desired_pos_w
            self._robot_orientations = self._desired_ori_w

            goal_pos = self._desired_pos_w.clone()
            goal_ori = self._desired_ori_w.clone()

            robot_pos = self._robot.data.root_pos_w.clone()
            robot_ori = self._robot.data.root_quat_w.clone()

            translation_pos = torch.cat([goal_pos, robot_pos], dim=0)
            translation_ori = torch.cat([goal_ori, robot_ori], dim=0)
            marker_indices = [0]*self.num_envs + [1]*self.num_envs
            self.frame_visualizer.visualize(translation_pos, translation_ori, marker_indices=marker_indices)