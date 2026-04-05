from __future__ import annotations

import math
import torch

# Isaac SDK imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    subtract_frame_transforms, 
    combine_frame_transforms,
    matrix_from_quat,
    quat_error_magnitude,
    random_orientation,
    quat_inv,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    quat_unique,
    yaw_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    rigid_body_twist_transform,
    euler_xyz_from_quat,
    matrix_from_euler,
    wrap_to_pi,
)
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.sim.spawners.shapes import SphereCfg, spawn_sphere
from isaaclab.sim.spawners.materials import VisualMaterialCfg, PreviewSurfaceCfg, spawn_preview_surface

# from isaaclab.sim.utils import get_prim_at_path
from pxr import Usd, UsdShade, Gf
# Local imports
import gymnasium as gym
import numpy as np
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_0DOF_CFG, AERIAL_MANIPULATOR_0DOF_DEBUG_CFG, AERIAL_MANIPULATOR_QUAD_ONLY_CFG
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_0DOF_LONG_ARM_COM_MIDDLE_CFG
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_V_CFG, AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_MIDDLE_CFG, AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_EE_CFG
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_2DOF_CFG

from utils.math_utilities import (
    yaw_from_quat,
    yaw_error_from_quats,
    quat_from_yaw,
    wrist_angle_error_from_quats,
    shoulder_angle_error_from_quats,
    yaw_error_from_quats,
    calculate_required_pos,
    aerial_manipulator_angle_errors,
    aerial_manipulator_angle_solns_2dof,
    vee_map,
    hat_map,
)
from utils.trajectory_utilities import eval_sinusoid
import utils.trajectory_utilities as traj_utils
import utils.math_utilities as math_utils
import utils.flatness_utilities as flatness_utils

class AerialManipulatorTrajectoryTrackingEnvWindow(BaseEnvWindow):
    """4Window manager for the Quadcopter environment."""

    def __init__(self, env: AerialManipulatorTrajectoryTrackingEnv, window_name: str = "Aerial Manipulator Trajectory Tracking - IsaacLab"):
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
class EventCfg:
    """Configuration for events - used for domain randomization."""
    randomize_endeffector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["endeffector"]),
            "mass_distribution_params": (-0.2, 0.0),
            "operation": "add",
        },
    )

@configclass
class NoEndEffectorEventCfg:
    """Helper class to set the endeffector mass to 0.0"""
    randomize_endeffector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["endeffector"]),
            "mass_distribution_params": (-0.2, -0.2),
            "operation": "add",
        },
    )


@configclass
class AerialManipulatorTrajectoryTrackingEnvBaseCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    sim_rate_hz = 100
    policy_rate_hz = 50
    decimation = sim_rate_hz // policy_rate_hz
    ui_window_class_type = AerialManipulatorTrajectoryTrackingEnvWindow
    state_space = 0
    debug_vis = True

    # added as a way to make curriculum learning easier to formulate in terms of iterations instead of timesteps,
    # changing this via hydra won't change the actual behavior of the algorithm
    num_steps_per_env = 32

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        #disable_contact_processing=True,
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
            static_friction=0.0,
            dynamic_friction=0.0,
            restitution=0.2,
        ),
        debug_vis=False,
    )

    events = NoEndEffectorEventCfg()

    action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    observation_space= gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,))
    state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

    # observation_noise_model: NoiseModelCfg = NoiseModelCfg(
    #     noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01),
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    traj_update_dt = 0.02

    trajectory_type = "lissaajous"
    trajectory_horizon = 5
    random_shift_trajectory = False
    # TODO: this is not changed at this level using hydra args, happens once __init__ called for env
    eval_trajectory = "" or trajectory_type
 
    # (x, y, z, roll, pitch, yaw)
    lissajous_amplitudes = [0.0] * 6#[1.0, 1.0, 1.0, 3*np.pi/4, 3*np.pi/4, 3*np.pi/4]
    lissajous_amplitudes_rand_ranges = [1.0, 1.0, 1.0, np.pi, np.pi, np.pi]#[1.0, 1.0, 1.0, np.pi, np.pi, np.pi]
    lissajous_frequencies = [0.0] * 6#[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    lissajous_frequencies_rand_ranges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]#[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    lissajous_phases = [0.0]*6
    lissajous_phases_rand_ranges = [np.pi]*6
    lissajous_offsets = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0] # Higher z offset just to avoid fake crashes
    lissajous_offsets_rand_ranges = [2.0, 2.0, 0.5, np.pi, np.pi, np.pi]

    reset_curriculum = 50
    reset_curriculum_rand_range = 1.0

    polynomial_x_coefficients= [0.5, 0.5]
    polynomial_y_coefficients= [0.5, 0.5]
    polynomial_z_coefficients= [0.5, 0.5]
    polynomial_roll_coefficients= [0.5, 0.5]
    polynomial_roll_rand_ranges = [0.5, 0.5]
    polynomial_pitch_coefficients= [0.5, 0.5]
    polynomial_pitch_rand_ranges = [0.5, 0.5]
    polynomial_yaw_coefficients= [0.5, 0.5]
    polynomial_yaw_rand_ranges = [0.5, 0.5]

    # Motor dynamics - would need to validate on hardware - these are rough estimates
    use_motor_dynamics = False
    rotor_arm_length = 0.12367  # named to avoid conflict with manipulator arm_length
    k_eta = 1.179e-6            # thrust coefficient
    k_m = 1.104e-8             # moment coefficient
    tau_m = 0.005             # motor time constant [s]
    motor_speed_min = 0.0
    motor_speed_max = 2393.0

    moment_scale_xy = 1.18
    moment_scale_z = 0.126 # 0.025 # 0.1
    thrust_to_weight = 2.75

    # reward scales
    body_pos_radius_start = 1.0
    body_pos_radius_curriculum = 75 #int(1e7) # 10e6
    body_pos_error_reward_scale = 0.0 # -1.0
    body_pos_distance_reward_scale = 1.0 #15.0

    ee_pos_radius_start = 1.0
    ee_pos_radius_curriculum = 75
    ee_pos_error_reward_scale = 0.0 # -1.0
    ee_pos_distance_reward_scale = 10.0 #15.0

    ori_radius_start = 1.5
    ori_radius_curriculum = 75
    ori_distance_reward_scale = 10.0 #15.0
    ori_error_reward_scale = 0.0 # -0.5

    lin_vel_reward_scale = -0.5 # -0.05
    lin_vel_radius_start = 2.0
    lin_vel_radius_curriculum = 0  # Set to > 0 to enable curriculum
    lin_vel_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward
    
    ang_vel_reward_scale = -1.0# -0.01
    ang_vel_radius_start = 5.0
    ang_vel_radius_curriculum = 0  # Set to > 0 to enable curriculum
    ang_vel_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward
    
    body_ang_vel_reward_scale = 0.0
    body_ang_vel_radius_start = 0.8
    body_ang_vel_radius_curriculum = 0  # Set to > 0 to enable curriculum
    body_ang_vel_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward

    jitter_reward_scale = 0.0 # Penalizes jittering of body angular velocity in the local x and y axes
    
    joint_vel_reward_scale = -0.2 # -0.01
    joint_vel_radius_start = 0.5
    joint_vel_radius_curriculum = 0  # Set to > 0 to enable curriculum
    joint_vel_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward

    action_norm_reward_scale = 0.0
    action_delta_reward_scale = 0.0
    
    action_norm_prop_reward_scale = -0.5 # -0.01
    action_joint_norm_reward_scale = -0.2 # 0.0a
    previous_action_prop_reward_scale = -0.5 # -0.01
    previous_action_joint_reward_scale = -0.2 # -0.01
    action_delta_prop_radius_start = 1.0
    action_delta_prop_radius_curriculum = 0  # Set to > 0 to enable curriculum
    action_delta_prop_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward
    action_delta_joint_radius_start = 1.0
    action_delta_joint_radius_curriculum = 0  # Set to > 0 to enable curriculum
    action_delta_joint_distance_reward_scale = 0.0  # Reward scale for curriculum-based distance reward
    
    yaw_error_reward_scale = 0.0 # -0.01
    yaw_distance_reward_scale = 0.0 # -0.01
    yaw_radius_start = 0.8
    yaw_radius_curriculum = int(0) 
    yaw_smooth_transition_scale = 0.0

    shoulder_error_reward_scale = 0.0
    shoulder_radius_start = 0.8
    shoulder_radius_curriculum = int(0)
    shoulder_distance_reward_scale = 0.0
    
    wrist_error_reward_scale = 0.0 #-2.0 
    wrist_radius_start = 0.8
    wrist_radius_curriculum = 0
    wrist_distance_reward_scale = 0.0#1.0

    axis_reward_scale = 0.0

    stay_alive_reward = 0.0
    crash_penalty = -1.0
    scale_reward_with_time = True
    square_reward_errors = False
    square_pos_error = True
    combined_alpha = 0.0
    combined_tolerance = 0.0
    combined_scale = 0.0

    # Control mode: "CTBM" (collective thrust + body moments, default) or "CTATT" (collective thrust + attitude setpoint)
    control_mode = "CTBM"

    # CTATT inner-loop PD gains (only active when control_mode == "CTATT")
    # kp_att = 1575
    # kd_att = 229.93
    # attitude_scale_xy = 0.2
    # attitude_scale_z = torch.pi - 1e-6

    # PD attitude loop runs at sim_rate_hz; pd_loop_decimation=1 means every physics step
    # pd_loop_rate_hz = sim_rate_hz
    # pd_loop_decimation = sim_rate_hz // pd_loop_rate_hz

    goal_pos_range = 2.0
    goal_yaw_range = 3.14159

    # Task condionionals for the environment - modifies the goal
    goal_cfg = "rand" # "rand", "fixed", or "initial"
    # "rand" - Random goal position and orientation
    # "fixed" - Fixed goal position and orientation set apriori
    # "initial" - Goal position and orientation is the initial position and orientation of the robot
    goal_pos = None
    goal_vel = None
    init_pos_ranges=[1.0, 1.0, 0.5]
    init_lin_vel_ranges=[0.0, 0.0, 0.0]
    init_yaw_ranges=[3.14159]
    init_ang_vel_ranges=[0.0, 0.0, 0.0]
    init_joint_ranges =[3.14159, 3.14159]
    init_joint_vel_ranges =[0.0, 0.0]

    init_cfg = "rand" # "default" or "rand"

    task_body = "endeffector" # "root" or "endeffector" or "vehicle" or "COM"
    goal_body = "endeffector" # "root" or "endeffector" or "vehicle" or "COM"
    reward_task_body = "endeffector"
    reward_goal_body = "endeffector"    
    body_name = "vehicle"
    has_end_effector = True
    use_grav_vector = True
    use_full_ori_matrix = True
    use_yaw_representation = False
    use_previous_actions = True
    use_yaw_representation_for_trajectory = False
    use_ang_vel_from_trajectory=True
    use_previous_velocities = False

    shoulder_joint_active = True
    wrist_joint_active = True

    eval_mode = False
    gc_mode = False
    viz_mode = "triad" # or robot
    viz_history_length = 100
    robot_color=[0.0, 0.0, 0.0]
    viz_ref_offset=[0.0,0.0,0.0]

@configclass
class AerialManipulator0DOFTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12

    # action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    # observation_space= gym.spaces.Box(low=-np.inf, high=np.inf, shape=(91,))
    # state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(33,))
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator0DOFLongArmTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_LONG_ARM_COM_MIDDLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator0DOFSmallArmCOMVehicleTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_V_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator0DOFSmallArmCOMMiddleTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_MIDDLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
@configclass
class AerialManipulator0DOFSmallArmCOMEndEffectorTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_SMALL_ARM_COM_EE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator0DOFQuadOnlyTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_QUAD_ONLY_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class AerialManipulator0DOFDebugTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 4
    num_joints = 0
    observation_space = 91 # TODO: Need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 9(ori) = 18
    # 3(vel) + 3(ang vel) + 3(pos) + 3(grav vector body frame) = 12

    # action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    # observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(91,))
    
    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_0DOF_DEBUG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot: ArticulationCfg = AERIAL_MANIPULATOR_QUAD_ONLY_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # robot.collision_group = 0
    # robot.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
    #     friction_combine_mode="multiply",
    #     restitution_combine_mode="multiply",
    #     static_friction=20.0,
    #     dynamic_friction=20.0,
    #     restitution=0.0,
    # )
    # robot.spawn.collision_props=sim_utils.CollisionPropertiesCfg(
    #     collision_enabled=True,
    #     contact_offset=0.02,
    #     torsional_patch_radius=0.04,
    #     min_torsional_patch_radius=0.0001,
    # ),
    # scene = AerialManipulatorTrajectoryTrackingSceneCfg()
    # scene.robot = AERIAL_MANIPULATOR_0DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class AerialManipulator2DOFTrajectoryTrackingEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    # env
    action_space = 6
    num_joints = 2
    observation_space = 16 # TODO: might need to update this..
    # 3(vel) + 3(ang vel) + 3(pos) + 3(ori) + 2(joint pos) + 2(joint vel) = 16
    # action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))

    # robot
    robot: ArticulationCfg = AERIAL_MANIPULATOR_2DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    
    shoulder_torque_scalar = robot.actuators["shoulder"].effort_limit
    wrist_torque_scalar = robot.actuators["wrist"].effort_limit

@configclass 
class AerialManipulatorWithMotorDynamicsCfg(AerialManipulator2DOFTrajectoryTrackingEnvCfg):

    use_motor_dynamics = True

@configclass
class AerialManipulatorWithEndEffectorMassCfg(AerialManipulator2DOFTrajectoryTrackingEnvCfg):

    events = EventCfg()

@configclass 
class AerialManipulatorWithMotorDynamicsAndEndEffectorMassCfg(AerialManipulator2DOFTrajectoryTrackingEnvCfg):

    use_motor_dynamics = True
    events = EventCfg()

class AerialManipulatorTrajectoryTrackingEnv(DirectRLEnv):
    cfg: AerialManipulatorTrajectoryTrackingEnvBaseCfg

    def __init__(self, cfg: AerialManipulatorTrajectoryTrackingEnvBaseCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_space= gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,))
        self.observation_space= gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,))
        self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(0,))

        # Actions / Actuation interfaces
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._joint_torques = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._body_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._body_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._previous_velocity_obs = torch.zeros(self.num_envs, 6 + self.cfg.num_joints, device=self.device) # body lin vel b, body ang vel b, joint velocities

        # Motor dynamics state
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)

        # Goal State   
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ori_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_pos_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._desired_ori_traj_w = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 4, device=self.device)
        self._pos_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._roll_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, device=self.device)
        self._pitch_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, device=self.device)
        self._yaw_traj = torch.zeros(5, self.num_envs, 1+self.cfg.trajectory_horizon, device=self.device)
        self._pos_shift = torch.zeros(self.num_envs, 3, device=self.device)
        self._roll_shift = torch.zeros(self.num_envs, 1, device=self.device)
        self._pitch_shift = torch.zeros(self.num_envs, 1, device=self.device)
        self._yaw_shift = torch.zeros(self.num_envs, 1, device=self.device)

        # Required body attribute for reaching next goal
        self._desired_body_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_body_pos_traj = torch.zeros(self.num_envs, 1+self.cfg.trajectory_horizon, 3, device=self.device)
        self._desired_com_pos = torch.zeros_like(self._desired_body_pos)
        self._desired_com_pos_traj = torch.zeros_like(self._desired_body_pos_traj)

        # self.amplitudes = torch.zeros(self.num_envs, 4, device=self.device)
        # self.frequencies = torch.zeros(self.num_envs, 4, device=self.device)
        # self.phases = torch.zeros(self.num_envs, 4, device=self.device)
        # self.offsets = torch.zeros(self.num_envs, 4, device=self.device)

        # Time(needed for trajectory tracking)
        self._time = torch.zeros(self.num_envs, 1, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "body_pos_error",
                "body_pos_distance",
                "endeffector_combined_error",
                "endeffector_lin_vel",
                "endeffector_lin_vel_distance",
                "endeffector_ang_vel",
                "endeffector_ang_vel_distance",
                "body_ang_vel",
                "body_ang_vel_distance",
                "endeffector_pos_error",
                "endeffector_pos_distance",
                "endeffector_ori_distance",
                "endeffector_ori_error",
                "body_yaw_error",
                "body_yaw_distance",
                "shoulder_joint_error",
                "shoulder_joint_distance",
                "wrist_joint_error",
                "wrist_joint_distance",
                "joint_vel",
                "joint_vel_distance",
                "action_norm",
                "action_delta",
                "action_norm_prop",
                "action_norm_joint",
                "action_delta_prop",
                "action_delta_joint",
                "action_delta_prop_distance",
                "action_delta_joint_distance",
                "jitter",
                "stay_alive",
                "crash_penalty",
                "axis_reward",
            ]
        }

        self._episode_error_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "body_pos_error",
                "combined_error",
                "ee_pos_error",
                "body_ang_vel",
                "jitter",
                "ori_error",
                "yaw_error",
                "shoulder_joint_error",
                "yaw_distance",
                "wrist_joint_error",
                "lin_vel",
                "ang_vel",
                "joint_vel",
                "action_norm",
                "action_delta",
                "action_norm_prop",
                "action_norm_joint",
                "action_delta_prop",
                "action_delta_joint",
                "stay_alive",
                "crash_penalty",
                "axis_reward",
            ]
        }

        # if self.cfg.goal_cfg == "fixed":
        #     assert self.cfg.goal_pos is not None and self.cfg.goal_vel is not None, "Goal position and orientation must be set for fixed goal task"

        # Robot specific data
        self._body_id = self._robot.find_bodies(self.cfg.body_name)[0]
        self._com_id = self._robot.find_bodies("COM")[0]

        assert len(self._body_id) == 1, "There should be only one body with the name \'vehicle\' or \'body\'"

        if self.cfg.has_end_effector:
            self._ee_id = self._robot.find_bodies("endeffector")[0] # also the root of the system
            self._ee_com_id = self._robot.find_bodies("endeffector_com")[0]
        
        if self.cfg.num_joints > 0:
            self._shoulder_joint_idx = self._robot.find_joints("joint_shoulder")[0][0]
        if self.cfg.num_joints > 1:
            self._wrist_joint_idx = self._robot.find_joints("joint_wrist")[0][0]
        # total mass that policy/model sees only accounts for actual robot
        self._total_mass = (self._robot.root_physx_view.get_masses()[0].sum() - self._robot.root_physx_view.get_masses()[0, self._ee_id]).item()
        print("Total Mass: ", self._total_mass)
        self.total_mass = self._total_mass
        self.quad_inertia = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).squeeze()
        if self.cfg.has_end_effector:
            self.arm_inertia = self._robot.root_physx_view.get_inertias()[0, self._ee_id, :].view(-1, 3, 3).squeeze()
        self.arm_offset = self._robot.root_physx_view.get_link_transforms()[0, self._body_id,:3].squeeze() - \
                            self._robot.root_physx_view.get_link_transforms()[0, self._ee_id,:3].squeeze() 
        
        # Compute position and orientation offset between the end effector and the vehicle
        quad_pos = self._robot.data.body_pos_w[0, self._body_id]
        quad_ori = self._robot.data.body_quat_w[0, self._body_id]

        com_pos = self._robot.data.body_pos_w[0, self._com_id]
        com_ori = self._robot.data.body_quat_w[0, self._com_id]

        ee_pos = self._robot.data.body_pos_w[0, self._ee_id]
        ee_ori = self._robot.data.body_quat_w[0, self._ee_id]
        self.model_ee_ori = ee_ori.clone()
        # self.initial_ee_ori = torch.zeros(self.num_envs, 4, device=self.device)
        # self.initial_quad_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        # self.initial_shoulder = torch.zeros_like(self.initial_quad_yaw)
        # self.initial_wrist = torch.zeros_like(self.initial_quad_yaw)
        self.last_yaw_cmd = torch.zeros(self.num_envs, 1, device=self.device)
    
        print("Quad Pos: ", quad_pos)
        print("Quad Ori: ", quad_ori)
        print("COM Pos: ", com_pos)
        print("COM Ori: ", com_ori)
        print("EE Pos: ", ee_pos)
        print("EE Ori: ", ee_ori)
        print("Trajectory type: ", self.cfg.eval_trajectory)


        # get center of mass of whole system (vehicle + end effector)
        self.vehicle_mass = self._robot.root_physx_view.get_masses()[0, self._body_id].sum()
        self.arm_mass = self._total_mass - self.vehicle_mass

        self.com_pos_w = torch.zeros(1, 3, device=self.device)
        for i in range(self._robot.num_bodies):
            self.com_pos_w += self._robot.root_physx_view.get_masses()[0, i] * self._robot.root_physx_view.get_link_transforms()[0, i, :3].squeeze()
        self.com_pos_w /= self._robot.root_physx_view.get_masses()[0].sum()
        self.com_offset = torch.linalg.norm(self.com_pos_w - ee_pos) # offset w.r.t to EE, slightly more useful than wrt to quad for calculating required position

        self.com_pos_e, self.com_ori_e = subtract_frame_transforms(ee_pos, ee_ori, com_pos, com_ori)

        self.arm_offset = self._robot.root_physx_view.get_link_transforms()[0, self._body_id,:3].squeeze() - \
                            self._robot.root_physx_view.get_link_transforms()[0, self._ee_id,:3].squeeze() 
        
        self.arm_length = torch.linalg.norm(self.arm_offset, dim=-1)

        print("Arm Length: ", self.arm_length)
        print("COM_pos_e: ", self.com_pos_e)
        print("Inertia: ", self.quad_inertia)

        # import code; code.interact(local=locals())


        self.position_offset = quad_pos
        # self.orientation_offset = quat_mul(quad_ori, quat_conjugate(ee_ori))
        self.orientation_offset = quad_ori


        self._gravity_magnitude = torch.tensor(self.cfg.sim.gravity, device=self.device).norm()
        self._robot_weight = (self._total_mass * self._gravity_magnitude).item()
        self._grav_vector_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device).tile((self.num_envs, 1))
        self._grav_vector = torch.tensor(self.cfg.sim.gravity, device=self.device).tile((self.num_envs, 1))

        # Rotor geometry for motor dynamics (matches Crazyflie layout)
        r2o2 = math.sqrt(2.0) / 2.0
        _rotor_positions = torch.cat(
            [
                self.cfg.rotor_arm_length * torch.tensor([[r2o2,  r2o2, 0]]),
                self.cfg.rotor_arm_length * torch.tensor([[r2o2, -r2o2, 0]]),
                self.cfg.rotor_arm_length * torch.tensor([[-r2o2, -r2o2, 0]]),
                self.cfg.rotor_arm_length * torch.tensor([[-r2o2,  r2o2, 0]]),
            ],
            dim=0,
        ).to(self.device)
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k_motor = self.cfg.k_m / self.cfg.k_eta  # torque-to-thrust ratio

        # Force-to-wrench mapping: [T, Mx, My, Mz] = f_to_TM @ [f1, f2, f3, f4]
        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(
                            _rotor_positions[i],
                            torch.tensor([0.0, 0.0, 1.0], device=self.device),
                        ).view(-1, 1)[0:2]
                        for i in range(4)
                    ],
                    dim=1,
                ).to(self.device),
                self.k_motor * self._rotor_directions.view(1, -1),
            ],
            dim=0,
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Per-environment vehicle inertia tensor (tiled for batched bmm)
        self.inertia_tensor = (
            self._robot.root_physx_view.get_inertias()[0, self._body_id, :]
            .view(-1, 3, 3)
            .tile(self.num_envs, 1, 1)
            .to(self.device)
        )

        # Hover motor speed: 4 * k_eta * omega^2 = robot_weight  =>  omega = sqrt(w / (4*k_eta))
        self._hover_motor_speed = math.sqrt(self._robot_weight / (4.0 * self.cfg.k_eta))
        self._motor_speeds[:] = self._hover_motor_speed

        # Visualization marker data
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
        self.reset_mask = torch.zeros(self.num_envs, 1, device=self.device)
        self.crash_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self.body_pos_radius = self.cfg.body_pos_radius_start
        self.ee_pos_radius = self.cfg.ee_pos_radius_start
        self.ori_radius = self.cfg.ori_radius_start
        self.yaw_radius = self.cfg.yaw_radius_start
        self.shoulder_radius = self.cfg.shoulder_radius_start
        self.wrist_radius = self.cfg.wrist_radius_start
        self.lin_vel_radius = self.cfg.lin_vel_radius_start
        self.ang_vel_radius = self.cfg.ang_vel_radius_start
        self.body_ang_vel_radius = self.cfg.body_ang_vel_radius_start
        self.joint_vel_radius = self.cfg.joint_vel_radius_start
        self.action_delta_prop_radius = self.cfg.action_delta_prop_radius_start
        self.action_delta_joint_radius = self.cfg.action_delta_joint_radius_start
        self._configure_trajectories()

        # import code; code.interact(local=locals())


    def _configure_trajectories(self):
        """Set lissajous parameters and polynomial coefficients based on the configuration."""
        eval_trajectory = self.cfg.eval_trajectory

        # Easy to visualize known trajectories for eval:
        l = self.arm_length
        if eval_trajectory in ("shoulder", "s"):
        # Trajectory where only shoulder angle should change:
            self.cfg.lissajous_amplitudes = [0.0, l, l, np.pi / 2, 0.0, 0.0]
            self.cfg.lissajous_amplitudes_rand_ranges = [0.0] * 6
            self.cfg.lissajous_frequencies = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            self.cfg.lissajous_frequencies_rand_ranges = [0.0] * 6
            self.cfg.lissajous_phases = [0.0]*6
            self.cfg.lissajous_phases_rand_ranges = [0.0]*6
            self.cfg.lissajous_offsets = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            self.cfg.lissajous_offsets_rand_ranges = [0.0] * 6

        elif eval_trajectory in ("wrist", "w"):
        # Trajectory where only wrist angle should change:
            self.cfg.lissajous_amplitudes = [0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0]
            self.cfg.lissajous_amplitudes_rand_ranges = [0.0] * 6
            self.cfg.lissajous_frequencies = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.cfg.lissajous_frequencies_rand_ranges = [0.0] * 6
            self.cfg.lissajous_phases = [0.0]*6
            self.cfg.lissajous_phases_rand_ranges = [0.0]*6
            self.cfg.lissajous_offsets = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            self.cfg.lissajous_offsets_rand_ranges = [0.0] * 6

        # Trajectory where only yaw angle should change:
        elif eval_trajectory in ("yaw", "y"):
            self.cfg.lissajous_amplitudes = [l, l, 0.0, 0.0, 0.0, np.pi / 2]
            self.cfg.lissajous_amplitudes_rand_ranges = [0.0] * 6
            self.cfg.lissajous_frequencies = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.cfg.lissajous_frequencies_rand_ranges = [0.0] * 6
            self.cfg.lissajous_phases = [0.0]*6
            self.cfg.lissajous_phases_rand_ranges = [0.0]*6
            self.cfg.lissajous_offsets = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            self.cfg.lissajous_offsets_rand_ranges = [0.0] * 6

        elif eval_trajectory in ("hover", "h"):
            self.cfg.lissajous_amplitudes = [0.0] * 6
            self.cfg.lissajous_amplitudes_rand_ranges = [0.0] * 6

        elif eval_trajectory in ("line", "l"):
            self.cfg.lissajous_amplitudes_rand_ranges = [0.0] * 6
            self.cfg.lissajous_frequencies_rand_ranges = [0.0] * 6
            horizontal_amplitude = 2 * (2 * np.random.rand() - 1) # [-2, 2]
            vertical_amplitude = 2 * np.random.rand() - 1 # [-1, 1]
            self.cfg.lissajous_amplitudes = [horizontal_amplitude, horizontal_amplitude, vertical_amplitude, 0.0, 0.0, 0.0]
            horizontal_frequency = 0.5 + 0.5 * np.random.rand() # [0.5, 1.0]
            vertical_frequency = 0.5 + 0.5 * np.random.rand() # [0.5, 1.0]
            self.cfg.lissajous_frequencies = [horizontal_frequency, horizontal_frequency, vertical_frequency, 0.0, 0.0, 0.0]

        self.lissajous_amplitudes = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_amplitudes_rand_ranges = torch.tensor(self.cfg.lissajous_amplitudes_rand_ranges, device=self.device).float()
        self.lissajous_frequencies = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_frequencies_rand_ranges = torch.tensor(self.cfg.lissajous_frequencies_rand_ranges, device=self.device).float()
        self.lissajous_phases = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_phases_rand_ranges = torch.tensor(self.cfg.lissajous_phases_rand_ranges, device=self.device).float()
        self.lissajous_offsets = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((self.num_envs, 1)).float()
        self.lissajous_offsets_rand_ranges = torch.tensor(self.cfg.lissajous_offsets_rand_ranges, device=self.device).float()

        max_coefficients = max(
            len(self.cfg.polynomial_x_coefficients),
            len(self.cfg.polynomial_y_coefficients),
            len(self.cfg.polynomial_z_coefficients),
            len(self.cfg.polynomial_roll_coefficients),
            len(self.cfg.polynomial_pitch_coefficients),
            len(self.cfg.polynomial_yaw_coefficients)
        )
        self.polynomial_coefficients = torch.zeros(self.num_envs, 6, max_coefficients, device=self.device)
        self.polynomial_coefficients[:, 0, :len(self.cfg.polynomial_x_coefficients)] = torch.tensor(self.cfg.polynomial_x_coefficients, device=self.device).tile((self.num_envs, 1))
        self.polynomial_coefficients[:, 1, :len(self.cfg.polynomial_y_coefficients)] = torch.tensor(self.cfg.polynomial_y_coefficients, device=self.device).tile((self.num_envs, 1))
        self.polynomial_coefficients[:, 2, :len(self.cfg.polynomial_z_coefficients)] = torch.tensor(self.cfg.polynomial_z_coefficients, device=self.device).tile((self.num_envs, 1))
        self.polynomial_coefficients[:, 3, :len(self.cfg.polynomial_roll_coefficients)] = torch.tensor(self.cfg.polynomial_roll_coefficients, device=self.device).tile((self.num_envs, 1))
        self.polynomial_coefficients[:, 4, :len(self.cfg.polynomial_pitch_coefficients)] = torch.tensor(self.cfg.polynomial_pitch_coefficients, device=self.device).tile((self.num_envs, 1))
        self.polynomial_coefficients[:, 5, :len(self.cfg.polynomial_yaw_coefficients)] = torch.tensor(self.cfg.polynomial_yaw_coefficients, device=self.device).tile((self.num_envs, 1))

        self.polynomial_roll_rand_ranges = torch.tensor(self.cfg.polynomial_roll_rand_ranges, device=self.device).float()
        self.polynomial_pitch_rand_ranges = torch.tensor(self.cfg.polynomial_pitch_rand_ranges, device=self.device).float()
        self.polynomial_yaw_rand_ranges = torch.tensor(self.cfg.polynomial_yaw_rand_ranges, device=self.device).float()


    def _compute_motor_speeds(self, wrench_des: torch.Tensor) -> torch.Tensor:
        """Convert a desired wrench [T, Mx, My, Mz] (num_envs, 4) to desired motor speeds (num_envs, 4)."""
        f_des = torch.matmul(self.TM_to_f, wrench_des.t()).t()
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
        return motor_speeds_des


    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0) # clamp the actions to [-1, 1]

        # Propulsion actions occupy indices [0:4]:
        #   Action[0] = Collective Thrust (normalized)
        #   CTBM:  Action[1] = Mx, Action[2] = My, Action[3] = Mz

        # Collective thrust is always the same regardless of control mode
        self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self.cfg.thrust_to_weight)

        if self.cfg.control_mode == "CTBM":
            self._wrench_des[:, 1:3] = self._actions[:, 1:3] * self.cfg.moment_scale_xy
            self._wrench_des[:, 3] = self._actions[:, 3] * self.cfg.moment_scale_z
            
        else:
            raise NotImplementedError(f"Control mode {self.cfg.control_mode} is not implemented.")

        if self.cfg.use_motor_dynamics:
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        else:
            self._body_forces[:, 0, 2] = ((self._actions[:, 0] + 1.0) / 2.0) * (self._robot_weight * self.cfg.thrust_to_weight)
            self._body_moment[:, 0, :2] = self._actions[:, 1:3] * self.cfg.moment_scale_xy
            self._body_moment[:, 0, 2] = self._actions[:, 3] * self.cfg.moment_scale_z

        if self.cfg.num_joints > 0:
            self._joint_torques[:, self._shoulder_joint_idx] = self._actions[:, 4] * self.cfg.shoulder_torque_scalar
        if self.cfg.num_joints > 1:
            self._joint_torques[:, self._wrist_joint_idx] = self._actions[:, 5] * self.cfg.wrist_torque_scalar


    def _apply_action(self):
        """
        Apply joint torques, then propagate motor dynamics and apply the resulting
        propulsion wrench to the vehicle body.
        """
        if self.cfg.num_joints > 0:
            self._robot.set_joint_effort_target(self._joint_torques[:,self._shoulder_joint_idx], joint_ids=self._shoulder_joint_idx)
        if self.cfg.num_joints > 1:
            self._robot.set_joint_effort_target(self._joint_torques[:,self._wrist_joint_idx], joint_ids=self._wrist_joint_idx)


        # First-order motor speed dynamics: tau_m * d(omega)/dt = omega_des - omega
        if self.cfg.use_motor_dynamics:
            motor_accel = (1.0 / self.cfg.tau_m) * (self._motor_speeds_des - self._motor_speeds)
            self._motor_speeds += motor_accel * self.physics_dt
            self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
            # Convert actual motor speeds to wrench and apply to vehicle body
            motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
            # if self.cfg.gc_mode:
            #     # assume we can get the motor speeds to be exact
            #     motor_forces = self._motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
            #     motor_forces = self.cfg.k_eta * motor_forces ** 2
            wrench = torch.matmul(self.f_to_TM, motor_forces.t()).t()

            self._body_forces[:, 0, 2] = wrench[:, 0]
            self._body_moment[:, 0, :] = wrench[:, 1:]

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._body_forces, torques=self._body_moment
        )

    def _apply_curriculum(self):
        """
        Apply the curriculum to the environment.
        """
        # print("[Isaac Env: Curriculum] Total Timesteps: ", total_timesteps, " Pos Radius: ", self.cfg.pos_radius)
        iteration = self.common_step_counter // self.cfg.num_steps_per_env
        if self.cfg.ee_pos_radius_curriculum > 0:
            # half the pos radius every pos_radius_curriculum timesteps
            self.ee_pos_radius = max(self.cfg.ee_pos_radius_start * (0.5 ** (iteration // self.cfg.ee_pos_radius_curriculum)), 0.01)
        if self.cfg.body_pos_radius_curriculum > 0:
            # half the pos radius every pos_radius_curriculum timesteps
            self.body_pos_radius = max(self.cfg.body_pos_radius_start * (0.5 ** (iteration // self.cfg.body_pos_radius_curriculum)), 0.1)
        if self.cfg.ori_radius_curriculum > 0:
            self.ori_radius = max(self.cfg.ori_radius_start * (0.5 ** (iteration // self.cfg.ori_radius_curriculum)), 1e-3)
        if self.cfg.yaw_radius_curriculum > 0:
            self.yaw_radius = max(self.cfg.yaw_radius_start * (0.5 ** (iteration // self.cfg.yaw_radius_curriculum)), 0.1)
        if self.cfg.shoulder_radius_curriculum > 0:
            self.shoulder_radius = max(self.cfg.shoulder_radius_start * (0.5 ** (iteration // self.cfg.shoulder_radius_curriculum)), 0.1)
        if self.cfg.wrist_radius_curriculum > 0:
            self.wrist_radius = max(self.cfg.wrist_radius_start * (0.5 ** (iteration // self.cfg.wrist_radius_curriculum)), 0.1)
        if self.cfg.lin_vel_radius_curriculum > 0:
            self.lin_vel_radius = max(self.cfg.lin_vel_radius_start * (0.5 ** (iteration // self.cfg.lin_vel_radius_curriculum)), 0.01)
        if self.cfg.ang_vel_radius_curriculum > 0:
            self.ang_vel_radius = max(self.cfg.ang_vel_radius_start * (0.5 ** (iteration // self.cfg.ang_vel_radius_curriculum)), 0.01)
        if self.cfg.body_ang_vel_radius_curriculum > 0:
            self.body_ang_vel_radius = max(self.cfg.body_ang_vel_radius_start * (0.5 ** (iteration // self.cfg.body_ang_vel_radius_curriculum)), 0.01)
        if self.cfg.joint_vel_radius_curriculum > 0:
            self.joint_vel_radius = max(self.cfg.joint_vel_radius_start * (0.5 ** (iteration // self.cfg.joint_vel_radius_curriculum)), 0.01)
        if self.cfg.action_delta_prop_radius_curriculum > 0:
            self.action_delta_prop_radius = max(self.cfg.action_delta_prop_radius_start * (0.5 ** (iteration // self.cfg.action_delta_prop_radius_curriculum)), 0.01)
        if self.cfg.action_delta_joint_radius_curriculum > 0:
            self.action_delta_joint_radius = max(self.cfg.action_delta_joint_radius_start * (0.5 ** (iteration // self.cfg.action_delta_joint_radius_curriculum)), 0.01)



        
    def update_goal_state(self):
        env_ids = (self.episode_length_buf % int(self.cfg.traj_update_dt*self.cfg.policy_rate_hz)== 0).nonzero(as_tuple=False)
        # print("Env IDs: ", env_ids, env_ids.squeeze(1))
        
        if len(env_ids) == 0 or env_ids.size(0) == 0:
            return
        

        current_time = self.episode_length_buf[env_ids]
        future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        # future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        time = (current_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt

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
            pos_traj, roll_traj, pitch_traj, yaw_traj = (
                traj_utils.eval_lissajous_curve_6dof(
                    time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4
                    )
            )
        elif self.cfg.trajectory_type == "polynomial":
            # TODO: add roll and pitch polynomials to poly curves
            pos_traj, yaw_traj = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
        elif self.cfg.trajectory_type == "combined":
            pos_lissajous, roll_lissajous, pitch_lissajous, yaw_lissajous = (
                traj_utils.eval_lissajous_curve_6dof(
                    time, self.lissajous_amplitudes, self.lissajous_frequencies, self.lissajous_phases, self.lissajous_offsets, derivatives=4
                )
            )
            # TODO: add roll and pitch polynomials to poly curves, otherwise this will break
            pos_poly, roll_poly, pitch_poly, yaw_poly = traj_utils.eval_polynomial_curve(time, self.polynomial_coefficients, derivatives=4)
            pos_traj = pos_lissajous + pos_poly
            roll_traj = roll_lissajous + roll_poly
            pitch_traj = pitch_lissajous + pitch_poly
            yaw_traj = yaw_lissajous + yaw_poly

            # print("Poly coefficients: ", self.polynomial_coefficients[0, :, :])
            # print("Pos Poly: ", pos_poly[:2, 0, :, 0])
            # print("Yaw poly: ", yaw_poly[:2, 0, 0])
        else:
            raise NotImplementedError("Trajectory type not implemented")
    
        self._pos_traj = pos_traj
        self._roll_traj = roll_traj
        self._pitch_traj = pitch_traj
        self._yaw_traj = yaw_traj
        if self.cfg.random_shift_trajectory:
            # Ensure the shapes are compatible for broadcasting
            pos_shift = self._pos_shift.unsqueeze(-1)
            roll_shift = self._roll_shift.unsqueeze(-1)
            pitch_shift = self._pitch_shift.unsqueeze(-1)
            yaw_shift = self._yaw_shift

            pos_traj[0, :, :, :] += pos_shift
            roll_traj[0, :, :] += roll_shift
            pitch_traj[0, :, :] += pitch_shift
            yaw_traj[0, :, :] += yaw_shift

        # we need to switch the last two dimensions of pos_traj since the _desired_pos_w is of shape (num_envs, horizon, 3) instead of (num_envs, 3, horizon)
        # print(self._desired_pos_traj_w.shape, pos_traj[0,env_ids.squeeze(1)].shape)
        self._desired_pos_traj_w[env_ids.squeeze(1)] = (pos_traj[0,env_ids.squeeze(1)]).transpose(1,2)
        # self._desired_pos_traj_w[env_ids.squeeze(1),:, :2] += self._terrain.env_origins[env_ids, :2] # shift the trajectory to the correct position for each environment
        # we need to convert from the yaw angle to a quaternion representation
        # print("Yaw Traj: ", yaw_traj[0, 0, :2])
        self._desired_ori_traj_w[env_ids.squeeze(1)] = quat_from_euler_xyz(
            roll_traj[0,env_ids.squeeze(1)],
            pitch_traj[0,env_ids.squeeze(1)],
            yaw_traj[0,env_ids.squeeze(1)],
        )
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
        

        # For IK, we need to calculate the required body position for the next goal
        # print("Shapes:")
        # print("Desired Ori: ", self._desired_ori_w.shape)
        # print("Desired Pos: ", self._desired_pos_w.shape)
        # print("Desired Body Pos: ", self._desired_body_pos.shape)
        # print("Arm Length: ", self.arm_length.shape)
        # print("Env IDs: ", env_ids.shape)
        # print("Desired Body Pos: ", self._desired_body_pos.shape)
        # print("Desired Body Pos: ", self._desired_body_pos.shape)
        self._desired_body_pos = calculate_required_pos(self._desired_ori_w, self._desired_pos_w, self._desired_body_pos, self.arm_length, env_ids.squeeze(1))
        self._desired_body_pos_traj = calculate_required_pos(self._desired_ori_traj_w, self._desired_pos_traj_w, self._desired_body_pos_traj, self.arm_length, env_ids.squeeze(1))
        self._desired_com_pos = calculate_required_pos(self._desired_ori_w, self._desired_pos_w, self._desired_com_pos, self.com_offset, env_ids)
        self._desired_com_pos_traj = calculate_required_pos(self._desired_ori_traj_w, self._desired_pos_traj_w, self._desired_com_pos_traj, self.com_offset, env_ids)
        # print("0th env: ", self._desired_pos_w[0], self._desired_ori_w[0])
        # print("[Isaac Env: Update Goal State] Desired Pos: ", self._desired_pos_w[env_ids[:5,0]])
        

    def _get_observations(self) -> torch.Dict[str, torch.Tensor | torch.Dict[str, torch.Tensor]]:
        """
        Returns the observation dictionary. Policy observations are in the key "policy".
        """
        self._apply_curriculum()
        self.update_goal_state()
        
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.goal_body)


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

        # wrist_error = wrist_angle_error_from_quats(base_ori_w, goal_ori_w)

        # Get vehicle frame info
        body_pos_w, body_ori_w, body_lin_vel_w, body_ang_vel_w = self.get_frame_state_from_task("vehicle")
        com_pos_w, com_ori_w, com_lin_vel_w, com_ang_vel_w = self.get_frame_state_from_task("COM")

        # For quad body, we only care about the position error, so can use any orientation for calculating the frame transform
        body_pos_error, _ = subtract_frame_transforms(body_pos_w, body_ori_w,
                                                          self._desired_body_pos, body_ori_w)
        
        # goal in body frame
        _, goal_to_body_ori = subtract_frame_transforms(body_pos_w, body_ori_w, goal_pos_w, goal_ori_w)
        goal_to_body_pos, _ = subtract_frame_transforms(body_pos_w, base_ori_w, goal_pos_w, goal_ori_w) # hack trying for now - express this in ee axes

        # end effector to body transfrom
        ee_to_body_pos, ee_to_body_ori = subtract_frame_transforms(body_pos_w, body_ori_w, base_pos_w, base_ori_w)
        body_to_ee_pos, body_to_ee_ori = subtract_frame_transforms(base_pos_w, base_ori_w, body_pos_w, body_ori_w)

        # com_pos_error, _ = subtract_frame_transforms(com_pos_w, body_ori_w, self._desired_com_pos, body_ori_w)

        # body_pos_error_2, _ = self.convert_ee_goal_from_task(goal_pos_w, goal_ori_w, "vehicle")
        # body_pos_error_2 = quat_apply_inverse(body_ori_w, body_pos_error_2)
        # print(body_pos_error[:5])
        # print('-'*20)
        # print(body_pos_error_2[:5])

        # End effector in body frame
        # ee_pos_body, ee_ori_body = subtract_frame_transforms(body_pos_w, body_ori_w,
        #                                                   base_pos_w, base_ori_w)

        # body_roll, body_pitch, _ = euler_xyz_from_quat(body_ori_w)
        # body_roll = torch.reshape(body_roll, (-1, 1))
        # body_pitch = torch.reshape(body_pitch, (-1, 1))

        yaw_error, shoulder_error, wrist_error = aerial_manipulator_angle_errors(base_ori_w, goal_ori_w)


        future_pos_error_b = []
        future_ori_error_b = []
        future_body_pos_error_b = []
        # future_com_pos_error_b = []
        # for i in range(self.cfg.trajectory_horizon):
        #     goal_pos_traj_w, goal_ori_traj_w = self.convert_ee_goal_from_task(self._desired_pos_traj_w[:, i+1].squeeze(1), self._desired_ori_traj_w[:, i+1].squeeze(1), self.cfg.goal_body)
        #     # goal_body_pos_traj_w, _ = self.convert_ee_goal_from_task(self._desired_pos_traj_w[:, i+1].squeeze(1), self._desired_ori_traj_w[:, i+1].squeeze(1), "vehicle")
        #     waypoint_pos_error_b, waypoint_ori_error_b = subtract_frame_transforms(base_pos_w, base_ori_w, goal_pos_traj_w, goal_ori_traj_w)
        #     waypoint_body_pos_error_b, _ = subtract_frame_transforms(body_pos_w, body_ori_w,
        #         self._desired_body_pos_traj[:, i+1].squeeze(1), body_ori_w
        #     )
        #     # waypoint_body_pos_error_b, _ = subtract_frame_transforms(body_pos_w, body_ori_w,
        #         # self._desired_pos_traj_w[:, i+1].squeeze(1), body_ori_w
        #     # )
        #     # waypoint_com_pos_error_b, _ = subtract_frame_transforms(
        #     #     com_pos_w, body_ori_w,self._desired_com_pos_traj[:, i+1].squeeze(1), body_ori_w
        #     # )
        #     future_pos_error_b.append(waypoint_pos_error_b) # append (n, 3) tensor
        #     future_ori_error_b.append(waypoint_ori_error_b) # append (n, 4) tensor
        #     future_body_pos_error_b.append(waypoint_body_pos_error_b)
        #     # future_com_pos_error_b.append(waypoint_com_pos_error_b)
        # if len(future_pos_error_b) > 0:
        #     future_pos_error_b = torch.stack(future_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor
        #     future_ori_error_b = torch.stack(future_ori_error_b, dim=1) # stack to (n, horizon, 4) tensor
        #     future_body_pos_error_b = torch.stack(future_body_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor
        #     # future_com_pos_error_b = torch.stack(future_com_pos_error_b, dim=1) # stack to (n, horizon, 3) tensor
        #     if self.cfg.use_yaw_representation_for_trajectory:
        #         future_ori_error_b = math_utils.yaw_from_quat(future_ori_error_b).reshape(self.num_envs, self.cfg.trajectory_horizon, 1)
        if self.cfg.trajectory_horizon > 0:
            future_pos_error_b, future_ori_error_b = subtract_frame_transforms(
                base_pos_w.unsqueeze(1).repeat(1, self.cfg.trajectory_horizon, 1),
                base_ori_w.unsqueeze(1).repeat(1, self.cfg.trajectory_horizon, 1),
                self._desired_pos_traj_w[:, 1:],
                self._desired_ori_traj_w[:, 1:]
            )
            future_ori_error_b = quat_unique(future_ori_error_b)
            future_body_pos_error_b, _ = subtract_frame_transforms(
                body_pos_w.unsqueeze(1).repeat(1, self.cfg.trajectory_horizon, 1),
                body_ori_w.unsqueeze(1).repeat(1, self.cfg.trajectory_horizon, 1),
                self._desired_body_pos_traj[:, 1:],
                body_ori_w.unsqueeze(1).repeat(1, self.cfg.trajectory_horizon, 1)
            )
        else:
            future_pos_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
            future_ori_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 4, device=self.device)
            future_body_pos_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
            # future_com_pos_error_b = torch.zeros(self.num_envs, self.cfg.trajectory_horizon, 3, device=self.device)
        

        # Compute the orientation error as a yaw error in the body frame
        # goal_yaw_w = yaw_quat(self._desired_ori_w)
        # goal_yaw_w = yaw_quat(goal_ori_w)
        # current_yaw_w = yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        # yaw_error_w = yaw_error_from_quats(current_yaw_w, goal_yaw_w, dof=self.cfg.num_joints).view(self.num_envs, 1)
        
        # if self.cfg.use_yaw_representation:
        #     yaw_representation = yaw_error_w
        # else:
        #     yaw_representation = torch.zeros(self.num_envs, 0, device=self.device)
        
        if self.cfg.use_full_ori_matrix:
            ori_representation_b = matrix_from_quat(ori_error_b).flatten(-2, -1)
            body_ori_representation = matrix_from_quat(body_ori_w).flatten(-2, -1)
            goal_to_body_ori_representation = matrix_from_quat(goal_to_body_ori).flatten(-2, -1)
            ee_to_body_ori_representation = matrix_from_quat(ee_to_body_ori).flatten(-2, -1)
        else:
            ori_representation_b = quat_unique(ori_error_b)
            body_ori_representation = quat_unique(body_ori_w)
            goal_to_body_ori_representation = quat_unique(goal_to_body_ori)
            ee_to_body_ori_representation = quat_unique(ee_to_body_ori)

        if self.cfg.use_grav_vector:
            grav_vector_b = quat_apply_inverse(base_ori_w, self._grav_vector_unit) # projected gravity vector in the cfg frame
        else:
            grav_vector_b = torch.zeros(self.num_envs, 0, device=self.device)
        
        # Compute the linear and angular velocities of the end-effector in body frame
        if self.cfg.trajectory_horizon > 0 and not self.cfg.gc_mode:
            lin_vel_des = self._pos_traj[1, :, :, 0]
            lin_vel_error_w = lin_vel_des - lin_vel_w
            # Also make have the body velocity be relative to the target
            body_lin_vel_error_w = lin_vel_des - body_lin_vel_w
        else:
            lin_vel_des = torch.zeros_like(lin_vel_w, device=self.device)
            lin_vel_error_w = lin_vel_w
            body_lin_vel_error_w = body_lin_vel_w

        lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_error_w)
        if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0 and not self.cfg.gc_mode:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_des[:, 0] = self._roll_traj[1, :, 0]
            ang_vel_des[:, 1] = self._pitch_traj[1, :, 0]
            ang_vel_des[:, 2] = self._yaw_traj[1, :, 0]
            ang_vel_error_w = ang_vel_des - ang_vel_w
            body_ang_vel_error_w = ang_vel_des - body_ang_vel_w
        else:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_error_w = ang_vel_w
            body_ang_vel_error_w = body_ang_vel_w
        ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_error_w)

        # Component of the desired angular velocity vector about the shoulder actuation axis (quadrotor forward axis)
        axis = quat_apply(body_ori_w, torch.tensor([[1.0, 0.0, 0.0]], device=body_ori_w.device).tile((body_ori_w.shape[0], 1)))
        ang_vel_des_shoulder = (ang_vel_des * axis).sum(dim=-1, keepdim=True)

        # Component of the desired angular velocity vector about the wrist actuation axis (end effector y-axis)
        axis = quat_apply(base_ori_w, torch.tensor([[0.0, 1.0, 0.0]], device=base_ori_w.device).tile((base_ori_w.shape[0], 1)))
        ang_vel_des_wrist = (ang_vel_des * axis).sum(dim=-1, keepdim=True)

        # Do the same for the body frame
        body_lin_vel_b = quat_apply_inverse(body_ori_w, body_lin_vel_w)
        body_ang_vel_b = quat_apply_inverse(body_ori_w, body_ang_vel_w)
        body_lin_vel_error_b = quat_apply_inverse(body_ori_w, body_lin_vel_error_w)
        body_ang_vel_error_b = quat_apply_inverse(body_ori_w, body_ang_vel_error_w)

        # Given the pose of the body relative to the end effector, compute the required linear velocity of the body (function also gives
        # angular velocity but rigid body assumption does not hold)
        # body_lin_vel_b_des, _ = rigid_body_twist_transform(lin_vel_des, ang_vel_des, body_to_ee_pos, body_to_ee_ori)
        # true_body_lin_vel_error_b = body_lin_vel_b_des - body_lin_vel_b

        # Compute the joint states
        shoulder_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        shoulder_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_pos = torch.zeros(self.num_envs, 0, device=self.device)
        wrist_joint_vel = torch.zeros(self.num_envs, 0, device=self.device)
        if self.cfg.num_joints > 0:
            shoulder_joint_pos = wrap_to_pi(self._robot.data.joint_pos[:, self._shoulder_joint_idx].unsqueeze(1))
            shoulder_joint_vel = self._robot.data.joint_vel[:, self._shoulder_joint_idx].unsqueeze(1)
        if self.cfg.num_joints > 1:
            wrist_joint_pos = wrap_to_pi(self._robot.data.joint_pos[:, self._wrist_joint_idx].unsqueeze(1))
            wrist_joint_vel = self._robot.data.joint_vel[:, self._wrist_joint_idx].unsqueeze(1)

        shoulder_vel_error = ang_vel_des_shoulder - shoulder_joint_vel
        wrist_vel_error = ang_vel_des_wrist - wrist_joint_vel

        # Embeddings
        shoulder_joint_pos_embedding = torch.cat([torch.cos(shoulder_joint_pos), torch.sin(shoulder_joint_pos)], dim=-1)
        wrist_joint_pos_embedding = torch.cat([torch.cos(wrist_joint_pos), torch.sin(wrist_joint_pos)], dim=-1)

        # Previous Action
        if self.cfg.use_previous_actions:
            previous_actions = self._previous_actions
        else:
            previous_actions = torch.zeros(self.num_envs, 0, device=self.device)

        if self.cfg.use_previous_velocities:
            previous_velocities = self._previous_velocity_obs
        else:
            previous_velocities = torch.zeros(self.num_envs, 0, device=self.device)

        obs = torch.cat(
            [
                pos_error_b,                                # (num_envs, 3) [0-2]
                # ori_error_b,                                # (num_envs, 3) [6-8]
                ori_representation_b,                     # (num_envs, 0) if not using full ori matrix, (num_envs, 9)
                # yaw_error,
                # shoulder_error,
                # wrist_error,
                # body_pos_error,                             # (num_envs, 3) [3-5]
                goal_to_body_pos,
                body_ori_representation,                                 # (num_envs, 4) [6-9]
                # goal_to_body_ori_representation,
                ee_to_body_ori_representation,
                # yaw_representation,                         # (num_envs, 4) if using yaw representation (quat), 0 otherwise (0 for 2DOF)
                # goal_ori_w_flattened_matrix,                # [12-20]
                grav_vector_b,                              # (num_envs, 3) if using gravity vector, 0 otherwise [21-23]
                lin_vel_b,                                  # (num_envs, 3) [24-26]
                ang_vel_b,                                  # (num_envs, 3) [27-29]
                body_lin_vel_b,
                # body_lin_vel_error_b,
                body_ang_vel_b,
                previous_velocities,
                # body_ang_vel_error_b,
                # wrist_error,                                # (num_envs, 1) [31]
                # shoulder_joint_pos,                         # (num_envs, 1) [30]
                # wrist_joint_pos,                            # (num_envs, 1) [34]
                # shoulder_joint_pos_embedding,
                # wrist_joint_pos_embedding,
                # wrist_error,  
                shoulder_joint_vel,                         # (num_envs, 1) [32]
                wrist_joint_vel,                            # (num_envs, 1) [33]
                # shoulder_vel_error,
                # wrist_vel_error,
                previous_actions,
                future_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                future_ori_error_b.flatten(-2, -1),          # (num_envs, horizon * 4) if use_yaw_representation_for_trajectory, else (num_envs, horizon, 1)
                # future_body_pos_error_b.flatten(-2, -1),     # (num_envs, horizon * 3)
            ],
            dim=-1                                          # (num_envs, 22 + 7*horizon + 3*horizon)
        )

        # Additional critic observations
        if self.cfg.num_joints == 2:
            # critic_obs = obs
            # give access to the randomized masses
            critic_obs = torch.cat(
                [
                    obs,
                    # pos_error_b,                                # (num_envs, 3) [0-2]
                    # ori_representation_b,                       # (num_envs, 0) if not using full ori matrix, (num_envs, 9) if using full ori matrix
                    # com_pos_error,
                    # body_pos_error,
                    # true_body_lin_vel_error_b,
                    # body_ori_representation,
                    # body_ori_error_b,
                    # yaw_representation,                         # (num_envs, 4) if using yaw representation (quat), 0 otherwise
                    # grav_vector_b,                              # (num_envs, 3) if using gravity vector, 0 otherwise
                    # lin_vel_b,                                  # (num_envs, 3)
                    # ang_vel_b,                                  # (num_envs, 3)
                    # body_lin_vel_b,
                    # body_ang_vel_b,                    
                    # shoulder_joint_pos,                         # (num_envs, 1)
                    # wrist_joint_pos,                            # (num_envs, 1)
                    # shoulder_joint_pos_embedding,
                    # wrist_joint_pos_embedding,
                    # yaw_error,
                    # shoulder_error,
                    # wrist_error,
                    # shoulder_joint_vel,                         # (num_envs, 1)
                    # wrist_joint_vel,
                    # shoulder_vel_error,
                    # wrist_vel_error,
                    # previous_actions,
                    # future_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                    # future_ori_error_b.flatten(-2, -1),          # (num_envs, horizon * 4) if use_yaw_representation_for_trajectory, else (num_envs, horizon, 1)
                    # future_body_pos_error_b.flatten(-2, -1),     # (num_envs, horizon * 3)
                    # future_com_pos_error_b.flatten(-2, -1),         # (num_envs, horizon * 3)
                    self._robot.root_physx_view.get_masses()[:, self._body_id].to(self.device), # body
                    self._robot.root_physx_view.get_masses()[:, self._ee_com_id].to(self.device), # arm
                    self._robot.root_physx_view.get_masses()[:, self._ee_id].to(self.device), # mass at end effector
                ],
                dim=-1
            )

        self._previous_velocity_obs = torch.cat(
            [
                body_lin_vel_b,
                body_ang_vel_b,
                shoulder_joint_vel,
                wrist_joint_vel,
            ],
            dim=-1
        )
        
        # We also need the state information for other controllers like the decoupled controller.
        # This is the full state of the robot
        # print("[Isaac Env: Observations] \"Frame\" Pos: ", base_pos_w)
        # quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("vehicle")
        quad_pos_w, quad_ori_w, quad_lin_vel_w, quad_ang_vel_w = self.get_frame_state_from_task("COM")
        ee_pos_w, ee_ori_w, ee_lin_vel_w, ee_ang_vel_w = self.get_frame_state_from_task("endeffector")
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

            # com_pos_w, com_ori_w, com_lin_vel_w, com_ang_vel_w = self.get_frame_state_from_task("COM")

            self.last_yaw_cmd, shoulder_req, wrist_req = aerial_manipulator_angle_solns_2dof(self.model_ee_ori.tile((self.num_envs, 1)), goal_ori_w, self.last_yaw_cmd)
            shoulder_error_2 = wrap_to_pi(shoulder_joint_pos - shoulder_req)
            wrist_error_2 = wrap_to_pi(wrist_joint_pos - wrist_req)
            gc_obs = torch.cat(
                [
                    # com_pos_w,
                    body_pos_w,
                    body_ori_w,
                    body_lin_vel_w,
                    body_ang_vel_b,
                    com_pos_w,
                    com_lin_vel_w,
                    self._desired_com_pos,
                    self._desired_body_pos,
                    # g
                    # oal_ori_w,
                    # goal_yaw_w.unsqueeze(1),
                    # yaw_from_quat(goal_ori_w).unsqueeze(1),
                    self.last_yaw_cmd,
                    shoulder_joint_pos,
                    wrist_joint_pos,
                    shoulder_joint_vel,
                    wrist_joint_vel,
                    shoulder_error_2,
                    wrist_error_2, 
                    self._desired_com_pos_traj.flatten(-2, -1), # (num_envs, (horizon + 1) * 3), +1 for current term
                    self._desired_ori_traj_w.flatten(-2, -1), # (num_envs, (horizon + 1) * 4)
                ],
                dim=-1
            )
        else:
            gc_obs = None

        if self.cfg.eval_mode:
            pos_traj = self._pos_traj[:3,:,:,0].permute(1,0,2).reshape(self.num_envs, -1)
            yaw_traj = self._yaw_traj[:2,:,0].permute(1,0).reshape(self.num_envs, -1)
            if not self.cfg.gc_mode:
                # Keep track of what the ideal angles should (almost) be for debugging
                self.last_yaw_cmd, shoulder_req, wrist_req = aerial_manipulator_angle_solns_2dof(self.model_ee_ori.tile((self.num_envs, 1)), goal_ori_w, self.last_yaw_cmd)
            full_state = torch.cat(
                [
                    quad_pos_w,                                 # (num_envs, 3) [0-2]
                    quad_ori_w,                                 # (num_envs, 4) [3-6]
                    quad_lin_vel_w,                             # (num_envs, 3) [7-9]
                    quad_ang_vel_w,                             # (num_envs, 3) [10-12]
                    ee_pos_w,                                   # (num_envs, 3) [13-15]
                    ee_ori_w,                                   # (num_envs, 4) [16-19]
                    ee_lin_vel_w,                               # (num_envs, 3) [20-22]
                    ee_ang_vel_w,                               # (num_envs, 3) [23-25]
                    shoulder_joint_pos,                         # (num_envs, 1) [26] 
                    wrist_joint_pos,                            # (num_envs, 1) [27]
                    shoulder_joint_vel,                         # (num_envs, 1) [28]
                    wrist_joint_vel,                            # (num_envs, 1) [29]
                    self._desired_pos_w,                        # (num_envs, 3) [30-32] 
                    self._desired_ori_w,                        # (num_envs, 4) [33-36] 
                    self._actions,                              # (num_envs, 6) [37-42]
                    wrist_error,                                 # (num_envs, 1) [43]
                    pos_error_b,                                # (num_envs, 3) [44-46]
                    body_pos_error,                             # (num_envs, 3) [47-49]
                    yaw_error,                                  # (num_envs, 1) [50]
                    shoulder_error,                             # (num_envs, 1) [51]
                    ang_vel_error_w,                            # (num_envs, 3) [52-54]
                    ang_vel_b,                                  # (num_envs, 3) [55-57]
                    self.last_yaw_cmd,
                    shoulder_req,
                    wrist_req,
                    # pos_traj,                                   # (num_envs, 3 * (horizon + 1)) [52-54] 
                    # yaw_traj,                                   # (num_envs, (horizon + 1)) [54-56]
                ],
                dim=-1                                          # (num_envs, 61)
            )
            self._state = full_state
        else:
            full_state = None

        return {"policy": obs, "gc": gc_obs, "full_state": full_state, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        Returns the reward tensor.
        """
        base_pos_w, base_ori_w, lin_vel_w, ang_vel_w = self.get_frame_state_from_task(self.cfg.reward_task_body)
        goal_pos_w, goal_ori_w = self.get_goal_state_from_task(self.cfg.reward_goal_body)
        
        body_pos_w , _, _, body_ang_vel_w = self.get_frame_state_from_task("vehicle")
        # com_pos_w, _, _, _ = self.get_frame_state_from_task("COM")
        # Computes the error from the desired position and orientation
        if self.cfg.num_joints != 2:
            body_pos_error = torch.linalg.norm(goal_pos_w - base_pos_w, dim=1)
            ee_pos_error = torch.zeros_like(body_pos_error)
        else:
            ee_pos_error = torch.linalg.norm(goal_pos_w - base_pos_w, dim=1)
            body_pos_error = torch.linalg.norm(self._desired_body_pos - body_pos_w, dim=1)
            # body_pos_error = torch.linalg.norm(com_pos_w - self._desired_com_pos, dim=1)
        if self.cfg.square_pos_error:
            ee_pos_distance = torch.exp(- (ee_pos_error **2) / self.ee_pos_radius)
            body_pos_distance = torch.exp(- (body_pos_error **2) / self.body_pos_radius)
        else:
            ee_pos_distance = torch.exp(- (ee_pos_error) / self.ee_pos_radius)
            body_pos_distance = torch.exp(- (body_pos_error) / self.body_pos_radius)


        ori_error = quat_error_magnitude(goal_ori_w, base_ori_w)
        
        ori_distance = torch.exp(-(ori_error ** 2) / self.ori_radius)

        goal_yaw_w = yaw_quat(goal_ori_w)
        current_yaw_w = yaw_quat(base_ori_w)
        # yaw_error_w = quat_mul(quat_inv(current_yaw_w), goal_yaw_w)
        # yaw_error = quat_error_magnitude(yaw_error_w, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).tile((self.num_envs, 1)))

        smooth_transition_func = 1.0 - torch.exp(-1.0 / torch.max(self.cfg.yaw_smooth_transition_scale*ee_pos_error - 10.0, torch.zeros_like(ee_pos_error)))

        # other_yaw_error = yaw_error_from_quats(goal_yaw_w, current_yaw_w, self.cfg.num_joints).unsqueeze(1)
        yaw_error = yaw_error_from_quats(goal_ori_w, base_ori_w, self.cfg.num_joints).unsqueeze(1)
        # other_yaw_error = torch.sum(torch.square(other_yaw_error), dim=1)
        yaw_error = torch.linalg.norm(yaw_error, dim=1)

        # yaw_distance = (1.0 - torch.tanh(yaw_error / self.cfg.yaw_radius)) * smooth_transition_func
        yaw_distance = torch.exp(- (yaw_error **2) / self.yaw_radius)
        yaw_error = yaw_error * smooth_transition_func

        # combined_error = (pos_error)**2 + (yaw_error * self.arm_length)**2
        combined_error = ee_pos_error/self.cfg.goal_pos_range + (yaw_error/self.cfg.goal_yaw_range)*self.arm_length
        combined_reward = (1 + torch.exp(self.cfg.combined_alpha * (combined_error - self.cfg.combined_tolerance)))**-1
        combined_distance = combined_reward

        # Axis reward as dot product of goal axis and current axis
        axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).tile((self.num_envs, 1))
        goal_axis = quat_apply(goal_ori_w, axis)
        current_axis = quat_apply(base_ori_w, axis)
        axis_reward = (goal_axis * current_axis).sum(dim=1)

        # More detailed joint error components
        if self.cfg.num_joints == 2:
            yaw_error, shoulder_joint_error, wrist_joint_error = aerial_manipulator_angle_errors(base_ori_w, goal_ori_w)
            yaw_error = yaw_error.abs().squeeze()
            shoulder_joint_error = shoulder_joint_error.abs().squeeze()
            wrist_joint_error = wrist_joint_error.abs().squeeze()
            shoulder_joint_distance = torch.exp(- (shoulder_joint_error **2) / self.shoulder_radius)
            wrist_joint_distance = torch.exp(- (wrist_joint_error **2) / self.wrist_radius)

        # Copied from hover - don't actually use this but needed for shapes when concatenating rewards:
        # yaw_error = torch.linalg.norm(yaw_error, dim=1)
        yaw_distance = torch.exp(- (yaw_error **2) / self.yaw_radius)
        yaw_error = yaw_error * smooth_transition_func

        # Velocity error components, used for stabliization tuning
        if self.cfg.trajectory_horizon > 0:
            lin_vel_error_w = self._pos_traj[1, :, :, 0] - lin_vel_w
        else:
            lin_vel_error_w = torch.zeros_like(lin_vel_w, device=self.device) - lin_vel_w
        lin_vel_b = quat_apply_inverse(base_ori_w, lin_vel_error_w)
        if self.cfg.use_ang_vel_from_trajectory and self.cfg.trajectory_horizon > 0:
            ang_vel_des = torch.zeros_like(ang_vel_w)
            ang_vel_des[:, 0] = self._roll_traj[1, :, 0]
            ang_vel_des[:, 1] = self._pitch_traj[1, :, 0]
            ang_vel_des[:, 2] = self._yaw_traj[1, :, 0]
            ang_vel_error_w = ang_vel_des - ang_vel_w
        else:
            ang_vel_error_w = torch.zeros_like(ang_vel_w) - ang_vel_w
        ang_vel_b = quat_apply_inverse(base_ori_w, ang_vel_error_w)
        # lin_vel_error = torch.linalg.norm(lin_vel_b, dim=-1)
        # ang_vel_error = torch.linalg.norm(ang_vel_b, dim=-1)
        # lin_vel_error = torch.sum(torch.square(lin_vel_b), dim=1)
        lin_vel_error = torch.norm(lin_vel_b, dim=1)
        # ang_vel_error = torch.sum(torch.square(ang_vel_b), dim=1)
        ang_vel_error = torch.norm(ang_vel_b, dim=1)
        # experiment with incorporatin the desired yaw rate 
        body_ang_vel_des = torch.zeros_like(body_ang_vel_w)
        # body_ang_vel_des[:, 2] = self._yaw_traj[1, :, 0]
        body_ang_vel_error = torch.norm(body_ang_vel_w - body_ang_vel_des, dim=1)
        # body_ang_vel_b = quat_apply_inverse(base_ori_w, body_ang_vel_w)
        joint_vel_error = torch.norm(self._robot.data.joint_vel, dim=1)
        
        # Curriculum-based distance rewards for velocity tracking
        if self.cfg.square_pos_error:
            lin_vel_distance = torch.exp(-(lin_vel_error ** 2) / self.lin_vel_radius)
            ang_vel_distance = torch.exp(-(ang_vel_error ** 2) / self.ang_vel_radius)
            body_ang_vel_distance = torch.exp(-(body_ang_vel_error ** 2) / self.body_ang_vel_radius)
            joint_vel_distance = torch.exp(-(joint_vel_error ** 2) / self.joint_vel_radius)
        else:
            lin_vel_distance = torch.exp(-lin_vel_error / self.lin_vel_radius)
            ang_vel_distance = torch.exp(-ang_vel_error / self.ang_vel_radius)
            body_ang_vel_distance = torch.exp(-body_ang_vel_error / self.body_ang_vel_radius)
            joint_vel_distance = torch.exp(-joint_vel_error / self.joint_vel_radius)

        actions = self._actions.clone()
        # For the thrust component of the action, rescale it since an action of -1 corresponds to a thrust of 0
        actions[:, 0] = 0.5 * actions[:, 0] + 0.5
        action_prop_error = torch.norm(actions[:, :4], dim=1)
        action_joint_error = torch.norm(actions[:, -2:], dim=1)
        action_norm_error = torch.norm(actions, dim=1)

        action_delta = self._actions - self._previous_actions
        action_delta_error = torch.norm(action_delta, dim=1)
        action_delta_prop_error = torch.norm(action_delta[:, :4], dim=1)
        action_delta_joint_error = torch.norm(action_delta[:, -2:], dim=1)
        jitter = torch.norm(action_delta[:, 1:3], dim=1)

        self._previous_actions = self._actions.clone()

        # Curriculum-based distance rewards for action delta
        if self.cfg.square_pos_error:
            action_delta_prop_distance = torch.exp(-(action_delta_prop_error ** 2) / self.action_delta_prop_radius)
            action_delta_joint_distance = torch.exp(-(action_delta_joint_error ** 2) / self.action_delta_joint_radius)
        else:
            action_delta_prop_distance = torch.exp(-action_delta_prop_error / self.action_delta_prop_radius)
            action_delta_joint_distance = torch.exp(-action_delta_joint_error / self.action_delta_joint_radius)

        if self.cfg.scale_reward_with_time:
            time_scale = 1.0 / self.cfg.policy_rate_hz
        else:
            time_scale = 1.0

        if self.cfg.square_reward_errors:
            # pos_error = pos_error ** 2
            # pos_distance = pos_distance ** 2
            # ori_error = ori_error ** 2
            # yaw_error = yaw_error ** 2
            # yaw_distance = yaw_distance ** 2
            # lin_vel_error = lin_vel_error ** 2
            # ang_vel_error = ang_vel_error ** 2
            # joint_vel_error = joint_vel_error ** 2
            # action_error = action_error ** 2
            # previous_action_error = previous_action_error ** 2
            # combined_distance = combined_distance ** 2

            # Copied from hover
            ee_pos_error = ee_pos_error ** 2
            ee_pos_distance = ee_pos_distance ** 2
            body_pos_error = body_pos_error ** 2
            body_pos_distance = body_pos_distance ** 2
            ori_error = ori_error ** 2
            yaw_error = yaw_error ** 2
            yaw_distance = yaw_distance ** 2
            lin_vel_error = lin_vel_error ** 2
            ang_vel_error = ang_vel_error ** 2
            body_ang_vel_error = body_ang_vel_error ** 2
            joint_vel_error = joint_vel_error ** 2
            lin_vel_distance = lin_vel_distance ** 2
            ang_vel_distance = ang_vel_distance ** 2
            body_ang_vel_distance = body_ang_vel_distance ** 2
            joint_vel_distance = joint_vel_distance ** 2
            action_delta_prop_distance = action_delta_prop_distance ** 2
            action_delta_joint_distance = action_delta_joint_distance ** 2
            action_error = action_error ** 2 # Artifact from hover, it's not being squared there either
            combined_distance = combined_distance ** 2
            # NOTE: didn't square the previous action error, that is in the traj tracking env though

        crash_penalty_time = self.cfg.crash_penalty * (self.max_episode_length - self.episode_length_buf)


        rewards = {
            "body_pos_error": body_pos_error * self.cfg.body_pos_error_reward_scale * time_scale,
            "body_pos_distance": body_pos_distance * self.cfg.body_pos_distance_reward_scale * time_scale,
            "endeffector_combined_error": combined_reward * self.cfg.combined_scale * time_scale,
            "endeffector_pos_error": ee_pos_error * self.cfg.ee_pos_error_reward_scale * time_scale,
            "endeffector_pos_distance": ee_pos_distance * self.cfg.ee_pos_distance_reward_scale * time_scale,
            "endeffector_ori_error": ori_error * self.cfg.ori_error_reward_scale * time_scale,
            "endeffector_ori_distance": ori_distance * self.cfg.ori_distance_reward_scale * time_scale,
            "body_yaw_error": yaw_error * self.arm_length * self.cfg.yaw_error_reward_scale * time_scale,
            "body_yaw_distance": yaw_distance * self.cfg.yaw_distance_reward_scale * time_scale,
            "endeffector_lin_vel": lin_vel_error * self.cfg.lin_vel_reward_scale * time_scale,
            "endeffector_lin_vel_distance": lin_vel_distance * self.cfg.lin_vel_distance_reward_scale * time_scale,
            "endeffector_ang_vel": ang_vel_error * self.arm_length * self.cfg.ang_vel_reward_scale * time_scale,
            "endeffector_ang_vel_distance": ang_vel_distance * self.cfg.ang_vel_distance_reward_scale * time_scale,
            "joint_vel": joint_vel_error * self.cfg.joint_vel_reward_scale * time_scale,
            "joint_vel_distance": joint_vel_distance * self.cfg.joint_vel_distance_reward_scale * time_scale,
            "shoulder_joint_error": shoulder_joint_error * self.cfg.shoulder_error_reward_scale * time_scale,
            "shoulder_joint_distance": shoulder_joint_distance * self.cfg.shoulder_distance_reward_scale * time_scale,
            "wrist_joint_error": wrist_joint_error * self.cfg.wrist_error_reward_scale * time_scale,
            "wrist_joint_distance": wrist_joint_distance * self.cfg.wrist_distance_reward_scale * time_scale,
            "body_ang_vel": body_ang_vel_error * self.cfg.body_ang_vel_reward_scale * time_scale, # use the same reward scale as the endeffector ang vel for now
            "body_ang_vel_distance": body_ang_vel_distance * self.cfg.body_ang_vel_distance_reward_scale * time_scale,
            # "joint_vel": combined_distance * self.cfg.joint_vel_reward_scale * time_scale,
            "action_norm": action_norm_error * self.cfg.action_norm_reward_scale * time_scale,
            "action_delta": action_delta_error * self.cfg.action_delta_reward_scale * time_scale,
            "action_norm_prop": action_prop_error * self.cfg.action_norm_prop_reward_scale * time_scale,
            "action_norm_joint": action_joint_error * self.cfg.action_joint_norm_reward_scale * time_scale,
            "action_delta_prop": action_delta_prop_error * self.cfg.previous_action_prop_reward_scale * time_scale,
            "action_delta_joint": action_delta_joint_error * self.cfg.previous_action_joint_reward_scale * time_scale,
            "action_delta_prop_distance": action_delta_prop_distance * self.cfg.action_delta_prop_distance_reward_scale * time_scale,
            "action_delta_joint_distance": action_delta_joint_distance * self.cfg.action_delta_joint_distance_reward_scale * time_scale,
            "stay_alive": torch.ones_like(ee_pos_error) * self.cfg.stay_alive_reward * time_scale,
            "crash_penalty": self.reset_terminated[:].float() * crash_penalty_time * time_scale,
            "axis_reward": axis_reward * self.cfg.axis_reward_scale * time_scale,
            "jitter": jitter * self.cfg.jitter_reward_scale * time_scale,
        }

        errors = {
            "body_pos_error": body_pos_error,
            "combined_error": combined_error,
            "ee_pos_error": ee_pos_error,
            "ori_error": ori_error,
            "yaw_error": yaw_error,
            "yaw_distance": yaw_distance,
            "shoulder_joint_error": shoulder_joint_error,
            "wrist_joint_error": wrist_joint_error,
            "lin_vel": lin_vel_error,
            "ang_vel": ang_vel_error,
            "body_ang_vel": body_ang_vel_error,
            "joint_vel": joint_vel_error,
            "action_norm": action_norm_error,
            "action_delta": action_delta_error,
            "action_norm_prop": action_prop_error,
            "action_norm_joint": action_joint_error,
            "action_delta_prop": action_delta_prop_error,
            "action_delta_joint": action_delta_joint_error,
            "stay_alive": torch.ones_like(ee_pos_error),
            "crash_penalty": self.reset_terminated[:].float(),
            "axis_reward": -1 * axis_reward,
            "jitter": jitter,
        }

        # 7 x 1024 -> 1024
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        for key, value in errors.items():
            self._episode_error_sums[key] += value
        return reward

    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tensors corresponding to termination and truncation. 
        """

        # Check if end effector or body has collided with the ground
        if self.cfg.has_end_effector:
            died = torch.logical_or(self._robot.data.body_state_w[:, self._ee_id, 2].squeeze() < 0.0, self._robot.data.body_state_w[:, self._body_id, 2].squeeze() < 0.0)
            died = died | (self._robot.data.root_pos_w[:, 2] > 10.0) # set a height limit
        else:
            died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)

        # Check if the robot is too high
        # died = torch.logical_or(died, self._robot.data.root_pos_w[:, 2] > 10.0)
        self.crash_mask = self.crash_mask | died
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Resets the environment at the specified indices.
        """

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        base_pos_w, base_ori_w, _, _ = self.get_frame_state_from_task(self.cfg.task_body)

        # Logging the episode sums
        final_distance_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - base_pos_w[env_ids], dim=1).mean()
        final_ori_error_to_goal = quat_error_magnitude(self._desired_ori_w[env_ids], base_ori_w[env_ids]).mean()
        final_yaw_error_to_goal = quat_error_magnitude(yaw_quat(self._desired_ori_w[env_ids]), yaw_quat(base_ori_w[env_ids])).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = self._episode_sums[key][env_ids].mean()
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for key in self._episode_error_sums.keys():
            episodic_sum_avg = self._episode_error_sums[key][env_ids].mean()
            extras["Episode Error/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_error_sums[key][env_ids] = 0.0
        extras["Metrics/Final Distance to Goal"] = final_distance_to_goal
        extras["Metrics/Final Orientation Error to Goal"] = final_ori_error_to_goal
        extras["Metrics/Final Yaw Error to Goal"] = final_yaw_error_to_goal
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/EE Position Radius"] = self.ee_pos_radius
        extras["Metrics/Quad Position Radius"] = self.body_pos_radius
        extras["Metrics/Ori Radius"] = self.ori_radius
        extras["Metrics/Wrist Radius"] = self.wrist_radius
        extras["Metrics/Lin Vel Radius"] = self.lin_vel_radius
        extras["Metrics/Ang Vel Radius"] = self.ang_vel_radius
        extras["Metrics/Body Ang Vel Radius"] = self.body_ang_vel_radius
        extras["Metrics/Joint Vel Radius"] = self.joint_vel_radius
        extras["Metrics/Action Delta Prop Radius"] = self.action_delta_prop_radius
        extras["Metrics/Action Delta Joint Radius"] = self.action_delta_joint_radius
        extras["Metrics/Unique Crashes"] = torch.count_nonzero(self.crash_mask).item()
        t = self.common_step_counter // self.cfg.num_steps_per_env
        extras["Metrics/Reset curriculum"] = min(self.cfg.reset_curriculum_rand_range + 0.1 * (t // self.cfg.reset_curriculum), 1.0)
        # extras["Metrics/Common Step Counter"] = self.common_step_counter
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.eval_mode:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        elif self.cfg.eval_mode:
            self.episode_length_buf[env_ids] = 0

        # Update the trajectories for the reset environments
        self.initialize_trajectories(env_ids)
        self.update_goal_state()

        # print("Goal State: ", self._desired_pos_w[env_ids[0]], self._desired_ori_w[env_ids[0]])

        # Reset Robot state
        self._robot.reset()
        
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # if self.cfg.num_joints > 0:
        #     # print("Resetting shoulder joint to pi/2")
        #     shoulder_joint_pos = torch.tensor(torch.pi/2, device=self.device, requires_grad=False).float()
        #     shoulder_joint_vel = torch.tensor(0.0, device=self.device, requires_grad=False).float()
        #     self._robot.write_joint_state_to_sim(shoulder_joint_pos, shoulder_joint_vel, joint_ids=self._shoulder_joint_idx, env_ids=env_ids)

        # Default initialization - robot placed on current target trajectory position and orientation
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] = self._desired_body_pos[env_ids]
        init_yaw, init_shoulder, init_wrist = aerial_manipulator_angle_errors(self._desired_ori_w[env_ids], self.model_ee_ori.tile((len(env_ids), 1)))
        # desired_joint_angles = torch.stack([init_shoulder.squeeze(1), init_wrist.squeeze(1)], dim=1)
        # default_root_state[:, 3:7] = math_utils.quat_from_yaw(init_yaw.squeeze(1))
        # default_root_state[:, 7:10] = self._pos_traj[1, env_ids, :, 0]
        default_root_state[:, -1] = self._yaw_traj[1, env_ids, 0] # set the body yaw velocity to the desired yaw velocity
        # desired_joint_vel = torch.stack([self._roll_traj[1, env_ids, 0], self._pitch_traj[1, env_ids, 0]], dim=1) # set the joint velocites to the desired angular velocities (is approximate)
        desired_joint_vel = torch.zeros(len(env_ids), 2, device=self.device)
        if self.cfg.init_cfg == "rand":
            # default_root_state = self._robot.data.default_root_state[env_ids]
            # Initialize the robot on the trajectory with the correct velocity
            # traj_pos_start = self._pos_traj[0, env_ids, :, 0]
            # traj_vel_start = self._pos_traj[1, env_ids, :, 0]
            # traj_yaw_start = self._yaw_traj[0, env_ids, 0]
            if not self.cfg.eval_mode:
                # t = self.common_step_counter * self.num_envs
                # decay = min(0.1 + 0.1 * (t // 10_000_000), 1.0)
                iteration = self.common_step_counter // self.cfg.num_steps_per_env
                decay = min(self.cfg.reset_curriculum_rand_range + 0.1 * (iteration // self.cfg.reset_curriculum), 1.0)
            else:
                decay = 1.0
            pos_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_pos_ranges, device=self.device).float() * decay
            vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_lin_vel_ranges, device=self.device).float() * decay
            yaw_rand = (torch.rand(len(env_ids), 1, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_yaw_ranges, device=self.device).float() * decay
            ang_vel_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_ang_vel_ranges, device=self.device).float() * decay
            init_yaw = init_yaw + yaw_rand
            joint_rand = (torch.rand(len(env_ids), 2, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_joint_ranges, device=self.device).float() * decay
            init_shoulder = init_shoulder + joint_rand[:, 0:1]
            init_wrist = init_wrist + joint_rand[:, 1:2]
            joint_vel_rand = torch.zeros(len(env_ids), 2, device=self.device)

            default_root_state[:, :3] = default_root_state[:, :3] + pos_rand
            # default_root_state[:, 3:7] = init_yaw
            default_root_state[:, 7:10] = default_root_state[:, 7:10] + vel_rand
            default_root_state[:, 10:13] = ang_vel_rand
            # desired_joint_angles = desired_joint_angles + joint_rand
            desired_joint_vel = desired_joint_vel + joint_vel_rand
            # default_root_state[:, :3] = traj_pos_start
            # default_root_state[:, 3:7] = math_utils.quat_from_yaw(traj_yaw_start)
            # default_root_state[:, 7:10] = traj_vel_start
            # default_root_state[:, 10:13] = torch.zeros_like(traj_vel_start)
        elif self.cfg.init_cfg == "fixed":
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, 2] = 3.0
            # desired_joint_angles = torch.zeros(len(env_ids), 2, device=self.device)
            desired_joint_vel = torch.zeros(len(env_ids), 2, device=self.device)
            init_yaw = torch.zeros_like(init_yaw)
            init_shoulder = torch.zeros_like(init_shoulder)
            init_wrist = torch.zeros_like(init_wrist)
            # default_root_state[:, 3:7] = self._desired_ori_w[env_ids]            
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # if self.cfg.num_joints > 0:
        #     default_root_state[:, 3:7] = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self.device, requires_grad=False).float().tile((env_ids.size(0), 1))
        desired_joint_angles = torch.cat([init_shoulder, init_wrist], dim=1)
        default_root_state[:, 3:7] = math_utils.quat_from_yaw(init_yaw.squeeze(1))
        default_root_state[:, 2] = torch.clamp(default_root_state[:, 2], min=0.25) # min height to avoid instant resets
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(desired_joint_angles, desired_joint_vel, env_ids=env_ids)
        # self.initial_shoulder[env_ids] = init_shoulder
        # self.initial_wrist[env_ids] = init_wrist
        # self.initial_quad_yaw[env_ids] = init_yaw
        # _, ee_ori_w, _, _ = self.get_frame_state_from_task(self.cfg.task_body)
        # self.initial_ee_ori[env_ids] = ee_ori_w[env_ids]
        self.last_yaw_cmd[env_ids] = init_yaw
 
        # Update viz_histories
        if self.cfg.viz_mode == "robot":
            self._robot_pos_history[env_ids] = default_root_state[:, :3].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._robot_ori_history[env_ids] = default_root_state[:, 3:7].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._goal_pos_history[env_ids] = self._desired_pos_w[env_ids].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)
            self._goal_ori_history[env_ids] = self._desired_ori_w[env_ids].unsqueeze(1).tile(1, self.cfg.viz_history_length, 1)

        # Reset motor state: start at hover equilibrium and clear previous controller error
        self._motor_speeds[env_ids] = self._hover_motor_speed
        self._motor_speeds_des[env_ids] = self._hover_motor_speed
        self._previous_omega_err[env_ids] = 0.0
        self._previous_velocity_obs[env_ids] = 0.0

    def initialize_trajectories(self, env_ids):
        """
        Initializes the trajectory for the environment ids.
        """
        num_envs = env_ids.size(0)

        # Randomize Lissajous parameters
        if not self.cfg.eval_mode:
            # t = self.common_step_counter * self.num_envs
            # decay = min(self.cfg.reset_curriculum_rand_range + 0.1 * (t // self.cfg.reset_curriculum), 1.0)
            decay = 1.0
        else:
            decay = 1.0
        random_amplitudes = ((torch.rand(num_envs, 6, device=self.device)) * 2.0 - 1.0) * self.lissajous_amplitudes_rand_ranges * decay
        random_frequencies = ((torch.rand(num_envs, 6, device=self.device))) * self.lissajous_frequencies_rand_ranges * decay
        random_phases = ((torch.rand(num_envs, 6, device=self.device)) * 2.0 - 1.0) * self.lissajous_phases_rand_ranges
        random_offsets = ((torch.rand(num_envs, 6, device=self.device)) * 2.0 - 1.0) * self.lissajous_offsets_rand_ranges

        # Randomize polynomial parameters
        random_poly_roll = ((torch.rand(num_envs, len(self.cfg.polynomial_roll_rand_ranges), device=self.device)) * 2.0 - 1.0) * self.polynomial_roll_rand_ranges * decay
        random_poly_pitch = ((torch.rand(num_envs, len(self.cfg.polynomial_pitch_rand_ranges), device=self.device)) * 2.0 - 1.0) * self.polynomial_pitch_rand_ranges * decay
        random_poly_yaw = ((torch.rand(num_envs, len(self.cfg.polynomial_yaw_rand_ranges), device=self.device)) * 2.0 - 1.0) * self.polynomial_yaw_rand_ranges * decay

        terrain_offsets = torch.zeros_like(random_offsets, device=self.device)
        terrain_offsets[:, :2] = self._terrain.env_origins[env_ids, :2]
        
        self.lissajous_amplitudes[env_ids] = torch.tensor(self.cfg.lissajous_amplitudes, device=self.device).tile((num_envs, 1)).float() + random_amplitudes
        self.lissajous_frequencies[env_ids] = torch.tensor(self.cfg.lissajous_frequencies, device=self.device).tile((num_envs, 1)).float() + random_frequencies
        self.lissajous_phases[env_ids] = torch.tensor(self.cfg.lissajous_phases, device=self.device).tile((num_envs, 1)).float() + random_phases
        self.lissajous_offsets[env_ids] = torch.tensor(self.cfg.lissajous_offsets, device=self.device).tile((num_envs, 1)).float() + random_offsets + terrain_offsets

        self.polynomial_coefficients[env_ids, 3] = torch.tensor(self.cfg.polynomial_roll_coefficients, device=self.device).tile((num_envs, 1)).float() + random_poly_roll
        self.polynomial_coefficients[env_ids, 4] = torch.tensor(self.cfg.polynomial_pitch_coefficients, device=self.device).tile((num_envs, 1)).float() + random_poly_pitch
        self.polynomial_coefficients[env_ids, 5] = torch.tensor(self.cfg.polynomial_yaw_coefficients, device=self.device).tile((num_envs, 1)).float() + random_poly_yaw
        # # Rerandomize the random shift if needed
        # if self.cfg.random_shift_trajectory:
        #     self._pos_shift[env_ids] = torch.zeros_like(self._pos_shift[env_ids]).uniform_(-self.cfg.goal_pos_range, self.cfg.goal_pos_range)
        #     self._yaw_shift[env_ids] = torch.zeros_like(self._yaw_shift[env_ids]).uniform_(-self.cfg.goal_yaw_range, self.cfg.goal_yaw_range)
    
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
        elif task_body == "vehicle":
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

    def get_goal_state_from_task(self, goal_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_body == "root":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "endeffector":
            goal_pos_w = self._desired_pos_w
            goal_ori_w = self._desired_ori_w
        elif goal_body == "COM":
            # desired_pos, desired_yaw = self.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e)
            desired_pos, desired_yaw = math_utils.compute_desired_pose_from_transform(self._desired_pos_w, self._desired_ori_w, self.com_pos_e, 0)
            goal_pos_w = desired_pos
            goal_ori_w = quat_from_yaw(desired_yaw)
        else:
            raise ValueError("Invalid goal body: ", goal_body)

        return goal_pos_w, goal_ori_w
    
    def convert_ee_goal_from_task(self, ee_pos_w, ee_ori_w, task_body:str) -> tuple[torch.Tensor, torch.Tensor]:
        if task_body == "root":
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

    def _setup_scene(self):
        if sum(self.cfg.robot_color) > 0:
            print("Setting robot color to: ", self.cfg.robot_color)
            print(self.cfg.robot.spawn.visual_material)
            self.cfg.robot.spawn.visual_material=sim_utils.GlassMdlCfg(glass_color=tuple(self.cfg.robot_color))
            print(self.cfg.robot.spawn.visual_material)
            
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                if self.cfg.viz_mode == "triad" or self.cfg.viz_mode == "frame": 
                    frame_marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                            markers={
                                            "frame": sim_utils.UsdFileCfg(
                                                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                                                scale=(0.1, 0.1, 0.1),
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
                                                radius=0.01,
                                                visual_material=sim_utils.GlassMdlCfg(glass_color=history_color),
                                            ),
                                            "goal_history": sim_utils.SphereCfg(
                                                radius=0.01,
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
            pos, ori, _, _ = self.get_frame_state_from_task(self.cfg.task_body)
            self._frame_positions[:, 0] = pos
            self._frame_positions[:, 1] = self._desired_pos_w
            # self._frame_positions[:, 2] = self._robot.data.body_pos_w[:, self._body_id].squeeze(1)
            # self._frame_positions[:, 2] = com_pos_w
            self._frame_orientations[:, 0] = ori
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

