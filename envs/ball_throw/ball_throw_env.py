"""Ball-throw evaluation environment.

Subclasses the 2-DOF trajectory-tracking environment so that the observation
space is *identical* to what trained policies expect.  Adds a ball attached to
the end-effector (released at time ``ball_release_time``) and a hoop spawned at
a random location.  The reward is purely binary: 1 if the ball passes through
the hoop after release, 0 otherwise.

Intended use: load a policy trained on
``Isaac-AerialManipulator-2DOF-TrajectoryTracking-*`` and run it in this
environment to see whether it can follow a throwing trajectory accurately
enough to score.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.sim.utils import get_current_stage
from pxr import Gf, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.shapes import SphereCfg, CylinderCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    euler_xyz_from_quat,
    wrap_to_pi,
)

from configs.aerial_manip_asset import AERIAL_MANIPULATOR_2DOF_CFG
from utils.math_utilities import calculate_required_pos, yaw_from_quat, quat_from_yaw

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

from envs.trajectory_tracking.trajectory_tracking_env_2dof import (
    AerialManipulatorTrajectoryTrackingEnv,
    AerialManipulatorTrajectoryTrackingEnvBaseCfg,
    EventCfg,
)

from utils.trajectory_utilities import eval_polynomial_curve_6dof


@configclass
class BallThrowEventCfg(EventCfg):
    """Zeros out the EE mass at startup (like NoEndEffectorEventCfg) but
    inherits from EventCfg so that ``isinstance(cfg.events, EventCfg)``
    is True and the critic observation includes the EE mass channel."""
    randomize_endeffector_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["endeffector"]),
            "mass_distribution_params": (-0.2, -0.2),
            "operation": "add",
        },
    )


# ---------------------------------------------------------------------------
# Asset configs for ball and hoop
# ---------------------------------------------------------------------------

THROW_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ThrowBall",
    spawn=SphereCfg(
        radius=0.03,
        visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
        ),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.1,  # 100 g
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            restitution=0.5,
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
    ),
    collision_group=0,
)

HOOP_RING_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/HoopRing",
    spawn=CylinderCfg(
        radius=0.2,
        height=0.02,
        axis="Z",
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    ),
    collision_group=0,
)

HOOP_POST_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/HoopPost",
    spawn=CylinderCfg(
        radius=0.015,
        height=1.5,
        axis="Z",
        visual_material=PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
    ),
    collision_group=-1,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class BallThrowEnvCfg(AerialManipulatorTrajectoryTrackingEnvBaseCfg):
    """Config that inherits everything from the 2-DOF trajectory-tracking
    base config so that observation construction is identical."""

    action_space = 6
    num_joints = 2
    observation_space = 16  # overridden at runtime, kept for compatibility

    robot: ArticulationCfg = AERIAL_MANIPULATOR_2DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    shoulder_torque_scalar = robot.actuators["shoulder"].effort_limit
    wrist_torque_scalar = robot.actuators["wrist"].effort_limit

    # Zeros the EE mass at startup while keeping isinstance(..., EventCfg)
    # true so the critic observation includes the EE mass channel.  The
    # actual ball mass is added/removed dynamically in the environment.
    events = BallThrowEventCfg()

    # Ball & hoop assets
    throw_ball: RigidObjectCfg = THROW_BALL_CFG.replace(
        prim_path="/World/envs/env_.*/ThrowBall"
    )
    hoop_ring: RigidObjectCfg = HOOP_RING_CFG.replace(
        prim_path="/World/envs/env_.*/HoopRing"
    )
    hoop_post: RigidObjectCfg = HOOP_POST_CFG.replace(
        prim_path="/World/envs/env_.*/HoopPost"
    )

    # Override scene to use wider spacing (ball can fly far)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(
    #     num_envs=64, env_spacing=8.0, replicate_physics=True
    # )

    # Drone fixed spawn
    drone_spawn_pos = [0.0, 0.0, 2.0]

    # Hoop spawn range
    hoop_pos_center = [0.0, 0.0, 2.0]
    hoop_pos_range = [0.0, 0.0, 0.0] #[1.0, 1.0, 0.5]
    

    # Trajectory parameters for phase 1: polynomial trajectory to a common point next to the hoop
    hoop_hover_offset = [6.0, -2.0, 1.5]
    hoop_hover_time = 2.0
    polynomial_degree = 3
    hoop_hover_yaw_angle = np.pi/2
    hoop_hover_shoulder_angle = 1*np.pi/6
    hoop_hover_wrist_angle = 0.0
    hoop_hover_vel = [0.0, 0.0, 0.0] # ee vel, will also be quad desired vel if no joint velocity
    

    # Phase 2 (throwing trajectory)
    # Ball release time in seconds
    ball_release_time = hoop_hover_time + 2.0
    hoop_throw_offset = [2.0, 0.0, 2.0] # drone pos relative to hoop
    throw_yaw_angle = np.pi/2
    throw_shoulder_angle = -1*np.pi/6 + 1*2*np.pi
    throw_wrist_angle = 0.0
    throw_vel_drone = [0.0, 0.0, 0.0] # x and y components will be calculated, z can be set
    throw_vel_yaw = 0.0
    throw_vel_shoulder = 3.0
    throw_vel_wrist = 0.0

    # Phase 3 (follow through trajectory/come to stationary point)
    follow_through_time = ball_release_time + 2.0
    follow_through_offset = [0.0, 2.0, 2.0] # drone pos specified
    follow_through_yaw_angle = 0.0
    follow_through_shoulder_angle = -np.pi/2
    follow_through_wrist_angle = 0.0


    # Ball mass added to the EE link while attached (kg)
    ball_mass = 0.1

    # Force eval mode so we get the full state logged
    eval_mode = True
    init_cfg = "rand"
    init_pos_ranges=[0.1, 0.1, 0.0]
    init_yaw_ranges=[0.1]
    init_joint_ranges =[0.1, 0.1]

    use_motor_dynamics = False

    # Reward: only hoop score matters
    ball_through_hoop_reward = 1.0


@configclass
class BallThrowWithMotorDynamicsCfg(BallThrowEnvCfg):
    use_motor_dynamics = True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BallThrowEnv(AerialManipulatorTrajectoryTrackingEnv):
    """Subclass that adds ball + hoop assets and ball-release logic on top of
    the trajectory-tracking environment.  Observations are identical to the
    parent so that pre-trained policies can be evaluated directly."""

    cfg: BallThrowEnvCfg

    def __init__(self, cfg: BallThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Parent init allocates _pos_traj as (..., 1+horizon, 3), while the
        # trajectory generator uses (..., 3, 1+horizon). Since this subclass
        # overrides initialize_trajectories(), normalize once here.
        if self._pos_traj.shape[-2] != 3:
            self._pos_traj = self._pos_traj.transpose(-1, -2).contiguous()

        # Ball state tracking
        self._ball_attached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._ball_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ball_passed_through_hoop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hoop positions per env
        self._hoop_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Release step
        self._release_step = int(self.cfg.ball_release_time * self.cfg.policy_rate_hz)

        # Base EE mass (without ball).  Initialised to 0 in _reset_idx since
        # BallThrowEventCfg zeros the EE link mass at startup.
        self._ee_base_mass = None

        # Episode tracking
        self._ball_throw_episode_sums = {
            "ball_through_hoop": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        self.init_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_quad_yaw = torch.zeros(self.num_envs, device=self.device)
        self.init_shoulder_angle = torch.zeros(self.num_envs, device=self.device)
        self.init_wrist_angle = torch.zeros(self.num_envs, device=self.device)
        self.ball_release_pos_targets = torch.zeros(self.num_envs, 3, device=self.device)

        self.hoop_hover_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.follow_through_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # might be a better way of doing this
        _, _, _, self.hoop_hover_ori = self.joint_to_rpy(
            torch.tensor(self.cfg.hoop_hover_yaw_angle, device=self.device).tile(self.num_envs, 1),
            torch.tensor(self.cfg.hoop_hover_shoulder_angle, device=self.device).tile(self.num_envs, 1),
            torch.tensor(self.cfg.hoop_hover_wrist_angle, device=self.device).tile(self.num_envs, 1)
        )
        self.hoop_throw_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.poly_coefficients = torch.zeros(self.num_envs, 6, self.cfg.polynomial_degree + 1, device=self.device) # matches format expected by eval_polynomial_curve_6dof
        self.poly_coefficients_throw = torch.zeros(self.num_envs, 6, self.cfg.polynomial_degree + 1, device=self.device) # matches format expected by eval_polynomial_curve_6dof
        self.poly_coefficients_follow_through = torch.zeros(self.num_envs, 6, self.cfg.polynomial_degree + 1, device=self.device) # matches format expected by eval_polynomial_curve_6dof

        self.hover_pos_drone_end = torch.zeros(self.num_envs, 3, device=self.device)
        self.in_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Pre-calculate some useful quantities for the throwing part - determine where the height of the drone and shoulder joint
        # velocity needed at the end of the throwing stage, given a desired drone velocity, horizontal distance of drone to hoop, and shoulder angle 
        ball_throw_x = self.cfg.hoop_throw_offset[0] - self.arm_length * np.cos(self.cfg.throw_shoulder_angle)
        ball_throw_z = self.cfg.hoop_throw_offset[2] + self.arm_length * np.sin(self.cfg.throw_shoulder_angle)
        s = np.sin(self.cfg.throw_shoulder_angle)
        c = np.cos(self.cfg.throw_shoulder_angle)
        vz_throw = self.arm_length * c * self.cfg.throw_vel_shoulder + self.cfg.throw_vel_drone[2]
        vx_throw = self.arm_length * s * self.cfg.throw_vel_shoulder # just the component from the angular velocity
        g = abs(self.cfg.sim.gravity[2])
        ball_fall_time = (-vz_throw - (vz_throw**2 + 2*g*ball_throw_z)**0.5) / -g
        vx_needed = -ball_throw_x / ball_fall_time - vx_throw # negative sign because throwing towards hoop coming from positive x
        self.cfg.throw_vel_drone[0] = vx_needed

        # in case of a y-offset:
        vy_needed = -self.cfg.hoop_throw_offset[1] / ball_fall_time
        self.cfg.throw_vel_drone[1] = vy_needed

        print(
            f"vz_throw: {vz_throw}, vx_throw: {vx_throw}, ball_throw_z: {ball_throw_z}, ball_throw_x: {ball_throw_x}, g: {g}, ball_fall_time: {ball_fall_time}, vx_needed: {vx_needed}, vy_needed: {vy_needed}"
        )


    # ------------------------------------------------------------------
    # Scene: add ball + hoop on top of parent scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()

        self._throw_ball = RigidObject(self.cfg.throw_ball)
        self.scene.rigid_objects["throw_ball"] = self._throw_ball

        self._hoop_ring = RigidObject(self.cfg.hoop_ring)
        self.scene.rigid_objects["hoop_ring"] = self._hoop_ring

        self._hoop_post = RigidObject(self.cfg.hoop_post)
        self.scene.rigid_objects["hoop_post"] = self._hoop_post

    # ------------------------------------------------------------------
    # Override update_goal_state to set a hover target at the drone
    # spawn position instead of following lissajous/polynomial curves.
    # This can be replaced later with a proper throwing trajectory.
    # ------------------------------------------------------------------

    def update_goal_state(self):
        env_ids = (
            self.episode_length_buf % int(self.cfg.traj_update_dt * self.cfg.policy_rate_hz) == 0
        ).nonzero(as_tuple=False)

        if len(env_ids) == 0 or env_ids.size(0) == 0:
            return

        ids = env_ids.squeeze(1)

        hover_pos_shifted = self.hoop_hover_pos.clone() #torch.tensor(self.cfg.drone_spawn_pos, device=self.device)
        # hover_pos_shifted = hover_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        # hover_pos_shifted[:, 2] += 0.5
        # hover_pos_shifted[:, :2] += self._terrain.env_origins[:, :2]

        # Phase 1: polynomial trajectory to a common point next to the hoop using the precomputed coefficients
        T = 1 + self.cfg.trajectory_horizon
        # curr_time = self.episode_length_buf.unsqueeze(-1)
        # future_timesteps = torch.arange(0, T, device=self.device)
        # time = ((curr_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt).unsqueeze(1) # (num_envs, 1, T)

        # t_offset = 1.0
        phase_1_time = self.cfg.hoop_hover_time
        phase_2_time = self.cfg.ball_release_time
        phase_3_time = self.cfg.follow_through_time
        curr_time = self.episode_length_buf.unsqueeze(-1)
        future_timesteps = torch.arange(0, T, device=self.device)
        time = (curr_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt # (num_envs, T)
        pos_traj_1, yaw_traj_1, shoulder_traj_1, wrist_traj_1 = eval_polynomial_curve_6dof(time, self.poly_coefficients, derivatives=1)
        pos_traj_2, yaw_traj_2, shoulder_traj_2, wrist_traj_2 = eval_polynomial_curve_6dof(time - phase_1_time, self.poly_coefficients_throw, derivatives=1)
        pos_traj_3, yaw_traj_3, shoulder_traj_3, wrist_traj_3 = eval_polynomial_curve_6dof(time - phase_2_time, self.poly_coefficients_follow_through, derivatives=1)
        # motion_mask = (time <= arrive_time).unsqueeze(1).tile(1, 3, 1) # mask on future timesteps to send observation
        phase_1_mask = (time <= phase_1_time)[0] # NOTE: this logic and the counting logic below only works if the parallel envs are at the same time step (which is only the case if resets always happen at the same time for all envs)
        n_phase_1 = phase_1_mask.count_nonzero()
        phase_2_mask = ((time <= phase_2_time) & (time > phase_1_time))[0]
        n_phase_2 = phase_2_mask.count_nonzero()
        phase_3_mask = ((time <= phase_3_time) & (time > phase_2_time))[0]
        n_phase_3 = phase_3_mask.count_nonzero()
        done_mask = (time > phase_3_time)[0]
        n_done = done_mask.count_nonzero()
        # target_pos = torch.where(motion_mask, pos_traj[0], hover_pos_shifted.unsqueeze(-1).tile(1, 1, T))
        # slopes = torch.where(motion_mask, pos_traj[1], torch.zeros_like(pos_traj[1]))
        target_pos = torch.zeros_like(pos_traj_1[0])
        target_pos[..., phase_1_mask] = pos_traj_1[0][..., phase_1_mask]
        target_pos[..., phase_2_mask] = pos_traj_2[0][..., phase_2_mask]
        target_pos[..., phase_3_mask] = pos_traj_3[0][..., phase_3_mask]
        target_pos[..., done_mask] = self.follow_through_pos.unsqueeze(-1).tile(1, 1, n_done)
        # num_in_motion = motion_mask.count_nonzero()
        # target_pos = pos_traj[0]
        # target_pos[..., ~motion_mask] = hover_pos_shifted.unsqueeze(-1).tile(1, 1, T - num_in_motion)
        slopes = torch.zeros_like(pos_traj_1[1])
        slopes[..., phase_1_mask] = pos_traj_1[1][..., phase_1_mask]
        # since for phase 2 & 3 the position trajectory is specified as a polynomial for the quadrotor's position, pos_traj_2[1] is not the true EE desired velocity, get from finite differences
        # note that only the desired velocity for the current time step is the one that gets used for the observation, so those are the only ones that need to be corrected
        finite_diff_mask = phase_2_mask | phase_3_mask
        slopes[..., phase_2_mask] = (pos_traj_2[0][..., torch.clamp(phase_2_mask.nonzero(as_tuple=True)[0] + 1, max=T-1)] - pos_traj_2[0][..., phase_2_mask]) / self.cfg.traj_update_dt # clamping to avoid index errors
        slopes[..., phase_3_mask] = (pos_traj_3[0][..., torch.clamp(phase_3_mask.nonzero(as_tuple=True)[0] + 1, max=T-1)] - pos_traj_3[0][..., phase_3_mask]) / self.cfg.traj_update_dt # clamping to avoid index errors
        # slopes[..., phase_2_mask] = pos_traj_2[1][..., phase_2_mask]
        slopes[..., done_mask] = torch.zeros_like(slopes[..., done_mask])
        # slopes = pos_traj[1]
        # slopes[..., ~motion_mask] = torch.zeros_like(slopes[..., ~motion_mask])

        # slopes = (hover_pos_shifted.unsqueeze(-1).tile(1, 1, T) - self.init_ee_pos.unsqueeze(-1).tile(1, 1, T)) / arrive_time
        # target_pos = self.init_ee_pos.unsqueeze(-1) + slopes * time.unsqueeze(1)
        # motion_mask = (time <= arrive_time).unsqueeze(1).tile(1, 3, 1) # mask on future timesteps to send observation
        # target_pos = torch.where(motion_mask, target_pos, hover_pos_shifted.unsqueeze(-1).tile(1, 1, T))
        # slopes = torch.where(motion_mask, slopes, torch.zeros_like(slopes))

        # motion_mask = time <= arrive_time

        # yaw_traj[0] = torch.where(motion_mask, yaw_traj[0], self.cfg.hoop_hover_yaw_angle * torch.ones_like(yaw_traj[0]))
        yaw_traj = torch.zeros_like(yaw_traj_1)
        shoulder_traj = torch.zeros_like(shoulder_traj_1)
        wrist_traj = torch.zeros_like(wrist_traj_1)
        yaw_traj[0, ..., phase_1_mask] = yaw_traj_1[0, ..., phase_1_mask]
        shoulder_traj[0, ..., phase_1_mask] = shoulder_traj_1[0, ..., phase_1_mask]
        wrist_traj[0, ..., phase_1_mask] = wrist_traj_1[0, ..., phase_1_mask]
        yaw_traj[0, ..., phase_2_mask] = yaw_traj_2[0, ..., phase_2_mask]
        shoulder_traj[0, ..., phase_2_mask] = shoulder_traj_2[0, ..., phase_2_mask]
        wrist_traj[0, ..., phase_2_mask] = wrist_traj_2[0, ..., phase_2_mask]
        yaw_traj[0, ..., phase_3_mask] = yaw_traj_3[0, ..., phase_3_mask]
        shoulder_traj[0, ..., phase_3_mask] = shoulder_traj_3[0, ..., phase_3_mask]
        wrist_traj[0, ..., phase_3_mask] = wrist_traj_3[0, ..., phase_3_mask]
        yaw_traj[0, ..., done_mask] = torch.ones_like(yaw_traj[0, ..., done_mask]) * self.cfg.follow_through_yaw_angle
        shoulder_traj[0, ..., done_mask] = torch.ones_like(shoulder_traj[0, ..., done_mask]) * self.cfg.follow_through_shoulder_angle
        wrist_traj[0, ..., done_mask] = torch.ones_like(wrist_traj[0, ..., done_mask]) * self.cfg.follow_through_wrist_angle
        
        # yaw_traj[0, ..., ~motion_mask] = self.cfg.hoop_hover_yaw_angle * torch.ones_like(yaw_traj[0, ..., ~motion_mask])
        # shoulder_traj[0, ..., ~motion_mask] = self.cfg.hoop_hover_shoulder_angle * torch.ones_like(shoulder_traj[0, ..., ~motion_mask])
        # wrist_traj[0, ..., ~motion_mask] = self.cfg.hoop_hover_wrist_angle * torch.ones_like(wrist_traj[0, ..., ~motion_mask])
        # shoulder_traj[0] = torch.where(motion_mask, shoulder_traj[0], self.cfg.hoop_hover_shoulder_angle * torch.ones_like(shoulder_traj[0]))
        # wrist_traj[0] = torch.where(motion_mask, wrist_traj[0], self.cfg.hoop_hover_wrist_angle * torch.ones_like(wrist_traj[0]))

        r, p, y, q = self.joint_to_rpy(yaw_traj[0], shoulder_traj[0], wrist_traj[0]) # idx 0 bc that's the one that contains the actual angles
        # r, p, y = wrap_to_pi(r), wrap_to_pi(p), wrap_to_pi(y)s
        # since these might be complex transforms, calculate the velocities with finite differences
        r_vel = torch.zeros_like(r)
        r_vel[:, :-1] = (r[:, 1:] - r[:, :-1]) / self.cfg.traj_update_dt
        p_vel = torch.zeros_like(p)
        p_vel[:, :-1] = (p[:, 1:] - p[:, :-1]) / self.cfg.traj_update_dt
        y_vel = torch.zeros_like(y)
        y_vel[:, :-1] = (y_vel[:, 1:] - y_vel[:, :-1]) / self.cfg.traj_update_dt

        # reformulate for end effector position in phase 2 and done
        target_pos[..., finite_diff_mask | done_mask] = self.get_ee_pos(target_pos[..., finite_diff_mask | done_mask], q[:, finite_diff_mask | done_mask]).transpose(1, 2)
        # slopes[]

        # additional masking just in case
        # r_vel = torch.where(motion_mask, r_vel, torch.zeros_like(r_vel))
        # p_vel = torch.where(motion_mask, p_vel, torch.zeros_like(p_vel))
        # y_vel = torch.where(motion_mask, y_vel, torch.zeros_like(y_vel))
        # motion_mask = motion_mask.unsqueeze(-1).tile(1, 1, 4) # for the quat trajectory
        # q = torch.where(motion_mask, q, self.hoop_hover_ori.tile(1, T, 1))
        # roll_traj = torch.stack([r, r_vel], dim=0)
        # pitch_traj = torch.stack([p, p_vel], dim=0)
        # yaw_traj = torch.stack([y, y_vel], dim=0)
 
 
 
        # Set variables needed for observation in parent class
        self._desired_pos_traj_w[ids] = target_pos[ids, :, :].transpose(1,2)

        self._pos_traj[:, ids] = 0.0
        self._pos_traj[0, ids] = target_pos[ids]
        self._pos_traj[1, ids] = slopes[ids]

        # self._roll_traj[:, ids] = 0.0
        self._roll_traj[0, ids] = r[ids]
        self._roll_traj[1, ids] = r_vel[ids]
        self._pitch_traj[0, ids] = p[ids]
        self._pitch_traj[1, ids] = p_vel[ids]
        # self._pitch_traj[0, ids] = target_pitch[ids]
        # self._pitch_traj[1, ids] = slope[ids]
        # self._yaw_traj[:, ids] = 0.0
        self._yaw_traj[0, ids] = y[ids]
        self._yaw_traj[1, ids] = y_vel[ids]
        self._desired_ori_traj_w[ids] = q
        self._desired_pos_w[ids] = self._desired_pos_traj_w[ids, 0]
        self._desired_ori_w[env_ids] = self._desired_ori_traj_w[env_ids, 0]
        # Derived body / COM desired positions
        self._desired_body_pos = calculate_required_pos(
            self._desired_ori_w, self._desired_pos_w,
            self._desired_body_pos, self.arm_length, ids,
        )
        self._desired_body_pos_traj = calculate_required_pos(
            self._desired_ori_traj_w, self._desired_pos_traj_w,
            self._desired_body_pos_traj, self.arm_length, ids,
        )
        self._desired_com_pos = calculate_required_pos(
            self._desired_ori_w, self._desired_pos_w,
            self._desired_com_pos, self.com_offset, env_ids,
        )
        self._desired_com_pos_traj = calculate_required_pos(
            self._desired_ori_traj_w, self._desired_pos_traj_w,
            self._desired_com_pos_traj, self.com_offset, env_ids,
        )

        # hack to set the robot's initial position to be relative to the env coordinate frame, required
        # since for init_cfg = "rand", the vehicle's starting position is a random offset w.r.t self._desired_body_pos
        # reset_idx = self.in_reset.nonzero(as_tuple=False).squeeze(-1)
        # self._desired_body_pos[reset_idx] = self._hoop_pos[reset_idx]
        # self._desired_body_pos[reset_idx, 2] += 0.5

    # ------------------------------------------------------------------
    # Also override initialize_trajectories (called by parent _reset_idx)
    # to be a no-op; our update_goal_state handles everything.
    # ------------------------------------------------------------------

    def initialize_trajectories(self, env_ids):
        pass

    # ------------------------------------------------------------------
    # Override _apply_action to also update ball state
    # ------------------------------------------------------------------

    def _apply_action(self):
        super()._apply_action()
        self._update_ball_state()

    # ------------------------------------------------------------------
    # Ball attachment / release with EE mass update
    # ------------------------------------------------------------------

    def _set_ee_mass(self, env_ids: torch.Tensor, mass: torch.Tensor):
        """Set the end-effector link mass for the given env indices.

        Args:
            env_ids: (N,) environment indices.
            mass: scalar or (N,) tensor of masses to write.
        """
        mass += 1e-8 # avoid Isaac 0 mass error
        all_masses = self._robot.root_physx_view.get_masses().clone()
        env_ids_cpu = env_ids.cpu()
        if mass.dim() == 0:
            all_masses[env_ids_cpu, self._ee_id] = mass.cpu()
        else:
            all_masses[env_ids_cpu, self._ee_id] = mass.cpu()
        self._robot.root_physx_view.set_masses(all_masses, env_ids_cpu)
        self.end_effector_mass[env_ids] = all_masses[env_ids_cpu, self._ee_id, None].to(self.device)

    def _update_ball_state(self):
        """Keep ball glued to EE while attached; on release, impart EE velocity
        and restore the original EE mass."""

        # --- Release transition ---
        release_mask = (self.episode_length_buf >= self._release_step) & self._ball_attached
        if release_mask.any():
            release_ids = release_mask.nonzero(as_tuple=False).squeeze(-1)
            self._ball_attached[release_ids] = False
            self._ball_released[release_ids] = True

            # Give ball the EE velocity at moment of release
            ee_vel_w = self._robot.data.body_lin_vel_w[:, self._ee_id].squeeze(1)
            ee_ang_vel_w = self._robot.data.body_ang_vel_w[:, self._ee_id].squeeze(1)
            ball_vel = torch.zeros_like(self._throw_ball.data.root_vel_w[release_ids])
            ball_vel[:, :3] = ee_vel_w[release_ids]
            ball_vel[:, 3:] = ee_ang_vel_w[release_ids]
            self._throw_ball.write_root_velocity_to_sim(ball_vel, env_ids=release_ids)

            # Restore EE mass (remove ball mass)
            self._set_ee_mass(release_ids, self._ee_base_mass[release_ids])

        # --- While attached: keep ball at EE pose ---
        attached_ids = self._ball_attached.nonzero(as_tuple=False).squeeze(-1)
        if attached_ids.numel() > 0:
            ee_pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
            ee_ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
            ee_vel = self._robot.data.body_lin_vel_w[:, self._ee_id].squeeze(1)
            ee_ang_vel = self._robot.data.body_ang_vel_w[:, self._ee_id].squeeze(1)

            ball_pose = self._throw_ball.data.default_root_state[attached_ids].clone()
            ball_pose[:, :3] = ee_pos[attached_ids]
            ball_pose[:, 3:7] = ee_ori[attached_ids]
            self._throw_ball.write_root_pose_to_sim(ball_pose[:, :7], env_ids=attached_ids)

            ball_vel = torch.zeros(len(attached_ids), 6, device=self.device)
            ball_vel[:, :3] = ee_vel[attached_ids]
            ball_vel[:, 3:] = ee_ang_vel[attached_ids]
            self._throw_ball.write_root_velocity_to_sim(ball_vel, env_ids=attached_ids)

    def set_hoop_color(self, env_ids: torch.Tensor, color: tuple[float, float, float]):
        """Set the hoop ring's PreviewSurface diffuse color for selected envs."""
        if env_ids is None or env_ids.numel() == 0:
            return

        # Shapes spawner typically creates: <prim>/geometry/material/Shader
        # but we keep a small set of fallbacks to be robust to USD layout changes.
        for env_id in env_ids:
            base = f"/World/envs/env_{env_id.item()}/HoopRing"
            candidate_shader_paths = (
                f"{base}/geometry/material/Shader",
                f"{base}/geometry/material/shader/Shader",
                f"{base}/geometry/material/PreviewSurface",
                f"{base}/Looks/Material/Shader",
                f"{base}/Looks/Material/previewShader",
            )
            shader_prim = None
            for prim_path in candidate_shader_paths:
                prim = get_current_stage().GetPrimAtPath(prim_path)
                if prim:
                    shader_prim = prim
                    break

            if shader_prim is None:
                continue

            shader = UsdShade.Shader(shader_prim)
            if shader:
                shader.GetInput("diffuseColor").Set(Gf.Vec3f(color))

    # ------------------------------------------------------------------
    # Rewards – binary hoop score only
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = self._throw_ball.data.root_pos_w
        ball_to_hoop_xy = torch.linalg.norm(
            ball_pos[:, :2] - self._hoop_pos[:, :2], dim=1
        )
        ball_height_diff = (ball_pos[:, 2] - self._hoop_pos[:, 2]).abs()

        hoop_radius = HOOP_RING_CFG.spawn.radius
        through_hoop = (
            self._ball_released
            & ~self._ball_passed_through_hoop
            & (ball_to_hoop_xy < hoop_radius)
            & (ball_height_diff < 0.15)
        )
        if through_hoop.any():
            self.set_hoop_color(through_hoop.nonzero(as_tuple=False).squeeze(-1), (0.0, 1.0, 0.0))
        self._ball_passed_through_hoop = self._ball_passed_through_hoop | through_hoop

        reward = through_hoop.float() * self.cfg.ball_through_hoop_reward

        self._ball_throw_episode_sums["ball_through_hoop"] += reward
        return reward

    def get_ee_pos(self, drone_pos: torch.Tensor, ee_ori: torch.Tensor):
        return calculate_required_pos(ee_ori, drone_pos.transpose(1, 2), torch.zeros_like(drone_pos.transpose(1, 2)), -self.arm_length, self._robot._ALL_INDICES)

    def joint_to_rpy(self, yaw, shoulder, wrist):
        """
        Convert quadrotor yaw, shoulder, and wrist angles to roll, pitch, and yaw. Also returns the quaternion for free
        Note that this is only approximate since the true rotation axis for the yaw is
        the quadrotor z, not global z - essentially dictating shoulder angles for an instantaneous hover
        """

        yaw_quat = quat_from_angle_axis(yaw, torch.tensor([[0.0, 0.0, 1.0]], device=self.device).tile(*yaw.shape, 1))
        # apply yaw rotation to quadrotor forward axis
        forward_axis_rotated = quat_apply(yaw_quat, torch.tensor([[1.0, 0.0, 0.0]], device=self.device).tile(*yaw_quat.shape[:-1], 1))

        shoulder_quat = quat_from_angle_axis(shoulder, forward_axis_rotated)

        # apply yaw and shoulder rotation to end effetor x-axis (y)
        q2 = quat_mul(shoulder_quat, yaw_quat)  
        ee_axis_rotated = quat_apply(q2, torch.tensor([[0.0, 1.0, 0.0]], device=self.device).tile(*yaw_quat.shape[:-1], 1))

        # quaternion for wrist rotation
        wrist_quat = quat_from_angle_axis(wrist, ee_axis_rotated)

        # calculate final quaternion as product of all 3
        q = quat_mul(wrist_quat, q2)
        res = euler_xyz_from_quat(q.view(-1, 4)) # need to reshape for function call
        return res[0].view(*q.shape[:-1]), res[1].view(*q.shape[:-1]), res[2].view(*q.shape[:-1]), q

    def set_polynomial_trajectory(self, env_ids: torch.Tensor):
        """ For trajectory phase 1 - calculate the 3rd degree polynomical coeffcients for the x, y, and z positon
        s.t. x(0) = init_ee_pos, v(0) = 0, x(t_hover) = hoop_hover_pos, v(t_hover) = 0
        """
        T = self.cfg.hoop_hover_time
        N = len(env_ids)

        degree = self.cfg.polynomial_degree
        if degree == 3:
            A_mat = torch.tensor([
                [1, 0, 0, 0], 
                [0, 1, 0, 0],
                [1, T, T**2, T**3],
                [0, 1, 2*T, 3*T**2],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)
        elif degree == 2:
            A_mat = torch.tensor([
                [1, 0, 0],
                [1, T, T**2],
                [0, 1, 2*T],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)

        B_mat = torch.zeros(N, degree + 1, 6, device=self.device) # 0 velocities at target position, and initial position if degree == 3
        B_mat[:, 0, :3] = self.init_ee_pos[env_ids] # initial position
        B_mat[:, -2, :3] = self.hoop_hover_pos[env_ids] # target position
        B_mat[:, -1, :3] = torch.tensor(self.cfg.hoop_hover_vel, device=self.device).unsqueeze(0).tile(N, 1) # drone velocity

        B_mat[:, 0, 3] = self.init_quad_yaw[env_ids] # initial yaw
        B_mat[:, -2, 3] = self.cfg.hoop_hover_yaw_angle # target yaw

        B_mat[:, -2, 4] = self.cfg.hoop_hover_shoulder_angle # target shoulder angle

        B_mat[:, 0, 5] = self.init_wrist_angle[env_ids] # initial wrist angle
        B_mat[:, -2, 5] = self.cfg.hoop_hover_wrist_angle # target wrist angle

        coeffs = torch.linalg.solve(A_mat, B_mat)
        self.poly_coefficients[env_ids, 0] = coeffs[..., 0]
        self.poly_coefficients[env_ids, 1] = coeffs[..., 1]
        self.poly_coefficients[env_ids, 2] = coeffs[..., 2]
        self.poly_coefficients[env_ids, 3] = coeffs[..., 3]
        self.poly_coefficients[env_ids, 4] = coeffs[..., 4]
        self.poly_coefficients[env_ids, 5] = coeffs[..., 5]

        # Throwing phase data
        T = self.cfg.ball_release_time - self.cfg.hoop_hover_time
        if degree == 3:
            A_mat = torch.tensor([
                [1, 0, 0, 0], 
                [0, 1, 0, 0],
                [1, T, T**2, T**3],
                [0, 1, 2*T, 3*T**2],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)
        elif degree == 2:
            A_mat = torch.tensor([
                [1, 0, 0],
                [1, T, T**2],
                [0, 1, 2*T],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)

        B_mat = torch.zeros(N, degree + 1, 6, device=self.device) # 0 velocities at target position, and initial position if degree == 3
        B_mat[:, 0, :3] = self.hover_pos_drone_end[env_ids] # initial position
        B_mat[:, -2, :3] = self.hoop_throw_pos[env_ids] # target position
        B_mat[:, 1, :3] = torch.tensor(self.cfg.hoop_hover_vel, device=self.device).unsqueeze(0).tile(N, 1) # drone velocity
        B_mat[:, -1, :3] = torch.tensor(self.cfg.throw_vel_drone, device=self.device).unsqueeze(0).tile(N, 1) # drone velocity

        B_mat[:, 0, 3] = self.cfg.hoop_hover_yaw_angle # initial yaw angle
        B_mat[:, -2, 3] = self.cfg.throw_yaw_angle # target yaw angle

        B_mat[:, 0, 4] = self.cfg.hoop_hover_shoulder_angle # initial shoulder angle
        B_mat[:, -2, 4] = self.cfg.throw_shoulder_angle # target shoulder angle
        B_mat[:, -1, 4] = self.cfg.throw_vel_shoulder # joint angle throwing speed
    
        B_mat[:, 0, 5] = self.cfg.hoop_hover_wrist_angle # initial wrist angle
        B_mat[:, -2, 5] = self.cfg.throw_wrist_angle # target wrist angle
        coeffs = torch.linalg.solve(A_mat, B_mat)
        self.poly_coefficients_throw[env_ids, 0] = coeffs[..., 0]
        self.poly_coefficients_throw[env_ids, 1] = coeffs[..., 1]
        self.poly_coefficients_throw[env_ids, 2] = coeffs[..., 2]
        self.poly_coefficients_throw[env_ids, 3] = coeffs[..., 3]
        self.poly_coefficients_throw[env_ids, 4] = coeffs[..., 4]
        self.poly_coefficients_throw[env_ids, 5] = coeffs[..., 5]

        T = self.cfg.follow_through_time - self.cfg.ball_release_time
        if degree == 3:
            A_mat = torch.tensor([
                [1, 0, 0, 0], 
                [0, 1, 0, 0],
                [1, T, T**2, T**3],
                [0, 1, 2*T, 3*T**2],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)
        elif degree == 2:
            A_mat = torch.tensor([
                [1, 0, 0],
                [1, T, T**2],
                [0, 1, 2*T],
            ], device=self.device).unsqueeze(0).tile(N, 1, 1)

        B_mat = torch.zeros(N, degree + 1, 6, device=self.device) # 0 velocities at target position, and initial position if degree == 3
        B_mat[:, 0, :3] = self.hoop_throw_pos[env_ids] # initial position
        B_mat[:, -2, :3] = self.follow_through_pos[env_ids] # target position
        B_mat[:, 1, :3] = torch.tensor(self.cfg.throw_vel_drone, device=self.device).unsqueeze(0).tile(N, 1) # drone vel at throw end = start of follow through drone vel


        B_mat[:, 0, 3] = self.cfg.throw_yaw_angle # initial yaw angle
        B_mat[:, -2, 3] = self.cfg.follow_through_yaw_angle # target yaw angle

        B_mat[:, 0, 4] = wrap_to_pi(torch.tensor(self.cfg.throw_shoulder_angle, device=self.device)) # initial shoulder angle
        B_mat[:, -2, 4] = self.cfg.follow_through_shoulder_angle # target shoulder angle
        B_mat[:, 1, 4] = self.cfg.throw_vel_shoulder # joint angle throwing speed continuity

        B_mat[:, 0, 5] = self.cfg.throw_wrist_angle # initial wrist angle
        B_mat[:, -2, 5] = self.cfg.follow_through_wrist_angle # target wrist angle
        B_mat[:, 1, 5] = self.cfg.throw_vel_wrist # joint angle throwing speed continuity
        coeffs = torch.linalg.solve(A_mat, B_mat)
        self.poly_coefficients_follow_through[env_ids, 0] = coeffs[..., 0]
        self.poly_coefficients_follow_through[env_ids, 1] = coeffs[..., 1]
        self.poly_coefficients_follow_through[env_ids, 2] = coeffs[..., 2]
        self.poly_coefficients_follow_through[env_ids, 3] = coeffs[..., 3]
        self.poly_coefficients_follow_through[env_ids, 4] = coeffs[..., 4]
        self.poly_coefficients_follow_through[env_ids, 5] = coeffs[..., 5]
    
    # ------------------------------------------------------------------
    # Reset – parent handles robot; we add ball + hoop + mass bookkeeping
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log ball-throw metrics before clearing
        extras = dict()
        extras["Metrics/BallThroughHoop"] = (
            self._ball_passed_through_hoop[env_ids].float().mean().item()
        )

         # Randomize hoop position
        hoop_center = torch.tensor(self.cfg.hoop_pos_center, device=self.device)
        hoop_range = torch.tensor(self.cfg.hoop_pos_range, device=self.device)
        random_offset = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * hoop_range
        self._hoop_pos[env_ids] = hoop_center + random_offset
        self._hoop_pos[env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # Place hoop ring at target
        hoop_ring_state = self._hoop_ring.data.default_root_state[env_ids].clone()
        hoop_ring_state[:, :3] = self._hoop_pos[env_ids]
        hoop_ring_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._hoop_ring.write_root_pose_to_sim(hoop_ring_state[:, :7], env_ids=env_ids)
        self._hoop_ring.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

        # Place hoop post under the ring
        post_state = self._hoop_post.data.default_root_state[env_ids].clone()
        post_height = 1.5
        post_state[:, :2] = self._hoop_pos[env_ids, :2]
        post_state[:, 2] = post_height / 2.0
        post_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._hoop_post.write_root_pose_to_sim(post_state[:, :7], env_ids=env_ids)
        self._hoop_post.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

        # Let the parent handle robot reset, trajectory init, logging, etc.
        # self.in_reset[env_ids] = True
        super()._reset_idx(env_ids)
        # self.in_reset[env_ids] = False

        # Override the initial position so that parent reset only used for metrics and setting some physics properties (ee mass, motor speeds)
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self._hoop_pos[env_ids] + torch.tensor([[6.0, 0.0, 0.2]], device=self.device)
        if self.cfg.init_cfg == "rand":
            pos_rand = (torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_pos_ranges, device=self.device).float()
            default_root_state[:, :3] = default_root_state[:, :3] + pos_rand
            default_root_state[:, 2] -= 0.2
            yaw_rand = (torch.rand(len(env_ids), 1, device=self.device) * 2.0 - 1.0) * torch.tensor(self.cfg.init_yaw_ranges, device=self.device).float()
            default_root_state[:, 3:7] = quat_from_yaw(yaw_rand.squeeze(1))
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)

        # Merge our extras into the parent's log
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Reset ball state flags
        self._ball_attached[env_ids] = True
        self._ball_released[env_ids] = False
        self._ball_passed_through_hoop[env_ids] = False
        self._ball_throw_episode_sums["ball_through_hoop"][env_ids] = 0.0

        # Restore hoop ring color to default (red) on reset.
        self.set_hoop_color(env_ids, (0.9, 0.1, 0.1))

        # --- EE mass bookkeeping ---
        # BallThrowEventCfg zeros the EE mass at startup.  On each reset we
        # set it to exactly ball_mass (ball is attached).  On release it gets
        # set back to zero.
        if self._ee_base_mass is None:
            self._ee_base_mass = torch.zeros(self.num_envs, device=self.device)
        self._ee_base_mass[env_ids] = 0.0

        self._set_ee_mass(
            env_ids,
            torch.full((len(env_ids),), self.cfg.ball_mass, device=self.device),
        )

        # Place ball at end-effector
        ee_pos = self._robot.data.body_pos_w[:, self._ee_id].squeeze(1)
        ee_ori = self._robot.data.body_quat_w[:, self._ee_id].squeeze(1)
        ball_state = self._throw_ball.data.default_root_state[env_ids].clone()
        ball_state[:, :3] = ee_pos[env_ids]
        ball_state[:, 3:7] = ee_ori[env_ids]
        ball_state[:, 7:] = 0.0
        self._throw_ball.write_root_pose_to_sim(ball_state[:, :7], env_ids=env_ids)
        self._throw_ball.write_root_velocity_to_sim(
            ball_state[:, 7:13], env_ids=env_ids
        )

        self.init_ee_pos[env_ids] = ee_pos[env_ids]
        self.init_quad_yaw[env_ids] = yaw_from_quat(self._robot.data.body_quat_w[env_ids, self._body_id].squeeze(1)[env_ids])
        self.init_shoulder_angle[env_ids] = wrap_to_pi(self._robot.data.joint_pos[env_ids, self._shoulder_joint_idx])
        self.init_wrist_angle[env_ids] = wrap_to_pi(self._robot.data.joint_pos[env_ids, self._wrist_joint_idx])
        self.hoop_hover_pos[env_ids] = self._hoop_pos[env_ids] + torch.tensor(self.cfg.hoop_hover_offset, device=self.device).unsqueeze(0)
        self.follow_through_pos[env_ids] = self._hoop_pos[env_ids] + torch.tensor(self.cfg.follow_through_offset, device=self.device).unsqueeze(0)
        self.hover_pos_drone_end = calculate_required_pos(
            self.hoop_hover_ori.squeeze(1), self.hoop_hover_pos, self.hover_pos_drone_end, self.arm_length, env_ids
        )
        self.hoop_throw_pos[env_ids] = self._hoop_pos[env_ids] + torch.tensor(self.cfg.hoop_throw_offset, device=self.device).unsqueeze(0)
        self.set_polynomial_trajectory(env_ids)


        