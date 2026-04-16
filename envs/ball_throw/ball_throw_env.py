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

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.shapes import SphereCfg, CylinderCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils.math import quat_from_euler_xyz

from configs.aerial_manip_asset import AERIAL_MANIPULATOR_2DOF_CFG
from utils.math_utilities import calculate_required_pos

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

from envs.trajectory_tracking.trajectory_tracking_env_2dof import (
    AerialManipulatorTrajectoryTrackingEnv,
    AerialManipulatorTrajectoryTrackingEnvBaseCfg,
    EventCfg,
)


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
        radius=0.1,
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
    hoop_pos_center = [3.0, 0.0, 2.5]
    hoop_pos_range = [1.0, 1.0, 0.5]

    # Ball release time in seconds
    ball_release_time = 3.0

    # Ball mass added to the EE link while attached (kg)
    ball_mass = 0.1

    # Force eval mode so we get the full state logged
    eval_mode = True
    init_cfg = "rand"

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

        hover_pos_shifted = self._hoop_pos.clone() #torch.tensor(self.cfg.drone_spawn_pos, device=self.device)
        # hover_pos_shifted = hover_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        hover_pos_shifted[:, 2] += 1.0
        # hover_pos_shifted[:, :2] += self._terrain.env_origins[:, :2]

        # Iteration 1: follow linear path to target position over hoop
        arrive_time = self.cfg.ball_release_time - 0.5
        arrive_step = int(arrive_time * self.cfg.policy_rate_hz)
        moving_mask = (self.episode_length_buf <= arrive_step)
        hover_mask = ~moving_mask
        curr_time = self.episode_length_buf.unsqueeze(-1)
        future_timesteps = torch.arange(0, 1+self.cfg.trajectory_horizon, device=self.device)
        time = (curr_time + future_timesteps.unsqueeze(0)) * self.cfg.traj_update_dt
        slopes = (hover_pos_shifted - self.init_ee_pos) / arrive_time
        target_pos = self.init_ee_pos.unsqueeze(-1) + slopes.unsqueeze(-1) * time.unsqueeze(1)

        # Current goal and full horizon: constant hover position, identity orientation
        self._desired_pos_w[hover_mask] = hover_pos_shifted[hover_mask]
        self._desired_pos_w[moving_mask] = target_pos[moving_mask, :, 0]
        self._desired_ori_w[ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self._desired_pos_traj_w[hover_mask] = hover_pos_shifted[hover_mask].unsqueeze(1).expand(
            -1, 1 + self.cfg.trajectory_horizon, -1
        )
        self._desired_pos_traj_w[moving_mask] = target_pos[moving_mask].transpose(1,2)
        self._desired_ori_traj_w[ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).expand(len(ids), 1 + self.cfg.trajectory_horizon, -1)

        # Zero-velocity trajectory references so the policy sees zero desired
        # velocity / angular velocity (hover).
        # _pos_traj shape: (5, num_envs, 3, 1+horizon)
        # _roll/pitch/yaw_traj shape: (5, num_envs, 1+horizon)
        self._pos_traj[:, ids] = 0.0
        self._pos_traj[0, hover_mask] = hover_pos_shifted[hover_mask].unsqueeze(-1).tile(
            1, 1, 1 + self.cfg.trajectory_horizon
        )
        self._pos_traj[0, moving_mask] = target_pos[moving_mask]
        self._pos_traj[1, moving_mask] = slopes[moving_mask].unsqueeze(2).expand(
            -1, -1, 1 + self.cfg.trajectory_horizon
        ) # reference trajectory velocity

        self._roll_traj[:, ids] = 0.0
        self._pitch_traj[:, ids] = 0.0
        self._yaw_traj[:, ids] = 0.0

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

    # ------------------------------------------------------------------
    # Rewards – binary hoop score only
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = self._throw_ball.data.root_pos_w
        ball_to_hoop_xy = torch.linalg.norm(
            ball_pos[:, :2] - self._hoop_pos[:, :2], dim=1
        )
        ball_height_diff = (ball_pos[:, 2] - self._hoop_pos[:, 2]).abs()

        hoop_radius = 0.25
        through_hoop = (
            self._ball_released
            & ~self._ball_passed_through_hoop
            & (ball_to_hoop_xy < hoop_radius)
            & (ball_height_diff < 0.15)
        )
        self._ball_passed_through_hoop = self._ball_passed_through_hoop | through_hoop

        reward = through_hoop.float() * self.cfg.ball_through_hoop_reward

        self._ball_throw_episode_sums["ball_through_hoop"] += reward
        return reward

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

        # Let the parent handle robot reset, trajectory init, logging, etc.
        super()._reset_idx(env_ids)

        # Merge our extras into the parent's log
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Reset ball state flags
        self._ball_attached[env_ids] = True
        self._ball_released[env_ids] = False
        self._ball_passed_through_hoop[env_ids] = False
        self._ball_throw_episode_sums["ball_through_hoop"][env_ids] = 0.0

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
