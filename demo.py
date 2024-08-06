# These imports need to go first
from omni.isaac.lab.app import AppLauncher
import argparse # Used for Isaac
parser = argparse.ArgumentParser(description="Run demo with Isaac Sim")
parser.add_argument("--time_step", type=float, default=1.0 / 60.0, help="Time step for the simulation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless=False
# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Local imports
from configs.aerial_manip_asset import AERIAL_MANIPULATOR_CFG

# Isaac Lab/Isaac Sim imports
# import isaacsim
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from omni.isaac.lab_assets import CRAZYFLIE_CFG


import tyro # used for everything else
import torch
import numpy as np


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Robots
    crazyflie_cfg = CRAZYFLIE_CFG
    crazyflie_cfg.spawn.func("/World/Crazyflie/Robot_1", crazyflie_cfg.spawn, translation=(0.0, 0.0, 0.0))

    aerial_manipulator_cfg = AERIAL_MANIPULATOR_CFG
    print("Soft Limits: ", aerial_manipulator_cfg.soft_joint_pos_limit_factor)
    aerial_manipulator_cfg.spawn.func("/World/AerialManipulator/Robot_1", aerial_manipulator_cfg.spawn, translation=(0.0, 0.0, 0.0))

    # create handles for the robots
    origins = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    crazyflie = Articulation(crazyflie_cfg.replace(prim_path="/World/Crazyflie/Robot.*"))
    aerial_manipulator = Articulation(aerial_manipulator_cfg.replace(prim_path="/World/AerialManipulator/Robot.*"))

    # Create Marker for visualization
    marker_cfg = VisualizationMarkersCfg(prim_path="/Visuals/Markers",
                                         markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),})
    marker = VisualizationMarkers(marker_cfg)

    scene_entities = {"crazyflie": crazyflie, "aerial_manipulator": aerial_manipulator, "marker": marker}
    return scene_entities, origins

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 1.0, 0.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    crazyflie = scene_entities["crazyflie"]
    aerial_manipulator = scene_entities["aerial_manipulator"]
    marker_frame = scene_entities["marker"]

    marker_location = torch.tensor([[1.0, 1.0, 0.5]], device=sim.device)
    marker_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    crazyflie_prop_body_ids = crazyflie.find_bodies("m.*_prop")[0]
    aerial_manipulator_body_ids = aerial_manipulator.find_bodies("vehicle.*")[0]
    aerial_manipulator_joints = aerial_manipulator.find_joints(".*joint.*")[0]
    crazyflie_mass = crazyflie.root_physx_view.get_masses().sum()
    aerial_manipulator_mass = aerial_manipulator.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()



    import code; code.interact(local=locals())

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            sim_time = 0.0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            # reset dof state
            joint_pos, joint_vel = crazyflie.data.default_joint_pos, crazyflie.data.default_joint_vel
            crazyflie.write_joint_state_to_sim(joint_pos, joint_vel)
            crazyflie.write_root_pose_to_sim(crazyflie.data.default_root_state[:, :7])
            crazyflie.write_root_velocity_to_sim(crazyflie.data.default_root_state[:, 7:])
            crazyflie.reset()

            joint_pos, joint_vel = aerial_manipulator.data.default_joint_pos, aerial_manipulator.data.default_joint_vel
            aerial_manipulator.write_joint_state_to_sim(joint_pos, joint_vel)
            aerial_manipulator.write_root_pose_to_sim(aerial_manipulator.data.default_root_state[:, :7])
            aerial_manipulator.write_root_velocity_to_sim(aerial_manipulator.data.default_root_state[:, 7:])
            aerial_manipulator.reset()
            print("[INFO]: Resetting Aerial manipulator: ", aerial_manipulator.data.default_root_state)

            desired_joint_pos = torch.tensor([np.random.uniform(-3, 3), np.random.uniform(-3, 3)], device=sim.device).float()
            desired_joint_vel = torch.tensor([0.0, 0.0], device=sim.device).float()
            desired_joint_effort = torch.tensor([0.0, 0.0], device=sim.device).float()
            print("Desired Joint Pos: ", desired_joint_pos)

            print("[INFO]: Resetting Crazyflie state...")

        # Update marker
        marker_frame.visualize(marker_location, marker_orientation)
        # apply action to the robot (make the robot float in place)
        forces = torch.zeros(1, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        forces[..., 2] = crazyflie_mass * gravity / 4.0
        crazyflie.set_external_force_and_torque(forces, torques, body_ids=crazyflie_prop_body_ids)
        crazyflie.write_data_to_sim()


        # aerial_manipulator.write_root_pose_to_sim(aerial_manipulator.data.default_root_state[:, :7])
        # aerial_manipulator.write_root_velocity_to_sim(aerial_manipulator.data.default_root_state[:, 7:])

        
        # print("Desired Joint Pos: ", desired_joint_pos)
        aerial_manipulator.set_joint_position_target(desired_joint_pos, joint_ids=aerial_manipulator_joints)
        aerial_manipulator.set_joint_velocity_target(desired_joint_vel, joint_ids=aerial_manipulator_joints)
        aerial_manipulator.set_joint_effort_target(desired_joint_effort, joint_ids=aerial_manipulator_joints)
        
        print("Joint Torques Computed: ", aerial_manipulator.data.computed_torque)
        print("Joint Torques Applied: ", aerial_manipulator.data.applied_torque)
        aerial_manipulator.update(sim_dt)
        aerial_manipulator.write_data_to_sim()

        # forces = torch.zeros(1, 1, 3, device=sim.device)
        # torques = torch.zeros_like(forces)
        # # forces[..., 2] = aerial_manipulator_mass * gravity
        # aerial_manipulator.set_external_force_and_torque(forces, torques, body_ids=aerial_manipulator_body_ids)
        # aerial_manipulator.write_data_to_sim()

        # Perform step
        sim.step()
        # Increment counter
        count += 1
        sim_time += sim_dt
        # Update buffers
        crazyflie.update(sim_dt)

if __name__ == "__main__":
    main()
    