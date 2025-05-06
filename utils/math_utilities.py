import torch
import omni.isaac.lab.utils.math as isaac_math_utils
from typing import Tuple

def exp_so3(S):
    pass

def matrix_log(S):
    pass

@torch.jit.script
def vee_map(S):
    """Convert skew-symmetric matrix to vector.

    Args:
        S: The skew-symmetric matrix. Shape is (N, 3, 3).

    Returns:
        The vector representation of the skew-symmetric matrix. Shape is (N, 3).
    """
    return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=1)

@torch.jit.script
def hat_map(v):
    """Convert vector to skew-symmetric matrix.

    Args:
        v: The vector. Shape is (N, 3).

    Returns:
        The skew-symmetric matrix representation of the vector. Shape is (N, 3, 3).
    """
    return isaac_math_utils.skew_symmetric_matrix(v)

@torch.jit.script
def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Get yaw angle from quaternion.

    Args:
        q: The quaternion. Shape is (..., 4).
        q = [w, x, y, z]

    Returns:
        The yaw angle. Shape is (...,).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    yaw = torch.atan2(2.0 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]), -1.0 + 2.0*(q[:,0]**2 + q[:,1]**2))
    # yaw = torch.atan2(2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]), q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2)
    # yaw3 = torch.atan2(2.0 * (q[:, 1] * q[:, 0] + q[:, 2] * q[:, 3]), 1.0 - 2.0*(q[:,0]**2 + q[:,1]**2))
    return yaw.reshape(shape[:-1])

def body_yaw_error_from_quats(q1: torch.Tensor, q2: torch.Tensor):
    '''
    compute the yaw error of the body for the 2DOF case
    q1 = body or ee quaternion, depending on implementation
    q2 = goal quaternion

    return values are error in radians
    '''
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    
    #Find vector "b2" that is the y-axis of the rotated frame
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((q2.shape[0], 1)))

    # perform z-correction on goal orientations that have at least one nonzero horizontal (x or y) component
    has_x = torch.nonzero(b2[:, 0])
    has_y = torch.nonzero(b2[:, 1])
    has_horiz = torch.cat((has_x, has_y)).unique()
    b2[has_horiz, 2] = 0.0
    b2 = torch.nn.functional.normalize(b2, dim=1)

    dots =(b1*b2).sum(dim=1)
    dots = torch.reshape(dots, (-1, 1))
    errors = torch.zeros_like(dots)
    errors[has_horiz] = torch.arccos(torch.clamp(dots[has_horiz], -1.0+1e-8, 1.0-1e-8))
    return torch.abs(errors)

def calculate_required_shoulder(q: torch.Tensor, angles: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the shoulder's required angle given the goal orientation of the end effector frame at specified
    environment ids

    Args: 
        q: Quaternions for the goal. Shape (..., 4)
        angles: Current estimates for the required angle. Shape (..., 1)
        env_ids: Indices where updates are required
    '''

    # Get local y vector of the target frame in world coords
    b = isaac_math_utils.quat_rotate(q[env_ids], torch.tensor([0.0, 1.0, 0.0], device=q.device).tile((q[env_ids].shape[0], 1)))

    # print('INDEX SHAPE: ', angles[env_ids].shape)
    # print('ARCSIN SHAPE: ', torch.arcsin(torch.clamp(b[:, -1], -1.0+1e-8, 1.0-1e-8)).shape)
    angles[env_ids] = torch.reshape(torch.arcsin(torch.clamp(b[:, -1], -1.0+1e-8, 1.0-1e-8)), (-1, 1))

    return angles

def shoulder_angle_error_from_quats(q1: torch.Tensor, q2: torch.Tensor):
    '''
    compute the shoulder joint angle error from the ee orientation (q1) and goal orientation (q2)

    returns the error in radians
    '''
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    
    #Find vector "b2" that is the y-axis of the rotated frame
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((q2.shape[0], 1)))

    b1_shoulder_angles = torch.arcsin(torch.clamp(b1[:, -1], -1.0+1e-8, 1.0-1e-8))
    b2_shoulder_angles = torch.arcsin(torch.clamp(b2[:, -1], -1.0+1e-8, 1.0-1e-8))

    return torch.abs(b2_shoulder_angles - b1_shoulder_angles)

def calculate_required_wrist(q: torch.Tensor, angles: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the wrist's required angle given the goal orientation of the end effector frame at specified
    environment ids

    Args: 
        q: Quaternions for the goal. Shape (..., 4)
        angles: Current estimates for the required angle. Shape (..., 1)
        env_ids: Indices where updates are required
    '''

    # Get local x vector of the target frame in world coords
    b = isaac_math_utils.quat_rotate(q[env_ids], torch.tensor([1.0, 0.0, 0.0], device=q.device).tile((q[env_ids].shape[0], 1)))

    angles[env_ids] = torch.reshape(torch.arcsin(torch.clamp(b[:, -1], -1.0+1e-8, 1.0-1e-8)), (-1, 1))

    return angles

def wrist_angle_error_from_quats(q1: torch.Tensor, q2: torch.Tensor):
    '''
    Args:
        q1: current EE rotation, (..., 4)
        q2: target EE rotation (..., 4)

    Returns abs value of the error of the wrist angle
    '''
    
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    # Step 1: Find the quaternion that will rotate the local y-vector of frame 1 to frame 2 
    # (this is the vector that points axially through the EE).
    # Based on https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    y_1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    y_2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q2.shape[0], 1)))
    # needed if y_1 and y_2 are aligned, plus for a later calculation
    x_1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    dot = (y_1 * y_2).sum(dim=1) # unit vectors, also gives cos(theta)
    dot = torch.clamp(dot, -1.0+1e-8, 1.0-1e-8)
    c_half = torch.sqrt((1.0 + dot) / 2)
    s_half = torch.sqrt((1.0 - dot) / 2).reshape((-1, 1))
    axis = torch.linalg.cross(y_1, y_2)
    axis = torch.nn.functional.normalize(axis, dim=1)
    q_12 = torch.zeros_like(q1)
    q_12[:, 0] = c_half
    q_12[:, 1:] = axis * s_half
    where_zero = dot < (-1 + 1e-6)
    q_12[where_zero, 0] = 0.0
    q_12[where_zero, 1:] = x_1[where_zero]

    x_1_transform = isaac_math_utils.quat_rotate(q_12, x_1)

    # Step 2: Now that we have rotated the EE x-axis onto the frame where the EE y-axis and goal y-axis are the
    # same, calculate the angular error between them
    x_2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))  
    dots = (x_1_transform * x_2).sum(dim=1)
    ans = torch.abs(torch.arccos(torch.clamp(dots, -1+1e-8, 1-1e-8))).reshape(-1, 1)
    if torch.any(torch.isnan(ans)):
        mask = torch.isnan(ans).squeeze()
        print('found nans')
        print('q_12 where nan:')
        print(q_12[mask])
        print('original dot product where nan:')
        print(dot[mask])
        print('calculated axis where nan:')
        print(axis[mask])
        print('s_half where nan:')
        print(s_half[mask])
    return ans






def calculate_required_yaw(q: torch.Tensor, yaw: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the quadrotor's required yaw angle given the goal orientation of the end effector frame at specified
    environment ids

    Args: 
        q: Quaternions for the goal. Shape (..., 4)
        yaw: Current estimates for the required yaw angle. Shape (..., 1)
        env_ids: Indices where updates are required
    '''

    ## NOTE: could probably use Isaac's yaw from quaternion function 

    # Get local y vector of the target frame in world coords
    b = isaac_math_utils.quat_rotate(q[env_ids], torch.tensor([0.0, 1.0, 0.0], device=q.device).tile((q[env_ids].shape[0], 1)))

    # if the local y vector is aligned with the global z vector, yaw can be any angle - use convention of angle = 0 in this case
    new_yaws = torch.zeros((b.shape[0], 1), device=yaw.device)
    has_x = torch.nonzero(b[:, 0])
    has_y = torch.nonzero(b[:, 1])
    has_horiz = torch.cat((has_x, has_y)).unique()
    b[has_horiz, 2] = 0.0
    b = torch.nn.functional.normalize(b, dim=1)

    global_y = torch.zeros_like(b[has_horiz], device=yaw.device)
    global_y[:, 1] = 1.0

    # only doing the calculation on indices where there is a horizontal component
    dots = (b[has_horiz]*global_y).sum(dim=1)
    dots = torch.reshape(dots, (-1, 1))
    new_yaws[has_horiz] = torch.arccos(torch.clamp(dots, -1.0+1e-8, 1.0-1e-8))
    yaw[env_ids] = new_yaws
    return yaw


def calculate_required_pos(q: torch.Tensor, p_goal: torch.Tensor, p_guess: torch.Tensor,
                            arm_length: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    '''
    Calculates the quadrotor's required position given the goal orientation and position
    of the end effector frame at specified environment ids

    Args: 
        q: Quaternions for the goal. Shape (..., 4)
        p_goal: Goal position. Shape (..., 3)
        p_estimate: Current estimates for the required position. Shape (..., 3)
        arm_length: EE arm length. Scalar
        env_ids: Indices where updates are required
    '''
    # Get local y vector of the target frame in world coords
    b = isaac_math_utils.quat_rotate(q[env_ids], torch.tensor([0.0, 1.0, 0.0], device=q.device).tile((q[env_ids].shape[0], 1)))

    # Subtract transformed vectors scaled by arm length from the goal position
    p_guess[env_ids] = p_goal[env_ids] - arm_length.item() * b
    return p_guess

def calculate_required_angles(q: torch.Tensor, angles: torch.Tensor, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Calculates the three required angles (yaw, shoulder, wrist) of the end effector given the goal orientation and specificed environement ids
    (likely incorrect)
    Args:
        q: Goal quaternion (... , 4)
        angles: Current estimate (..., 3)
        env_ids: Update indices
    '''

    roll, pitch, yaw = isaac_math_utils.euler_xyz_from_quat(q[env_ids])
    angles[env_ids, 0] = roll
    angles[env_ids, 1] = pitch
    angles[env_ids, 2] = yaw

    return angles




def yaw_error_from_quats(q1: torch.Tensor, q2: torch.Tensor, dof:int) -> torch.Tensor:
    """Get yaw error between two quaternions.

    Args:
        q1: The first quaternion. Shape is (..., 4).
        q2: The second quaternion. Shape is (..., 4).

    Returns:
        The yaw error. Shape is (...,).
    """
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    
    #Find vector "b2" that is the y-axis of the rotated frame
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q2.shape[0], 1)))

    # changed this so that yaw always only looks at horizontal (xy) components
    # if dof == 0:
    b1[:,2] = 0.0
    b2[:,2] = 0.0
    dot = (b1*b2).sum(dim=1)
    reward = torch.ones_like(dot)
    has_horiz = torch.logical_and(torch.logical_or(b1[:, 0] != 0.0, b1[:, 1] != 0.0),
                                  torch.logical_or(b2[:, 0] != 0.0, b2[:, 1] != 0.0))
    b1_norm = torch.norm(b1, dim=-1)
    b2_norm = torch.norm(b2, dim=-1)
    prod = b1_norm * b2_norm
    # operand = (b1*b2).sum(dim=1) / (b1_norm * b2_norm)
    reward[has_horiz] = dot[has_horiz] / prod[has_horiz]
    return torch.arccos(torch.clamp(reward, -1.0+1e-8, 1.0-1e-8)).view(shape1[:-1])

@torch.jit.script
def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Get quaternion from yaw angle.

    Args:
        yaw: The yaw angle. Shape is (...,).

    Returns:
        The quaternion. Shape is (..., 4).
    """
    shape = yaw.shape
    yaw = yaw.view(-1)
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2.0)
    q[:, 1] = 0.0
    q[:, 2] = 0.0
    q[:, 3] = torch.sin(yaw / 2.0)
    return q.view(shape + (4,))


@torch.jit.script
def compute_desired_pose_from_transform(
    goal_pos_w: torch.Tensor,
    goal_ori_w: torch.Tensor,
    pos_transform: torch.Tensor,
    num_joints: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the desired position and yaw from the given transform.
    
    Args:
        goal_pos_w (Tensor): Goal positions in world frame (batch_size, 3).
        goal_ori_w (Tensor): Goal orientations as quaternions (batch_size, 4).
        pos_transform (Tensor): Position transforms (batch_size, 3).
        num_joints (int): Number of joints.

    Returns:
        Tuple[Tensor, Tensor]: Desired positions and yaws.
    """
    batch_size = goal_ori_w.shape[0]

    # Rotate the y-axis vector by the goal orientations
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=goal_ori_w.device).unsqueeze(0).expand(batch_size, -1)
    b2 = isaac_math_utils.quat_rotate(goal_ori_w, y_axis)

    # Set the z-component to zero if num_joints == 0
    if num_joints == 0:
        b2 = b2.clone()  # Avoid modifying the original tensor
        b2[:, 2] = 0.0

    b2 = isaac_math_utils.normalize(b2)

    # Compute the desired yaw angle
    yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi / 2
    yaw_desired = isaac_math_utils.wrap_to_pi(yaw_desired)

    # Compute the desired position
    pos_transform_norm = torch.linalg.norm(pos_transform, dim=1, keepdim=True)
    displacement = pos_transform_norm * (-b2)
    pos_desired = goal_pos_w + displacement
    # pos_desired, _ = isaac_math_utils.combine_frame_transforms(goal_pos_w, goal_ori_w, pos_transform)
    # yaw_desired = yaw_from_quat(goal_ori_w)

    return pos_desired, yaw_desired

    