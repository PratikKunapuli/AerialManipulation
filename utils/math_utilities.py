import torch
import numpy as np
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

    # Apply roatations to local y-axis for each frame. For the body frame, we'll also do this for the 
    # negative y-axis since alignment can be achieved by having the body be either parallel or antiparallel
    # to the target frame's y-axis
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    b1_neg = -1.0 * b1
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((q2.shape[0], 1)))

    # Only care about the horizontal components of the y-axes
    b1[:, 2] = 0.0
    b1_neg[:, 2] = 0.0
    b2[:, 2] = 0.0

    # Renormalize the vectors

    # Use functional.normalize so that vectors below a certain magnitude normalize ot a length less than 1 
    b1 = torch.nn.functional.normalize(b1, dim=1, eps=1e-8)
    b1_neg = torch.nn.functional.normalize(b1_neg, dim=1, eps=1e-8)
    b2 = torch.nn.functional.normalize(b2, dim=1, eps=1e-8)

    dot = (b1*b2).sum(dim=1)
    dot_neg = (b1_neg*b2).sum(dim=1)

    yaw_error = torch.zeros_like(dot)
    # If b2 is small in magnitude, this means that the target end effector frame is near-vertical, so aligntment
    # can be achieved with any yaw angle. Only calculate the error if b2 is large enough
    mask = torch.norm(b2, dim=1) < 1.0-1e-10

    pos_error = torch.arccos(torch.clamp(dot, -1.0+1e-8, 1.0-1e-8))
    neg_error = torch.arccos(torch.clamp(dot_neg, -1.0+1e-8, 1.0-1e-8))

    yaw_error[torch.abs(pos_error) < torch.abs(neg_error)] = pos_error[torch.abs(pos_error) < torch.abs(neg_error)]
    yaw_error[torch.abs(pos_error) >= torch.abs(neg_error)] = neg_error[torch.abs(pos_error) >= torch.abs(neg_error)]
    yaw_error[mask] = 0.0

    return yaw_error.reshape(-1, 1)

    
    #Find vector "b2" that is the y-axis of the rotated frame
    # b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    # b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((q2.shape[0], 1)))

    # # perform z-correction on goal orientations that have at least one nonzero horizontal (x or y) component
    # has_x = torch.nonzero(b2[:, 0])
    # has_y = torch.nonzero(b2[:, 1])
    # has_horiz = torch.cat((has_x, has_y)).unique()
    # b2[has_horiz, 2] = 0.0
    # b2 = torch.nn.functional.normalize(b2, dim=1)

    # dots =(b1*b2).sum(dim=1)
    # dots = torch.reshape(dots, (-1, 1))
    # errors = torch.zeros_like(dots)
    # errors[has_horiz] = torch.arccos(torch.clamp(dots[has_horiz], -1.0+1e-8, 1.0-1e-8))
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
    Compute the signed shoulder joint angle error from the ee orientation (q1) and goal orientation (q2)

    Returns the signed error in radians
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

    return (b1_shoulder_angles - b2_shoulder_angles).reshape(-1, 1)

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

    Returns the signed error of the wrist angle in radians
    '''
    # NOTE: need to check whether the signs are technically correct, but if you are using this simply as say a metric or observation, it shouldn't matter
    # as long as you're using the same sign convention when comparing trials
    
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    # Step 1: Determine yaw and shoulder errors, apply those rotations to frame 1 to align end effector axis with target
    yaw_to_apply = -yaw_error_from_quats(q1, q2, 2)
    shoulder_to_apply = -shoulder_angle_error_from_quats(q1, q2)
    yaw_to_apply = quat_from_yaw(yaw_to_apply)
    q1 = isaac_math_utils.quat_mul(yaw_to_apply, q1)

    # axis for the shoulder rotation would be the local x-axis rotated by the yaw rotation
    shoulder_axis = isaac_math_utils.quat_rotate(yaw_to_apply, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    shoulder_to_apply = isaac_math_utils.quat_from_angle_axis(shoulder_to_apply.squeeze(), shoulder_axis)
    q1 = isaac_math_utils.quat_mul(shoulder_to_apply, q1)
    

    # Step 2: Now that we have rotated the EE x-axis onto the frame where the EE y-axis and goal y-axis are the
    # same, calculate the angular error between them
    x_1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    x_2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[1.0, 0.0, 0.0]], device=q2.device).tile((q2.shape[0], 1)))  
    dots = (x_1 * x_2).sum(dim=1)
    ans = torch.arccos(torch.clamp(dots, -1+1e-8, 1-1e-8)).reshape(-1, 1)

    # Step 3: Get the sign of the error by taking cross products of the local x-axes (of the frames where the y-axes are aligned)
    # and use the sign of the resultant product's y-component
    cross = torch.linalg.cross(x_2, x_1)
    cross = isaac_math_utils.quat_rotate_inverse(q1, cross)
    sign = torch.sign(cross[:, 1])
    ans[:, 0] *= sign
    return ans

def _aerial_manipulator_angle_errors(
        q1: torch.Tensor,
        q2: torch.Tensor,
        shape_vec: torch.Tensor = None,
        two_way_yaw: bool = False,
        eps: float = 1e-8,
        debug: bool = False
    ) -> torch.Tensor:
    """
    q1: current, q2: desired, shape_vec: desired shape vector, defaults to z-axis, two_way_yaw: whether to consider the case for -b1,
    eps: small value threshold for considering vectors to be zero
    """
    if debug:
        breakpoint()

    if shape_vec is None:
        shape_vec = torch.zeros((*q1.shape[:-1], 3), device=q1.device)
        shape_vec[..., 2] = 1.0
    else:    
        shape_vec = shape_vec / (shape_vec.norm(dim=-1, keepdim=True) + 1e-8)

    # Step 1, calculate the yaw error - this will be a yaw about the desired shape vector, instead of projecting the desired/actual
    # end effector axes onto the xy plane, project it onto the plane normal to the shape vector.

    #Find vector "b2" that is the y-axis of the rotated frame
    if two_way_yaw:
        b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, -1.0, 0.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    else:
        b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((*q2.shape[:-1], 1)))

    # shape, b1, b2 all unit vectors shape (..., 3)
    b1_proj = b1 - (b1 * shape_vec).sum(dim=-1, keepdim=True) * shape_vec # projection of b1 onto the plane normal to the shape vector
    b2_proj = b2 - (b2 * shape_vec).sum(dim=-1, keepdim=True) * shape_vec # projection of b2 onto the plane normal to the shape vector

    # 2 special cases to consider: if b2_proj is 0, that means b2 is aligned with the shape vector. Any yaw is valid, so we'll set 0 error.
    # If b1_proj is 0 and b2_proj is not, b1 is aligned with the shape vector, but to be able to perform the calculation, we'll need to project
    # it another way: we'll rotate it by pi/2 about the cross product of the shape vector and b2_proj.
    b1_proj_norm = torch.norm(b1_proj, dim=-1, keepdim=True)
    b2_proj_norm = torch.norm(b2_proj, dim=-1, keepdim=True)

    no_error_mask = b2_proj_norm < eps
    degen_mask = (b1_proj_norm < eps) & (~no_error_mask)

    if degen_mask.any():
        degen_mask_vectors = degen_mask.tile(3) # needed since degen_mask is [..., 1]
        degen_axis = torch.linalg.cross(shape_vec[degen_mask_vectors], b2_proj[degen_mask_vectors])
        degen_angle = np.pi/2 * torch.ones(*degen_axis.shape[:-1], 1, device=q1.device)
        quat_to_apply = isaac_math_utils.quat_from_angle_axis(degen_angle, degen_axis)
        b1_proj[degen_mask_vectors] = isaac_math_utils.quat_rotate(quat_to_apply, b1_proj[degen_mask_vectors])
        b1_proj_norm = torch.norm(b1_proj, dim=-1, keepdim=True) # recalculate, degenerate cases should be 1 anyway


    dot = (b1_proj*b2_proj).sum(dim=-1, keepdim=True)
    yaw_error = torch.ones_like(dot)
    prod = b1_proj_norm * b2_proj_norm
    yaw_error[(~no_error_mask).squeeze(-1)] = dot[(~no_error_mask).squeeze(-1)] / prod[(~no_error_mask).squeeze(-1)]
    yaw_error = torch.arccos(torch.clamp(yaw_error, -1.0+1e-8, 1.0-1e-8)).view(q1.shape[:-1])


    cross = torch.linalg.cross(b2_proj, b1_proj)
    # for sign error, rotate the shape vector onto z axis and get sign from z-component of the cross product
    shape_vec_rot_axis = torch.linalg.cross(shape_vec, torch.tensor([[0.0, 0.0, 1.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    shape_vec_rot_angle = torch.arccos(
        torch.clamp((shape_vec * shape_vec_rot_axis).sum(dim=-1, keepdim=True), -1.0+1e-8, 1.0-1e-8)
    ).squeeze(-1)
    shape_vec_rot_to_apply = isaac_math_utils.quat_from_angle_axis(shape_vec_rot_angle, shape_vec_rot_axis)
    cross = isaac_math_utils.quat_rotate(shape_vec_rot_to_apply, cross)
    sign = torch.sign(cross[..., 2]) # z-component of the cross product determins the sign of the error
    yaw_error *= sign
    # yaw_error[mask] *= -1.0 # sign swap for the masked cases

    # Step 2, calculate the shoulder error by appling the yaw rotation to q1 and calculating error as a dot product
    yaw_to_apply = quat_from_yaw(-yaw_error) # negative sign since the convention we are following for the angle errors is actual - desired
    q1 = isaac_math_utils.quat_mul(yaw_to_apply, q1)

    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q2.device).tile((*q2.shape[:-1], 1)))
    dot = (b1*b2).sum(dim=-1)
    shoulder_error = torch.arccos(torch.clamp(dot, -1.0+1e-8, 1.0-1e-8)).view(q1.shape[:-1])
    cross = torch.linalg.cross(b2, b1) # result of cross product will be aligned with the forward axis of the quadrotor, get sign by looking at x-component
    cross = isaac_math_utils.quat_rotate_inverse(yaw_to_apply, cross)
    sign = torch.sign(cross[..., 0])
    shoulder_error *= sign

    # Step 3, calculate the wrist error by appling the yaw and shoulder rotations to q1 and calculating error as a dot product

    # Axis for the shoulder rotation would be the quadrotor's forward axis rotated by the yaw rotation
    shoulder_axis = isaac_math_utils.quat_rotate(yaw_to_apply, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    shoulder_to_apply = isaac_math_utils.quat_from_angle_axis(-shoulder_error, shoulder_axis)
    q1 = isaac_math_utils.quat_mul(shoulder_to_apply, q1)
    

    # Now that we have rotated the EE x-axis onto the frame where the EE y-axis and goal y-axis are the
    # same, calculate the angular error between them
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[1.0, 0.0, 0.0]], device=q1.device).tile((*q1.shape[:-1], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[1.0, 0.0, 0.0]], device=q2.device).tile((*q2.shape[:-1], 1)))  
    dots = (b1 * b2).sum(dim=-1)
    wrist_error = torch.arccos(torch.clamp(dots, -1+1e-8, 1-1e-8)).view(q1.shape[:-1])

    # Get the sign of the error by taking cross products of the local x-axes (of the frames where the y-axes are aligned)
    # and use the sign of the resultant product's y-component
    cross = torch.linalg.cross(b2, b1)
    cross = isaac_math_utils.quat_rotate_inverse(q1, cross)
    sign = torch.sign(cross[..., 1])
    wrist_error *= sign
    return yaw_error[..., None], shoulder_error[..., None], wrist_error[..., None] # each is (..., 1)

def aerial_manipulator_angle_errors(
    q1: torch.Tensor,
    q2: torch.Tensor,
    shape_vec: torch.Tensor = None,
    eps: float = 1e-8
) -> torch.Tensor:
    '''
    Calculates the yaw, shoulder, and wrist angle errors for the aerial manipulator in the 2DOF case in the (actual - desired) convention

    Args:
        q1: current EE rotation, (..., 4)
        q2: target EE rotation (..., 4)

    Returns the signed error of the each angle in radians
    '''

    sol1 = torch.cat(_aerial_manipulator_angle_errors(q1, q2, shape_vec, True, eps), dim=-1)
    sol2 = torch.cat(_aerial_manipulator_angle_errors(q1, q2, shape_vec, False, eps), dim=-1)

    sol = torch.zeros_like(sol1)
    sol1_norm = torch.norm(sol1, dim=-1)
    sol2_norm = torch.norm(sol2, dim=-1)
    sol[sol1_norm < sol2_norm] = sol1[sol1_norm < sol2_norm]
    sol[sol1_norm >= sol2_norm] = sol2[sol1_norm >= sol2_norm]
    return sol[..., :1], sol[..., 1:2], sol[..., 2:3]

def aerial_manipulator_angle_solns_2dof(
    initial_ee_ori: torch.Tensor,
    target_ee_ori: torch.Tensor,
    last_yaw: torch.Tensor,
    shape_vec: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    '''
    Args:
        initial_ee_ori: initial EE rotation, (..., 4)
        target_ee_ori: target EE rotation (..., 4)
        last_yaw: last yaw angle command (..., 1), used to get a continuous trajectory of yaw commands
        shape_vec: desired shape vector, (..., 3)
        eps: small value threshold for considering vectors to be zero
    '''
    sol1 = -torch.cat(_aerial_manipulator_angle_errors(initial_ee_ori, target_ee_ori, shape_vec, True, eps), dim=-1)
    sol2 = -torch.cat(_aerial_manipulator_angle_errors(initial_ee_ori, target_ee_ori, shape_vec, False, eps), dim=-1)

    delta_1 = isaac_math_utils.wrap_to_pi(sol1[..., :1] - last_yaw)
    delta_2 = isaac_math_utils.wrap_to_pi(sol2[..., :1] - last_yaw)

    sol = torch.zeros_like(sol1)
    sol[torch.norm(delta_1, dim=-1) < torch.norm(delta_2, dim=-1)] = sol1[torch.norm(delta_1, dim=-1) < torch.norm(delta_2, dim=-1)]
    sol[torch.norm(delta_1, dim=-1) >= torch.norm(delta_2, dim=-1)] = sol2[torch.norm(delta_1, dim=-1) >= torch.norm(delta_2, dim=-1)]
    return sol[..., :1], sol[..., 1:2], sol[..., 2:3]

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
    # print(q.shape)
    # print(q[env_ids].shape, torch.tensor([0.0, 1.0, 0.0], device=q.device).tile((*q[env_ids].shape[:-1], 1)).shape)
    b = isaac_math_utils.quat_rotate(q[env_ids], torch.tensor([0.0, 1.0, 0.0], device=q.device).tile((*q[env_ids].shape[:-1], 1)))

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
    """Get signed yaw error between two quaternions.

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
    error = torch.arccos(torch.clamp(reward, -1.0+1e-8, 1.0-1e-8)).view(shape1[:-1])
    cross = torch.linalg.cross(b2, b1)
    sign = torch.sign(cross[:, 2]) # z-component of the cross product determins the sign of the error
    error *= sign
    return error

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

    