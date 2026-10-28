"""Part 1: AprilTag map geometry and pose from a planar homography."""

import numpy as np
from scipy.spatial.transform import Rotation

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose

EDGE_TO_EDGE_GAP = 0.152
SQUARE_SIZE = 0.152
EDGE_TO_EDGE_JUMP = 0.178 - 0.152

def get_corners(ids: np.ndarray) -> np.ndarray:
    """Return (N, 4, 3) world coordinates in metres for the given tag IDs.

    Preserve ID order. Each tag's corners are bottom left, bottom right,
    top right, top left (p1, p2, p3, p4); the mat is the world z=0 plane.
    """
    
    p_w = np.array([
        get_square_corner(id) for id in ids
    ])
    
    assert p_w.shape[1] == 4 and p_w.shape[2] == 3, f"Expected shape (N, 4, 3) got {p_w.shape})"
    return p_w

def get_square_corner(id: int) -> np.ndarray:
    """Return (4, 3) position of the respective tag ID
    """
    # down the mat
    n_row = id % 12 
    n_col = (id // 12)
    p4_x = (EDGE_TO_EDGE_GAP + SQUARE_SIZE) * n_row
    
    p4_y = (EDGE_TO_EDGE_GAP + SQUARE_SIZE) * n_col
    p4_y+= EDGE_TO_EDGE_JUMP if n_col + 1 > 3 else 0
    p4_y+= EDGE_TO_EDGE_JUMP if n_col + 1 > 6 else 0
    
    p4 = [p4_x, p4_y, 0]
    p3 = [p4_x, p4_y + SQUARE_SIZE, 0]
    p2 = [p4_x + SQUARE_SIZE, p4_y + SQUARE_SIZE, 0]
    p1 = [p4_x + SQUARE_SIZE, p4_y, 0]
    
    return np.array([p1, p2, p3, p4])
    
    
def get_homography(frame: Frame) -> np.ndarray:
    """Computes the Homography from world to camera coordinates.
    Ah = 0, H = h.reshape((3, 3))

    Returns:
        np.ndarray: (3, 3) 
    """
    
    corners_b = frame.corners
    corners_w = get_corners(frame.ids)[:, :, :2]
    assert corners_b.shape == corners_w.shape, f"Missaligned corners got {corners_b.shape} and {corners_w.shape}"
    
    # identify A for Ah = 0 
    A = np.array([
            [
                *pi_w_xy, 1, 0, 0, 0, *(-pi_w_xy * pi_c_xy[0]), -pi_c_xy[0], 
                0, 0, 0, *pi_w_xy, 1, *(-pi_w_xy * pi_c_xy[1]), -pi_c_xy[1]  
            ] 
            
        for p_c, p_w in zip(corners_b, corners_w, strict=True)
        for pi_c_xy, pi_w_xy in zip(p_c, p_w, strict=True)
        ]
    ).reshape(-1, 9)
            
    _, _, VT = np.linalg.svd(A)
    
    H = VT[-1, :].reshape((3, 3))
    
    # Gurarantee positive Depth
    H *= -1 if np.sum(H[-1, :]) < 0 else 1
    
    return H
    
    
def get_rotatation(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns Rotation Matrix and translation from the world to the body frame

    Args:
        H (np.ndarray): Homography (w to c)

    Returns:
        tuple[np.ndarray, np.ndarray]: R_wc,, twc
    """
    
    R12t = np.linalg.inv(CAMERA_MATRIX) @ H
    R13 = np.linalg.cross(R12t[:, 0], R12t[:, 1])
    R = np.hstack([R12t[:, :2], R13[:, None]])
    
    # compute the svd:
    U, _, VT = np.linalg.svd(R)
    S = np.eye(3)
    S[-1, :] = np.array([0, 0, np.linalg.det(U@VT)])
    R_scaled = U @ S @ VT
    
    t_scaled = R12t[:, -1] / np.linalg.norm(R12t[:, :2], ord=2)
    
    assert np.isclose(d:=np.linalg.det(R_scaled), 1), f"Expected det(R) == 1 instead got {d}"
    
    
    return R_scaled, t_scaled
    

def estimate_pose(frame: Frame) -> Pose | None:
    """Estimate the body origin and body-to-world rotation from tag corners.

    Use get_corners(), the calibrated camera, and a homography estimated by
    SVD. Choose the sign giving positive camera depths and a proper rotation.
    Apply BODY_TO_CAMERA to recover the body pose. Return None if no tags
    are visible. Pixel coordinates are zero-based (u right, v down).
    """
    if len(frame.ids) == 0: return None
    
    H = get_homography(frame)
    
    R_cw, t_cw = get_rotatation(H)
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    
    
    
    
    R_wb = R_wc @ BODY_TO_CAMERA[:3, :3]
    t_wb = t_wc + R_wc @ BODY_TO_CAMERA[3, :3]
    
    assert t_wb[-1] > 0, f"Camera exibits negative depth"
    
    return Pose(
        position=t_wb, 
        rotation=R_wb
    )

