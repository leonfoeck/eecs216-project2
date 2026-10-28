"""Part 1: AprilTag map geometry and pose from a planar homography."""

import numpy as np
from scipy.spatial.transform import Rotation

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose


def get_corners(ids: np.ndarray) -> np.ndarray:
    """Return (N, 4, 3) world coordinates in metres for the given tag IDs.

    Preserve ID order. Each tag's corners are bottom left, bottom right,
    top right, top left (p1, p2, p3, p4); the mat is the world z=0 plane.
    """
    corners = np.zeros((len(ids), 4, 3))
    for idx, tag_id in enumerate(ids):
        i = tag_id % 12
        j = tag_id // 12
        if j > 5:
            width = 0.152 * (2 * j - 2) + 2 * 0.178
        elif j > 2:
            width = 0.152 * (2 * j - 1) + 0.178
        else:
            width = 2 * j * 0.152
        upper_left_corner = np.array([2 * 0.152 * i, width, 0])
        corners[idx, 0] = upper_left_corner + np.array([0.152, 0, 0])
        corners[idx, 1] = upper_left_corner + np.array([0.152, 0.152, 0])
        corners[idx, 2] = upper_left_corner + np.array([0, 0.152, 0])
        corners[idx, 3] = upper_left_corner

    return corners


def estimate_pose(frame: Frame) -> Pose | None:
    """Estimate the body origin and body-to-world rotation from tag corners.

    Use get_corners(), the calibrated camera, and a homography estimated by
    SVD. Choose the sign giving positive camera depths and a proper rotation.
    Apply BODY_TO_CAMERA to recover the body pose. Return None if no tags
    are visible. Pixel coordinates are zero-based (u right, v down).
    """
    ids = frame.ids
    if len(ids) < 1:
        return None
    picture_tags_corners = frame.corners
    marker_tags_corners = get_corners(ids)
    # Ah = 0
    A = []
    marker_points = []
    for picture_tag_corners, marker_tag_corners in zip(picture_tags_corners, marker_tags_corners):
        for picture_tag_corner, marker_tag_corner in zip(picture_tag_corners, marker_tag_corners):
            marker_points.append(marker_tag_corner)
            X = marker_tag_corner[0]
            Y = marker_tag_corner[1]
            u = picture_tag_corner[0]
            v = picture_tag_corner[1]
            A.append(np.array([X, Y, 1, 0, 0, 0, -X * u, -Y * u, -u]))
            A.append(np.array([0, 0, 0, X, Y, 1, -X * v, -Y * v, -v]))

    A = np.array(A)

    # Keep the full SVD: a single tag gives an 8x9 system, so the right null
    # space is only the last row of a 9x9 Vh, not of a thin factorisation.
    Vh = np.linalg.svd(A)[2]

    h = Vh[-1, :]
    H = h.reshape(3, 3)
    B = np.linalg.inv(CAMERA_MATRIX) @ H
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    gamma = 2 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1_pos = gamma * b1
    r2_pos = gamma * b2
    r3_pos = np.cross(r1_pos, r2_pos)
    t_CM_pos = gamma * b3
    R_pos = np.array([r1_pos, r2_pos, r3_pos]).T  # R_CM with positive gamma
    U, S, Vt = np.linalg.svd(R_pos)
    R_pos = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    r1_neg = -gamma * b1
    r2_neg = -gamma * b2
    r3_neg = np.cross(r1_neg, r2_neg)
    t_CM_neg = -gamma * b3
    R_neg = np.array([r1_neg, r2_neg, r3_neg]).T  # R_CM with negative gamma
    U, S, Vt = np.linalg.svd(R_neg)
    R_neg = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    translation_neg = np.asarray(t_CM_neg, dtype=np.float64).reshape(3)
    translation_pos = np.asarray(t_CM_pos, dtype=np.float64).reshape(3)
    camera_points_neg = np.asarray(marker_points) @ R_neg.T + translation_neg
    camera_points_pos = np.asarray(marker_points) @ R_pos.T + translation_pos
    front_count_neg = np.count_nonzero(camera_points_neg[..., 2] > 0)
    front_count_pos = np.count_nonzero(camera_points_pos[..., 2] > 0)
    if front_count_neg <= front_count_pos:
        R = R_pos
        t_CM = t_CM_pos
    else:
        R = R_neg
        t_CM = t_CM_neg

    body_to_camera_rotation = BODY_TO_CAMERA[:3, :3]
    body_to_camera_translation = BODY_TO_CAMERA[:3, 3]

    body_to_marker_rotation = R.T @ body_to_camera_rotation
    body_to_marker_translation = - R.T @ (t_CM - body_to_camera_translation)

    return Pose(body_to_marker_translation, body_to_marker_rotation)
