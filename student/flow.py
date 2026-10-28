"""Part 2: feature tracking, optical-flow velocity, and three-point RANSAC."""

import cv2 as cv
import numpy as np
import math

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose

# Seeded so a given dataset always produces the same plots.
_RNG = np.random.default_rng(0)

# RANSAC tuning. The inlier test is a budget on KLT tracking error in pixels
# (see estimate_velocity); the inlier fraction and hit probability set how many
# trials are needed to draw an all-inlier sample.
PIXEL_TOLERANCE = 0.3
INLIER_FRACTION = 0.5
HIT_PROBABILITY = 0.99

# A minimal sample is three points; give up on a trial after this many redraws
# rather than looping forever when every triple is rank deficient.
_MAX_REDRAWS = 20


def velocity_ransac(points: np.ndarray, flow: np.ndarray, depth: np.ndarray, *,
                    ransac: bool, threshold: float, iterations: int) -> np.ndarray:
    """Return camera-frame [vx, vy, vz, wx, wy, wz] (6,).

    points and flow are (N, 2) normalized image coordinates and their rates
    (1/s); depth is (N,) positive camera-z depth in metres, with N >= 3.
    With ransac=False, fit all points by least squares. Otherwise sample
    three distinct points per trial, reject rank-deficient systems, score
    each point by its 2D flow residual norm (< threshold, in 1/s), keep the
    best consensus, and refit using all its inliers.
    """
    if len(points) < 3:
        return None
    if not ransac:
        return calculate_velocity(points, flow, depth)

    # Built once and reused: scoring every trial against it costs one matrix
    # product instead of rebuilding a 2x6 block per point per iteration.
    AB = build_AB(points, depth)
    measured = flow.reshape(-1)

    best_inliers = None
    for _ in range(iterations):
        sample = _sample_well_posed(points, depth)
        if sample is None:
            continue
        velocity = calculate_velocity(points[sample], flow[sample], depth[sample])
        residual = np.linalg.norm((AB @ velocity - measured).reshape(-1, 2), axis=1)
        inliers = residual < threshold
        if inliers.sum() >= 3 and (best_inliers is None
                                   or inliers.sum() > best_inliers.sum()):
            best_inliers = inliers

    if best_inliers is None:
        return None
    return calculate_velocity(points[best_inliers], flow[best_inliers], depth[best_inliers])


def _sample_well_posed(points: np.ndarray, depth: np.ndarray) -> np.ndarray | None:
    """Draw three distinct points whose 6x6 system has full rank, else None.

    Three collinear or coincident features leave some of the six degrees of
    freedom unconstrained, so such a sample cannot generate a hypothesis.
    """
    for _ in range(_MAX_REDRAWS):
        sample = _RNG.choice(len(points), 3, replace=False)
        if np.linalg.matrix_rank(build_AB(points[sample], depth[sample])) == 6:
            return sample
    return None


def calculate_velocity(points: np.ndarray, flow: np.ndarray, depth: np.ndarray) -> np.ndarray:
    q = flow.flatten()
    AB = build_AB(points, depth)
    M, N = AB.shape
    U, S, Vt = np.linalg.svd(AB)

    S_inv_values = np.where(S > 1e-10, 1.0 / S, 0.0)
    Sigma_inv = np.zeros((N, M))
    np.fill_diagonal(Sigma_inv, S_inv_values)

    return Vt.T @ Sigma_inv @ U.T @ q


def build_AB(points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    AB_list = []
    for point, Z in zip(points, depth):
        AB_i = build_AB_i(point, Z)
        AB_list.append(AB_i)

    return np.vstack(AB_list)


def build_AB_i(point: np.ndarray, depth: float) -> np.ndarray:
    u, v = point
    A_i = np.array([[-1 / depth, 0, u / depth], [0, -1 / depth, v / depth]])
    B_i = np.array([[u * v, -(1 + u ** 2), v], [1 + v ** 2, -u * v, -u]])
    return np.hstack((A_i, B_i))


def estimate_velocity(previous: Frame, current: Frame, pose: Pose, *,
                      ransac: bool) -> np.ndarray | None:
    """Return [world linear velocity, body angular velocity] (6,).

    pose is the body pose at previous.time. Detect corners in previous.image
    (including scribble), track to current.image with KLT, and discard failed
    tracks. Normalize pixels with CAMERA_MATRIX; divide displacements by
    current.time - previous.time. Intersect previous-frame rays with world
    z=0 using the camera pose to obtain depth. Fit camera motion with
    velocity_ransac(), then account for the camera/body rotation AND offset.
    Choose detector/tracker and RANSAC parameters here. Return None when
    fewer than three usable tracks remain. The runner plots interval averages
    at midpoint timestamps; use the previous pose for frame transformations.
    """
    dt = current.time - previous.time
    if dt <= 0:
        return None
    prev_image = previous.image
    curr_image = current.image

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_corners = cv.goodFeaturesToTrack(prev_image, mask=None, **feature_params)

    if prev_corners is None or len(prev_corners) < 3:
        return None

    # calculate optical flow
    curr_corners, st, err = cv.calcOpticalFlowPyrLK(prev_image, curr_image, prev_corners, None, **lk_params)

    if curr_corners is not None:
        good_curr = curr_corners[st == 1]
        good_prev = prev_corners[st == 1]
    else:
        return None

    if len(good_curr) < 3 or len(good_prev) < 3:
        return None

    normalized_optical_flow = (good_curr - good_prev) / (np.array([CAMERA_MATRIX[0][0], CAMERA_MATRIX[1][1]]) * dt)
    normalized_points = (good_prev - [CAMERA_MATRIX[0][2], CAMERA_MATRIX[1][2]]) / (
        [CAMERA_MATRIX[0][0], CAMERA_MATRIX[1][1]])
    s_CB_C = BODY_TO_CAMERA[0:3, 3]
    W_R_B = pose.rotation
    C_R_B = BODY_TO_CAMERA[0:3, 0:3]
    W_R_C = W_R_B @ C_R_B.T
    s_BW_W = pose.position
    rays = np.column_stack((normalized_points, np.ones(len(normalized_points))))
    ray_world_z = rays @ W_R_C[2, :]
    camera_height = s_BW_W[2] - np.dot(W_R_C[2, :], s_CB_C)
    depth = np.full(len(normalized_points), np.nan)
    np.divide(-camera_height, ray_world_z, out=depth, where=ray_world_z != 0)

    # Keep the same valid tracks in points, flow, and camera-z depths.
    valid_depth = np.isfinite(depth) & (depth > 0)
    normalized_points = normalized_points[valid_depth]
    normalized_optical_flow = normalized_optical_flow[valid_depth]
    depth = depth[valid_depth]
    if len(depth) < 3:
        return None

    # The residual is a flow residual, so its units are 1/s and its scale goes
    # as 1/dt. Image timestamps in these datasets jitter between roughly 3 and
    # 52 ms, so a constant threshold in 1/s would be several times stricter on
    # a short interval than on a long one. Fixing the tolerance in pixels of
    # KLT tracking error and converting keeps the test consistent across
    # intervals; 0.3 px is about the sub-pixel accuracy of the tracker.
    focal = 0.5 * (CAMERA_MATRIX[0, 0] + CAMERA_MATRIX[1, 1])
    threshold = PIXEL_TOLERANCE / (focal * dt)

    M = 3           # minimal sample size: three points give six equations
    iterations = math.ceil(np.log(1 - HIT_PROBABILITY)
                           / np.log(1 - INLIER_FRACTION ** M))

    velocity = velocity_ransac(normalized_points, normalized_optical_flow, depth,
                               ransac=ransac, threshold=threshold, iterations=iterations)
    if velocity is None:
        return None
    W_v_C_C = velocity[0:3]
    W_omega_B_C = velocity[3:6]
    W_omega_B_B = C_R_B.T @ W_omega_B_C
    W_v_B_W = W_R_C @ (W_v_C_C + np.cross(W_omega_B_C, s_CB_C))

    return np.array([W_v_B_W, W_omega_B_B]).flatten()
