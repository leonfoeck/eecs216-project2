"""Part 2: feature tracking, optical-flow velocity, and three-point RANSAC."""

import cv2
import numpy as np

from support import BODY_TO_CAMERA, CAMERA_MATRIX, Frame, Pose


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
    raise NotImplementedError("Complete flow.py: velocity_ransac")


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
    raise NotImplementedError("Complete flow.py: estimate_velocity")
