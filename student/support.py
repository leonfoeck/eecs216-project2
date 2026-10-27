"""Provided calibration, data loading, containers, and plotting."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CAMERA_MATRIX = np.array([[311.0520, 0, 201.8724],
                          [0, 311.3885, 113.6210], [0, 0, 1]])
# Homogeneous transform: point_camera = BODY_TO_CAMERA @ point_body.
# Translation is the body origin expressed in camera coordinates, in metres.
_c = np.sqrt(0.5)
BODY_TO_CAMERA = np.array([[_c, -_c, 0, -0.04],
                           [-_c, -_c, 0, 0],
                           [0, 0, -1, -0.03],
                           [0, 0, 0, 1]])


@dataclass
class Frame:
    time: float
    image: np.ndarray       # (height, width), rectified uint8 grayscale
    ids: np.ndarray         # (N,), AprilTag IDs
    corners: np.ndarray     # (N, 4, 2), pixel (u, v); p1, p2, p3, p4 per tag


@dataclass
class Pose:
    position: np.ndarray    # (3,), body origin in world coordinates, metres
    rotation: np.ndarray    # (3, 3), body-to-world rotation


@dataclass
class Dataset:
    frames: list[Frame]
    reference_time: np.ndarray   # (M,), mocap timestamps, seconds
    vicon: np.ndarray            # (12, M), original mocap measurements
    number: int


@dataclass
class Result:
    time: np.ndarray        # (N,), seconds; image times or interval midpoints
    values: np.ndarray      # (6, N); pose [position, rpy] or velocity [v_W, omega_B]


def load_dataset(number):
    """Read all images; keep the mocap stream on its own timestamps."""
    path = Path(__file__).parent / "data" / f"studentdata{number}.mat"
    contents = loadmat(path, struct_as_record=False, squeeze_me=True,
                       variable_names=("data", "time", "vicon"))
    frames = []
    for packet in np.atleast_1d(contents["data"]):
        ids = np.asarray(packet.id, dtype=int).reshape(-1)
        corners = np.stack([np.asarray(getattr(packet, f"p{i}"), dtype=float)
                            .reshape(2, ids.size).T for i in (1, 2, 3, 4)], axis=1)
        frames.append(Frame(float(packet.t), np.ascontiguousarray(packet.img), ids, corners))
    return Dataset(frames, np.asarray(contents["time"]).reshape(-1),
                   contents["vicon"], number)


def plot_result(result, dataset, *, part, ransac=True):
    """Plot estimates and mocap measurements on their respective timestamps."""
    import matplotlib.pyplot as plt

    title = f"Part {part} - Dataset {dataset.number}"
    if part == 2:
        title += " - RANSAC" if ransac else " - Least squares"
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    reference = dataset.vicon[:6].copy() if part == 1 else dataset.vicon[6:12]
    values = result.values.copy()
    if part == 1:
        valid = np.all(np.isfinite(values[3:]), axis=0)
        values[3:, valid] = np.unwrap(values[3:, valid], axis=1)
        reference[3:] = np.unwrap(reference[3:], axis=1)
        if np.any(valid):
            first = np.flatnonzero(valid)[0]
            ref = np.array([np.interp(result.time[first], dataset.reference_time, row)
                            for row in reference[3:]])
            values[3:, valid] += (2 * np.pi * np.round((ref-values[3:, first])/(2*np.pi)))[:, None]
    groups = (("Position (world)", "m"), ("Orientation", "rad")) if part == 1 else (
        ("Linear velocity (world)", "m/s"), ("Angular velocity (body)", "rad/s"))
    for row, (group, unit) in enumerate(groups):
        components = ("Roll", "Pitch", "Yaw") if part == 1 and row == 1 else ("X", "Y", "Z")
        for col, component in enumerate(components):
            ax = axes[row, col]
            index = row * 3 + col
            ax.plot(result.time, values[index], "r", linewidth=1, label="Vision estimate")
            ax.plot(dataset.reference_time, reference[index], "b", linewidth=1,
                    label="Mocap measurement")
            ax.set_title(f"{group}: {component}")
            ax.set_ylabel(unit)
            if row == 1:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend()
    return fig
