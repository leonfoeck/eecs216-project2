"""Run either part of the vision estimation exercise."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from pose import estimate_pose
from flow import estimate_velocity
from support import Result, load_dataset, plot_result


def run_estimation(dataset, *, part=1, ransac=True):
    """Use images/tags for estimation; mocap data is used only for plotting."""
    frames = dataset.frames
    time = np.array([frame.time for frame in frames])
    if part == 1:
        values = np.full((6, len(frames)), np.nan)
        for index, frame in enumerate(frames):
            pose = estimate_pose(frame)
            if pose is not None:
                values[:3, index] = pose.position
                values[3:, index] = Rotation.from_matrix(pose.rotation).as_euler("ZYX")[::-1]
    else:
        values = np.full((6, len(frames)-1), np.nan)
        time = (time[:-1] + time[1:]) / 2
        for index, (previous, current) in enumerate(zip(frames[:-1], frames[1:])):
            pose = estimate_pose(previous)
            if pose is not None:
                velocity = estimate_velocity(previous, current, pose, ransac=ransac)
                if velocity is not None:
                    values[:, index] = velocity
    return Result(time, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dataset", type=int, choices=(1, 4), required=True)
    parser.add_argument("--ransac", choices=("on", "off"), default="on")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    ransac = args.ransac == "on"
    result = run_estimation(dataset, part=args.part, ransac=ransac)
    plot_result(result, dataset, part=args.part, ransac=ransac)
    plt.show()
