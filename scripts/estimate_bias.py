"""Script to estimate IMU bias parameters."""

# pylint: disable=invalid-name,unused-import

import argparse

import gtsam
import numpy as np
from lrse import datasets
from lrse.arguments import parse_args, parse_config
from lrse.state_estimator import GRAVITY
from lrse.utils import load_imu_params

POSES_FIG = 1
FIG_RESULT = 2

np.set_printoptions(precision=18, suppress=True)
np.set_string_function(lambda x: repr(x).replace('(', '').replace(')', '').
                       replace('array', '').replace("       ", ' '),
                       repr=False)


def main():
    """Main runner"""
    args = parse_args()
    config = parse_config(args.config_yaml)

    imu_params = load_imu_params(config.imu_yaml)

    data = datasets.SCSDataset(config.measurements, config.ground_truth)

    # freq = imu_params["rate_hz"]

    args.start_time = 0.0
    args.time = 1.0
    end_time = args.start_time + args.time

    measurements = data.measurements
    measurements = measurements[measurements[:, 1] >=
                                args.start_time * 1e9]  # 1.7
    measurements = measurements[measurements[:, 1] <= end_time * 1e9]  # 9.5
    print("Start index:", measurements[0, 0])
    print("End index:", measurements[-1, 0])

    bTs = imu_params["pose"]
    wTb = gtsam.Pose3()
    wTs = wTb.compose(bTs)

    # gravity in the navigational frame with Z up
    n_gravity = gtsam.Point3(0, 0, -GRAVITY)

    s_gravity = wTs.rotation().unrotate(n_gravity)

    mean_measurements = measurements[:, 2:].mean(axis=0)
    print("original mean measurements: ", mean_measurements)
    print("gyroscope bias:\t", mean_measurements[0:3])
    print("accelerometer bias:\t", mean_measurements[3:6] + s_gravity)


if __name__ == "__main__":
    print("Running IMU Bias Estimator")

    main()
