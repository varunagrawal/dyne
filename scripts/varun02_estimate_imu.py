"""
Main script to estimate IMU state.

python scripts/varun02_estimate_imu.py -T 4.995 data/atlas/straight_line.yaml
"""

# pylint: disable=invalid-name,unused-import

import gtsam
import numpy as np

from dyne import datasets
from dyne.arguments import parse_args, parse_config
from dyne.state_estimator import BaseImu, RobotImu
from dyne.utils import load_imu_params

POSES_FIG = 1
FIG_RESULT = 2

np.set_printoptions(precision=18, suppress=True)


def run(estimator, dataset: datasets.Dataset, T: float):
    estimator.set_prior_state(dataset.states[0])

    # simulate the data loop
    for k, _ in enumerate(np.arange(0, T - estimator.dt, estimator.dt)):
        measured_omega = dataset.angular_velocity[k]
        measured_acc = dataset.linear_acceleration[k]

        estimator.step(k, measured_omega, measured_acc)

    result = estimator.optimize()
    return result


def main():
    """Main runner"""
    args = parse_args()
    config = parse_config(args.config_yaml)

    imu_params = load_imu_params(config.imu_yaml)

    acc_bias = np.asarray(imu_params["accelerometer"]["bias"]["mean"])
    gyro_bias = np.asarray(imu_params["gyroscope"]["bias"]["mean"])

    bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

    dataset = datasets.SCSDataset(config.measurements,
                                  config.ground_truth,
                                  skip=config.skip,
                                  bRs=imu_params["pose"].rotation())

    base_imu = RobotImu(
        imu_params["rate_hz"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"],
        bias=bias)
    estimator = BaseImu(base_imu=base_imu)

    run(estimator, dataset, args.time)


if __name__ == "__main__":
    print("Starting up")

    main()
