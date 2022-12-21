"""
Main script to estimate IMU state.

python scripts/varun01_estimate_constant_twist.py --num_poses 20 data/pybullet/config.yaml -T 15 -p
"""

# pylint: disable=invalid-name

import gtsam
import numpy as np

from dyne import arguments, datasets, plot, utils
from dyne.state_estimator import BaseImu, RobotImu

POSES_FIG = 1
FIG_RESULT = 2

np.set_printoptions(precision=3, suppress=True)


def pose_noise(stddev=0.1):
    """Random gaussian Pose3 noise."""
    return gtsam.Pose3.Expmap(np.random.randn(6) * stddev)


def velocity_noise(stddev=0.1):
    """Random gaussian velocity noise."""
    return gtsam.Pose3.Expmap(np.random.randn(3) * stddev)


def estimate_initial_biases(measured_acc, measured_omega):
    """Estimate initial bias means and variances from when the IMU is stationary."""
    omega_mean = np.mean(measured_omega, axis=0)
    # omega_var = np.var(measured_omega, axis=0)
    acc_mean = np.mean(measured_acc, axis=0)
    # acc_var = np.var(measured_acc, axis=0)
    return acc_mean, omega_mean


def parse_args():
    """Define command line arguments for this script."""
    parser = arguments.get_parser()
    parser.add_argument("--num_poses", default=120, type=int)
    return parser.parse_args()


def run(estimator: BaseImu, dataset: datasets.Dataset, T: float):
    estimator.set_prior_state(dataset.states[0])

    for k, _ in enumerate(dataset.timestamps):
        measured_omega = dataset.angular_velocity[k]
        measured_acc = dataset.linear_acceleration[k]

        estimator.step(k, measured_omega, measured_acc)

    results = estimator.optimize()
    return results


def main():
    """Main runner"""
    args = parse_args()

    T = args.time

    config = arguments.parse_config(args.config_yaml)
    imu_params = utils.load_imu_params(config.imu_yaml)

    gyro_bias = imu_params["gyroscope"]["bias"]["mean"]
    acc_bias = imu_params["accelerometer"]["bias"]["mean"]
    bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

    imu = RobotImu(
        imu_params["rate_hz"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"],
        bias)
    pim_params = imu.pim_params

    twist_data = datasets.ConstantTwist(imu_params,
                                        pim_params,
                                        num_poses=args.num_poses,
                                        bias=bias)

    estimator = BaseImu(imu, estimation_rate=20)

    results = run(estimator, twist_data, T)

    utils.save_results(estimator, twist_data, results, args.output, "base_imu")

    if args.plot:
        plot.plot_results(results, estimator, twist_data, T)


if __name__ == "__main__":
    print("Running IMU State Estimator")

    main()
