"""
Script to estimate trajectory from Pybullet data using the BaseImu State Estimator.

python scripts/varun03_base_estimator_pybullet.py data/pybullet/config.yaml --robot models/a1.urdf -r 10 --output results/base_estimator_turn
"""

import pathlib

import numpy as np

from lrse import datasets, utils
from lrse.arguments import parse_args, parse_config
from lrse.plot import plot_results
from lrse.state_estimator import BaseImu, RobotImu

DATA_DIR = pathlib.Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"


def run(estimator: BaseImu, dataset: datasets.Dataset, T: float):
    estimator.set_prior_state(dataset.states[0])

    # simulate the data loop
    for t, _ in enumerate(dataset.timestamps):
        measured_omega = dataset.angular_velocity[t]
        measured_acc = dataset.linear_acceleration[t]

        estimator.step(t, measured_omega, measured_acc)

    results = estimator.optimize()
    return results


def main():
    """Do state estimation on Pybullet A1 simulation."""

    args = parse_args()
    config = parse_config(args.config_yaml)
    T = config.total_time

    imu_params = utils.load_imu_params(config.imu_yaml)

    base_imu = RobotImu(
        imu_params["freq"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"])

    pim_params = base_imu.pim_params()
    pybullet = datasets.PybulletDataset(config.measurements,
                                        config.ground_truth,
                                        args.robot,
                                        pim_params=pim_params,
                                        bRs=imu_params["pose"].rotation())

    estimator = BaseImu(base_imu=base_imu, estimation_rate=args.rate)

    results = run(estimator, pybullet, T)

    ape, rpe = utils.save_results(estimator, pybullet, results, args.output,
                                  "base_imu")
    stats = utils.add_stats(ape, rpe)

    print("Statistics")
    utils.print_table(stats, tablefmt=args.tablefmt)

    if args.plot:
        plot_results(results, estimator, pybullet, T)


if __name__ == "__main__":
    main()
