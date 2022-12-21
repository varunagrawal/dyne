"""
Script to estimate trajectory from Pybullet data using the Dyne State Estimator.

python scripts/varun06_bloesch2_estimator_pybullet.py data/pybullet/config.yaml --robot models/a1.urdf -T 5 -o results/bloesch_estimator
"""

from pathlib import Path

import gtsam
import numpy as np

from lrse import datasets, plot, utils
from lrse.arguments import parse_args, parse_config
from lrse.noise_models import a1_noise
from lrse.state_estimator import Dyne, RobotImu

DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = Path(__file__).parent.parent.absolute() / "models"

np.set_printoptions(precision=8, suppress=True)


def run(estimator, pybullet: datasets.PybulletDataset, T: float):
    estimator.set_prior_state(pybullet.states[0])

    # simulate the data loop
    for k, _ in enumerate(pybullet.timestamps):
        measured_omega = pybullet.angular_velocity[k]
        measured_acc = pybullet.linear_acceleration[k]

        contacts_at_k = {
            foot: contacts[k]
            for foot, contacts in pybullet.contacts.items()
        }

        estimator.step(k,
                       measured_omega,
                       measured_acc,
                       joint_angles=pybullet.joint_angles[k],
                       contacts=contacts_at_k)

    result = estimator.optimize()
    return result


def main():
    """Do state estimation on Pybullet A1 simulation."""

    args = parse_args()
    config = parse_config(args.config_yaml)
    T = args.time

    imu_params = utils.load_imu_params(config.imu_yaml)
    pybullet = datasets.PybulletDataset(config.measurements,
                                        config.ground_truth,
                                        robot_file=args.robot,
                                        bRs=imu_params["pose"].rotation())

    bias = gtsam.imuBias.ConstantBias(
        imu_params["accelerometer"]["bias"]["mean"],
        imu_params["gyroscope"]["bias"]["mean"])

    base_imu = RobotImu(
        imu_params["freq"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"],
        bias)

    # Define the toe point on the leg link.
    contact_in_com = config.contact_in_com

    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    bias_prior_sigmas = np.asarray(
        [imu_params["accelerometer"]["bias"]["stddev"]] * 3 +
        [imu_params["gyroscope"]["bias"]["stddev"]] * 3)
    # a1_noise.bias_prior = bias_prior_sigmas
    a1_noise.pose_prior = 1e-3
    a1_noise.imu_between_sigmas = np.ones(6) * 1e-3
    a1_noise.contact_point = 1e-6

    estimator = Dyne(base_imu=base_imu,
                     model_file=str(args.robot),
                     estimation_rate=args.rate,
                     noise_params=a1_noise,
                     base_name="trunk",
                     feet=("FR_toe", "FL_toe", "RR_toe", "RL_toe"),
                     contact_in_com=contact_in_com)

    results = run(estimator, pybullet, T)

    # Write out ground truth data and the estimated trajectory.
    ape, rpe = utils.save_results(estimator, pybullet, results, args.output,
                                  "gtdyne")
    stats = utils.add_stats(ape, rpe)

    print("Statistics")
    utils.print_table(stats, tablefmt=args.tablefmt)

    if args.plot:
        plot.plot_estimated_contacts(results, estimator)
        plot.plot_ground_truth_contacts(pybullet, contact_in_com)
        plot.plot_results(results, estimator, pybullet, T)


if __name__ == "__main__":
    main()
