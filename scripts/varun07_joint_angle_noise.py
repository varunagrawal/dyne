"""
Script to estimate trajectory from Pybullet data using the Dyne State Estimator.
This script adds noise to the joint angle measurements.

python scripts/varun07_joint_angle_noise.py data/pybullet/config.yaml --output results/joint_angle_noise
"""

from pathlib import Path
from typing import List

import gtsam
import numpy as np

from lrse import datasets, plot, utils
from lrse.arguments import get_parser, parse_config
from lrse.noise_models import a1_noise
from lrse.state_estimator import Bloesch, Dyne, RobotImu

DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = Path(__file__).parent.parent.absolute() / "models"

np.set_printoptions(precision=8, suppress=True)

# Set the random seed
rng = np.random.default_rng(110791)

estimators = {"bloesch": Bloesch, "dyne": Dyne}


def parse_args():
    parser = get_parser()
    parser.add_argument("--add_noise",
                        '-a',
                        default=False,
                        action='store_true')
    parser.add_argument(
        "--stddev",
        default=1,
        type=float,
        help=
        "The standard deviation of the gaussian noise to add to joint angles (in degrees)."
    )
    return parser.parse_args()


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


def get_estimator(estimator_name, config, dataset, robot_file, rate):
    """Get the estimator as specified by `estimator_name`."""
    base_imu = utils.robot_imu_from_yaml(config.imu_yaml)

    # Define the toe point on the leg link.
    contact_in_com = config.contact_in_com

    Estimator = estimators[estimator_name]
    estimator = Estimator(base_imu=base_imu,
                          model_file=str(robot_file),
                          estimation_rate=rate,
                          noise_params=a1_noise,
                          base_name="trunk",
                          feet=dataset.feet,
                          contact_in_com=contact_in_com)

    return estimator


def perturb_joint_angles(joint_angles_list: List[gtsam.Values],
                         stddev_deg: float = 0):
    """Add some gaussian noise with sigma=stddev_deg to the joint angles."""
    for joint_angles in joint_angles_list:
        for k in joint_angles.keys():
            angle_rad = joint_angles.atDouble(k)
            angle_deg = np.rad2deg(angle_rad)
            new_angle_deg = angle_deg + (rng.standard_normal(1) *
                                         stddev_deg)[0]
            new_angle_rad = np.deg2rad(new_angle_deg)
            joint_angles.update(k, new_angle_rad)


def main():
    """Do state estimation on Pybullet A1 simulation."""

    args = parse_args()
    config = parse_config(args.config_yaml)
    T = config.total_time

    robot_file = str(MODEL_DIR / config.robot_file)

    imu_params = utils.load_imu_params(config.imu_yaml)
    pim_params = RobotImu(
        imu_params["rate_hz"],  #
        imu_params["gyroscope"]["noise"]["stddev"],  #
        imu_params["accelerometer"]["noise"]["stddev"],  #
        imu_params["pose"]).pim_params()

    robot_file = str(MODEL_DIR / config.robot_file)

    pybullet = datasets.PybulletDataset(config.measurements,
                                        config.ground_truth,
                                        robot_file=robot_file,
                                        pim_params=pim_params,
                                        bRs=imu_params["pose"].rotation())

    if args.add_noise:
        perturb_joint_angles(pybullet.joint_angles, args.stddev)

    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    stats = {}

    for name in estimators:
        print("\n===== Running {} Estimator".format(name.capitalize()))
        estimator = get_estimator(name, config, pybullet, robot_file,
                                  args.rate)
        results = run(estimator, pybullet, T)

        ape, rpe = utils.save_results(estimator, pybullet, results, output_dir,
                                      name)

        stats = utils.add_stats(ape, rpe, stats)

        if args.plot:
            plot.plot_estimated_contacts(results, estimator)
            plot.plot_ground_truth_contacts(pybullet, estimator.contact_in_com)
            plot.plot_results(results, estimator, pybullet, T)

    headers = ("Metric", "Units", "Bloesch APE", "Bloesch RPE", "Dyne APE",
               "Dyne RPE")

    print("Statistics")
    utils.print_table(stats, headers=headers)

    utils.save_table(stats,
                     headers=headers,
                     output_file=output_dir / "results.md")


if __name__ == "__main__":
    main()
