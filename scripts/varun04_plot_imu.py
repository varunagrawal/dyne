"""
Main script to estimate IMU state.

python scripts/varun04_plot_imu.py data/pybullet/config.yaml --robot models/a1.urdf
"""

# pylint: disable=invalid-name,unused-import

import gtdynamics as gtd
import numpy as np
from lrse import datasets
from lrse.arguments import parse_args, parse_config

POSES_FIG = 1
FIG_RESULT = 2

np.set_printoptions(precision=3, suppress=True)


def main():
    """Main runner"""
    args = parse_args()
    config = parse_config(args.config_yaml)

    data = datasets.PybulletDataset(config.measurements,
                                    config.ground_truth,
                                    robot_file=args.robot)

    data.plot_data(show=True)


if __name__ == "__main__":
    print("Plotting IMU Data")

    main()
