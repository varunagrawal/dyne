"""
Script to export pybullet npz data to CSV format.

python scripts/export_pybullet_csv.py data/pybullet/config.yaml --robot models/a1.urdf
"""

from pathlib import Path

import numpy as np

from dyne import datasets, utils
from dyne.arguments import parse_args, parse_config

DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = Path(__file__).parent.parent.absolute() / "models"

np.set_printoptions(precision=8, suppress=True)


def main():
    """Main runner"""

    args = parse_args()
    config = parse_config(args.config_yaml)
    T = args.time

    imu_params = utils.load_imu_params(config.imu_yaml)
    pybullet = datasets.PybulletDataset(config.measurements,
                                        config.ground_truth,
                                        robot_file=args.robot,
                                        bRs=imu_params["pose"].rotation())

    pybullet.export_csv("pybullet_measurements.csv",
                        "pybullet_ground_truth.csv")


if __name__ == "__main__":
    main()
