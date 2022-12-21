"""Script to compute IMU frequency."""

# pylint: disable=invalid-name,unused-import

import numpy as np

from lrse import datasets
from lrse.arguments import parse_args, parse_config
from lrse.utils import load_imu_params

np.set_printoptions(precision=18, suppress=True)


def main():
    """Main runner"""
    args = parse_args()
    config = parse_config(args.config_yaml)

    imu_params = load_imu_params(config.imu_yaml)
    data = datasets.SCSDataset(config.measurements,
                              config.ground_truth,
                              skip=config.skip,
                              bRs=imu_params["pose"].rotation())

    freq = data.compute_frequency()
    print(freq.round())


if __name__ == "__main__":
    print("Computing IMU measurement frequency")
    main()
