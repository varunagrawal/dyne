"""
Utilities for parsing command line arguments.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
from yaml import CLoader


def get_parser():
    """
    Generate an argument parser.
    Useful when we wish to add more arguments downstream.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("config_yaml", help="YAML file with all the arguments")
    parser.add_argument("--time",
                        "-T",
                        default=20,
                        type=float,
                        help="Total measurement time in seconds")
    parser.add_argument("--rate",
                        "-r",
                        default=50,
                        type=int,
                        help="The estimation rate for the estimator.")
    parser.add_argument("--start_time",
                        "-S",
                        default=0,
                        type=float,
                        help="Measurement time start in seconds")
    parser.add_argument("--robot", default="", help="Path to robot URDF/SDF.")
    parser.add_argument("--plot", "-p", action="store_true", default=False)
    parser.add_argument("--output",
                        "-o",
                        default="output",
                        help="Directory to save results in.")
    parser.add_argument("--tablefmt",
                        default="github",
                        help="The formatting to use when printing metrics.",
                        choices=('github', 'latex'))
    return parser


def parse_args():
    """Parse command line arguments."""
    parser = get_parser()
    return parser.parse_args()


def parse_config(config_file):
    """
    Parse configuration YAML file.
    Automatically prepends the path of the config file to all path related variables.
    """
    config_file = Path(config_file)
    parent = config_file.parent.resolve()

    config = yaml.load(open(config_file), Loader=CLoader)

    config = SimpleNamespace(**config)

    config.imu_yaml = parent / config.imu_yaml
    config.measurements = parent / config.measurements
    config.ground_truth = parent / config.ground_truth

    return config
