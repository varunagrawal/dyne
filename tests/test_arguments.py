"""Tests for dyne/arguments.py"""

import pathlib
from unittest import mock

from gtsam.utils.test_case import GtsamTestCase

from dyne import arguments


class TestArguments(GtsamTestCase):
    DATA_DIR = pathlib.Path(__file__).parent.parent.absolute() / "data" / "pybullet"
    CONFIG_FILE = DATA_DIR / "config.yaml"

    @mock.patch('argparse.ArgumentParser.parse_args',
              return_value=arguments.argparse.Namespace(config_yaml=CONFIG_FILE, time=1))
    def test_parse_args(self, actual_args):
        actual_args = arguments.parse_args()
        self.assertIsInstance(actual_args, arguments.argparse.Namespace)

    def test_parse_config(self):
        actual_config = arguments.parse_config(self.CONFIG_FILE)
        self.assertEqual(actual_config.imu_yaml, self.DATA_DIR / "imu.yaml")
        self.assertEqual(actual_config.measurements, self.DATA_DIR / "a1_walking_straight.npz")
        self.assertEqual(actual_config.ground_truth, self.DATA_DIR / "a1_walking_straight.npz")
        self.assertEqual(actual_config.robot_file, "a1.urdf")
        self.assertEqual(
            actual_config.contact_in_com, {
                "FR_toe": [0, 0, 0],
                "FL_toe": [0, 0, 0],
                "RR_toe": [0, 0, 0],
                "RL_toe": [0, 0, 0]
            })
        self.assertEqual(actual_config.skip, 0)
