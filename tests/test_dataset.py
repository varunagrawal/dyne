"""Tests for data loading."""

import pathlib
import unittest

import gtdynamics as gtd
import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from dyne import datasets
from dyne.state_estimator import GRAVITY, RobotImu
from dyne.utils import load_imu_params

RESOURCES = pathlib.Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"


class TestDataset(GtsamTestCase):
    """Test the base Dataset class."""

    def setUp(self):
        self.dataset = datasets.Dataset()

    def test_constructor(self):
        self.assertFalse(self.dataset.states)
        self.assertFalse(self.dataset.timestamps.size)
        self.assertFalse(self.dataset.angular_velocity)
        self.assertFalse(self.dataset.linear_acceleration)

    def test_load_measurements(self):
        self.assertRaises(NotImplementedError, self.dataset.load_measurements,
                          "")
        self.assertRaises(NotImplementedError, self.dataset.load_measurements,
                          "")

    def test_load_ground_truth(self):
        self.assertRaises(NotImplementedError, self.dataset.load_ground_truth,
                          "")
        self.assertRaises(NotImplementedError, self.dataset.load_ground_truth,
                          "")


class TestPybulletDataset(GtsamTestCase):
    """Tests for the data loader for the Pybullet simulation."""
    A1 = RESOURCES / "pybullet"

    def setUp(self):
        imu_yaml = self.A1 / "imu.yaml"
        self.imu_params = load_imu_params(imu_yaml)

        self.robot_file = str(MODEL_DIR / "a1.urdf")
        self.feet = ("FR_toe", "FL_toe", "RR_toe", "RL_toe")

        self.measurements_file = self.A1 / "a1_walking_straight.npz"
        self.ground_truth_file = self.A1 / "a1_walking_straight.npz"
        self.dataset = datasets.PybulletDataset(
            self.measurements_file,
            self.ground_truth_file,
            robot_file=self.robot_file,
            bRs=self.imu_params["pose"].rotation())

    def test_constructor(self):
        """Test constructor."""
        pybullet = datasets.PybulletDataset(
            self.measurements_file,
            self.ground_truth_file,
            robot_file=self.robot_file,
            bRs=self.imu_params["pose"].rotation())

        self.assertEqual(1000, len(pybullet.states))
        self.assertEqual((1000, ), pybullet.timestamps.shape)

    def test_dataset(self):
        """Test whether the data is loaded."""
        self.assertEqual((1000, 3), self.dataset.angular_velocity.shape)
        self.assertEqual((1000, 3), self.dataset.linear_acceleration.shape)

        self.assertSequenceEqual(self.feet,
                                 tuple(self.dataset.contacts.keys()))
        self.assertEqual(12, len(self.dataset.joint_names))
        self.assertEqual(1000, len(self.dataset.joint_angles))

    def test_imu_measurement(self):
        """
        Test to check if the angular velocity and the angular acceleration
        are in the correct frames and are numerically correct.
        """
        k = 300  # 300 * 0.005 = 1.5 seconds
        dt = 1.0 / self.imu_params["freq"]
        nTb299 = self.dataset.states[k - 1]
        nTb300 = self.dataset.states[k]
        nTb301 = self.dataset.states[k + 1]
        self.gtsamAssertEquals(nTb300.position() + nTb300.velocity() * dt,
                               nTb301.position(),
                               tol=1e-4)

        predicted_acceleration_n = (nTb301.velocity() -
                                    nTb299.velocity()) / (2 * dt)
        nRb300 = nTb300.attitude()
        gravity_n = np.asarray([0, 0, -GRAVITY])

        np.testing.assert_allclose(np.degrees(nRb300.rpy()), [4.7, -2.5, 0.77],
                                   atol=1e-1)
        predicted_acceleration_b = nRb300.unrotate(predicted_acceleration_n)
        gravity_b = nRb300.unrotate(gravity_n)
        np.testing.assert_allclose(gravity_b, [-0.42, -0.78, -9.7], atol=1e-1)

        self.gtsamAssertEquals(self.dataset.linear_acceleration[k],
                               predicted_acceleration_b - gravity_b)

    def test_joint_angles(self):
        """Test if joint angles are correctly loaded."""
        true_joint_angles_0 = {
            "FR_hip_joint": 0.00017253942628727773,
            "FR_upper_joint": 0.9241253681585302,
            "FR_lower_joint": -1.8330203558347935,
            "FL_hip_joint": 0.0001743035632169786,
            "FL_upper_joint": 0.9230325389728722,
            "FL_lower_joint": -1.8338113705185008,
            "RR_hip_joint": 0.00014003652467009726,
            "RR_upper_joint": 0.8778316003433424,
            "RR_lower_joint": -1.8520048879456086,
            "RL_hip_joint": 0.0001371670610410984,
            "RL_upper_joint": 0.8782774194143699,
            "RL_lower_joint": -1.8528367618107737
        }

        for joint_name, angle in true_joint_angles_0.items():
            actual_angle = gtd.JointAngle(
                self.dataset.joint_angles[0],
                self.dataset.robot.joint(joint_name).id(), 0)
            self.assertEqual(actual_angle, angle)

    @unittest.skip("Pybullet doesn't account for fixed links")
    def test_foot_poses(self):
        """Test if foot poses (in world/navigation frame) are loaded correctly."""

        # Test at different timesteps in the trajectory
        for k in range(0, 1000, 100):
            for foot in self.feet:
                actual_wTtoe = self.dataset.foot_poses[foot][k]

                wTb = self.dataset.states[k].pose()
                fk = self.dataset.robot.forwardKinematics(
                    self.dataset.joint_angles[k], k, "trunk")

                # Pose of leg CoM in world frame
                wTleg = wTb * gtd.Pose(fk,
                                       self.dataset.robot.link(foot).id(), k)
                # Transform from leg CoM to toe frame
                legTtoe = self.dataset.legTtoe

                expected_wTtoe = wTleg * legTtoe
                self.gtsamAssertEquals(actual_wTtoe, expected_wTtoe, tol=1e-2)

                toe_mass = 0.06
                shank_mass = 0.166
                combined_com = (
                    (actual_wTtoe.translation() * toe_mass) +
                    (self.dataset.lower_links[foot][k].translation()) *
                    shank_mass) / (toe_mass + shank_mass)
                self.gtsamAssertEquals(combined_com, wTleg.translation(), 1e-2)


if __name__ == "__main__":
    unittest.main()
