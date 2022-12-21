"""Tests for the Dyne state estimator which uses a chain of factors for each leg."""

import pathlib
import unittest

import gtdynamics as gtd
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X
from gtsam.utils.test_case import GtsamTestCase

from dyne.arguments import parse_config
from dyne.datasets import PybulletDataset
from dyne.noise_models import a1_noise
from dyne.state_estimator import Dyne, RobotImu
from dyne.utils import load_imu_params

DATA_DIR = pathlib.Path(__file__).parent.absolute() / "fixtures"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"


class TestBloesch2StateEstimator(GtsamTestCase):
    """Test the Bloesch 2 state estimator."""
    CONFIG = DATA_DIR / "config.yaml"
    ROBOT_MODEL = MODEL_DIR / "a1.urdf"

    def setUp(self):
        """
        Create robot, dataset and load configuration into the state estimator
        for running various tests.
        """
        config = parse_config(self.CONFIG)
        self.robot_file = str(self.ROBOT_MODEL)
        self.robot = gtd.CreateRobotFromFile(self.robot_file, "", True)
        self.feet = ("FR_toe", "FL_toe", "RR_toe", "RL_toe")

        imu_params = load_imu_params(config.imu_yaml)
        self.dataset = PybulletDataset(config.measurements,
                                       config.ground_truth,
                                       self.robot_file,
                                       bRs=imu_params["pose"].rotation())
        self.base_imu = RobotImu(
            imu_params["rate_hz"],  #
            imu_params["gyroscope"]["noise"]["stddev"],  #
            imu_params["accelerometer"]["noise"]["stddev"],  #
            imu_params["pose"])
        self.ground_truth_states = self.dataset.states

        self.dt = 1 / self.base_imu.freq

        # Define the toe point on the lower leg link for the A1.
        self.contact_in_com = {
            'FR_toe': [0, 0, 0],
            'FL_toe': [0, 0, 0],
            'RR_toe': [0, 0, 0],
            'RL_toe': [0, 0, 0]
        }

    def test_add_forward_kinematics_factors(self):
        """Test if adding of forward kinematics factor is correct."""
        estimator = Dyne(self.base_imu,
                         estimation_rate=20,
                         model_file=str(self.ROBOT_MODEL),
                         noise_params=a1_noise,
                         base_name="trunk",
                         feet=self.feet,
                         contact_in_com=self.contact_in_com)
        k = 200

        joint_angles = self.dataset.joint_angles[k]
        estimator.add_forward_kinematics_factors(k, joint_angles)

        # Get the first (actuated) joint's factor
        jm_factor = estimator.graph().at(0)

        # joint_measurement = jm_factor.measured()

        base_key = gtd.DynamicsSymbol(jm_factor.keys()[0])
        leg_key = gtd.DynamicsSymbol(jm_factor.keys()[1])
        self.assertEqual(base_key.time(), 0)
        self.assertEqual(base_key.linkIdx(), 1)
        self.assertEqual(leg_key.time(), 0)
        self.assertEqual(leg_key.linkIdx(), 2)

    def test_add_single_leg(self):
        """
        Add a single leg for forward kinematics computation.
        """
        estimator = Dyne(base_imu=self.base_imu,
                         model_file=str(self.ROBOT_MODEL),
                         estimation_rate=20,
                         noise_params=a1_noise,
                         base_name="trunk",
                         feet=self.feet,
                         contact_in_com=self.contact_in_com)
        k = 0
        # Set only 1 foot to be in contact
        for foot in self.feet:
            if foot != "FL_toe":
                self.dataset.contacts[foot][k] = 0

        contacts_at_k = {
            foot: int(contact[k])
            for foot, contact in self.dataset.contacts.items()
        }

        # Run step to add the leg factors
        estimator.step(k,
                       measured_omega_b=np.zeros(3),
                       measured_acceleration_b=np.zeros(3),
                       joint_angles=self.dataset.joint_angles[k],
                       contacts=contacts_at_k)

        # Run step to add the IMU factors
        estimator.step(int(estimator.rate() - 1),
                       measured_omega_b=np.zeros(3),
                       measured_acceleration_b=np.zeros(3),
                       joint_angles=self.dataset.joint_angles[k],
                       contacts=contacts_at_k)

        # Get all the PoseFactors
        joint_measurement_factors = [
            estimator.graph().at(idx)
            for idx in range(estimator.graph().nrFactors()) if isinstance(
                estimator.graph().at(idx), gtd.JointMeasurementFactor)
        ]
        # regression
        self.assertEqual(len(joint_measurement_factors), 21)
        # regression
        num_bias_factors = 1
        self.assertEqual(estimator.initial().size(),
                         4 * 3 + 9 + 2 + 1 + num_bias_factors)

    def test_add_two_legs(self):
        """
        Add two legs for forward kinematics computation.
        """
        estimator = Dyne(base_imu=self.base_imu,
                         model_file=str(self.ROBOT_MODEL),
                         estimation_rate=20,
                         noise_params=a1_noise,
                         base_name="trunk",
                         feet=self.feet,
                         contact_in_com=self.contact_in_com)
        k = 0
        # Set only 1 foot to be in contact
        for foot in self.feet:
            if foot != "FL_toe" and foot != "RL_toe":
                self.dataset.contacts[foot][k] = 0

        contacts_at_k = {
            foot: int(contact[k])
            for foot, contact in self.dataset.contacts.items()
        }

        # Run step to add the leg factors
        estimator.step(k,
                       measured_omega_b=np.zeros(3),
                       measured_acceleration_b=np.zeros(3),
                       joint_angles=self.dataset.joint_angles[k],
                       contacts=contacts_at_k)

        # Run step to add the IMU factors
        estimator.step(int(estimator.rate() - 1),
                       measured_omega_b=np.zeros(3),
                       measured_acceleration_b=np.zeros(3),
                       joint_angles=self.dataset.joint_angles[k],
                       contacts=contacts_at_k)

        # Get all the JointMeasurementFactors
        joint_measurement_factors = [
            estimator.graph().at(idx)
            for idx in range(estimator.graph().nrFactors()) if isinstance(
                estimator.graph().at(idx), gtd.JointMeasurementFactor)
        ]
        # regression
        self.assertEqual(len(joint_measurement_factors), 3 + 3 + 3 + 3 + 9)
        # regression: 4*3 factors for each leg + 9 fixed joint factors + factors for 2 contact points + factors for pose and velocity + 1 factor for bias chain
        num_bias_factors = 1
        self.assertEqual(estimator.initial().size(),
                         4 * 3 + 9 + 2 + 2 + num_bias_factors)

    def test_state_estimation(self):
        """Integration test for the full batch state estimation over a 200 ms time window at a frame rate of 10 Hz."""
        T = 0.2

        estimator = Dyne(base_imu=self.base_imu,
                         model_file=str(self.ROBOT_MODEL),
                         base_name="trunk",
                         feet=self.feet,
                         estimation_rate=10,
                         noise_params=a1_noise,
                         contact_in_com=self.contact_in_com)

        # Set the origin for the estimator as well as add bias priors
        estimator.set_prior_state(self.ground_truth_states[0],
                                  add_bias_prior=True)

        # simulate the data loop to generate the full factor graph
        for k, _ in enumerate(np.arange(0, T, self.dt)):
            # Get IMU measurements
            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            # Get contact state as a map from foot to boolean.
            contacts_at_k = {
                foot: int(contact[k])
                for foot, contact in self.dataset.contacts.items()
            }
            # Add factors into the graph for the current timestep
            estimator.step(k,
                           measured_omega,
                           measured_acc,
                           joint_angles=self.dataset.joint_angles[k],
                           contacts=contacts_at_k)

        result = estimator.optimize()

        self.gtsamAssertEquals(result.atPose3(X(0)),
                               self.ground_truth_states[0].pose(),
                               tol=1e-2)

        self.gtsamAssertEquals(result.atPose3(X(1)),
                               self.ground_truth_states[10].pose(),
                               tol=1e-1)

    def test_incremental(self):
        """Test incremental state estimation."""
        T = 3
        estimator = Dyne(base_imu=self.base_imu,
                         model_file=str(self.ROBOT_MODEL),
                         base_name="trunk",
                         feet=self.feet,
                         estimation_rate=10,
                         noise_params=a1_noise,
                         contact_in_com=self.contact_in_com)

        estimator.set_prior_state(self.ground_truth_states[0])

        # simulate the data loop
        for k, _ in enumerate(np.arange(0, T, self.dt)):
            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            contacts_at_k = {
                foot: int(contact[k])
                for foot, contact in self.dataset.contacts.items()
            }

            estimator.step(k,
                           measured_omega,
                           measured_acc,
                           joint_angles=self.dataset.joint_angles[k],
                           contacts=contacts_at_k)

            if estimator.imu_factor_added():
                result = estimator.update()

        self.assertIsInstance(result, gtsam.Values)
        self.gtsamAssertEquals(result.atPose3(X(0)),
                               self.ground_truth_states[0].pose(),
                               tol=1e-1)

        t1 = int(estimator.rate()) - 1
        self.gtsamAssertEquals(result.atPose3(X(1)),
                               self.ground_truth_states[t1].pose(),
                               tol=1e-1)

        t2 = 2 * int(estimator.rate()) - 1
        self.gtsamAssertEquals(result.atPose3(X(2)),
                               self.ground_truth_states[t2].pose(),
                               tol=1e-1)

        t3 = 3 * int(estimator.rate())
        self.gtsamAssertEquals(result.atPose3(X(2)),
                               self.ground_truth_states[t3].pose(),
                               tol=1e-1)
