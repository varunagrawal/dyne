"""Tests for the Bloesch state estimation baselines."""

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
from dyne.state_estimator import Bloesch, RobotImu
from dyne.utils import load_imu_params

DATA_DIR = pathlib.Path(__file__).parent.absolute() / "fixtures"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"


class TestBloeschStateEstimator(GtsamTestCase):
    """Test the state estimator described in Bloesch13iros."""
    CONFIG = DATA_DIR / "config.yaml"
    ROBOT_MODEL = MODEL_DIR / "a1.urdf"

    def setUp(self):
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

    def test_constructor(self):
        """Test the constructor."""
        estimator = Bloesch(base_imu=self.base_imu,
                            estimation_rate=20,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet,
                            contact_in_com=self.contact_in_com)
        self.assertIsInstance(estimator, Bloesch)
        self.assertIsInstance(estimator.base_imu(), RobotImu)
        self.assertIsInstance(estimator.robot(), gtd.Robot)

    def test_add_forward_kinematics_factors(self):
        """Test if adding of forward kinematics factor is correct."""
        estimator = Bloesch(self.base_imu,
                            estimation_rate=20,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet,
                            contact_in_com=self.contact_in_com)
        k = 200
        foot = "FR_toe"

        joint_angles = self.dataset.joint_angles[k]

        estimator.add_forward_kinematics_factors(k, joint_angles)

        # Get the pose of the leg lower link as computed in the factor.
        fk_factor = estimator.graph().at(0)

        actual_bTleg = fk_factor.measured()

        # Get the measurement from `robot.forwardKinematics`.
        foot_id = estimator.robot().link(foot).id()
        fk = estimator.robot().forwardKinematics(joint_angles, k, foot)
        wTb = gtd.Pose(fk,
                       estimator.robot().link(estimator.base_name()).id(), k)
        wTleg = gtd.Pose(fk, foot_id, k)

        expected_bTleg = wTb.between(wTleg)

        self.gtsamAssertEquals(actual_bTleg, expected_bTleg)

        base_key = gtsam.Symbol(fk_factor.keys()[0])
        leg_key = gtd.DynamicsSymbol(fk_factor.keys()[1])
        self.assertEqual(base_key.index(), 0)
        self.assertEqual(leg_key.time(), 0)

    @unittest.skip("We shouldn't need to add the contact height prior")
    def test_add_contact_height_prior(self):
        k = 0
        joint_angles = self.dataset.joint_angles[k]
        estimator = Bloesch(self.base_imu,
                            str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet)
        estimator.add_factors(k, joint_angles, self.dataset.contacts)

        # Robot is stationary so it should have
        # 4 FK factors + 4 contact priors + 4 contact point factors
        # regression
        self.assertEqual(estimator.graph().size(), 12)

        k = 14
        estimator = Bloesch(self.base_imu,
                            str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet)
        estimator.add_factors(k, joint_angles, self.dataset.contacts)

        # Robot has 2 legs in contacts so graph should have
        # 2 FK factors + 2 contact priors + 2 contact point factors.
        # regression
        self.assertEqual(estimator.graph().size(), 6)

    def test_continuous_stance(self):
        """
        Test to ensure that Contact Equality factors are only added when the foot
        is in continuous stance since the last time step at which the
        forward kinematics factors were added.
        """
        estimator = Bloesch(base_imu=self.base_imu,
                            estimation_rate=20,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet,
                            contact_in_com=self.contact_in_com)

        N = 12

        # Generate toy contact data
        contacts = {}
        for foot in self.feet:
            contacts[foot] = [True] * N
        # Set intermediate subset to false to mark break in stance phase
        for foot in self.feet:
            contacts[foot][6:8] = [False] * 2

        for k, _ in enumerate(np.arange(0, N)):

            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            contacts_at_k = {
                foot: contact[k]
                for foot, contact in contacts.items()
            }

            estimator.step(k,
                           measured_omega,
                           measured_acc,
                           joint_angles=self.dataset.joint_angles[k],
                           contacts=contacts_at_k)

        # IMU factors are added at k = {9,}
        # Leg factors are added at k = {0, 10}
        # Height priors are added at k = {0, 10} due to break in contact
        # The graph should have 0 priors, 1 IMU factor, 1 bias chain factor, 4+4 FK factors,
        # 4+4 contact point factors, and 0 between factors.
        # regression
        num_bias_factors = 1
        self.assertEqual(estimator.graph().size(),
                         0 + 1 + num_bias_factors + 8 + 8 + 0)

    def test_update_foot_phase(self):
        T = 1.0
        estimator = Bloesch(base_imu=self.base_imu,
                            estimation_rate=20,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            base_name="trunk",
                            feet=self.feet,
                            contact_in_com=self.contact_in_com)

        # Pick out a short subset of contacts.
        # Assume each contact happens at a second mark.
        # FL_toe (id=5) and RR_toe (id=20) are always in stance
        # FR_toe (id=10) is in swing
        # RL_toe (id=15) goes from stance to swing at 0-1
        start = 30
        end = 61
        contacts = {}
        for foot in self.feet:
            contacts[foot] = self.dataset.contacts[foot][start:end]

        estimator.set_prior_state(self.ground_truth_states[0])

        # Iterate over the smaller set of contacts
        for k, _ in enumerate(np.arange(start, end)):

            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            contacts_at_k = {
                foot: int(contact[k])
                for foot, contact in contacts.items()
            }

            estimator.step(k,
                           measured_omega,
                           measured_acc,
                           joint_angles=self.dataset.joint_angles[k],
                           contacts=contacts_at_k)

        # IMU factors are added at k = {9, 19, 29}
        # Leg factors are added at k = {0, 10, 20, 30}
        # The graph should have 3 priors (pose, velocity, bias), 1+1+1 IMU factors, 1+1+1 bias factors,
        # 4+4+4+4 FK factors, 2+2+2+4 contact point factors
        # regression
        num_bias_factors = 3
        self.assertEqual(estimator.graph().size(),
                         3 + 3 + num_bias_factors + 16 + 10)

    def test_state_estimation(self):
        """Test the Bloesch state estimator."""
        T = 0.2

        estimator = Bloesch(base_imu=self.base_imu,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            estimation_rate=10,
                            base_name="trunk",
                            feet=self.feet,
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

        result = estimator.optimize()

        # Check if the resultant 1st pose is the same as the ground truth
        self.gtsamAssertEquals(result.atPose3(X(0)),
                               self.ground_truth_states[0].pose(),
                               tol=1e-2)

        # Check if the resultant 2nd pose is the same as the ground truth
        self.gtsamAssertEquals(result.atPose3(X(1)),
                               self.ground_truth_states[10].pose(),
                               tol=1e-1)

    def test_incremental(self):
        """Test incremental state estimation."""
        T = 3
        estimator = Bloesch(base_imu=self.base_imu,
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
                               tol=1e-2)

        t1 = int(estimator.rate()) - 1
        self.gtsamAssertEquals(result.atPose3(X(1)),
                               self.ground_truth_states[t1].pose(),
                               tol=1e-2)

        t2 = 2 * int(estimator.rate()) - 1
        self.gtsamAssertEquals(result.atPose3(X(2)),
                               self.ground_truth_states[t2].pose(),
                               tol=1e-2)

        t3 = 3 * int(estimator.rate())
        self.gtsamAssertEquals(result.atPose3(X(2)),
                               self.ground_truth_states[t3].pose(),
                               tol=1e-1)
