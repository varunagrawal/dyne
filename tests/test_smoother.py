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
from dyne.state_estimator import Bloesch
from dyne.utils import load_imu_params, robot_imu_from_yaml

DATA_DIR = pathlib.Path(__file__).parent.absolute() / "fixtures"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"


class TestSmoother(GtsamTestCase):
    """Test the state estimator described in Bloesch13iros."""
    CONFIG = DATA_DIR / "config.yaml"
    ROBOT_MODEL = MODEL_DIR / "a1.urdf"

    def setUp(self):
        config = parse_config(self.CONFIG)
        self.robot_file = str(self.ROBOT_MODEL)
        self.robot = gtd.CreateRobotFromFile(self.robot_file, "", True)

        imu_params = load_imu_params(config.imu_yaml)
        self.dataset = PybulletDataset(config.measurements,
                                       config.ground_truth,
                                       self.robot_file,
                                       bRs=imu_params["pose"].rotation())
        self.base_imu = robot_imu_from_yaml(config.imu_yaml)
        self.ground_truth_states = self.dataset.states

        self.feet: tuple = self.dataset.feet

        self.dt = 1 / self.base_imu.freq

        # Define the toe point on the lower leg link for the A1.
        self.contact_in_com = config.contact_in_com

    def initialize(self, estimator):
        """Set up the estimator with the initial factors and values."""

        estimator.set_prior_state(self.ground_truth_states[0], add_bias_prior=True)

        measured_omega = self.dataset.angular_velocity[0]
        measured_acc = self.dataset.linear_acceleration[0]
        joint_angles = self.dataset.joint_angles[0]
        contacts = {
            foot: int(contact[0])
            for foot, contact in self.dataset.contacts.items()
        }

        estimator.step(0,
                       measured_omega,
                       measured_acc,
                       joint_angles=joint_angles,
                       contacts=contacts)
        gfg = self.linearize(estimator)
        return gfg

    def linearize(self, estimator):
        """Linearize the factor graph in the state estimator."""
        gfg = estimator.graph().linearize(estimator.initial())
        return gfg

    def optimize(self, gfg: gtsam.GaussianFactorGraph, t: int):
        constrain_last = gtsam.KeyVector()
        constrain_last.append(X(t))

        ordering = gtsam.Ordering.ColamdConstrainedLastGaussianFactorGraph(
            gfg, constrain_last)
        bn: gtsam.GaussianBayesNet = gfg.eliminateSequential(ordering)
        bn.print("", gtd.GTDKeyFormatter)

        vector_values: gtsam.VectorValues = bn.optimize()
        # vector_values.print("", gtd.GTDKeyFormatter)
        return bn

    def incremental(self, T: float):
        estimator = Bloesch(base_imu=self.base_imu,
                            model_file=str(self.ROBOT_MODEL),
                            noise_params=a1_noise,
                            estimation_rate=10,
                            base_name="trunk",
                            feet=self.dataset.feet,
                            contact_in_com=self.contact_in_com)

        gfg0: gtsam.GaussianFactorGraph = self.initialize(estimator)
        bn = self.optimize(gfg0, 0)

        # simulate the data loop
        # Skip the first one since we ran `initialize`.
        for k, _ in enumerate(np.arange(1, T, estimator.dt())):
            add_leg_factors = k % int(estimator.estimation_rate) == 0

            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]
            joint_angles = self.dataset.joint_angles[k]

            if add_leg_factors:
                contacts = {
                    foot: int(contact[k])
                    for foot, contact in self.dataset.contacts.items()
                }

            else:
                # If we're not adding leg factors then we just preintegrate IMU measurements.
                contacts = None

            estimator.step(k,
                           measured_omega,
                           measured_acc,
                           joint_angles=joint_angles,
                           contacts=contacts)

    @unittest.skip
    def test_state_estimation(self):
        """Test the Bloesch state estimator with a single chain for the MHS smoother."""
        T = 0.2

        self.incremental(T)

        # # simulate the data loop
        # for k, _ in enumerate(np.arange(0, T, self.dt)):
        #     measured_omega = self.dataset.angular_velocity[k]
        #     measured_acc = self.dataset.linear_acceleration[k]

        #     contacts_at_k = {
        #         foot: int(contact[k])
        #         for foot, contact in self.dataset.contacts.items()
        #     }

        #     estimator.step(k,
        #                    measured_omega,
        #                    measured_acc,
        #                    joint_angles=self.dataset.joint_angles[k],
        #                    contacts=contacts_at_k)

        # result = estimator.optimize()

        # # Check if the resultant 1st pose is the same as the ground truth
        # self.gtsamAssertEquals(result.atPose3(X(0)),
        #                        self.ground_truth_states[0].pose(),
        #                        tol=1e-2)

        # # Check if the resultant 2nd pose is the same as the ground truth
        # self.gtsamAssertEquals(result.atPose3(X(1)),
        #                        self.ground_truth_states[10].pose(),
        #                        tol=1e-1)
