"""Tests for the IMU-only state estimator."""

import unittest
from pathlib import Path

import gtsam
import numpy as np
import numpy.testing as npt
from gtsam.symbol_shorthand import B, X
from gtsam.utils import test_case

from dyne import datasets
from dyne.arguments import parse_config
from dyne.state_estimator import BaseImu, RobotImu
from dyne.utils import load_imu_params

# from dyne.utils import estimate_initial_biases, load_imu_params

np.set_printoptions(precision=8, suppress=True)
np.set_string_function(lambda x: repr(x).replace('(', '').replace(')', '').
                       replace('array', '').replace("       ", ' '),
                       repr=False)

DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
FIXTURES_DIR = Path(__file__).parent.absolute() / "fixtures"
MODEL_DIR = Path(__file__).parent.parent.absolute() / "models"


class TestBaseStateEstimatorA1(test_case.GtsamTestCase):
    """Test the state_estimator.BaseImu class on the A1 dataset."""

    def setUp(self):
        test_data = FIXTURES_DIR
        config_file = test_data / "config.yaml"

        config = parse_config(config_file)

        imu_yaml = test_data / config.imu_yaml
        self.imu_params = load_imu_params(imu_yaml)

        self.bTs = self.imu_params["pose"]

        measurements_file = test_data / "a1_walking_straight.npz"
        ground_truth_file = test_data / "a1_walking_straight.npz"
        self.robot_file = str(MODEL_DIR / "a1.urdf")

        self.dataset = datasets.PybulletDataset(measurements_file,
                                                ground_truth_file,
                                                skip=config.skip,
                                                robot_file=self.robot_file,
                                                bRs=self.bTs.rotation())

        self.ground_truth_states = self.dataset.states

        self.acc_bias = np.asarray(
            self.imu_params["accelerometer"]["bias"]["mean"])
        self.gyro_bias = np.asarray(
            self.imu_params["gyroscope"]["bias"]["mean"])
        self.bias = gtsam.imuBias.ConstantBias(self.acc_bias, self.gyro_bias)

        self.robot_file = Path(config.robot_file)

        self.base_imu = RobotImu(
            self.imu_params["rate_hz"],  #
            self.imu_params["gyroscope"]["noise"]["stddev"],  #
            self.imu_params["accelerometer"]["noise"]["stddev"],  #
            self.imu_params["pose"],
            bias=self.bias)
        self.estimator = BaseImu(self.base_imu)

        self.dt = 1 / self.base_imu.freq

    def test_constructor(self):
        """Test state estimator constructor."""
        estimator = BaseImu(self.base_imu)
        self.assertIsInstance(estimator, BaseImu)

    def integrate_measurement(self, measured_omega, measured_acc, nRb, bTs,
                              nPb_i, nVb_i):
        """Utility function to perform measurement preintegration."""
        unbiased_acc = gtsam.Point3(measured_acc - self.bias.accelerometer())
        unbiased_omega = gtsam.Point3(measured_omega - self.bias.gyroscope())

        # correct measurement by sensor pose
        bRs = bTs.rotation()
        corrected_acc = bRs.rotate(unbiased_acc)
        corrected_omega = bRs.rotate(unbiased_omega)

        def skew(x):
            """Convert ndarray `x` to a skew-symmetric matrix."""
            return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],
                             [-x[1], x[0], 0]])

        # account for centrifugal acceleration
        body_omega_body = skew(corrected_omega)
        b_arm = bTs.translation()
        b_velocity_bs = body_omega_body @ b_arm
        corrected_acc = gtsam.Point3(corrected_acc -
                                     (body_omega_body @ b_velocity_bs))

        # time to update the preintegrated measurements
        a_body = corrected_acc

        # convert acceleration from sensor frame to nav frame
        a_nav = nRb.rotate(a_body)

        nPb_j = nPb_i + (nVb_i * self.dt) + \
            (a_nav * 0.5 * self.dt * self.dt)

        nVb_j = nVb_i + (a_nav * self.dt)

        return nPb_j, nVb_j

    def test_preintegrate_single_measurement(self):
        """Test preintegration of a single measuremnt."""
        pim = gtsam.PreintegratedImuMeasurements(
            self.estimator.base_imu().pim_params(),
            self.estimator.base_imu().previous_bias)
        measured_omega = self.dataset.angular_velocity[0]
        measured_acc = self.dataset.linear_acceleration[0]
        body_P_sensor = self.bTs

        dV_i = pim.deltaVij()
        nPb_i = pim.deltaPij()
        nRb = pim.deltaRij()

        pim.integrateMeasurement(measured_acc, measured_omega, self.dt)

        self.assertEqual(pim.deltaTij(), self.dt)

        nPb_j, nVb_j = self.integrate_measurement(measured_omega, measured_acc,
                                                  nRb, body_P_sensor, nPb_i,
                                                  dV_i)

        # velocity in the nav frame
        dV = pim.deltaVij()
        npt.assert_allclose(dV, nVb_j)

        npt.assert_allclose(nPb_j, pim.deltaPij())

        # the valkyrie slow walk is in a straight line
        # so no rotation expected if bias is zero
        expected_rotation_ypr = np.asarray([0.0000007, 0.00017101, -0.0000056])
        npt.assert_allclose(pim.deltaRij().ypr(),
                            expected_rotation_ypr,
                            atol=1e-6)

        bias = pim.biasHat()
        npt.assert_allclose(bias.accelerometer(), self.acc_bias)
        npt.assert_allclose(bias.gyroscope(), self.gyro_bias)

        expected_measurement_covariance = np.zeros((9, 9))
        expected_measurement_covariance[0:3, 0:3] = np.eye(3) * 1e-8
        expected_measurement_covariance[-3:, -3:] = np.eye(3) * 1e-8
        npt.assert_allclose(pim.preintMeasCov(),
                            expected_measurement_covariance,
                            atol=1e-8)

    def test_preintegrate_measurements(self):
        """Test measurement preintegration."""
        T = 5  # second(s)
        pim = gtsam.PreintegratedImuMeasurements(
            self.estimator.base_imu().pim_params(),
            self.estimator.base_imu().previous_bias)

        nRb = pim.deltaRij()
        nPb = pim.deltaPij()
        nVb = pim.deltaVij()

        for k, _ in enumerate(np.arange(0, T, self.dt)):
            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            nRb = pim.deltaRij()

            pim.integrateMeasurement(measured_acc, measured_omega, self.dt)

            nPb, nVb = self.integrate_measurement(measured_omega, measured_acc,
                                                  nRb, self.bTs, nPb, nVb)

        npt.assert_allclose(nVb, pim.deltaVij())
        npt.assert_allclose(nPb, pim.deltaPij())

    def test_preintegrate_measurements_with_bias(self):
        """Test preintegration with bias."""
        acc_bias = np.asarray(self.imu_params["accelerometer"]["bias"]["mean"])
        gyro_bias = np.asarray(self.imu_params["gyroscope"]["bias"]["mean"])
        bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)
        imu = RobotImu(
            self.imu_params["rate_hz"],  #
            self.imu_params["gyroscope"]["noise"]["stddev"],  #
            self.imu_params["accelerometer"]["noise"]["stddev"],  #
            self.imu_params["pose"],
            bias=bias)
        self.estimator = BaseImu(imu)

    def test_state_estimation(self):
        """Test the base state estimator."""
        T = 5
        estimator = BaseImu(base_imu=self.base_imu, estimation_rate=20)

        estimator.set_prior_state(self.ground_truth_states[0])

        # simulate the data loop
        for k, _ in enumerate(np.arange(0, T - self.dt, self.dt)):
            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            estimator.step(k, measured_omega, measured_acc)

        result = estimator.optimize()

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

        # regression
        npt.assert_allclose(estimator.graph().error(result), 1e-10, atol=1e-10)

    def test_incremental(self):
        """Test incremental state estimation."""
        T = 3
        estimator = BaseImu(base_imu=self.base_imu, estimation_rate=20)

        estimator.set_prior_state(self.ground_truth_states[0])

        # simulate the data loop
        for k, _ in enumerate(np.arange(0, T - self.dt, self.dt)):
            measured_omega = self.dataset.angular_velocity[k]
            measured_acc = self.dataset.linear_acceleration[k]

            estimator.step(k, measured_omega, measured_acc)

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


if __name__ == "__main__":
    unittest.main()
