"""Dataset for simulated Constant Twist scenario."""

import gtsam
import numpy as np

from .dataset import Dataset


class ConstantTwist(Dataset):
    """
    Dataset which uses GTSAM's built-in IMU data simulator for generating twist motions.
    """
    def __init__(self, imu_params, pim_params, num_poses=10, bias=None):
        super().__init__("", "", 0)
        # W = np.array([np.deg2rad(30), -np.deg2rad(30), 0])
        W = np.array([0, -np.deg2rad(30), 0])
        # simply move in a straight line
        # W = np.zeros(3)
        V = np.array([2, 0, 0])
        self.scenario = gtsam.ConstantTwistScenario(W, V)

        self.pim_params = pim_params
        self.freq = imu_params["rate_hz"]
        self.dt = 1 / self.freq

        if bias:
            self.bias = bias
        else:
            acc_bias = np.array([-0.3, 0.1, 0.2])
            gyro_bias = np.array([0.1, 0.3, -0.1])
            self.bias = gtsam.imuBias.ConstantBias(acc_bias, gyro_bias)

        self.runner = gtsam.ScenarioRunner(self.scenario, self.pim_params,
                                           1 / self.freq, self.bias)

        self.num_poses = num_poses
        self.load_measurements()
        self.load_ground_truth()

    def load_measurements(self, measurements_file=None):
        """Load the measurement data."""
        for t in np.arange(0, self.num_poses, 1 / self.freq):
            measured_omega = self.runner.measuredAngularVelocity(t)
            measured_acc = self.runner.measuredSpecificForce(t)
            self.angular_velocity.append(measured_omega)
            self.linear_acceleration.append(measured_acc)

        self.angular_velocity = np.asarray(self.angular_velocity)
        self.linear_acceleration = np.asarray(self.linear_acceleration)

    def load_ground_truth(self, ground_truth_file=None, bRs=None):
        """
        Load ground truth data.
        Since this is a simulation, the parameters are not used.
        """
        for i in np.arange(0, self.num_poses, 1 / self.freq):
            state_i = self.scenario.navState(float(i))
            self.states.append(state_i)
