"""Test the various metrics"""

import pathlib
from copy import deepcopy

import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from dyne import datasets, metrics

RESOURCES = pathlib.Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"

np.set_printoptions(precision=18, suppress=True)


class TestAlign(GtsamTestCase):
    """Class for testing the `align` function."""
    def setUp(self):
        self.measurements_file = RESOURCES / "pybullet" / "a1_walking_straight.npz"
        self.ground_truth_file = RESOURCES / "pybullet" / "a1_walking_straight.npz"
        self.robot_file = str(MODEL_DIR / "a1.urdf")

        self.dataset = datasets.PybulletDataset(self.measurements_file,
                                                self.ground_truth_file,
                                                robot_file=self.robot_file)

    def test_identity(self):
        """Test if equal trajectories are the same after alignment."""
        T = 5
        dt = 0.01
        values = self.dataset.ground_truth_as_values()
        reference_traj = metrics.get_evo_trajectory(values, np.arange(0, T, dt))
        estimated_traj = deepcopy(reference_traj)
        reference_traj, estimated_traj, _ = metrics.align_trajectories(
            reference_traj, estimated_traj)
        # The trajectories should be the same after alignment.
        assert estimated_traj == reference_traj

    def test_alignment_pose2(self):
        """Test the transform of the points in SE(2)."""
        T = 5
        dt = 0.01
        values = self.dataset.ground_truth_as_values()
        reference_traj = metrics.get_evo_trajectory(values, np.arange(0, T, dt))

        estimated_traj = deepcopy(reference_traj)
        rTe = gtsam.Pose2(11, 7, np.pi / 3)

        for idx in range(estimated_traj.positions_xyz.shape[0]):
            estimated_traj.positions_xyz[idx, 0:2] = rTe.transformTo(
                estimated_traj.positions_xyz[idx, 0:2])

        reference_traj, estimated_traj, rTe_estimated = metrics.align_trajectories(
            reference_traj, estimated_traj)

        # The trajectories should be the same after alignment.
        assert estimated_traj == reference_traj

        self.gtsamAssertEquals(rTe_estimated, rTe)

    def test_alignment_z(self):
        """Test the shift of z."""
        T = 5
        dt = 0.01
        values = self.dataset.ground_truth_as_values()
        reference_traj = metrics.get_evo_trajectory(values, np.arange(0, T, dt))

        estimated_traj = deepcopy(reference_traj)

        z = 10

        for idx in range(estimated_traj.positions_xyz.shape[0]):
            estimated_traj.positions_xyz[idx, 2] += z

        reference_traj, estimated_traj, rTe_estimated = metrics.align_trajectories(
            reference_traj, estimated_traj)

        # The trajectories should be the same after alignment.
        assert estimated_traj == reference_traj

    def test_alignment_pose2_and_z(self):
        """Test the transform of the points in SE(2) with shift of z."""
        T = 5
        dt = 0.01
        values = self.dataset.ground_truth_as_values()
        reference_traj = metrics.get_evo_trajectory(values, np.arange(0, T, dt))

        estimated_traj = deepcopy(reference_traj)

        rTe = gtsam.Pose2(11, 7, np.pi / 3)
        z = 10

        for idx in range(estimated_traj.positions_xyz.shape[0]):
            estimated_traj.positions_xyz[idx, 0:2] = rTe.transformTo(
                estimated_traj.positions_xyz[idx, 0:2])
            estimated_traj.positions_xyz[idx, 2] += z

        reference_traj, estimated_traj, rTe_estimated = metrics.align_trajectories(
            reference_traj, estimated_traj)

        # The trajectories should be the same after alignment.
        assert estimated_traj == reference_traj

        self.gtsamAssertEquals(rTe_estimated, rTe)

    def test_metric(self):
        T = 5
        dt = 0.01
        values = self.dataset.ground_truth_as_values()
        reference_traj = metrics.get_evo_trajectory(values, np.arange(0, T, dt))

        estimated_traj = deepcopy(reference_traj)

        rTe = gtsam.Pose2(11, 7, np.pi / 3)
        z = 10

        for idx in range(estimated_traj.positions_xyz.shape[0]):
            estimated_traj.positions_xyz[idx, 0:2] = rTe.transformTo(
                estimated_traj.positions_xyz[idx, 0:2])
            estimated_traj.positions_xyz[idx, 2] += z

        reference_traj, estimated_traj, rTe_estimated = metrics.align_trajectories(
            reference_traj, estimated_traj)

        self.gtsamAssertEquals(rTe_estimated, rTe)

        ape_stats, rpe_stats = metrics.evaluate(reference_traj, estimated_traj)

        self.assertAlmostEqual(ape_stats["rmse"], 0, 10)
        self.assertAlmostEqual(rpe_stats["rmse"], 0, 10)
