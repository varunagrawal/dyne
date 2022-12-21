"""Functions to compute metrics."""

import gtsam
import numpy as np
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D


def align_trajectories(reference_traj: PoseTrajectory3D,
                       estimated_traj: PoseTrajectory3D,
                       sampling_rate=1):
    """
    Use our custom alignment scheme to align the reference and estimated trajectories.

    We have 4 degrees of freedom (gauge freedom), which is (x, y, z, yaw).
    We use gtsam::Pose2::align to align the (x, y, yaw) and subtract the mean of z
    to align the z axis.

    Returns:
        The updated trajectories where the estimated trajectory has been transformed
        to the reference trajectory for the x & y translation values,
        and where the z-translation is zero-mean.
        Also returned is the `Pose2` transform for the x & y translation values.
    """
    reference_z_mean = reference_traj.positions_xyz[:, 2].mean()
    estimated_z_mean = estimated_traj.positions_xyz[:, 2].mean()

    reference_traj.positions_xyz[:, 2] -= reference_z_mean
    estimated_traj.positions_xyz[:, 2] -= estimated_z_mean

    reference_points = reference_traj.positions_xyz[::int(sampling_rate), 0:2]
    estimated_points = estimated_traj.positions_xyz[:, 0:2]

    # transform from estimated trajectory to reference trajectory
    rTe = gtsam.Pose2.Align(
        gtsam.Point2Pairs(list(zip(reference_points, estimated_points))))
    for idx in range(estimated_points.shape[0]):
        estimated_traj.positions_xyz[idx, 0:2] = rTe.transformFrom(
            estimated_traj.positions_xyz[idx, 0:2])

    return reference_traj, estimated_traj, rTe


def evaluate(reference_traj: PoseTrajectory3D,
             estimated_traj: PoseTrajectory3D,
             max_diff=0.01,
             pose_relation=metrics.PoseRelation.full_transformation,
             delta=1.0,
             delta_unit=metrics.Unit.frames,
             sync_trajectories=True,
             all_pairs=False):
    """
    Function to compute the APE and RPE metric statistics between the reference trajectory
    and the estimated trajectory.

    The evo documentation explains the various parameters in much greater detail:
    https://github.com/MichaelGrupp/evo/blob/ffda6b4b91a2c5739f0d39cd3491ed5f4011d499/notebooks/metrics.py_API_Documentation.ipynb

    Args:
        reference_traj: The ground-truth reference trajectory.
        estimated_traj: The estimated trajectory.
        max_diff: The amount of difference in time allowed when synchronizing the trajectories.
        pose_relation: The relation to use between each pose to compute the metrics.
            Default is full_transformation.
        delta: The distance between pose pairs along the trajectories.
        delta_unit: The unit to use for the delta. Default is `frames`.
        sync_trajectories: Flag indicating if trajectories should be synced before processing.
        all_pairs: Flag for all-pairs mode when computing relative error between pose pairs.
    """
    if sync_trajectories:
        data = sync.associate_trajectories(reference_traj,
                                           estimated_traj,
                                           max_diff=max_diff)
    else:
        data = (reference_traj, estimated_traj)

    ape_result, ape_stats = ape(data)
    rpe_result, rpe_stats = rpe(data, pose_relation, delta, delta_unit,
                                all_pairs)

    return ape_stats, rpe_stats


def ape(data):
    """
    Helper method to compute APE.
    Use the `full_transformation` pose relation which is dimensionless.
    """
    ape_metric = metrics.APE(metrics.PoseRelation.full_transformation)
    ape_metric.process_data(data)
    ape_result = ape_metric.get_result("reference", "estimate")
    ape_stats = ape_metric.get_all_statistics()

    return ape_result, ape_stats


def rpe(data, pose_relation, delta, delta_unit, all_pairs):
    """Helper method to compute RPE."""
    rpe_metric = metrics.RPE(pose_relation, delta, delta_unit, all_pairs)
    rpe_metric.process_data(data)
    rpe_result = rpe_metric.get_result("reference", "estimate")
    rpe_stats = rpe_metric.get_all_statistics()

    return rpe_result, rpe_stats


def get_evo_trajectory(values, dataset_timestamps, sampling_rate=1):
    """Get the trajectory in evo.PoseTrajectory3D format."""
    quaternions, xyz, timestamps = np.empty((0, 4)), np.empty(
        (0, 3)), np.empty((0, 1))
    i = 0
    for k, t in enumerate(dataset_timestamps):
        if k % int(sampling_rate) == 0:
            wTb = values.atPose3(gtsam.symbol_shorthand.X(i))
            quaternions = np.vstack(
                (quaternions,
                 wTb.rotation().toQuaternion().coeffs()[[3, 0, 1, 2]]))
            xyz = np.vstack((xyz, wTb.translation()))
            timestamps = np.vstack((timestamps, t))

            i += 1

    return PoseTrajectory3D(xyz, quaternions, timestamps)
