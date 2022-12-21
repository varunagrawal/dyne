"""Plotting utilities."""

import platform
from pathlib import Path

import gtdynamics as gtd
import gtsam
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import roboplot
from gtsam.symbol_shorthand import X

from dyne import datasets
from dyne.state_estimator import BaseImu

if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')


def plot_trajectory(values,
                    fignum=2,
                    title="Trajectory",
                    show=True,
                    image_dir="images"):
    """Plot poses of trajectory in values."""
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')

    # Set marker on first pose
    pose_0 = values.atPose3(X(0))
    t = pose_0.translation()
    axes.plot(t[0:1], t[1:2], t[2:3], color='k', marker='o')

    poses = gtsam.utilities.allPose3s(values)
    keys = gtsam.KeyVector(poses.keys())

    pose_values = gtsam.Values()
    for key in keys:
        sym = gtsam.Symbol(key)
        if chr(sym.chr()) == 'x':
            pose = values.atPose3(key)
            pose_values.insert(key, pose)

    # Plot the trajectory
    axes = roboplot.plot_trajectory(pose_values,
                                    fignum=fignum,
                                    title=title,
                                    azimuth=-0,
                                    elevation=90,
                                    show=show)

    fig.savefig(
        Path(image_dir) / fig.canvas.get_window_title().replace(' ', '_'))


def plot_contacts(values, T, point_on_links, fignum=0, show=False):
    """Plot the foot contacts."""
    foot_contacts = {}
    # initialize the dictionary
    for pol in point_on_links:
        foot_contacts[pol.link.name()] = []

    color_legend = [pol.link.name() for pol in point_on_links]

    for k in range(T):
        for point_on_link in point_on_links:
            try:
                wTl = gtd.Pose(values, point_on_link.link.id(), k)
                contact = wTl.transformFrom(point_on_link.point)
                foot_contacts[point_on_link.link.name()].append(contact)
            except:
                continue

    for pol in point_on_links:
        foot_contacts[pol.link.name()] = np.asarray(
            foot_contacts[pol.link.name()])

    roboplot.plot_foot_contacts(foot_contacts,
                                fignum,
                                color_legend=color_legend,
                                show=show)


def plot_estimated_contacts(result, estimator, fignum=5, show=False):
    point_on_links = [
        gtd.PointOnLink(estimator.robot().link(foot),
                        estimator.contact_in_com(foot))
        for foot in estimator.feet()
    ]

    plot_contacts(result,
                  estimator.index,
                  point_on_links,
                  fignum=fignum,
                  show=show)


def plot_ground_truth_contacts(data: datasets.Dataset,
                               contact_in_com: gtsam.Point3,
                               show=False):
    """Plot the ground truth contact points."""
    contact_points = {}
    for foot in data.feet:
        contact_points[foot] = np.empty((0, 3))

    robot = data.robot

    for k, _ in enumerate(data.timestamps):
        if k % 10 != 0:
            continue

        joint_angles = data.joint_angles[k]
        fk = robot.forwardKinematics(joint_angles, k, data.base)

        wTb = data.states[k].pose()

        for foot in data.feet:
            bTl = gtd.Pose(fk, robot.link(foot).id(), k)
            wTl = wTb * bTl
            wTcontact = wTl.transformFrom(contact_in_com[foot])

            if data.contacts[foot][k]:
                contact_points[foot] = np.vstack(
                    (contact_points[foot], wTcontact))

    roboplot.plot_foot_contacts(contact_points,
                                fignum=6,
                                title="Ground Truth Contact Points",
                                color_legend=data.feet,
                                show=show)


def plot_trajectories(estimator: BaseImu,
                      result: gtsam.Values,
                      estimate_title="Estimated Trajectory",
                      initial_title="Initial Trajectory",
                      image_dir="images"):
    """Plot the ground truth and estimated trajectories."""
    plot_trajectory(estimator.initial(),
                    fignum=1,
                    title=initial_title,
                    show=False,
                    image_dir=image_dir)
    plot_trajectory(result,
                    fignum=2,
                    title=estimate_title,
                    image_dir=image_dir)


def plot_results(result: gtsam.Values,
                 estimator: BaseImu,
                 dataset: datasets.Dataset,
                 T: int,
                 image_dir="images/pybullet"):
    #TODO Use evo_traj to plot these
    ground_truth = gtsam.Values()
    idx = 0
    for i, _ in enumerate(np.arange(0, T, estimator.dt())):
        if i % estimator.rate() == 0:
            state = dataset.states[i]
            ground_truth.insert(X(idx), state.pose())
            idx += 1

    plot_trajectory(ground_truth,
                    fignum=0,
                    title="Ground Truth Trajectory",
                    show=False,
                    image_dir=image_dir)

    estimator_name = estimator.__class__.__name__
    plot_trajectories(estimator,
                      result,
                      estimate_title=f"{estimator_name} Estimated Trajectory",
                      initial_title=f"{estimator_name} Initial Trajectory",
                      image_dir=image_dir)


from gtsam.symbol_shorthand import B
from plotly.subplots import make_subplots


def plot_foot_positions_comparison(dataset, contact_in_com, show=True):
    estimated_contact_points = {}
    for foot in dataset.feet:
        estimated_contact_points[foot] = np.empty((0, 3))

    robot = dataset.robot

    timestamps = dataset.timestamps - dataset.timestamps[0]

    for k, _ in enumerate(timestamps):
        wTb = dataset.states[k].pose()
        joint_angles = dataset.joint_angles[k]

        for foot in dataset.feet:
            # Define a FK factor with an arbitrary noise model.
            fk_factor = gtd.ForwardKinematicsFactor(
                robot, dataset.base, foot, joint_angles,
                gtsam.noiseModel.Isotropic.Sigma(3, 1.0), k)

            bTl = fk_factor.measured()
            wTl = wTb * bTl
            wTcontact = wTl.transformFrom(contact_in_com[foot])

            estimated_contact_points[foot] = np.vstack(
                (estimated_contact_points[foot], wTcontact))

    fig = make_subplots(rows=3,
                        cols=1,
                        subplot_titles=("X axis", "Y axis", "Z axis"))

    for foot in dataset.feet[:1]:
        dataset_foot_points = np.asarray(
            [pose.translation() for pose in dataset.foot_poses[foot]])

        for row in range(1, 4):
            fig.add_scatter(x=timestamps,
                            y=dataset_foot_points[:, row - 1],
                            name=f"{foot} From Dataset",
                            row=row,
                            col=1)
            fig.add_scatter(x=timestamps,
                            y=estimated_contact_points[foot][:, row - 1],
                            name=f"{foot} Computed",
                            row=row,
                            col=1)

    fig.show()


def plot_bias(estimator, values: gtsam.Values, dataset: datasets.Dataset,
              K: int):

    fig = make_subplots(rows=3,
                        cols=2,
                        subplot_titles=("Acc X axis", "Gyro X axis",
                                        "Acc Y axis", "Gyro Y axis",
                                        "Acc Z axis", "Gyro Z axis"))

    bias = np.empty((K, 6))  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    marginals = gtsam.Marginals(estimator.graph(), values)
    stddev = np.empty((K, 6))
    for i in range(K):
        bias[i] = values.atConstantBias(B(i)).vector()
        covariance = marginals.marginalCovariance(B(i))
        stddev[i] = np.sqrt(np.diag(covariance))

    timestamps = dataset.timestamps[:K * estimator.rate():estimator.rate()]

    for col, sensor in enumerate(('accelerometer', 'gyroscope')):
        for row, axis in enumerate(('x', 'y', 'z')):
            x = list(timestamps)
            y = list(bias[:, row + 3 * col])
            y_upper = list(bias[:, row + 3 * col] + stddev[:, row + 3 * col])
            y_lower = list(bias[:, row + 3 * col] - stddev[:, row + 3 * col])

            fig.add_scatter(x=x,
                            y=y,
                            name=f"{sensor} {axis} mean",
                            row=row + 1,
                            col=col + 1)

            fig.add_scatter(x=x + x[::-1],
                            y=y_upper + y_lower[::-1],
                            fill='toself',
                            fillcolor='rgba(99,110,250,.3)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            row=row + 1,
                            col=col + 1)

    fig.show()


def plot_trajectory_components(dataset: datasets.Dataset, estimator,
                               estimated_traj, reference_traj):
    """
    Function to plot each component (x, y, z, r, p, y) of the trajectory as a separate subplot.
    Useful for analyzing individual components.
    """

    estimated_timestamps = dataset.timestamps[::estimator.rate()]
    fig_xyz = make_subplots(rows=3,
                            cols=1,
                            subplot_titles=("X axis", "Y axis", "Z axis"))

    for row in range(1, 4):
        fig_xyz.add_scatter(x=dataset.timestamps,
                            y=reference_traj._positions_xyz[:, row - 1],
                            name="TSIF",
                            row=row,
                            col=1)
        fig_xyz.add_scatter(x=estimated_timestamps,
                            y=estimated_traj._positions_xyz[:, row - 1],
                            name="Estimated",
                            row=row,
                            col=1)

    fig_xyz.show()

    fig_rpy = make_subplots(rows=3,
                            cols=1,
                            subplot_titles=("Roll angle", "Pitch angle",
                                            "Yaw angle"))

    reference_rpy = reference_traj.get_orientations_euler()
    estimated_rpy = estimated_traj.get_orientations_euler()
    for row in range(1, 4):
        fig_rpy.add_scatter(x=dataset.timestamps,
                            y=reference_rpy[:, row - 1],
                            name="TSIF",
                            row=row,
                            col=1)
        fig_rpy.add_scatter(x=estimated_timestamps,
                            y=estimated_rpy[:, row - 1],
                            name="Estimated",
                            row=row,
                            col=1)
    fig_rpy.show()
