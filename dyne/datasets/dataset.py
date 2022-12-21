"""Base class for datasets."""

import gtdynamics as gtd
import numpy as np
import roboplot
from gtsam.symbol_shorthand import X
from gtsam.utils import plot
from matplotlib import pyplot as plt


class Dataset:
    """
    Base class for loading data.
    """

    def __init__(self,
                 measurements_file=None,
                 ground_truth_file=None,
                 skip=0,
                 bRs=None):
        self.states = []
        self.timestamps = np.empty(0)
        self.angular_velocity = []
        self.linear_acceleration = []
        self.bRs = bRs

        self.skip = skip

        if ground_truth_file:
            self.load_ground_truth(ground_truth_file, bRs=bRs)
        if measurements_file:
            self.load_measurements(measurements_file)

    def load_measurements(self, measurements_file):
        """Load measurements from the `measurements_file` CSV."""
        raise NotImplementedError()

    def load_ground_truth(self, ground_truth_file, bRs=None):
        """Load ground truth from the `ground_truth_file` CSV."""
        raise NotImplementedError()

    def size(self):
        """Return the number of measurements."""
        return self.timestamps.shape[0]

    def ground_truth_as_values(self, sampling_freq=1):
        """Convert the ground truth data matrix to a gtdynamics.Values object and return it."""
        values = gtd.Values()
        k = 0
        for i, state in enumerate(self.states):
            if i % int(sampling_freq) == 0:
                values.insert(X(k), state.pose())
                k += 1
        return values

    def contacts_at_idx(self, k):
        """Get the contact values for all the feet as a dict at time index `k`."""
        contacts_at_k = {
            foot: contacts[k]
            for foot, contacts in self.contacts.items()
        }
        return contacts_at_k

    def compute_frequency(self):
        """
        Compute the frequency of the IMU from the timestamps.
        This is because the ROS topic does not guarantee messages at the specified frequency.
        """
        count = 0
        # sum of all the time deltas
        sum_of_differences = 0

        for i in range(1, self.timestamps.shape[0]):
            count += 1
            sum_of_differences += (self.timestamps[i] -
                                   self.timestamps[i - 1]) * 1e-9
        return count / sum_of_differences

    def plot_ground_truth_trajectory(self,
                                     fignum=1,
                                     sampling_freq=1,
                                     title="Ground Truth Trajectory",
                                     show=False):
        """Plot the ground truth trajectory."""
        values = self.ground_truth_as_values(sampling_freq=sampling_freq)
        plot.plot_trajectory(fignum, values, scale=0.5)
        plot.set_axes_equal(fignum)
        fig = plt.figure(fignum)

        # plot the start point
        pose0 = values.atPose3(X(0))
        ax = fig.gca(projection='3d')
        t = pose0.translation()
        ax.plot(t[0:1], t[1:2], t[2:3], color='k', marker='o')

        ax.view_init(75, -90)

        fig.suptitle(title)
        fig.canvas.set_window_title(title.lower())

        if show:
            plt.show()

    def plot_ground_truth(self,
                          GT_FIG=2,
                          title="Ground Truth State",
                          show=False):
        """Plot translation and velocities."""
        fig, axes = plt.subplots(2, 3, num=GT_FIG, constrained_layout=True)
        fig.suptitle(title)
        fig.canvas.set_window_title(title.lower())

        labels = list('xyz')
        colors = list('rgb')

        # plot translation
        translation = np.asarray([
            self.states[i].pose().translation()
            for i in range(len(self.states))
        ])
        min_val, max_val = translation.min(), translation.max()
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax = axes[0][i]
            ax.scatter(
                self.timestamps / 10**9,  # timestamps are in nanosecs
                translation[:, i],
                color=color,
                marker='.',
                s=2)
            ax.set_title('P' + label)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Translation (m)')

            ax.set_ylim(min_val - 1, max_val + 1)

        # plot velocity
        velocity = np.asarray(
            [self.states[i].velocity() for i in range(len(self.states))])
        min_val, max_val = translation.min(), translation.max()
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax = axes[1][i]
            ax.scatter(
                self.timestamps / 10**9,  # timestamps are in nanosecs
                velocity[:, i],
                color=color,
                marker='.',
                s=2)
            ax.set_title('V' + label)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (m/s)')

            ax.set_ylim(min_val - 1, max_val + 1)

        if show:
            plt.show()

    def plot_measurements(self,
                          IMU_FIG=3,
                          title="IMU Measurement Data",
                          show=False):
        """Plot the measurement data from the IMU."""
        roboplot.plot_imu_measurements(self.timestamps,
                                       self.angular_velocity,
                                       self.linear_acceleration,
                                       fignum=IMU_FIG,
                                       title=title,
                                       show=show)

    def plot_data(self, show=False):
        """Plot the ground truth data, measurement data and complete trajectory."""
        self.plot_ground_truth()
        self.plot_measurements()
        self.plot_ground_truth_trajectory()
        if show:
            plt.show()
