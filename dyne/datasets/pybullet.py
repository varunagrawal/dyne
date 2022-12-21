"""Dataset for the A1 from the PyBullet simulator."""

from pathlib import Path
from typing import Optional, Sequence, Union

import gtdynamics as gtd
import gtsam
import numpy as np

from dyne.state_estimator import GRAVITY

from .dataset import Dataset


class Sampler:
    def __init__(self, model, seed):
        self.rng = np.random.default_rng(seed)
        self.sigmas = model.sigmas()

    def sampleDiagonal(self, sigmas):
        d = sigmas.shape[0]
        result = np.zeros(d)
        for i in range(d):
            if sigmas[i] == 0.0:
                result[i] = 0.0
            else:
                result[i] = self.rng.normal(0.0, sigmas[i])
        return result

    def sample(self):
        return self.sampleDiagonal(self.sigmas)


class PybulletDataset(Dataset):
    """
    Load data collected from PyBullet simulation.
    """
    def __init__(
        self,
        measurements_file: Union[str, Path],
        ground_truth_file: Union[str, Path],
        robot_file: Union[str, Path],
        skip: int = 0,
        bRs: Optional[gtsam.Pose3] = None,
        pim_params=None,
        freq: int = 200,
        base: str = "trunk",
        feet: Sequence[str] = ("FR_lower", "FL_lower", "RR_lower", "RL_lower"),
        joint_names: Sequence[str] = ('FR_hip_joint', 'FR_upper_joint',
                                      'FR_lower_joint', 'FL_hip_joint',
                                      'FL_upper_joint', 'FL_lower_joint',
                                      'RR_hip_joint', 'RR_upper_joint',
                                      'RR_lower_joint', 'RL_hip_joint',
                                      'RL_upper_joint', 'RL_lower_joint'),
        gravity_n: np.ndarray = np.asarray([0, 0, -GRAVITY])):
        """Class for loading Pybullet simulation data.

        Args:
            measurements_file (str, Path): The npz file with the measurement data.
            ground_truth_file (str, Path): A npz file with the ground truth data.
            robot_file (str, Path): The SDF/URDF file for the robot used in the simulation.
            skip (int): The number of lines to skip.
            bRs (gtsam.Pose3, optional): Rotation for IMU sensor frame in pelvis frame
            pim_params ([type], optional): Parameters for Pre-Integrated Measurements. Defaults to None.
            freq (int, optional): The IMU measurement frequency. Defaults to 200.
            base (str, optional): The name of the base link. Defaults to "trunk".
            feet (tuple, optional): [description]. Defaults to list of A1 robot joints.
            gravity_n ([type], optional): The gravity vector in the navigation frame.
                Defaults to np.asarray([0, 0, -GRAVITY]).
        """
        self.base = base
        # The foot links
        self.feet = feet
        self.robot_file = robot_file
        self.joint_names = joint_names
        self.freq = freq
        self.dt = 1 / freq
        self.gravity_n = gravity_n

        self.pim_params = pim_params

        self.robot = gtd.CreateRobotFromFile(robot_file, "", True)

        # These are the transforms from leg CoM to toe CoM for the A1
        self.legTtoe = gtsam.Pose3(gtsam.Rot3(),
                                   gtsam.Point3(-0.00472659, 0, -0.0680247))

        super().__init__(measurements_file,
                         ground_truth_file,
                         skip=skip,
                         bRs=bRs)

    def add_noise(self):
        """
        Add noise to the angular velocity and linear acceleration
        measurements if the IMU parameters are provided.
        """
        if self.pim_params is not None:
            gyroSampler = Sampler(
                gtsam.noiseModel.Gaussian.Covariance(
                    self.pim_params.getGyroscopeCovariance()), 10)
            accelSampler = Sampler(
                gtsam.noiseModel.Gaussian.Covariance(
                    self.pim_params.getAccelerometerCovariance()), 29284)

            sqrt_dt = np.sqrt(self.dt)

            # Add gyro noise to the angular velocity
            for idx in range(self.angular_velocity.shape[0]):
                self.angular_velocity[idx] += gyroSampler.sample() / sqrt_dt
            # Add acceleration noise to the linear acceleration
            for idx in range(self.linear_acceleration.shape[0]):
                self.linear_acceleration[idx] += accelSampler.sample(
                ) / sqrt_dt

    def load_measurements(self, measurements_file):
        """
        Load the measurement data which includes foot contact and joint angles.

        Args:
            measurements_file: npz file containing the measurement data.
        """
        data = np.load(measurements_file, allow_pickle=True)
        # Keep track of original data.
        self.data = data

        self.timestamps = data["timesteps"]

        # Angular velocity is already in the body frame
        self.angular_velocity = data["omega_b"]

        # Compute acceleration from velocity via the Sobel filter.
        self.linear_acceleration = np.zeros((len(self.states), 3))
        for idx, _ in enumerate(self.states):
            if idx == 0 or idx == len(self.states) - 1:
                acc_n = np.zeros(3)
            else:
                # Apply Sobel derivative
                acc_n = (self.states[idx + 1].velocity() -
                         self.states[idx - 1].velocity()) / (2 * self.dt)

            nRb = self.states[idx].attitude()
            # Convert from nav frame to body frame
            acc_b = nRb.unrotate(acc_n - self.gravity_n)
            self.linear_acceleration[idx, :] = acc_b

        # Maybe add noise to the measurements.
        self.add_noise()

        foot_contacts = data["foot_contacts"]
        self.contacts = {}
        for fidx, foot in enumerate(self.feet):
            # We use _toe instead of _lower in the estimator
            key = foot.replace("lower", "toe")
            self.contacts[key] = foot_contacts[self.skip:, fidx]

            # Trim the foot poses
            self.foot_poses[foot] = self.foot_poses[foot][self.skip:]

        self.joint_angle_data = data["joint_angles"]

        self.joint_angles = []

        # For each time step, convert vector of joint angles to a joint angles Values.
        for k in range(self.timestamps.shape[0]):
            joint_angles = gtd.Values()
            for joint in self.robot.joints():
                # Pybullet didn't account for fixed joints so we set the angle to 0 by default.
                if joint.name() not in self.joint_names:
                    angle = 0.0
                else:
                    angle = self.joint_angle_data[
                        k, self.joint_names.index(joint.name())]
                # Store joint angles as Values dict
                gtd.InsertJointAngle(joint_angles, joint.id(), k, angle)

            self.joint_angles.append(joint_angles)

        # Update foot poses to be the contact pose since the toe pose from the simulator does not consider the ball from the toe CoM to the ground.
        # This is done by adding a translation for the sole of the foot.
        for k in range(self.timestamps.shape[0]):
            for foot in self.feet:
                original_foot_pose = self.foot_poses[foot][k]

                # # Directly project tz down to 0 for the ground.
                # # This should also handle the cases where the toe is rotated.
                # t = original_foot_pose.translation()
                # t[2] = 0.0

                # new_foot_pose = gtsam.Pose3(original_foot_pose.rotation(), t)
                # self.foot_poses[foot][k] = new_foot_pose

    def load_ground_truth(self, ground_truth_file, bRs=None):
        """
        Load ground truth data.

        The ground truth is of the robot pelvis.
        Thus, we modify it to be the pose of the IMU
        in the navigation frame before loading it into the NavState.

        Args:
            ground_truth_file: A npz file with the ground truth data.
            bRs: Rotation for IMU sensor frame in pelvis frame
        """
        self.states.clear()

        data = np.load(ground_truth_file, allow_pickle=True)

        base_position = data["base_position"]
        base_rotation = data["base_rotation"]
        base_vels = data["base_vels"]

        for idx in range(base_position.shape[0]):
            nPb = gtsam.Point3(*base_position[idx])
            # rotation of pelvis frame in nav frame
            nRb = gtsam.Rot3.Quaternion(*base_rotation[idx])
            # nRb is the imu in the navigation frame.
            nTb = gtsam.Pose3(nRb, nPb)

            nVb = base_vels[idx]
            state = gtsam.NavState(nTb, nVb)

            if bRs:
                nPs = bRs.unrotate(nPb)
                nRs = nRb.compose(bRs)
                nTs = gtsam.Pose3(nRs, nPs)
                nVs = bRs.unrotate(nVb)
                state = gtsam.NavState(nTs, nVs)

            self.states.append(state)

        foot_positions = data["foot_positions_world"]
        foot_orientations = data["foot_orientations_world"]

        # foot_pose is the pose of the toe link of the A1 in the world frame
        # This is later updated in the load_measurements method to the contact point.
        self.foot_poses = {foot: [] for foot in self.feet}
        for idx in range(base_position.shape[0]):
            for fidx, foot in enumerate(self.feet):
                wPc = gtsam.Point3(*foot_positions[idx, fidx])
                wRc = gtsam.Rot3.Quaternion(*foot_orientations[idx, fidx])
                wTc = gtsam.Pose3(wRc, wPc)
                self.foot_poses[foot].append(wTc)

        foot_positions = data["lower_link_positions_world"]
        foot_orientations = data["lower_link_orientations_world"]
        self.lower_links = {foot: [] for foot in self.feet}
        for idx in range(base_position.shape[0]):
            for fidx, foot in enumerate(self.feet):
                wPc = gtsam.Point3(*foot_positions[idx, fidx])
                wRc = gtsam.Rot3.Quaternion(*foot_orientations[idx, fidx])
                wTc = gtsam.Pose3(wRc, wPc)
                self.lower_links[foot].append(wTc)

    def export_csv(self, measurements_file, ground_truth_file):
        """
        Helper method to export data as CSV.

        Example use:
        pybullet.export_csv("pybullet_measurements.csv",
                        "pybullet_ground_truth.csv")
        """
        # Get timestamps, angular velocity and linear acceleration.
        data = np.empty((self.timestamps.shape[0], 1))
        data[:, 0] = self.timestamps
        data = np.hstack((data, self.angular_velocity))
        data = np.hstack((data, self.linear_acceleration))

        header = "timestamps,wx,wy,wz,ax,ay,az"

        # Get contact information
        contacts = np.empty((self.timestamps.shape[0], 4))
        for f, foot in enumerate(self.feet):
            fname = foot.replace("lower", "toe")
            contacts[:, f] = self.contacts[fname]
            header += f",{fname[:2]}_contact"
        data = np.hstack((data, contacts))

        # Get all joint angles
        joint_angles = np.zeros(
            (self.timestamps.shape[0], self.robot.numJoints()))

        for joint in self.robot.joints():
            header += f",{joint.name()}"

        for k in range(self.timestamps.shape[0]):
            angles = self.joint_angles[k]
            for j, joint in enumerate(self.robot.joints()):
                angle = gtd.JointAngle(angles, joint.id(), k)
                joint_angles[k, j] = angle

        data = np.hstack((data, joint_angles))

        np.savetxt(measurements_file, data, delimiter=',', header=header)
        print(f"{measurements_file} saved!")

        #################

        # Get base states
        header = "timestamps,qw,qx,qy,qz,tx,ty,tz,vx,vy,vz"
        data = np.empty((self.timestamps.shape[0], 11))

        data[:, 0] = self.timestamps
        for k, state in enumerate(self.states):
            q = state.attitude().toQuaternion()
            data[k, 1] = q.w()
            data[k, 2] = q.x()
            data[k, 3] = q.y()
            data[k, 4] = q.z()
            t = state.position()
            data[k, 5] = t[0]
            data[k, 6] = t[1]
            data[k, 7] = t[2]
            v = state.velocity()
            data[k, 8] = v[0]
            data[k, 9] = v[1]
            data[k, 10] = v[2]

        # Get foot contact positions
        foot_data = np.empty((self.timestamps.shape[0], len(self.feet) * 7))
        for f, foot in enumerate(self.feet):
            header += ",{0}_qw,{0}_qx,{0}_qy,{0}_qz,{0}_tx,{0}_ty,{0}_tz".format(
                foot[:2])
            for k, wTc in enumerate(self.foot_poses[foot]):
                q = wTc.rotation().toQuaternion()
                foot_data[k, f * 7 + 0] = q.w()
                foot_data[k, f * 7 + 1] = q.x()
                foot_data[k, f * 7 + 2] = q.y()
                foot_data[k, f * 7 + 3] = q.z()

                t = wTc.translation()
                foot_data[k, f * 7 + 4] = t[0]
                foot_data[k, f * 7 + 5] = t[1]
                foot_data[k, f * 7 + 6] = t[2]

        data = np.hstack((data, foot_data))

        np.savetxt(ground_truth_file, data, delimiter=',', header=header)
        print(f"{ground_truth_file} saved!")
