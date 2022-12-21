"""
Script to update the foot contact positions from `toe` links to `lower` links
so that GTDynamics can handle it correctly.

Currently, GTDynamics' use of SDFormat8 causes links with fixed joints to be truncated into a single link.
Thus GTDynamics cannot get the pose of the toe link since it doesn't support fixed joints, it can only get
the `lower` link, e.g. `FR_lower`. 
This script replaces the contact pose from that of XX_toe to XX_lower so we can add contact point constraints.
"""

import argparse

import gtdynamics as gtd
import gtsam
import kinpy as kp
import numpy as np
from lrse.datasets import PybulletDataset


def parse_args():
    """Parse command line args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input npz file to load.")
    parser.add_argument("robot", help="Path to the robot URDF/SDF file")
    parser.add_argument("output", help="Path to output npz file")
    return parser.parse_args()


def save_dataset(dataset, robot, output_path):
    """Save the dataset to a npz file."""
    foot_positions_world = np.empty((dataset.size(), 4, 3))
    foot_orientations_world = np.empty((dataset.size(), 4, 4))
    for idx in range(dataset.size()):
        for fidx, foot in enumerate(dataset.feet):
            pose = dataset.foot_poses[foot][idx]
            foot_positions_world[idx, fidx] = pose.translation()
            q = pose.rotation().toQuaternion().coeffs()[[3, 0, 1, 2]]
            foot_orientations_world[idx, fidx] = q

    np.savez(output_path,
             timesteps=dataset.timestamps,
             base_position=dataset.data["base_position"],
             base_rotation=dataset.data["base_rotation"],
             base_vels=dataset.data["base_vels"],
             omega_b=dataset.data["omega_b"],
             joint_angles=dataset.data["joint_angles"],
             true_joint_angles=dataset.data["true_joint_angles"],
             foot_positions_base=dataset.data["foot_positions_base"],
             foot_orientations_base=dataset.data["foot_orientations_base"],
             foot_positions_world=foot_positions_world,
             foot_orientations_world=foot_orientations_world,
             foot_contacts=dataset.data["foot_contacts"])
    print("Updated dataset saved.")


def transform_to_pose(transform: kp.Transform):
    """Convert from kinpy Transform to Pose3"""
    return gtsam.Pose3(gtsam.Rot3.Quaternion(*transform.rot), transform.pos)


def pose_to_transform(pose: gtsam.Pose3):
    """Convert from Pose3 to kinpy Transform"""
    return kp.Transform(pose.rotation().toQuaternion().coeffs()[[3, 0, 1, 2]],
                        pose.translation())


def main():
    """Main runner."""
    args = parse_args()

    dataset = PybulletDataset(args.input, args.input, robot_file=args.robot)
    robot = dataset.robot

    robot_chain = kp.build_chain_from_urdf(open(args.robot).read())

    for k in range(dataset.size()):
        wTb = dataset.states[k].pose()

        joint_angles = dataset.original_joint_angles[k, :]

        th = {}
        joints = [
            "FR_hip_joint", "FR_upper_joint", "FR_lower_joint", "FL_hip_joint",
            "FL_upper_joint", "FL_lower_joint", "RR_hip_joint",
            "RR_upper_joint", "RR_lower_joint", "RL_hip_joint",
            "RL_upper_joint", "RL_lower_joint"
        ]

        for idx, joint in enumerate(joints):
            th[joint] = joint_angles[idx]

        fk = robot.forwardKinematics(dataset.joint_angles[k], k, "trunk")
        ret = robot_chain.forward_kinematics(th, world=pose_to_transform(wTb))

        for foot in dataset.feet:
            print(foot)
            toe = foot.replace("lower", "toe")
            # print(transform_to_pose(ret[foot]))
            wTknee = wTb * gtd.Pose(fk, robot.link(foot).id(), k)
            toeTknee = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, -0.2))
            print(toeTknee.inverse())
            print(wTknee * toeTknee.inverse())
            wTtoe = transform_to_pose(ret[toe])
            print(wTtoe)
            # print(dataset.foot_poses[foot][k])
            print(wTtoe.between(wTknee))

            dataset.contafoot_posesct_poses[foot][k] = wTtoe

    save_dataset(dataset, robot, args.output)


if __name__ == "__main__":
    main()
