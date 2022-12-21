"""Robot specific tests"""

import pathlib

import gtdynamics as gtd
import gtsam
import numpy as np
from gtsam.utils.test_case import GtsamTestCase

from dyne import datasets
from dyne.arguments import parse_config
from dyne.utils import load_imu_params

RESOURCES = pathlib.Path(__file__).parent.parent.absolute() / "data"
MODEL_DIR = pathlib.Path(__file__).parent.parent.absolute() / "models"

np.set_printoptions(precision=18, suppress=True)


class TestA1(GtsamTestCase):

    ROBOT_MODEL = MODEL_DIR / "a1.urdf"

    def setUp(self):
        self.base = "trunk"
        self.robot = gtd.CreateRobotFromFile(str(self.ROBOT_MODEL), "", True)

        self.config_yaml = RESOURCES / "pybullet" / "config.yaml"

    @staticmethod
    def get_point_in_com(foot, bTlower, bTlower_com):
        """
        Compute the transform for the contact point in the corresponding leg's shank/lower leg CoM.
        This is done by first manually pulling the transform values between the shank and the toe, and the toe and point of contact.
        E.g. in the A1, this would be the joint `FR_toe_fixed` and `FR_toe`'s collision sphere radius.
        """

        # This is the `FR_toe` geometry sphere radius (same for all feet)
        # We're taking the toe as the contact point and ignoring the -0.02 radius
        #TODO(Varun) add the radius of the foot sphere.
        toeTcontact = gtsam.Pose3(gtsam.Rot3(), np.asarray([0, 0, 0]))

        # Don't add lowerTtoe transform if we're looking from the toe link.
        #NOTE if testing for _lower link, be sure to also adjust the dataset loading.
        if "_lower" in foot:
            # This pose is directly from the URDF e.g. FL_toe_fixed joint in a1
            # and is the same for all feet
            lowerTtoe = gtsam.Pose3(gtsam.Rot3(), np.asarray([0.0, 0.0, -0.2]))
        else:
            lowerTtoe = gtsam.Pose3()

        # Now compute the transform from the leg CoM to the poin of contact
        bTcontact = bTlower * lowerTtoe * toeTcontact
        lower_comTcontact = bTlower_com.inverse() * bTcontact
        return lower_comTcontact

    def get_foot_transforms(self, robot, foot, dataset, fk, t):
        foot_link = robot.link(foot)
        bTfoot_com = fk.atPose3(gtd.PoseKey(foot_link.id(), t))

        # Contact pose in base frame from dataset
        dataset_wTcontact = dataset.foot_poses[foot.replace("_toe",
                                                            "_lower")][t]
        wTb = dataset.states[t].pose()
        dataset_bTcontact = wTb.inverse() * dataset_wTcontact
        # print("Values for foot:", foot)
        # print("Contact pose in base frame:", f"{dataset_bTcontact=}")

        # Manually compute comTcontact for the foot
        computed_foot_comTcontact = self.get_point_in_com(
            foot, foot_link.bMlink(), foot_link.bMcom())
        # print(">>> This is the value that needs to go into the config file:")
        # print(f"{computed_foot_comTcontact=}")

        # The contact pose in body frame at rest.
        # The translation should be all the joint translations summed up.
        # E.g. t = [0.183, 0.13205, -0.4]
        side = (-1)**int(foot[1] == 'R')
        fore_legs = (-1)**int(foot[0] == 'R')

        bTcontact_at_rest = foot_link.bMcom() * computed_foot_comTcontact
        self.gtsamAssertEquals(gtsam.Pose3(
            gtsam.Rot3(), np.asarray([fore_legs * 0.183, side * 0.13205,
                                      -0.4])),
                               bTcontact_at_rest,
                               tol=1e-5)

        # The two poses, from the dataset and as computed, should be roughly equal
        self.gtsamAssertEquals(wTb * bTfoot_com * computed_foot_comTcontact,
                               wTb * dataset_bTcontact,
                               tol=1e-3)

        return computed_foot_comTcontact

    def test_contact_frame(self):
        """
        Test whether the contact point from the URDF matches our calculations.
        """
        config = parse_config(self.config_yaml)

        imu_params = load_imu_params(config.imu_yaml)
        dataset = datasets.PybulletDataset(config.measurements,
                                           config.ground_truth,
                                           robot_file=str(self.ROBOT_MODEL),
                                           bRs=imu_params["pose"].rotation())

        t = 0
        fk = self.robot.forwardKinematics(dataset.joint_angles[t], t,
                                          self.base)

        for foot in ("FL", "FR", "RL", "RR"):
            toe_comTcontact = self.get_foot_transforms(self.robot,
                                                       f"{foot}_toe", dataset,
                                                       fk, t)
            self.gtsamAssertEquals(toe_comTcontact, gtsam.Pose3(), 1e-6)

