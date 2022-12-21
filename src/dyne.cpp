/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file dyne.cpp
 * @date February 2022
 * @author Varun Agrawal
 * @brief Bloesch et. al. state estimator using IMU and leg kinematics.
 */

#include "src/dyne.h"

#include <gtdynamics/factors/JointMeasurementFactor.h>
#include <gtdynamics/utils/values.h>

using gtsam::symbol_shorthand::X;

namespace dyne {

/*****************************************************************************/
Dyne::Dyne(const RobotImu& base_imu, const std::string& model_file,
           uint64_t estimation_rate, const std::string& base_name,
           const std::vector<std::string>& feet,
           const std::map<std::string, gtsam::Point3>& contact_in_com,
           const NoiseParams& noise_params, const FootType& foot_type,
           bool use_foot_height_constraint)
    : Bloesch(base_imu, model_file, estimation_rate, base_name, feet,
              contact_in_com, noise_params, foot_type,
              use_foot_height_constraint) {}

/*****************************************************************************/
void Dyne::add_forward_kinematics_factors(uint64_t k,
                                          const gtsam::Values& joint_angles) {
  uint64_t t = index;
  auto fk = robot_.forwardKinematics(joint_angles, k, base_name_);
  gtsam::Pose3 wTb = state_i.pose();

  auto pose_model = gtsam::noiseModel::Isotropic::Sigma(6, noise_params_.pose);

  for (auto&& joint : robot_.joints()) {
    auto parent_link = joint->parent();
    auto child_link = joint->child();

    gtsam::Key parent_link_key;

    if (parent_link->name() == base_name_) {
      parent_link_key = X(t);
      // NOTE We don't add an initial estimate for the base link
      // since it is added by the superclass
    } else {
      parent_link_key = gtdynamics::PoseKey(parent_link->id(), t);
    }

    gtsam::Key child_link_key = gtdynamics::PoseKey(child_link->id(), t);

    // Add initial estimate for all the children which is every other link
    gtsam::Pose3 bTl = gtdynamics::Pose(fk, child_link->id(), k);
    initial_.insert<gtsam::Pose3>(child_link_key, wTb * bTl);

    // Get the joint angle value
    double joint_angle = gtdynamics::JointAngle(joint_angles, joint->id(), k);

    // Add unary measurements on joint angles
    graph_.emplace_shared<gtdynamics::JointMeasurementFactor>(
        parent_link_key, child_link_key, pose_model, joint, joint_angle);
  }
}

/*****************************************************************************/
void Dyne::print() const {
  graph_.print("", gtdynamics::GTDKeyFormatter);
  std::cout << "\n======================" << std::endl;
  initial_.print("", gtdynamics::GTDKeyFormatter);
  std::cout << "\n======================" << std::endl;
}

}  // namespace dyne
