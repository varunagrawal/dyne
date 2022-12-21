/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file bloesch.cpp
 * @date February 2022
 * @author Varun Agrawal
 * @brief Bloesch et. al. state estimator using IMU and leg kinematics.
 */

#include "src/bloesch.h"

#include <gtdynamics/factors/ContactHeightFactor.h>
#include <gtdynamics/factors/ContactPointFactor.h>
#include <gtdynamics/factors/ForwardKinematicsFactor.h>
#include <gtdynamics/utils/values.h>

using gtsam::symbol_shorthand::X;

namespace dyne {

/*****************************************************************************/
Bloesch::Bloesch(const RobotImu& base_imu, const std::string& model_file,
                 uint64_t estimation_rate, const std::string& base_name,
                 const std::vector<std::string>& feet,
                 const std::map<std::string, gtsam::Point3>& contact_in_com,
                 const NoiseParams& noise_params, const FootType& foot_type,
                 bool use_foot_height_constraint)
    : BaseImu(base_imu, estimation_rate, noise_params),
      base_name_(base_name),
      feet_(feet),
      robot_(gtdynamics::CreateRobotFromFile(model_file, "", true)),
      contact_in_com_(contact_in_com),
      foot_type_(foot_type),
      use_foot_height_constraint_(use_foot_height_constraint) {
  for (auto&& foot : feet_) {
    // Set previously_in_stance_ to false initially
    // so we can add the first contact state.
    previously_in_stance_[foot] = false;
    step_index[foot] = 0;
  }
}

/*****************************************************************************/
void Bloesch::add_forward_kinematics_factors(
    uint64_t k, const gtsam::Values& joint_angles) {
  for (auto&& foot_link_name : feet_) {
    uint64_t t = index;
    auto foot_link = robot_.link(foot_link_name);
    auto factor = gtdynamics::ForwardKinematicsFactor(
        X(t),  // Add key for wTb (base in the world frame)
        gtdynamics::PoseKey(foot_link->id(), t),  // wTleg
        robot_, base_name_, foot_link_name, joint_angles,
        gtsam::noiseModel::Isotropic::Sigma(6, noise_params_.pose), k);

    graph_.push_back(factor);

    // Get the current estimate of the base pose.
    gtsam::Pose3 wTb = state_i.pose();

    // Add the leg lower pose in the world frame
    gtsam::Pose3 wTl = wTb * factor.measured();
    gtdynamics::InsertPose(&initial_, foot_link->id(), t, wTl);
  }
}

/*****************************************************************************/
void Bloesch::add_contact_height_prior(uint64_t k,
                                       const std::string& foot_link_name,
                                       double ground_plane_height) {
  uint64_t t = index;
  auto foot_link = robot_.link(foot_link_name);
  gtsam::Vector3 gravity_n = base_imu_.pim_params()->n_gravity;

  auto factor = gtdynamics::ContactHeightFactor(
      gtdynamics::PoseKey(foot_link->id(), t),  // wTleg
      gtsam::noiseModel::Isotropic::Sigma(1, noise_params_.contact_height),
      contact_in_com_[foot_link_name], gravity_n, ground_plane_height);
  graph_.push_back(factor);
}

/*****************************************************************************/
gtsam::Key Bloesch::add_stance_foot_factor(const std::string& foot) {
  auto foot_link = robot_.link(foot);

  gtsam::Key link_pose_key = gtdynamics::PoseKey(foot_link->id(), index);
  gtsam::Key point_key = gtdynamics::DynamicsSymbol::SimpleSymbol(
      foot.substr(0, 2), step_index[foot]);

  // The noise model is a function of the deformation of the
  // contact point when it is in contact with the environment.
  uint noise_model_dim;
  if (foot_type_ == FootType::POINT) {
    noise_model_dim = 3;
  } else {
    noise_model_dim = 6;
  }
  auto noise_model = gtsam::noiseModel::Isotropic::Sigma(
      noise_model_dim, noise_params_.contact_point);

  gtsam::Point3 leg_contact_in_com = contact_in_com_.at(foot);

  // Depending on the type of foot, add the appropriate factor.
  if (foot_type_ == FootType::POINT) {
    graph_.emplace_shared<gtdynamics::ContactPointFactor>(
        link_pose_key, point_key, noise_model, leg_contact_in_com);

  } else if (foot_type_ == FootType::FLAT) {
    gtsam::Pose3 comTcontact(gtsam::Rot3(), leg_contact_in_com);
    graph_.emplace_shared<gtdynamics::ContactPoseFactor>(
        link_pose_key, point_key, noise_model, comTcontact);
  }

  // Only add the initial value for the first stance time index.
  if (!previously_in_stance_[foot]) {
    // Calculate the initial stance point in the world frame.
    gtsam::Point3 wPcontact =
        gtdynamics::PointOnLink(robot_.link(foot), contact_in_com_[foot])
            .predict(initial_, index);

    if (foot_type_ == FootType::POINT) {
      initial_.insert(point_key, wPcontact);
    } else {
      // Flat foot constraint. Same rotation but translated contact.
      initial_.insert(point_key, gtsam::Pose3(gtsam::Rot3(), wPcontact));
    }
  }

  return point_key;
}

/*****************************************************************************/
void Bloesch::add_factors(uint64_t k, const gtsam::Values& joint_angles,
                          const std::map<std::string, int>& contacts) {
  // Flag indicating if we should add leg factors.
  bool add_leg_factors = (k % rate_ == 0);
  if (add_leg_factors) {
    // Add the forward kinematics factors for all legs
    add_forward_kinematics_factors(k, joint_angles);

    for (size_t fidx = 0; fidx < feet_.size(); fidx++) {
      std::string foot = feet_[fidx];
      // If add_leg_factors is True, it means we are at the next time step
      // so we have to check if the foot was in stance at the last index.

      // We only check contact info when
      // we have to add factors for the contact point.
      // NOTE The assumption is that the foot does not switch
      // contact states between 2 time steps.
      if (contacts.at(foot) == ContactState::STANCE) {
        // If the foot was not in stance before but is now in stance, we
        // increment the step index. This is assuming all feet were on the
        // ground at the start.
        if (!previously_in_stance_[foot]) {
          step_index[foot] += 1;
        }

        // The foot is in stance phase so we add a contact point factor.
        add_stance_foot_factor(foot);

        // Add foot height constraint
        if (use_foot_height_constraint_) {
          add_contact_height_prior(k, foot, 0.0);
        }

        // We now set the contact flag for future timesteps
        previously_in_stance_[foot] = true;

      } else {
        // Foot is no longer continuously in stance so set flag to false
        // This records that we will have a new contact point
        // in the future.
        previously_in_stance_[foot] = false;
      }
    }
  }
}

/*****************************************************************************/
void Bloesch::step(uint64_t k, const gtsam::Vector3& measured_omega_b,
                   const gtsam::Vector3& measured_acceleration_b,
                   const gtsam::Values& joint_angles,
                   const std::map<std::string, int>& contacts) {
  add_factors(k, joint_angles, contacts);

  // Base class perform IMU preintegration
  // The time `index` is incremented in the base class
  BaseImu::step(k, measured_omega_b, measured_acceleration_b);
}

}  // namespace dyne
