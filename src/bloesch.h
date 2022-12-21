/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file bloesch.h
 * @date February 2022
 * @author Varun Agrawal
 * @brief Bloesch et. al. state estimator using IMU and leg kinematics.
 */

#pragma once

#include <gtdynamics/universal_robot/Robot.h>
#include <gtdynamics/universal_robot/sdf.h>

#include <vector>

#include "src/base.h"
#include "src/contact.h"

namespace dyne {
class Bloesch : public BaseImu {
 protected:
  std::string base_name_;
  std::vector<std::string> feet_;
  gtdynamics::Robot robot_;
  /// The type of foot constraint to use (point or flat foot).
  FootType foot_type_;

  std::map<std::string, gtsam::Point3> contact_in_com_;
  bool use_foot_height_constraint_;

  // Flags which tell us what the contact state of each leg was
  // in the previous time step.
  std::map<std::string, bool> previously_in_stance_;

  // Keep track of the contact point index for each step.
  // Used to define the contact point keys.
  std::map<std::string, uint64_t> step_index;

 public:
  Bloesch() : BaseImu() {}

  Bloesch(const RobotImu& base_imu, const std::string& model_file,
          uint64_t estimation_rate, const std::string& base_name,
          const std::vector<std::string>& feet,
          const std::map<std::string, gtsam::Point3>& contact_in_com,
          const NoiseParams& noise_params = NoiseParams(),
          const FootType& foot_type = FootType::POINT,
          bool use_foot_height_constraint = false);

  /**
   * @brief Add forward kinematics factor for leg specified by `end_link_name`.
   * The FK is enforced at the leg link CoM and not at the contact point.
   *
   * @param k The time step for the body.
   * @param joint_angles Values dict containing all the joint angles of the
   * robot.
   */
  virtual void add_forward_kinematics_factors(
      uint64_t k, const gtsam::Values& joint_angles);

  /**
   * @brief Add a factor to enforce the contact to be on the ground plane.
   *
   * @param k The time step.
   * @param end_link_name The name of the link whose position we want to compute
   * via FK.
   * @param ground_plane_height The ground plane height at the contact point.
   */
  virtual void add_contact_height_prior(uint64_t k,
                                        const std::string& end_link_name,
                                        double ground_plane_height = 0.0);

  /**
   * @brief If the foot is in the stance position, add a constraint between the
   * current pose and the contact point.
   *
   * @param foot The foot name for which the factor should be added.
   * @return gtsam::Key
   */
  virtual gtsam::Key add_stance_foot_factor(const std::string& foot);

  /**
   * @brief Add the various factors for each leg, and insert initial values.
   *
   * This method also takes care of keeping track of the phase for each foot.
   *
   * @param k Discrete time index (t = k*dt).
   * @param joint_angles Values dict with the joint angles for each joint.
   * @param contacts Dict giving the contact state at time `k` for each foot.
   */
  virtual void add_factors(uint64_t k, const gtsam::Values& joint_angles,
                           const std::map<std::string, int>& contacts);

  /**
   * @brief A measurement step which adds IMU and leg factors. The leg factors
   * are only added at the desired estimation rate.
   *
   * @param k Discrete time index (t = k*dt).
   * @param measured_omega_b The angular velocity in the body frame.
   * @param measured_acceleration_b The linear acceleration in the body frame.
   * @param joint_angles Values dict with the joint angles for each joint.
   * @param contacts Dictionary giving the contact state at time `k` for each
   * foot.
   */
  virtual void step(uint64_t k, const gtsam::Vector3& measured_omega_b,
                    const gtsam::Vector3& measured_acceleration_b,
                    const gtsam::Values& joint_angles,
                    const std::map<std::string, int>& contacts);

  std::string base_name() const { return base_name_; }
  std::vector<std::string> feet() const { return feet_; }
  gtdynamics::Robot robot() const { return robot_; }
  gtsam::Point3 contact_in_com(const std::string& foot) const {
    return contact_in_com_.at(foot);
  }
};
}  // namespace dyne
