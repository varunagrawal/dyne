/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file dyne.h
 * @date February 2022
 * @author Varun Agrawal
 * @brief Bloesch et. al. state estimator using IMU and leg kinematic chain
 * modeling.
 */

#pragma once

#include "src/bloesch.h"

namespace dyne {
class Dyne : public Bloesch {
 public:
  Dyne() {}

  /**
   * @brief Construct a new Dyne state estimator.
   *
   * @param base_imu
   * @param model_file
   * @param estimation_rate
   * @param base_name
   * @param feet
   * @param contact_in_com
   * @param noise_params
   * @param use_foot_height_constraint
   */
  Dyne(const RobotImu& base_imu, const std::string& model_file,
       uint64_t estimation_rate, const std::string& base_name,
       const std::vector<std::string>& feet,
       const std::map<std::string, gtsam::Point3>& contact_in_com,
       const NoiseParams& noise_params = NoiseParams(),
       const FootType& foot_type = FootType::POINT,
       bool use_foot_height_constraint = false);

  void add_forward_kinematics_factors(
      uint64_t k, const gtsam::Values& joint_angles) override;

  void print() const override;
};

}  // namespace dyne
