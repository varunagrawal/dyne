/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file noise_params.h
 * @date February 2022
 * @author Varun Agrawal
 * @brief Noise parameters struct.
 */
#pragma once

#include <gtsam/base/Vector.h>

namespace dyne {

/// All the noise parameters used in the state estimator.
struct NoiseParams {
  double pose_prior;
  double velocity_prior;
  /// Bias for [acc_x acc_y acc_z gyro_x gyro_y gyro_z]
  gtsam::Vector6 bias_prior;
  // Sigma value for IMU Markov chain. [acc_x acc_y acc_z gyro_x gyro_y gyro_z]
  gtsam::Vector6 imu_between_sigmas;
  double joint_angle;
  double joint_velocity;
  double pose;  // Sigma for pose of links
  double contact_point;
  double contact_height;

  NoiseParams(double pose_prior = 0.01, double velocity_prior = 0.01,
              gtsam::Vector6 bias_prior = gtsam::Vector6::Ones() * 1e-3,
              gtsam::Vector6 imu_between_sigmas = gtsam::Vector6::Ones() * 1e-4,
              double joint_angle = 0.0, double joint_velocity = 0.0,
              double pose = 1, double contact_point = 1e-8,
              double contact_height = 1e-8)
      : pose_prior(pose_prior),
        velocity_prior(velocity_prior),
        bias_prior(bias_prior),
        imu_between_sigmas(imu_between_sigmas),
        joint_angle(joint_angle),
        joint_velocity(joint_velocity),
        pose(pose),
        contact_point(contact_point),
        contact_height(contact_height) {}
};

}  // namespace dyne
