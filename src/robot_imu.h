/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file robot_imu.h
 * @date February 2022
 * @author Varun Agrawal
 * @brief Helper class for Robot IMU sensors.
 */

#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

namespace dyne {

static const double GRAVITY = 9.81;

/**
 * @brief Helper class which encapsulates the IMU on a robot and computes
 * Preintegration parameters directly from provided stddev values.
 */
class RobotImu {
 private:
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> pim_params_;

 public:
  double freq;
  gtsam::imuBias::ConstantBias previous_bias;

  RobotImu() {}

  /**
   * @brief Construct a new Robot Imu object.
   *
   * @param freq IMU sampling frequency.
   * @param gyroscope_noise_stddev The standard deviation of the noise on the
   * gyroscope.
   * @param accelerometer_noise_stddev The standard deviation of the noise on
   * the accelerometer.
   * @param body_P_sensor The pose of this IMU in the body frame of the robot.
   * @param bias The IMU bias as [acc, gyro].
   * @param gravity The gravity value to use. We consider gravity to be in
   * negative Z-axis.
   */
  RobotImu(double freq, double gyroscope_noise_stddev,
           double accelerometer_noise_stddev,
           const gtsam::Pose3& body_P_sensor = gtsam::Pose3(),
           const gtsam::imuBias::ConstantBias& bias =
               gtsam::imuBias::ConstantBias(gtsam::Vector3::Zero(),
                                            gtsam::Vector3::Zero()),
           double gravity = 9.81)
      : pim_params_(parse_pim_params(gyroscope_noise_stddev,
                                     accelerometer_noise_stddev, body_P_sensor,
                                     gravity)),
        freq(freq),
        previous_bias(bias) {}

  /// Parse Preintegration params and generate a
  /// gtsam::PreintegrationCombinedParams object
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> parse_pim_params(
      double gyroscope_noise_stddev, double accelerometer_noise_stddev,
      const gtsam::Pose3& body_P_sensor = gtsam::Pose3(),
      double gravity = 9.81) {
    auto params = gtsam::PreintegrationCombinedParams::MakeSharedU(gravity);
    gtsam::Matrix3 I3x3 = gtsam::Matrix3::Identity();

    double gyro_sigma = gyroscope_noise_stddev;
    auto gyro_cov = I3x3 * std::pow(gyro_sigma, 2);

    double acc_sigma = accelerometer_noise_stddev;
    auto acc_cov = I3x3 * std::pow(acc_sigma, 2);

    params->setGyroscopeCovariance(gyro_cov);
    params->setAccelerometerCovariance(acc_cov);

    auto integration_error_cov = I3x3 * std::pow(1e-7, 2);
    params->setIntegrationCovariance(integration_error_cov);

    params->setBiasAccCovariance(I3x3 * 1e-4);
    params->setBiasOmegaCovariance(I3x3 * 1e-4);
    params->setBiasAccOmegaInit(gtsam::Matrix6::Identity() * 1e-8);

    params->setBodyPSensor(body_P_sensor);

    return params;
  }

  /// Get the preintegration params.
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> pim_params() const {
    return pim_params_;
  }

  void print() const { pim_params_->print(); }
};

}  // namespace dyne
