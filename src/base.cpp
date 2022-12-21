/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file base.cpp
 * @date February 2022
 * @author Varun Agrawal
 * @brief Base estimator using only the IMU.
 */

#include "src/base.h"

#include <gtsam/slam/BetweenFactor.h>

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace dyne {

BaseImu::BaseImu(const RobotImu& base_imu, uint64_t estimation_rate,
                 NoiseParams noise_params)
    : base_imu_(base_imu), noise_params_(noise_params) {
  pose_prior_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(6, noise_params.pose_prior);
  velocity_prior_noise_ =
      gtsam::noiseModel::Isotropic::Sigma(3, noise_params.velocity_prior);
  bias_prior_noise_ =
      gtsam::noiseModel::Diagonal::Sigmas(noise_params.bias_prior);

  freq_ = base_imu.freq;
  dt_ = 1 / freq_;

  // If estimation rate is 20 Hz, we want to estimate at every 10th timestep for
  // a 200 Hz IMU.
  estimation_rate_ = estimation_rate;
  rate_ = freq_ / estimation_rate;

  // The state index indicating when a state was added to the graph
  index = 0;
  bias_index_ = 0;

  // initialize data structure for pre-integrated IMU measurements
  pim_ = gtsam::PreintegratedImuMeasurements(base_imu.pim_params(),
                                             base_imu.previous_bias);
}

gtsam::NonlinearFactorGraph BaseImu::add_prior(
    uint64_t k, gtsam::NonlinearFactorGraph& graph,
    const gtsam::NavState& state, bool add_bias_prior) {
  graph.addPrior<gtsam::Pose3>(X(k), state.pose(), pose_prior_noise_);
  graph.addPrior<gtsam::Vector3>(V(k), state.velocity(), velocity_prior_noise_);

  if (add_bias_prior) {
    graph.addPrior<gtsam::imuBias::ConstantBias>(B(k), base_imu_.previous_bias,
                                                 bias_prior_noise_);
  }

  return graph;
}

gtsam::Values BaseImu::insert_values(gtsam::Values& values, uint64_t i,
                                     const gtsam::NavState& initial_estimate,
                                     bool add_bias) {
  gtsam::Pose3 initial_pose = initial_estimate.pose();
  gtsam::Vector3 initial_velocity = initial_estimate.velocity();
  values.insert(X(i), initial_pose);
  values.insert(V(i), initial_velocity);

  if (add_bias) {
    if (!values.exists(B(bias_index_))) {
      values.insert(B(bias_index_), base_imu_.previous_bias);
    }
  }
  return values;
}

gtsam::NonlinearFactorGraph BaseImu::add_imu_factor(
    uint64_t i, gtsam::NonlinearFactorGraph& graph,
    const gtsam::PreintegratedImuMeasurements& pim) {
  graph.emplace_shared<gtsam::ImuFactor>(X(i), V(i), X(i + 1), V(i + 1),
                                         B(bias_index_), pim);
  return graph;
}

gtsam::NonlinearFactorGraph BaseImu::add_bias_chain(
    uint64_t i, gtsam::NonlinearFactorGraph& graph) {
  gtsam::imuBias::ConstantBias zero_bias;  // default is zero bias
  auto noise_model = gtsam::noiseModel::Diagonal::Sigmas(
      noise_params_.imu_between_sigmas * std::sqrt(pim_.deltaTij()));

  graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
      B(i), B(i + 1), zero_bias, noise_model));

  return graph;
}

void BaseImu::set_prior_state(const gtsam::NavState& state_0,
                              bool add_bias_prior) {
  // add prior on the first state
  graph_ = add_prior(0, graph_, state_0, add_bias_prior);
  // insert state into values
  initial_ = insert_values(initial_, 0, state_0, true);
  state_i = state_0;
}

void BaseImu::step(uint64_t k, const gtsam::Vector3& measured_omega,
                   const gtsam::Vector3& measured_acceleration, bool add_bias) {
  // Perform IMU preintegration
  pim_.integrateMeasurement(measured_acceleration, measured_omega, dt_);

  // create IMU factor every `rate_` seconds
  add_imu_factor_ = ((k + 1) % rate_ == 0);

  if (add_imu_factor_) {
    graph_ = add_imu_factor(index, graph_, pim_);

    if (add_bias) {
      graph_ = add_bias_chain(bias_index_, graph_);
      bias_index_ += 1;
    }

    // Get the next state prediction from the IMU preintegration
    gtsam::NavState state_j = pim_.predict(state_i, base_imu_.previous_bias);

    initial_ = insert_values(initial_, index + 1, state_j, add_bias);

    // Set the current state to the next state
    state_i = state_j;

    // Reset the preintegration
    pim_.resetIntegration();

    index += 1;
  }
}

void BaseImu::print() const {
  graph_.print("");
  std::cout << "======================" << std::endl;
  initial_.print("");
  std::cout << "======================" << std::endl;
}

gtsam::Values BaseImu::optimize(bool verbose) const {
  std::cout << "Initial error: " << graph_.error(initial_) << std::endl;

  // optimize using Levenberg-Marquardt optimization
  gtsam::LevenbergMarquardtParams params;
  if (verbose) {
    params.setVerbosityLM("SUMMARY");
  }
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_, params);

  gtsam::Values result = optimizer.optimize();

  std::cout << "Optimization completed" << std::endl;
  std::cout << "Optimized error: " << graph_.error(result) << std::endl;

  return result;
}

gtsam::Values BaseImu::update() {
  // Update ISAM
  isam_.update(this->graph_, this->initial_);

  // Clear the current graph and values
  this->graph_.resize(0);
  this->initial_.clear();

  // Return the new estimate
  return isam_.calculateEstimate();
}

}  // namespace dyne
