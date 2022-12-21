/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file base.h
 * @date February 2022
 * @author Varun Agrawal
 * @brief Base estimator using only the IMU.
 */

#pragma once

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include "src/noise_params.h"
#include "src/robot_imu.h"

namespace dyne {

class BaseImu {
 protected:
  boost::shared_ptr<gtsam::noiseModel::Isotropic> pose_prior_noise_,
      velocity_prior_noise_;
  boost::shared_ptr<gtsam::noiseModel::Diagonal> bias_prior_noise_;

  const RobotImu base_imu_;
  const NoiseParams noise_params_;

  // The IMU frequency.
  double freq_;
  // The time delta between measurements
  double dt_;
  /** The rate (in Hz) at which to perform estimation.
   * E.g. we may want to estimate at 50 Hz rather than at IMU frequency.
   */
  uint64_t estimation_rate_;
  /**
   * Defines after how many timesteps we want to estimate.
   * This means that if freq=200 Hz and estimation_rate=10 Hz,
   * we want to estimate every 200/10=20th step.
   */
  uint64_t rate_;

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_;
  gtsam::ISAM2 isam_;

  /// Set initial state to identity NavState
  gtsam::NavState state_i;

  gtsam::PreintegratedImuMeasurements pim_;

  bool add_imu_factor_;

 public:
  /// The state index indicating when a state was added to the graph
  uint64_t index;
  uint64_t bias_index_;

  BaseImu() {}

  BaseImu(const RobotImu& base_imu, uint64_t estimation_rate = 20,
          NoiseParams noise_params = NoiseParams());

  virtual gtsam::NonlinearFactorGraph add_prior(
      uint64_t k, gtsam::NonlinearFactorGraph& graph,
      const gtsam::NavState& state, bool add_bias_prior = false);

  virtual gtsam::Values insert_values(gtsam::Values& values, uint64_t i,
                                      const gtsam::NavState& initial_estimate,
                                      bool add_bias = true);

  /**
   * @brief Add an IMU factor for index `i` using the
   * preintegrated measurements `pim`.
   *
   * @param i The state index at which to add the factor.
   * @param graph The factor graph.
   * @param pim The preintegrated measurements up to that particular index.
   * @return gtsam::NonlinearFactorGraph
   */
  virtual gtsam::NonlinearFactorGraph add_imu_factor(
      uint64_t i, gtsam::NonlinearFactorGraph& graph,
      const gtsam::PreintegratedImuMeasurements& pim);

  /// Extend the bias markov chain.
  gtsam::NonlinearFactorGraph add_bias_chain(
      uint64_t i, gtsam::NonlinearFactorGraph& graph);

  /// Add a prior factor on the first state and insert the values.
  virtual void set_prior_state(const gtsam::NavState& state_0,
                               bool add_bias_prior = true);

  /**
   * @brief Preintegrate IMU measurement and add the necessary factors and
   * initial values.
   *
   * @param k The discrete time index for the measurement.
   * @param measured_omega The measured angular velocity.
   * @param measured_acceleration The measured linear acceleration.
   * @param add_bias Flag indicating if a new bias value should be added to the
   * bias markov chain.
   */
  void step(uint64_t k, const gtsam::Vector3& measured_omega,
            const gtsam::Vector3& measured_acceleration, bool add_bias = true);

  virtual void print() const;

  /// Optimize the factor graph to get the results.
  virtual gtsam::Values optimize(bool verbose = false) const;

  /**
   * @brief Run an incremental update step. Returns the updated values based on
   * the optimized BayesTree.
   *
   * @return gtsam::Values
   */
  virtual gtsam::Values update();

  gtsam::NonlinearFactorGraph& graph() { return graph_; }

  gtsam::Values& initial() { return initial_; }

  void setGraph(const gtsam::NonlinearFactorGraph& graph) { graph_ = graph; }

  void setInitial(const gtsam::Values& values) { initial_ = values; }

  dyne::RobotImu base_imu() const { return base_imu_; }

  /// Get the Preintegrated IMU measurements for states at IMU frequency.
  gtsam::PreintegratedImuMeasurements pim() const { return pim_; }
  /// The IMU frequency.
  double freq() const { return freq_; }
  /// The time delta between two measurements.
  double dt() const { return dt_; }
  /// The number of timesteps after which to perform estimation.
  uint64_t rate() const { return rate_; }
  /// The estimation frequency, e.g. 50 Hz.
  uint64_t estimation_rate() const { return estimation_rate_; }
  /// The current preintegrated NavState
  gtsam::NavState state() const { return state_i; }
  /// Check if IMU factor was added
  bool imu_factor_added() const { return add_imu_factor_; }
};

}  // namespace dyne
