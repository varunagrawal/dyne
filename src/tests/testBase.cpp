#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>

#include "src/base.h"
#include "src/robot_imu.h"
#include "src/utils.h"

using gtsam::symbol_shorthand::X;

dyne::DataLoader LoadDataset() {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::filesystem::path measurements_file =
      cwd / "../../../data/pybullet/pybullet_measurements.csv";
  std::filesystem::path ground_truth_file =
      cwd / "../../../data/pybullet/pybullet_ground_truth.csv";

  std::string robot_file = "../../../models/a1.urdf";

  dyne::DataLoader dataset(measurements_file, ground_truth_file, robot_file);
  return dataset;
}

dyne::BaseImu GetEstimator() {
  gtsam::imuBias::ConstantBias bias(gtsam::Vector3::Zero(),
                                    gtsam::Vector3::Zero());
  dyne::RobotImu base_imu(200, 5.886e-4, 1.745e-4, gtsam::Pose3(), bias);

  dyne::BaseImu estimator(base_imu, 50);
  return estimator;
}

/* ************************************************************************/
TEST(BaseImu, Constructor) {
  dyne::RobotImu imu;
  dyne::BaseImu estimator(imu);

  // dyne::NoiseParams a1(0.01, 0.01, 1e-3, 1, 3.5e-2, 1e-2, 1e-6, 1e-6);
}

/* ************************************************************************/
TEST(BaseImu, Preintegrate) {
  dyne::DataLoader dataset = LoadDataset();
  dyne::BaseImu estimator = GetEstimator();

  estimator.set_prior_state(dataset.ground_truth_state(0));

  // simulate the data loop
  for (size_t k = 0; k < dataset.size(); k += 1) {
    gtsam::Vector3 measured_omega = dataset.angular_velocity(k);
    gtsam::Vector3 measured_acc = dataset.linear_acceleration(k);

    estimator.step(k, measured_omega, measured_acc);
  }

  gtsam::Values result = estimator.optimize();

  EXPECT(gtsam::assert_equal(dataset.ground_truth_state(0).pose(),
                             result.at<gtsam::Pose3>(X(0)), 1e-6));
  EXPECT(gtsam::assert_equal(dataset.ground_truth_state(3).pose(),
                             result.at<gtsam::Pose3>(X(3)), 1e-2));
  EXPECT(gtsam::assert_equal(dataset.ground_truth_state(7).pose(),
                             result.at<gtsam::Pose3>(X(7)), 1e-1));
  EXPECT(gtsam::assert_equal(dataset.ground_truth_state(11).pose(),
                             result.at<gtsam::Pose3>(X(11)), 1e-1));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
