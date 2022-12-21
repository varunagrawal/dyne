#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>

#include "src/robot_imu.h"

TEST(RobotImu, Constructor) { dyne::RobotImu base_imu(250, 1e-3, 1e-3); }

TEST(RobotImu, Freq) {
  dyne::RobotImu base_imu(250, 1e-3, 1e-3);

  EXPECT_LONGS_EQUAL(250, base_imu.freq);
}

TEST(RobotImu, Gravity) {
  dyne::RobotImu base_imu(250, 1e-3, 1e-3, gtsam::Pose3(),
                          gtsam::imuBias::ConstantBias(), 10);
  auto pim_params = base_imu.pim_params();
  EXPECT(gtsam::assert_equal(gtsam::Vector3(0, 0, -10), pim_params->n_gravity));
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
