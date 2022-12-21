#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/slam/BetweenFactor.h>

#include "src/hybrid.h"
#include "src/robot_imu.h"

using gtsam::symbol_shorthand::M;  // discrete mode
using gtsam::symbol_shorthand::P;  // landmark point
using gtsam::symbol_shorthand::X;  // state

using MotionModel = gtsam::BetweenFactor<double>;

TEST(Quadruped, Estimation) { dyne::Hybrid estimator; }

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
