#include <CppUnitLite/TestHarness.h>
#include <gtdynamics/universal_robot/Robot.h>
#include <gtdynamics/universal_robot/sdf.h>
#include <gtsam/base/Testable.h>

#include "src/dyne.h"
#include "src/robot_imu.h"

TEST(Dyne, Constructor) {
  dyne::RobotImu imu;
  dyne::NoiseParams a1;
  gtdynamics::Robot robot =
      gtdynamics::CreateRobotFromFile(std::string("../../../models/a1.urdf"));

  std::map<std::string, gtsam::Vector3> contact_in_com = {
      {"FR_toe", gtsam::Vector3::Zero()},
      {"FL_toe", gtsam::Vector3::Zero()},
      {"RR_toe", gtsam::Vector3::Zero()},
      {"RL_toe", gtsam::Vector3::Zero()}};

  dyne::Dyne estimator(imu, std::string("../../../models/a1.urdf"), 50, "trunk",
                       {"FR_lower", "FL_lower", "RR_lower", "RL_lower"},
                       contact_in_com, a1);
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
