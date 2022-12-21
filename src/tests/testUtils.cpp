#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>

#include <filesystem>

#include "src/utils.h"

/* *********************************************************************** */
TEST(Utils, ReadCsv) {
  std::filesystem::path cwd = std::filesystem::current_path() /
                              "../../../data/simulation/measurements.csv";
  std::vector<std::vector<std::string>> data = dyne::ReadCsv(cwd.string());

  // Test header values
  EXPECT(data[0][0] == "seq_id");
  EXPECT(data[0][1] == "monotonic_time");
  EXPECT(data[0][2] == "omega_x");
  EXPECT(data[0][3] == "omega_y");
  EXPECT(data[0][4] == "omega_z");
  EXPECT(data[0][5] == "acc_x");
  EXPECT(data[0][6] == "acc_y");
  EXPECT(data[0][7] == "acc_z");

  // Test 2nd data row
  EXPECT(data[2][0] == "1");
  EXPECT(data[2][1] == "0.01");
  EXPECT(data[2][2] == "0.09855460202823862");
  EXPECT(data[2][3] == "0.29872166172989606");
  EXPECT(data[2][4] == "-0.09973250398647296");
  EXPECT(data[2][5] == "-0.3411316438282974");
  EXPECT(data[2][6] == "0.0882941022117128");
  EXPECT(data[2][7] == "10.03580028410325");
}

/* *********************************************************************** */
TEST(Utils, DataLoader) {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::filesystem::path measurements_file =
      cwd / "../../../data/pybullet/pybullet_measurements.csv";
  std::filesystem::path ground_truth_file =
      cwd / "../../../data/pybullet/pybullet_ground_truth.csv";

  std::string robot_file = "../../../models/a1.urdf";

  dyne::DataLoader dataset(measurements_file, ground_truth_file, robot_file,
                           "trunk",
                           {"FR_lower", "FL_lower", "RR_lower", "RL_lower"});
  EXPECT_LONGS_EQUAL(1000, dataset.size());
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
