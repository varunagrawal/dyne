#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>

#include "src/noise_params.h"

TEST(NoiseParams, Constructor) {
  dyne::NoiseParams n;

  dyne::NoiseParams a1(0.01, 0.01, gtsam::Vector6::Ones() * 1e-3,
                       gtsam::Vector6::Ones() * 1, 3.5e-2, 1e-2, 1e-6, 1e-6);
}

//******************************************************************************
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
//******************************************************************************
