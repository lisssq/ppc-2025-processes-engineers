#include <gtest/gtest.h>

#include <cmath>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaERunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr double kStep = 0.01;
  static constexpr double kXMax = 4.0;
  static constexpr double kYMax = 6.0;

  InType input_data{};

  void SetUp() override {
    input_data = InType{0.0, kXMax, 0.0, kYMax, kStep};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [x_min, y_min, f_min] = output_data;
    return std::abs(x_min - 2.0) < 1e-6 && std::abs(y_min - 3.0) < 1e-6 && std::abs(f_min - 0.0) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(PopovaERunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PopovaEGlobalOptimizationMPI, PopovaEOptimisationSEQ>(
    PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaERunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaERunPerfTest, kGtestValues, kPerfTestName);

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
