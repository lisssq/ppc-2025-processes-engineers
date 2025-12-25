#include <gtest/gtest.h>

#include <cmath>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaEGlobalOptimozationRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const double k_step = 0.005;
  const double k_x_min = -5.0;
  const double k_x_max = 10.0;
  const double k_y_min = -5.0;
  const double k_y_max = 10.0;

  InType input_data{};

  void SetUp() override {
    input_data = InType{.x_min = k_x_min,
                        .x_max = k_x_max,
                        .y_min = k_y_min,
                        .y_max = k_y_max,
                        .step = k_step,
                        .func_id = FunctionType::kParabola1};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [x_min, y_min, f_min] = output_data;
    return std::abs(x_min - 2.0) < 1e-6 && std::abs(y_min - 3.0) < 1e-6 && std::abs(f_min - 0.0) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(PopovaEGlobalOptimozationRunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PopovaEGlobalOptimizationDividingSearchAreaMPI, PopovaEOptimisationSEQ>(
        PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaEGlobalOptimozationRunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEGlobalOptimozationRunPerfTest, kGtestValues, kPerfTestName);

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
