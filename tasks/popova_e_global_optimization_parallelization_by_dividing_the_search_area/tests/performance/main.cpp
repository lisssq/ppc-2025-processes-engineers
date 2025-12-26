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
    input_data = InType{k_x_min, k_x_max, k_y_min, k_y_max, k_step, FunctionType::kParabola1};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [x, y, f] = output_data;
    const auto &in = input_data;

    switch (in.func_id) {
      case FunctionType::kParabola1:
        return std::abs(x - 2.0) < 1e-5 && std::abs(y - 3.0) < 1e-5 && std::abs(f) < 1e-5;
      case FunctionType::kParabola2:
        return std::abs(x) < 1e-5 && std::abs(y) < 1e-5 && std::abs(f) < 1e-5;
      case FunctionType::kParabola3:
        return std::abs(x - 1.0) < 1e-5 && std::abs(y - 1.0) < 1e-5 && std::abs(f - 1.0) < 1e-5;
      case FunctionType::kParabola4:
        return std::abs(x + 1.0) < 1e-5 && std::abs(y + 1.0) < 1e-5 && std::abs(f) < 1e-5;
      default:
        return false;
    }
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
