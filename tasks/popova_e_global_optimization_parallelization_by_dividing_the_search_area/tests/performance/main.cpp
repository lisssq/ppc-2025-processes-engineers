#include <gtest/gtest.h>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaEGlobalOptimozationRunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const double kStep_ = 0.005;  
  const double kXMin_ = -5.0;   
  const double kXMax_ = 10.0;
  const double kYMin_ = -5.0;
  const double kYMax_ = 10.0;

  InType input_data_{};

  void SetUp() override {
    input_data_ = InType{kXMin_, kXMax_, kYMin_, kYMax_, kStep_, FunctionType::kParabola1};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [x_min, y_min, f_min] = output_data;
    return std::abs(x_min - 2.0) < 1e-6 && std::abs(y_min - 3.0) < 1e-6 && std::abs(f_min - 0.0) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_data_;
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
