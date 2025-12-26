#include <gtest/gtest.h>

#include <cmath>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaERunPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const double kStep_ = 0.01;
  const double kXMax_ = 4.0;
  const double kYMax_ = 6.0;

  InType input_data_{};

  void SetUp() override {
    input_data_ = InType{0.0, kXMax_, 0.0, kYMax_, kStep_};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto [x_min, y_min, f_min] = output_data;
    return std::abs(x_min - 2.0) < 1e-6 && std::abs(y_min - 3.0) < 1e-6 && std::abs(f_min - 0.0) < 1e-6;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PopovaERunPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PopovaEOptimisationFieldMPI, PopovaEOptimisationSEQ>(
    PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaERunPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaERunPerfTest, kGtestValues, kPerfTestName);

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
