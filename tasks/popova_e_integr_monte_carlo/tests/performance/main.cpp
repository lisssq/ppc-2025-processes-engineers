#include <gtest/gtest.h>

#include <cmath>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaEIntegrMonteCarloRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 1000;
  InType input_data_{};

  void SetUp() override {
    // InType is std::tuple<double, double, int>
    // For Monte Carlo integration: (a, b, n) where a and b are integration bounds, n is point count
    input_data_ = std::make_tuple(0.0, 1.0, kCount_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // OutType is double (the computed integral value)
    // For f(x) = x^2 on [0, 1], expected value is 1/3 ≈ 0.333...
    const auto& [a, b, n] = input_data_;
    double expected = (b * b * b - a * a * a) / 3.0;  // точный интеграл f(x)=x^2
    double epsilon = 0.1 * expected;  // допустимая погрешность для performance теста
    return std::abs(output_data - expected) < epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PopovaEIntegrMonteCarloRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PopovaEIntegrMonteCarloSEQ>(PPC_SETTINGS_popova_e_integr_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaEIntegrMonteCarloRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEIntegrMonteCarloRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace popova_e_integr_monte_carlo
