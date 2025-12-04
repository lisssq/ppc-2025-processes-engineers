#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tuple>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaEIntegrMonteCarloRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_tuple(0.0, 2.0, 100000, quadratic_func);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[a, b, n, func_id] = input_data_;

    // double exp_integral = ((b * b * b * b) / 4 - 2 * b * b) - ((a * a * a * a) / 4 - 2 * a * a);
    double exp_integral = 0.0;
    exp_integral = FunctionPair::integral(func_id, b) - FunctionPair::integral(func_id, a);

    // switch (func_id) {
    //   case linear_func:
    //     exp_integral = ((b * b) + (7 * b)) - ((a * a) + (7 * a));
    //     break;
    //   case quadratic_func:
    //     exp_integral = (2.5 * b * b - b * b * b + 7 * b) - (2.5 * a * a - a * a * a + 7 * a);
    //     break;
    //   case cubic_func:
    //     exp_integral = ((b * b * b * b) / 4 - 2 * b * b) - ((a * a * a * a) / 4 - 2 * a * a);
    //     break;
    //   case cos_func:
    //     exp_integral = 0.5 * (std::sin(2 * b) - std::sin(2 * a));
    //     break;
    //   case exp_func:
    //     exp_integral = (-(b + 0.5) * std::exp(-2 * b) + 4 * b) - (-(a + 0.5) * std::exp(-2 * a) + 4 * a);
    //     break;
    // }

    double sredn = exp_integral / (b - a);
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(sredn), 1.0);
    double epsilon = std::max(10.0 * std_dev, 1e-2);
    return std::abs(output_data - exp_integral) <= epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PopovaEIntegrMonteCarloRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PopovaEIntegrMonteCarloMPI, PopovaEIntegrMonteCarloSEQ>(
    PPC_SETTINGS_popova_e_integr_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaEIntegrMonteCarloRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEIntegrMonteCarloRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace popova_e_integr_monte_carlo
