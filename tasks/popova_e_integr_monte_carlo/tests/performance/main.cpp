#include <gtest/gtest.h>
#include <cmath>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
// #include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaEIntegrMonteCarloRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    // Для performance тестов используем большие значения n
    // a=0.0, b=2.0, n=1000000 - большое количество точек для измерения производительности
    input_data_ = std::make_tuple(0.0, 2.0, 1000000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto& [a, b, n] = input_data_;
    
    // Ожидаемое значение интеграла: ((b^4)/4 - 2b^2) - ((a^4)/4 - 2a^2)
    double exp_integral = ((b*b*b*b)/4 - 2*b*b) - ((a*a*a*a)/4 - 2*a*a);
    
    // Эмпирическая погрешность Monte Carlo
    double avg = exp_integral / (b - a);
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(avg), 1.0);
    double epsilon = std::max(3.0 * std_dev, 1e-3);
    
    return std::abs(output_data - exp_integral) <= epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(PopovaEIntegrMonteCarloRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}


// Пока используем только SEQ, так как MPI еще не реализован
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PopovaEIntegrMonteCarloSEQ>(PPC_SETTINGS_popova_e_integr_monte_carlo);


// Когда MPI будет готов, раскомментируйте и используйте:

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, PopovaEIntegrMonteCarloMPI, PopovaEIntegrMonteCarloSEQ>(PPC_SETTINGS_popova_e_integr_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaEIntegrMonteCarloRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEIntegrMonteCarloRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace popova_e_integr_monte_carlo
