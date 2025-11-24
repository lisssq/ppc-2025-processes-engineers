#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[a, b, n] = input_data_;

#ifdef BUILD_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
#endif

    double exp_integral = ((b * b * b * b) / 4 - 2 * b * b) - ((a * a * a * a) / 4 - 2 * a * a);

    double sredn = exp_integral / (b - a);
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(sredn), 1.0);
    double epsilon = std::max(3.0 * std_dev, 1e-3);

    return std::abs(output_data - exp_integral) <= epsilon;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PopovaERunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {{std::make_tuple(std::make_tuple(0.0, 1.0, 1000), "test1"),
                                              std::make_tuple(std::make_tuple(0.0, 2.0, 1000), "test2"),
                                              std::make_tuple(std::make_tuple(1.0, 3.0, 5000), "test3"),
                                              std::make_tuple(std::make_tuple(0.0, 1.0, 2000), "test4"),
                                              std::make_tuple(std::make_tuple(-2.0, -0.5, 1500), "test5"),
                                              std::make_tuple(std::make_tuple(-1.0, 5.0, 80000), "test6"),
                                              std::make_tuple(std::make_tuple(-1.0, 5.0, 450000), "test7"),
                                              std::make_tuple(std::make_tuple(4.5, 5.0, 100), "test8"),
                                              std::make_tuple(std::make_tuple(4.5, 5.0, 8000), "test9"),
                                              std::make_tuple(std::make_tuple(-2.0, 5.0, 5), "test10")}};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PopovaEIntegrMonteCarloMPI, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo),
    ppc::util::AddFuncTask<PopovaEIntegrMonteCarloSEQ, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PopovaERunFuncTestsProcesses::PrintFuncTestName<PopovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, PopovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace popova_e_integr_monte_carlo
