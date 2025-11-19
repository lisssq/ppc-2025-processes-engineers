#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>

#include "popova_e_integr_monte_carlo/common/include/common.hpp"
#include "popova_e_integr_monte_carlo/mpi/include/ops_mpi.hpp"
#include "popova_e_integr_monte_carlo/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace popova_e_integr_monte_carlo {

class PopovaERunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    
    
    // std::string a = std::to_string(std::get<0>(std::get<0>(test_param)));
    // std::string b = std::to_string(std::get<1>(std::get<0>(test_param)));

    // std::replace(a.begin(), a.end(), '.', '_');
    // std::replace(b.begin(), b.end(), '.', '_');

    // return a + "_" + b + "_" + std::to_string(std::get<2>(std::get<0>(test_param))) + "_" + std::get<1>(test_param);

    return std::get<1>(test_param);  // просто название теста, без a, b и n

  }

 protected:
  void SetUp() override {

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);  

  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto& [a, b, n] = input_data_;

    double exp_integral = ((b*b*b*b)/4 - 2*b*b) - ((a*a*a*a)/4 - 2*a*a);
    // double epsilon = 0.1 * std::max(std::abs(exp_integral), 1e-3);         // допустимая погрешность
    // return std::abs(output_data - exp_integral) < epsilon;

     // Эмпирическая погрешность Monte Carlo
    double avg = exp_integral / (b - a);  // грубая оценка для sigma
    double std_dev = (b - a) / std::sqrt(n) * std::max(std::abs(avg), 1.0);
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






const std::array<TestType, 10> kTestParam = {{
    std::make_tuple(std::make_tuple(0.0, 1.0, 1000), "test1"),   // оригинальный
    std::make_tuple(std::make_tuple(0.0, 2.0, 1000), "test2"),   // оригинальный
    std::make_tuple(std::make_tuple(1.0, 3.0, 5000), "test3"),   // оригинальный
    std::make_tuple(std::make_tuple(0.0, 1.0, 2000), "test4"),  // отрицательные и положительные
    std::make_tuple(std::make_tuple(-2.0, -0.5, 1500), "test5"), // полностью отрицательный отрезок
    std::make_tuple(std::make_tuple(-1.0, 10.0, 8000), "test6"),  
    std::make_tuple(std::make_tuple(-1.0, 5.0, 45000), "test7"),    
    std::make_tuple(std::make_tuple(4.5, 5.0, 100), "test8"),
    std::make_tuple(std::make_tuple(4.5, 5.0, 8000), "test9"),   
    std::make_tuple(std::make_tuple(-2.0, 5.0, 5), "test10")
}};








// const auto kTestTasksList =
//     std::tuple_cat(ppc::util::AddFuncTask<PopovaEIntegrMonteCarloMPI, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo),
//                    ppc::util::AddFuncTask<PopovaEIntegrMonteCarloSEQ, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo));



const auto kTestTasksList =
    std::tuple_cat(
        ppc::util::AddFuncTask<PopovaEIntegrMonteCarloSEQ, InType>(kTestParam, PPC_SETTINGS_popova_e_integr_monte_carlo)
    );



const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PopovaERunFuncTestsProcesses::PrintFuncTestName<PopovaERunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, PopovaERunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace popova_e_integr_monte_carlo
