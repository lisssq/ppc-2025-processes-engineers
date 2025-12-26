#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"
#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

class PopovaEGlobalOptimizationRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
    auto [x, y, f] = output_data;
    const auto &in = input_data_;

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
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PopovaEGlobalOptimizationRunFuncTestsProcesses, Test) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    // (x-2)^2 + (y-3)^2, минимум в (2, 3)
    std::make_tuple(
        OptimizationInput{
            .x_min = 0.0, .x_max = 4.0, .y_min = 0.0, .y_max = 6.0, .step = 0.5, .func_id = FunctionType::kParabola1},
        "func1_step_05"),
    std::make_tuple(
        OptimizationInput{
            .x_min = 0.0, .x_max = 4.0, .y_min = 0.0, .y_max = 6.0, .step = 0.1, .func_id = FunctionType::kParabola1},
        "func1_step_01"),
    std::make_tuple(
        OptimizationInput{
            .x_min = 0.0, .x_max = 4.0, .y_min = 0.0, .y_max = 6.0, .step = 0.01, .func_id = FunctionType::kParabola1},
        "func1_step_001"),

    // x^2 + y^2, минимум в (0, 0)
    std::make_tuple(
        OptimizationInput{
            .x_min = -2.0, .x_max = 2.0, .y_min = -2.0, .y_max = 2.0, .step = 0.1, .func_id = FunctionType::kParabola2},
        "func2_step_01"),
    std::make_tuple(OptimizationInput{.x_min = -2.0,
                                      .x_max = 2.0,
                                      .y_min = -2.0,
                                      .y_max = 2.0,
                                      .step = 0.05,
                                      .func_id = FunctionType::kParabola2},
                    "func2_step_005"),

    // (x-1)^2 + (y-1)^2 + 1, минимум в (1, 1)
    std::make_tuple(
        OptimizationInput{
            .x_min = -1.0, .x_max = 3.0, .y_min = -1.0, .y_max = 3.0, .step = 0.1, .func_id = FunctionType::kParabola3},
        "func3_step_01"),

    // (x+1)^2 + (y+1)^2, минимум в (-1, -1)
    std::make_tuple(
        OptimizationInput{
            .x_min = -3.0, .x_max = 1.0, .y_min = -3.0, .y_max = 1.0, .step = 0.1, .func_id = FunctionType::kParabola4},
        "func4_step_01"),
    std::make_tuple(OptimizationInput{.x_min = -3.0,
                                      .x_max = 1.0,
                                      .y_min = -3.0,
                                      .y_max = 1.0,
                                      .step = 0.05,
                                      .func_id = FunctionType::kParabola4},
                    "func4_step_005")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PopovaEOptimisationSEQ, InType>(
        kTestParam, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area),
    ppc::util::AddFuncTask<PopovaEGlobalOptimizationDividingSearchAreaMPI, InType>(
        kTestParam, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(
    GlobalOptimizationTests, PopovaEGlobalOptimizationRunFuncTestsProcesses, kGtestValues,
    PopovaEGlobalOptimizationRunFuncTestsProcesses::PrintFuncTestName<PopovaEGlobalOptimizationRunFuncTestsProcesses>);

}  // namespace

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
