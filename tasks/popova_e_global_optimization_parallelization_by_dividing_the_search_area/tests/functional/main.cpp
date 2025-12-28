#include <gtest/gtest.h>

#include <algorithm>
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

class PopovaRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &label = std::get<1>(test_param);
    return label;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &[x_min, y_min, f_min] = output_data;
    const double target_x = std::clamp(2.0, input_data_.x_min, input_data_.x_max);
    const double target_y = std::clamp(3.0, input_data_.y_min, input_data_.y_max);
    const double f_expected = ((target_x - 2.0) * (target_x - 2.0)) + ((target_y - 3.0) * (target_y - 3.0));
    const double tol_pos = std::max(1e-6, input_data_.step);
    const double tol_f = std::max(1e-6, input_data_.step * input_data_.step * 2.0);

    bool x_ok = std::abs(x_min - target_x) < tol_pos;
    bool y_ok = std::abs(y_min - target_y) < tol_pos;
    bool f_ok = std::abs(f_min - f_expected) < tol_f;

    return x_ok && y_ok && f_ok;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(PopovaRunFuncTests, FindsGridMinimum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParams = {std::make_tuple(InType{0.0, 4.0, 0.0, 6.0, 1.0}, "test_1"),
                                             std::make_tuple(InType{0.0, 4.0, 0.0, 6.0, 0.50}, "test_2"),
                                             std::make_tuple(InType{-1.0, 4.0, -1.0, 7.0, 0.25}, "test_3"),
                                             std::make_tuple(InType{-2.0, 2.0, -2.0, 2.0, 0.50}, "test_4"),
                                             std::make_tuple(InType{1.0, 3.0, 2.0, 4.0, 0.20}, "test_5"),
                                             std::make_tuple(InType{0.0, 5.0, 0.0, 8.0, 0.33}, "test_6")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PopovaEGlobalOptimizationMPI, InType>(
        kTestParams, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area),
    ppc::util::AddFuncTask<PopovaEOptimisationSEQ, InType>(
        kTestParams, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = PopovaRunFuncTests::PrintFuncTestName<PopovaRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(GridSearchTests, PopovaRunFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
