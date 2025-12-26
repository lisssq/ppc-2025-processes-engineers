#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
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
    const auto &input = std::get<0>(test_param);
    const auto &label = std::get<1>(test_param);
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << input.step;
    auto step_str = ss.str();
    std::replace(step_str.begin(), step_str.end(), '.', 'p');
    return label + "_step_" + step_str;
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
    const double f_expected = (target_x - 2.0) * (target_x - 2.0) + (target_y - 3.0) * (target_y - 3.0);
    const double tol_pos = std::max(1e-6, input_data_.step);
    const double tol_f = std::max(1e-6, input_data_.step * input_data_.step * 2.0);
    return (std::abs(x_min - target_x) < tol_pos) && (std::abs(y_min - target_y) < tol_pos) &&
           (std::abs(f_min - f_expected) < tol_f);
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

const std::array<TestType, 6> kTestParams = {std::make_tuple(InType{0.0, 4.0, 0.0, 6.0, 1.0}, "unit"),
                                             std::make_tuple(InType{0.0, 4.0, 0.0, 6.0, 0.50}, "coarse"),
                                             std::make_tuple(InType{-1.0, 4.0, -1.0, 7.0, 0.25}, "shifted"),
                                             std::make_tuple(InType{-2.0, 2.0, -2.0, 2.0, 0.50}, "symmetric"),
                                             std::make_tuple(InType{1.0, 3.0, 2.0, 4.0, 0.20}, "tight"),
                                             std::make_tuple(InType{0.0, 5.0, 0.0, 8.0, 0.33}, "fraction")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PopovaEOptimisationFieldMPI, InType>(
        kTestParams, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area),
    ppc::util::AddFuncTask<PopovaEOptimisationSEQ, InType>(
        kTestParams, PPC_SETTINGS_popova_e_global_optimization_parallelization_by_dividing_the_search_area));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = PopovaRunFuncTests::PrintFuncTestName<PopovaRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(GridSearchTests, PopovaRunFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
