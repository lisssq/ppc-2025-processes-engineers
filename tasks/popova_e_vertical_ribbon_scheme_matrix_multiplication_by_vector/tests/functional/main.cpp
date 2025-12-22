#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"
#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

std::vector<double> CalculateExpectedResult(const InType &input) {
  int rows = input.first;
  int cols = input.second;
  std::vector<double> expected(rows, 0.0);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      double matrix_value = (i + j) * 1.5;
      double vector_value = j * 2.0;
      expected[i] += matrix_value * vector_value;
    }
  }

  return expected;
}

class PopovaEMatrixVectorRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    auto input = std::get<0>(test_param);
    std::string description = std::get<1>(test_param);
    return std::to_string(input.first) + "x" + std::to_string(input.second) + "_" + description;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rows = input_data_.first;

    if (output_data.size() != static_cast<size_t>(rows)) {
      return false;
    }

    auto expected = CalculateExpectedResult(input_data_);

    double epsilon = 1e-10;
    for (size_t i = 0; i < expected.size(); i++) {
      if (std::abs(output_data[i] - expected[i]) > epsilon) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(PopovaEMatrixVectorRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {std::make_tuple(std::make_pair(1, 1), "square_matrix_1x1"),
                                             std::make_tuple(std::make_pair(2, 2), "square_matrix_2x2"),
                                             std::make_tuple(std::make_pair(3, 3), "square_matrix_3x3"),
                                             std::make_tuple(std::make_pair(4, 4), "square_matrix_4x4"),
                                             std::make_tuple(std::make_pair(5, 5), "square_matrix_5x5"),
                                             std::make_tuple(std::make_pair(10, 10), "square_matrix_10x10"),
                                             std::make_tuple(std::make_pair(16, 16), "square_matrix_16x16"),
                                             std::make_tuple(std::make_pair(17, 17), "square_matrix_17x17"),
                                             std::make_tuple(std::make_pair(3, 5), "rectangular_matrix_3x5"),
                                             std::make_tuple(std::make_pair(5, 3), "rectangular_matrix_5x3"),
                                             std::make_tuple(std::make_pair(15, 3), "rectangular_matrix_15x3"),
                                             std::make_tuple(std::make_pair(3, 15), "rectangular_matrix_3x15"),
                                             std::make_tuple(std::make_pair(84, 11), "rectangular_matrix_84x11"),
                                             std::make_tuple(std::make_pair(11, 84), "rectangular_matrix_11x84")};
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI, InType>(
                       kTestParam, PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector),
                   ppc::util::AddFuncTask<PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ, InType>(
                       kTestParam, PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector));

// const auto kTestTasksList =
//     std::tuple_cat(ppc::util::AddFuncTask<PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ, InType>(
//         kTestParam, PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PopovaEMatrixVectorRunFuncTestsProcesses::PrintFuncTestName<PopovaEMatrixVectorRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMultiplicationTests, PopovaEMatrixVectorRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
