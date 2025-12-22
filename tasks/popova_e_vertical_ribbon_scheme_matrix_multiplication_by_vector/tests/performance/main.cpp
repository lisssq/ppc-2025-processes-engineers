#include <gtest/gtest.h>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"
#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

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

class PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_pair(200, 200);
    // input_data_ = std::make_pair(2000,2000);
    // input_data_ = std::make_pair(1500, 2000);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rows = input_data_.first;

    if (output_data.size() != static_cast<size_t>(rows)) {
      return false;
    }

    auto expected = CalculateExpectedResult(input_data_);

    double epsilon = 1e-8;
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
};

TEST_P(PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI,
                                PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ>(
        PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector);

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ>(
//         PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
