// #include <gtest/gtest.h>

// #include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
// #include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"
// #include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"
// #include "util/include/perf_test_util.hpp"

// namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

// class PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
//   const int kCount_ = 100;
//   InType input_data_{};

//   void SetUp() override {
//     input_data_ = kCount_;
//   }

//   bool CheckTestOutputData(OutType &output_data) final {
//     return input_data_ == output_data;
//   }

//   InType GetTestInputData() final {
//     return input_data_;
//   }
// };

// TEST_P(PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses, RunPerfModes) {
//   ExecuteTest(GetParam());
// }

// const auto kAllPerfTasks =
//     ppc::util::MakeAllPerfTasks<InType, PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI, PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ>(PPC_SETTINGS_popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector);

// const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

// const auto kPerfTestName = PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses::CustomPerfTestName;

// INSTANTIATE_TEST_SUITE_P(RunModeTests, PopovaEMatrixMultiplicationByVectorRunPerfTestProcesses, kGtestValues, kPerfTestName);

// }  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
