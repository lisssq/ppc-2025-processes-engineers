#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"

#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::ValidationImpl() {
  int rows = GetInput().first;
  int cols = GetInput().second;
  return (rows > 0 && cols > 0) && GetOutput().empty();
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::PreProcessingImpl() {
  rows_ = GetInput().first;
  cols_ = GetInput().second;

  GetOutput().resize(rows_, 0.0);

  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::RunImpl() {
  int rows = rows_;
  int cols = cols_;
  auto &result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += (i + j) * 1.5 * (j * 2.0);
    }
    result[i] = sum;
  }

  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
