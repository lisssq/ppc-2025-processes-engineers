#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"
#include "util/include/util.hpp"

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
  int rows = GetInput().first;
  int cols = GetInput().second;

  matrix_.resize(rows, std::vector<double>(cols, 0.0));
  vector_.resize(cols, 0.0);
  GetOutput().resize(rows, 0.0);

  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::RunImpl() {
  int rows = GetInput().first;
  int cols = GetInput().second;
  auto &result = GetOutput();

  // матрица
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix_[i][j] = (i + j) * 1.5;
    }
  }
  // вектор
  for (int j = 0; j < cols; j++) {
    vector_[j] = j * 2.0;
  }
  // умножение
  for (int i = 0; i < rows; i++) {
    double sum = 0.0;
    for (int j = 0; j < cols; j++) {
      sum += matrix_[i][j] * vector_[j];
    }
    result[i] = sum;
  }
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
