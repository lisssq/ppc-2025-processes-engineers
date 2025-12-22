#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/seq/include/ops_seq.hpp"

#include <iostream>
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
  rows_ = GetInput().first;
  cols_ = GetInput().second;

  matrix_.resize(cols_);
  for (int j = 0; j < cols_; j++) {
    matrix_[j].resize(rows_, 0.0);
  }

  vector_.resize(cols_, 0.0);
  GetOutput().resize(rows_, 0.0);

  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::RunImpl() {
  // заполняем матрицу
  for (int j = 0; j < cols_; j++) {
    for (int i = 0; i < rows_; i++) {
      matrix_[j][i] = (i + j) * 1.5;
    }
  }

  // заполняем вектор
  for (int j = 0; j < cols_; j++) {
    vector_[j] = j * 2.0;
  }

  auto &result = GetOutput();

  // умножаем
  for (int i = 0; i < rows_; i++) {
    double sum = 0.0;
    for (int j = 0; j < cols_; j++) {
      sum += matrix_[j][i] * vector_[j];
    }
    result[i] = sum;
  }

  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
