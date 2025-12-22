#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::
    PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::ValidationImpl() {
  int rows = GetInput().first;
  int cols = GetInput().second;
  return (rows > 0 && cols > 0);
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PreProcessingImpl() {
  rows_ = GetInput().first;
  cols_ = GetInput().second;
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (cols_ < size) {
    if (rank == 0) {
      std::vector<std::vector<double>> matrix(cols_, std::vector<double>(rows_));
      std::vector<double> vec(cols_);

      for (int j = 0; j < cols_; ++j) {
        vec[j] = j * 2.0;
        for (int i = 0; i < rows_; ++i) {
          matrix[j][i] = (i + j) * 1.5;
        }
      }

      std::vector<double> result(rows_, 0.0);
      for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
          result[i] += matrix[j][i] * vec[j];
        }
      }

      GetOutput() = result;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int base_cols = cols_ / size;
  int remainder = cols_ % size;

  int local_cols = base_cols + (rank < remainder ? 1 : 0);

  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    start_col += base_cols + (i < remainder ? 1 : 0);
  }

  std::vector<double> local_matrix(local_cols * rows_);
  std::vector<double> local_vector(local_cols);

  for (int j = 0; j < local_cols; ++j) {
    int global_col = start_col + j;
    local_vector[j] = global_col * 2.0;

    for (int i = 0; i < rows_; ++i) {
      local_matrix[j * rows_ + i] = (i + global_col) * 1.5;
    }
  }

  std::vector<double> local_result(rows_, 0.0);

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < local_cols; ++j) {
      local_result[i] += local_matrix[j * rows_ + i] * local_vector[j];
    }
  }

  std::vector<double> global_result;
  if (rank == 0) {
    global_result.resize(rows_, 0.0);
  }

  MPI_Reduce(local_result.data(),
             rank == 0 ? global_result.data() : nullptr,
             rows_,
             MPI_DOUBLE,
             MPI_SUM,
             0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_result;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
