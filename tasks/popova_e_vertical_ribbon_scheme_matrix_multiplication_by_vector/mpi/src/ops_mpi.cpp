#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <utility>
#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

std::pair<int, int> PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::GetLocalColumnsCounts(int cols,
                                                                                                      int rank,
                                                                                                      int size) {
  int base_cols = cols / size;
  int remainder = cols % size;

  int local_cols = base_cols;
  if (rank < remainder) {
    local_cols = local_cols + 1;
  }

  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    int cols_for_i = base_cols;
    if (i < remainder) {
      cols_for_i = cols_for_i + 1;
    }
    start_col = start_col + cols_for_i;
  }

  return {local_cols, start_col};
}

void PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::CountSeq(int rows, int cols,
                                                                          std::vector<double> &result) {
  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      double matrix_value = (i + j) * 1.5;
      double vector_value = j * 2.0;
      sum += matrix_value * vector_value;
    }
    result[i] = sum;
  }
}

void PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::CountMpi(int rows, int cols, int rank, int size,
                                                                          std::vector<double> &result) {
  auto [local_cols, start_col] = GetLocalColumnsCounts(cols, rank, size);

  std::vector<double> local_result(rows, 0.0);

  for (int i = 0; i < rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < local_cols; ++j) {
      int global_col = start_col + j;
      double matrix_value = (i + global_col) * 1.5;
      double vector_value = global_col * 2.0;
      sum += matrix_value * vector_value;
    }
    local_result[i] = sum;
  }

  MPI_Allreduce(local_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI(
    const InType &in) {
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

  GetOutput().resize(rows_, 0.0);
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (cols_ < size) {
    if (rank == 0) {
      CountSeq(rows_, cols_, GetOutput());
    }
    MPI_Bcast(GetOutput().data(), rows_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    CountMpi(rows_, cols_, rank, size, GetOutput());
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
