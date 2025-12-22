#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <iostream>
#include <vector>

#include "popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector/common/include/common.hpp"

namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector {

PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI(
    const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::ValidationImpl() {
  int rows = GetInput().first;
  int cols = GetInput().second;
  return (rows > 0 && cols > 0) && GetOutput().empty();
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PreProcessingImpl() {
  rows_ = GetInput().first;
  cols_ = GetInput().second;

  GetOutput().resize(rows_, 0.0);
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (cols_ < size) {
    if (rank == 0) {
      std::vector<std::vector<double>> matrix(cols_);
      std::vector<double> vector(cols_, 0.0);

      for (int j = 0; j < cols_; ++j) {
        matrix[j].resize(rows_, 0.0);
        for (int i = 0; i < rows_; ++i) {
          matrix[j][i] = (i + j) * 1.5;
        }
      }

      for (int j = 0; j < cols_; ++j) {
        vector[j] = j * 2.0;
      }

      auto &result = GetOutput();
      for (int i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols_; ++j) {
          sum += matrix[j][i] * vector[j];
        }
        result[i] = sum;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  // определяем кол-во столбцов на каждый процесс
  int cols_per_proc = cols_ / size;
  int remainder = cols_ % size;

  // кол-во на текущий процесс
  int local_cols = cols_per_proc;
  if (rank < remainder) {
    local_cols = local_cols + 1;
  }

  // начальный столбец
  int start_col = 0;
  for (int i = 0; i < rank; ++i) {
    int cols_for_i = cols_per_proc;
    if (i < remainder) {
      cols_for_i = cols_for_i + 1;
    }
    start_col = start_col + cols_for_i;
  }

  // генерируем данные для каждого столбца
  std::vector<double> local_matrix(local_cols * rows_, 0.0);
  std::vector<double> local_vector(local_cols, 0.0);

  for (int j = 0; j < local_cols; ++j) {
    int global_col = start_col + j;
    local_vector[j] = global_col * 2.0;

    for (int i = 0; i < rows_; ++i) {
      local_matrix[j * rows_ + i] = (i + global_col) * 1.5;
    }
  }

  // вычисление результатат
  std::vector<double> local_result(rows_, 0.0);

  for (int i = 0; i < rows_; ++i) {
    double sum = 0.0;
    for (int j = 0; j < local_cols; ++j) {
      sum += local_matrix[j * rows_ + i] * local_vector[j];
    }
    local_result[i] = sum;
  }

  std::vector<double> global_result(rows_, 0.0);

  MPI_Allreduce(local_result.data(), global_result.data(), rows_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  auto &result = GetOutput();
  for (int i = 0; i < rows_; ++i) {
    result[i] = global_result[i];
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool PopovaEVerticalRibbonSchemeMatrixMultiplicationByVectorMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_vertical_ribbon_scheme_matrix_multiplication_by_vector
