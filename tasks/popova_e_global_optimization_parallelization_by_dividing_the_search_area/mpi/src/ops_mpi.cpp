#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

#include "popova_e_global_optimization_parallelization_by_dividing_the_search_area/common/include/common.hpp"

namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area {

PopovaEGlobalOptimizationDividingSearchAreaMPI::PopovaEGlobalOptimizationDividingSearchAreaMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(0.0, 0.0, std::numeric_limits<double>::max());
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::ValidationImpl() {
  const auto &in = GetInput();
  return ((in.x_max > in.x_min) && (in.y_max > in.y_min) && (in.step > 0));
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PreProcessingImpl() {
  return true;
}

double PopovaEGlobalOptimizationDividingSearchAreaMPI::FunctionToOptimize(double x, double y) {
  const auto &in = GetInput();

  switch (in.func_id) {
    case FunctionType::kParabola1:
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
    case FunctionType::kParabola2:
      return (x * x) + (y * y);
    case FunctionType::kParabola3:
      return (((x - 1.0) * (x - 1.0)) + ((y - 1.0) * (y - 1.0))) + 1.0;
    case FunctionType::kParabola4:
      return ((x + 1.0) * (x + 1.0)) + ((y + 1.0) * (y + 1.0));
    default:
      return ((x - 2.0) * (x - 2.0)) + ((y - 3.0) * (y - 3.0));
  }
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::ComputeProcessArea(int rank, int size, double &x_start,
                                                                        double &x_end, double &y_start, double &y_end) {
  const auto &in = GetInput();

  double x_range = in.x_max - in.x_min;
  double x_step_per_process = x_range / size;

  x_start = in.x_min + rank * x_step_per_process;
  x_end = (rank == size - 1) ? in.x_max : (in.x_min + (rank + 1) * x_step_per_process);

  y_start = in.y_min;
  y_end = in.y_max;
}

void PopovaEGlobalOptimizationDividingSearchAreaMPI::FindGlobalMinimum(double local_x, double local_y,
                                                                       double local_value, double &global_x,
                                                                       double &global_y, double &global_value) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> all_values(size);
  std::vector<double> all_x(size);
  std::vector<double> all_y(size);

  MPI_Gather(&local_value, 1, MPI_DOUBLE, all_values.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_x, 1, MPI_DOUBLE, all_x.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_y, 1, MPI_DOUBLE, all_y.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int min_index = 0;
    for (int i = 1; i < size; ++i) {
      if (all_values[i] < all_values[min_index]) {
        min_index = i;
      }
    }

    global_x = all_x[min_index];
    global_y = all_y[min_index];
    global_value = all_values[min_index];
  }

  MPI_Bcast(&global_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::RunImpl() {
  const auto &in = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double x_start = 0.0;
  double x_end = 0.0;
  double y_start = 0.0;
  double y_end = 0.0;
  ComputeProcessArea(rank, size, x_start, x_end, y_start, y_end);

  const double epsilon = 1e-12;
  if (rank > 0) {
    x_start -= epsilon;
  }
  if (rank < size - 1) {
    x_end += epsilon;
  }
  if (rank > 0) {
    y_start -= epsilon;
  }
  if (rank < size - 1) {
    y_end += epsilon;
  }

  double best_x = x_start;
  double best_y = y_start;
  double min_value = std::numeric_limits<double>::max();

  double x = x_start;
  while (x <= x_end + epsilon) {
    double current_x = std::min(x, x_end);

    double y = y_start;
    while (y <= y_end + epsilon) {
      double current_y = std::min(y, y_end);

      double value = FunctionToOptimize(current_x, current_y);
      if (value < min_value) {
        min_value = value;
        best_x = current_x;
        best_y = current_y;
      }

      y += in.step;
    }

    x += in.step;
  }

  double global_x = 0.0;
  double global_y = 0.0;
  double global_value = 0.0;
  FindGlobalMinimum(best_x, best_y, min_value, global_x, global_y, global_value);

  GetOutput() = std::make_tuple(global_x, global_y, global_value);
  return true;
}

bool PopovaEGlobalOptimizationDividingSearchAreaMPI::PostProcessingImpl() {
  return true;
}

}  // namespace popova_e_global_optimization_parallelization_by_dividing_the_search_area
